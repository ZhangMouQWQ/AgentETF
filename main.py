#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ETF 动量轮动量化交易系统
========================
单文件完整实现 · 可直接运行

模块:
  1. 关键参数配置
  2. ETF池管理
  3. 数据获取 (腾讯日线 + 新浪实时)
  4. 策略计算 (动量/波动率/资金流/质量评分)
  5. 择时与选股
  6. 回测引擎 (严格无未来函数)
  7. 交易成本计算
  8. 日志与记录
  9. 主程序

运行:
  python main.py              # 完整回测
  python main.py --signal     # 仅生成今日信号
  python main.py --live       # 实盘信号 (需盘中)
"""

import os, json, time, argparse, traceback
from datetime import datetime, timezone, timedelta

import pandas as pd
import numpy as np

from config import get_config
from logger import TradeLogger
from data_fetcher import DataFetcher

cfg = get_config()
log = TradeLogger()
_fetcher = DataFetcher()

# ╔══════════════════════════════════════════════════════════════╗
# ║              1. 参数配置 (统一使用 config.py)                  ║
# ╚══════════════════════════════════════════════════════════════╝

# 所有可调参数集中在 config.py, 此处仅做本地别名引用
# cfg.MOM_LONG, cfg.MOM_SHORT, cfg.VOL_WINDOW, cfg.MIN_VOL,
# cfg.MAX_DAILY_RISE, cfg.STOP_LOSS_PCT, cfg.DATA_LEN, cfg.LOT_SIZE 等

# ── 路径 ──
DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data')
os.makedirs(DATA_DIR, exist_ok=True)

BJ = timezone(timedelta(hours=8))


# ╔══════════════════════════════════════════════════════════════╗
# ║              2. ETF池管理 (统一使用 config.py)                ║
# ╚══════════════════════════════════════════════════════════════╝

ETF_POOL = cfg.get_etf_pool()

# 构建索引
ETF_BY_CODE = {c: (s, n, sec) for c, (s, n, sec) in ETF_POOL.items()}
ETF_BY_NAME = {n: (c, s, sec) for c, (s, n, sec) in ETF_POOL.items()}

def code_to_name(code):
    """代码 → 名称"""
    info = ETF_BY_CODE.get(code)
    return info[1] if info else None

def name_to_code(name):
    """名称 → 代码"""
    info = ETF_BY_NAME.get(name)
    return info[0] if info else None


# ╔══════════════════════════════════════════════════════════════╗
# ║              3. 数据获取 (委托 data_fetcher 模块)            ║
# ╚══════════════════════════════════════════════════════════════╝


def fetch_all_data(max_workers=5):
    """拉取全部39只ETF日线, 构建 close 矩阵 + extra 字典"""
    codes = list(ETF_POOL.keys())
    print(f"[数据] 拉取 {len(codes)} 只 ETF 日线...")

    results = _fetcher.fetch_batch(codes, datalen=cfg.DATA_LEN, max_workers=max_workers)

    all_close = {}
    extra = {}
    failed = []
    for code, df in results.items():
        try:
            info = ETF_BY_CODE.get(code, ('', '', ''))
            name = info[1]
            if name and len(df) >= cfg.MOM_LONG + 5:
                all_close[name] = df.set_index('day')['close']
                ex_cols = [c for c in ['open', 'high', 'low', 'volume', 'amount', 'turnover'] if c in df.columns]
                extra[name] = df[['day'] + ex_cols].set_index('day')
            else:
                failed.append(code)
        except Exception as e:
            print(f"  [WARN] {code} 数据处理异常: {e}")
            failed.append(code)

    if failed:
        print(f"  [WARN] {len(failed)}只ETF失败: {failed}")

    if not all_close:
        print("[数据] 拉取完全失败!")
        return None, None

    price = pd.DataFrame(all_close).sort_index().ffill()
    valid = [c for c in price.columns if price[c].notna().sum() >= cfg.MOM_LONG + 2]
    price = price[valid]
    print(f"[数据] 完成: {price.shape[0]}行 x {price.shape[1]}列, 区间 {price.index[0]}~{price.index[-1]}")
    return price, extra


def get_latest_prices(price):
    """获取最新收盘价映射 {name: price}"""
    if price is None or len(price) == 0:
        return {}
    row = price.iloc[-1]
    return {n: float(row[n]) for n in row.index if not pd.isna(row[n])}


# ╔══════════════════════════════════════════════════════════════╗
# ║              4. 策略计算 (核心引擎)                           ║
# ╚══════════════════════════════════════════════════════════════╝

def calc_metrics(price, extra, data_date=None):
    """计算所有ETF的截面指标 (用于排名/筛选/quality_score)

    注意: 此处负责截面对比(momentum ranking, Z-score),
    StrategyEngine.calc_daily_indicators() 负责单只时序信号(MA/MACD/RSI).
    两者分工不同, 不做合并.

    返回: {name: {mom_long, mom_short, vol, quality_score, ...}}
    """
    metrics = {}
    if price is None or len(price) < cfg.MOM_LONG + 2:
        return metrics

    for name in price.columns:
        s = price[name].dropna()
        if len(s) < cfg.MOM_LONG + 2:
            continue

        # 动量: 严格使用历史数据, 无未来泄露
        latest = s.iloc[-1]
        prev = s.iloc[-2]
        mom_long = (latest / s.iloc[-cfg.MOM_LONG - 1] - 1) * 100
        mom_short = (latest / s.iloc[-cfg.MOM_SHORT - 1] - 1) * 100
        daily_change = (latest / prev - 1) * 100

        # 波动率 (Parkinson: 用High/Low)
        ex = extra.get(name)
        if ex is not None and 'high' in ex.columns and 'low' in ex.columns and len(ex) >= 20:
            hl = ex[['high', 'low']].dropna().tail(cfg.VOL_WINDOW)
            if len(hl) >= 20:
                hl_r = np.log(hl['high'] / hl['low'])
                vol = np.sqrt(1 / (4 * np.log(2)) * (hl_r ** 2).mean()) * np.sqrt(252) * 100
            else:
                vol = 999
        else:
            rets = s.pct_change().dropna().iloc[-cfg.VOL_WINDOW:]
            vol = rets.std() * np.sqrt(252) * 100 if len(rets) >= 20 else 999
        vol = max(vol, cfg.MIN_VOL)

        # 原始动量得分
        raw_score = (mom_long / vol) if mom_long > 0 else 0

        # 资金流数据
        amount = None
        turnover = None
        volume_ratio = None
        if ex is not None and len(ex) >= 6:
            last_r = ex.iloc[-1]
            amount = last_r.get('amount')
            turnover = last_r.get('turnover')
            if 'volume' in ex.columns:
                vols = ex['volume'].dropna()
                if len(vols) >= 6:
                    avg_v = vols.iloc[-6:-1].mean()
                    latest_v = vols.iloc[-1]
                    if avg_v > 0:
                        volume_ratio = ((latest_v / avg_v) - 1) * 100

        metrics[name] = {
            'code': ETF_BY_NAME.get(name, ('', '', ''))[0],
            'sector': ETF_BY_NAME.get(name, ('', '', ''))[2],
            'latest': round(latest, 3),
            'daily_change': round(daily_change, 2),
            'mom_long': round(mom_long, 2),
            'mom_short': round(mom_short, 2),
            'vol': round(vol, 1),
            'amount': round(amount / 1e8, 2) if amount is not None and not (isinstance(amount, float) and np.isnan(amount)) else None,
            'turnover': round(turnover, 2) if turnover is not None and not (isinstance(turnover, float) and np.isnan(turnover)) else None,
            'volume_ratio': round(volume_ratio, 2) if volume_ratio is not None else None,
            '_raw_score': raw_score,
            '_flow': {'turnover': turnover, 'volume_ratio': volume_ratio},
        }

    if not metrics:
        return metrics

    # ── 截面标准化 ──
    # 得分归一化
    scores = [m['_raw_score'] for m in metrics.values() if m['_raw_score'] > 0]
    for m in metrics.values():
        if m['_raw_score'] > 0 and scores:
            lo, hi = min(scores), max(scores)
            m['score'] = 100.0 if hi == lo else round((m['_raw_score'] - lo) / (hi - lo) * 100, 1)
        else:
            m['score'] = 0.0

    # 资金流 Z-score
    def _z(arr, v):
        if len(arr) < 3 or v is None or (isinstance(v, float) and np.isnan(v)):
            return 0.0
        std = np.std(arr) or 1.0
        return (v - np.mean(arr)) / std

    to_vals = np.array([m['_flow']['turnover'] for m in metrics.values()
                        if m['_flow']['turnover'] is not None and not (isinstance(m['_flow']['turnover'], float) and np.isnan(m['_flow']['turnover']))])
    vr_vals = np.array([m['_flow']['volume_ratio'] for m in metrics.values()
                        if m['_flow']['volume_ratio'] is not None and not (isinstance(m['_flow']['volume_ratio'], float) and np.isnan(m['_flow']['volume_ratio']))])

    for m in metrics.values():
        z_to = _z(to_vals, m['_flow']['turnover'])
        z_vr = _z(vr_vals, m['_flow']['volume_ratio'])
        m['flow_signal'] = round(z_to * 0.5 + z_vr * 0.5, 2)
        del m['_raw_score'], m['_flow']

    # quality_score: Z(动量)*0.4 + Z(得分)*0.35 + Z(资金流)*0.25
    non_broad = [m for m in metrics.values() if m['sector'] != '宽基指数']
    if non_broad:
        m_vals = np.array([m['mom_long'] for m in non_broad])
        s_vals = np.array([m['score'] for m in non_broad])
        f_vals = np.array([m['flow_signal'] for m in non_broad])
        for m in metrics.values():
            if m['sector'] == '宽基指数':
                m['quality_score'] = 0.0
            else:
                z_m = _z(m_vals, m['mom_long'])
                z_s = _z(s_vals, m['score'])
                z_f = _z(f_vals, m['flow_signal'])
                m['quality_score'] = round(z_m * 0.4 + z_s * 0.35 + z_f * 0.25, 2)

    return metrics


# ╔══════════════════════════════════════════════════════════════╗
# ║              5. 择时与选股                                   ║
# ╚══════════════════════════════════════════════════════════════╝

def market_timing(metrics):
    """择时: 基于HS300动量 + 板块宽度

    返回: (仓位比 0.0~1.0, 描述文本)
    """
    hs300 = metrics.get('沪深300ETF', {})
    hs300_mom = hs300.get('mom_long', -999)

    # 板块宽度
    sectors = {}
    for m in metrics.values():
        s = m.get('sector', '')
        if s == '宽基指数':
            continue
        sectors.setdefault(s, []).append(m['mom_long'] > 0)

    pos_sec = sum(1 for v in sectors.values() if sum(v) > len(v) / 2)
    breadth = pos_sec / max(len(sectors), 1)

    if hs300_mom > 2:
        return 1.0, "满仓", f"HS300动量{hs300_mom:+.1f}%强势"
    elif hs300_mom > 0:
        return 0.7, "7成", f"HS300动量{hs300_mom:+.1f}%偏多"
    elif hs300_mom > -2:
        # 弱市: 看板块宽度
        if breadth >= 0.5:
            return 0.5, "半仓", f"HS300动量{hs300_mom:+.1f}%弱势但{pos_sec}/{len(sectors)}板块强势"
        elif breadth >= 0.3:
            return 0.3, "3成", f"HS300动量{hs300_mom:+.1f}%弱势,{pos_sec}/{len(sectors)}板块活跃"
        return 0.0, "空仓", f"HS300动量{hs300_mom:+.1f}%弱势,板块宽度不足"
    else:
        # 极弱: 仅保留超优标的轻仓试探
        super_q = sum(1 for m in metrics.values()
                      if m.get('quality_score', 0) > 2.0
                      and m.get('sector') != '宽基指数'
                      and m.get('mom_long', 0) > 0)
        if super_q >= 2:
            return 0.3, "3成(试探)", f"HS300动量{hs300_mom:+.1f}%极弱但有{super_q}只超优标的"
        return 0.0, "空仓", f"HS300动量{hs300_mom:+.1f}%极弱,无超优标的"


def select_etfs(metrics, position_ratio):
    """选股: quality_score排序, 单只ETF, 排除宽基

    返回: [(name, weight, quality_score, mom_long), ...]
    """
    if position_ratio <= 0:
        return []

    candidates = []
    for name, m in metrics.items():
        price = m.get('latest', 0)
        quality = m.get('quality_score', 0)
        mom = m.get('mom_long', -999)
        change = m.get('daily_change', 0)

        # 过滤
        if m.get('sector') == '宽基指数':
            continue
        if mom <= 0:
            continue
        if quality <= 0:
            continue
        if change >= cfg.MAX_DAILY_RISE:  # 当日涨幅过大不追
            continue
        if price < 0.5 or price > 5.0:
            continue

        candidates.append((name, quality, mom, m.get('score', 0)))

    if not candidates:
        return []

    # 按 quality_score 排序, 选最优
    candidates.sort(key=lambda x: x[1], reverse=True)
    name, quality, mom, score = candidates[0]

    return [(name, round(position_ratio, 2), quality, mom)]


# ╔══════════════════════════════════════════════════════════════╗
# ║              6. 回测引擎 (内联, 集成 strategy_engine + trade_executor) ║
# ╚══════════════════════════════════════════════════════════════╝


# ╔══════════════════════════════════════════════════════════════╗
# ║              7. 日志与记录                                   ║
# ╚══════════════════════════════════════════════════════════════╝

def print_signal_report(metrics, ratio, pos_text, reason, picks):
    """打印完整信号报告 (中文)"""
    print(f"\n{'='*60}")
    print(f"  ETF 动量轮动 · 信号报告")
    print(f"  生成时间: {datetime.now(BJ).strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'='*60}")

    # 市场状态
    print(f"\n  [市场状态]")
    print(f"    择时: {pos_text} | {reason}")

    # 选股结果
    print(f"\n  [选股结果]")
    if picks:
        for name, weight, quality, mom in picks:
            code = name_to_code(name)
            print(f"    {name}({code}) 权重={weight*100:.0f}% quality={quality:+.2f} 动量={mom:+.1f}%")
    else:
        print(f"    无符合条件的标的")

    # 板块全景
    print(f"\n  [板块全景]")
    sectors = {}
    for name, m in metrics.items():
        s = m.get('sector', '其他')
        if s == '宽基指数':
            continue
        sectors.setdefault(s, []).append(m['mom_long'])

    for sname, moms in sorted(sectors.items(), key=lambda x: np.mean(x[1]), reverse=True):
        avg = np.mean(moms)
        pos = sum(1 for m in moms if m > 0)
        bar = 'UP' if avg > 2 else '--' if avg > 0 else 'DN'
        print(f"    {bar} {sname:10s} 均动量{avg:+.1f}%  {pos}/{len(moms)}正")

    # TOP动量ETF
    print(f"\n  [动量TOP10 (排除宽基)]")
    ranked = sorted(
        [(n, m) for n, m in metrics.items() if m.get('sector') != '宽基指数'],
        key=lambda x: x[1].get('quality_score', 0), reverse=True
    )[:10]
    for n, m in ranked:
        print(f"    {n:12s} qual={m['quality_score']:+.2f} "
              f"mom={m['mom_long']:+.1f}% vol={m['vol']:.0f}% "
              f"chg={m['daily_change']:+.2f}%")


# ╔══════════════════════════════════════════════════════════════╗
# ║              8. 主程序                                       ║
# ╚══════════════════════════════════════════════════════════════╝

def main():
    parser = argparse.ArgumentParser(description='ETF动量轮动量化交易系统')
    parser.add_argument('--signal', action='store_true', help='仅生成今日信号')
    parser.add_argument('--live', action='store_true', help='实盘信号(盘中)')
    parser.add_argument('--no-cache', action='store_true', help='强制刷新数据')
    args = parser.parse_args()

    if args.no_cache:
        from data_fetcher import clear_all_cache; clear_all_cache()
        cfg.CACHE_ENABLED = False

    t0 = time.time()
    log.info("=" * 40)
    log.info(f"系统启动 | 模式: {'回测' if not args.signal else '信号'} | 资金: {cfg.INITIAL_CASH:,.0f}")

    # ── 1. 获取数据 ──
    price, extra = fetch_all_data()
    if price is None:
        log.error("数据获取失败, 退出")
        return

    # ── 2. 计算指标 ──
    metrics = calc_metrics(price, extra)
    if not metrics:
        log.error("指标计算失败")
        return

    # ── 3. 择时选股 ──
    ratio, pos_text, reason = market_timing(metrics)
    picks = select_etfs(metrics, ratio)

    # ── 4. 打印信号 ──
    print_signal_report(metrics, ratio, pos_text, reason, picks)

    # ═══════════════════════════════════════
    # 5. 回测 (集成 strategy_engine + trade_executor + logger)
    # ═══════════════════════════════════════
    if not args.signal and not args.live:
        from strategy_engine import StrategyEngine
        from trade_executor import TradeExecutor

        log.info("开始回测...")
        strat = StrategyEngine()
        executor = TradeExecutor(mode='SIM', initial_cash=cfg.INITIAL_CASH, lot_size=cfg.LOT_SIZE)
        dates = price.index.tolist()
        start_idx = cfg.MOM_LONG + 5

        # ── 回测区间过滤 ──
        if cfg.BACKTEST_START:
            bt_start = pd.Timestamp(cfg.BACKTEST_START)
            start_idx = max(start_idx, next((i for i, d in enumerate(dates) if d >= bt_start), start_idx))
        if cfg.BACKTEST_END:
            bt_end = pd.Timestamp(cfg.BACKTEST_END)
            dates = [d for d in dates if d <= bt_end]

        # ── 拉取60分钟K线 (小时线定时用) ──
        log.info("拉取60分钟K线...")
        codes = list(ETF_POOL.keys())
        # ── 60分钟K线数据长度: 自动对齐日线范围 ──
        hourly_datalen = cfg.DATA_LEN_60MIN if cfg.DATA_LEN_60MIN > 0 else 2000  # Sina上限
        hourly_data = _fetcher.fetch_batch_60min(codes, datalen=hourly_datalen, max_workers=3)
        # 构建 name → hour_df 映射
        hourly_by_name = {}
        for code, hdf in hourly_data.items():
            name = code_to_name(code)
            if name and hdf is not None and len(hdf) >= 20:
                hourly_by_name[name] = hdf

        for i in range(start_idx, len(dates)):
            td = str(dates[i])[:10]
            # ── 当日价格 ──
            prices_today = {n: float(price.iloc[i][n]) for n in price.columns
                           if not pd.isna(price.iloc[i][n])}
            # ── prices_by_code: trade_executor 持仓用 code 作 key ──
            prices_by_code = {}
            for n, p in prices_today.items():
                c = name_to_code(n)
                if c:
                    prices_by_code[c] = p
            # ── 当日开盘价 (涨跌停判断用) ──
            opens_today = {}
            for n in price.columns:
                if n in extra:
                    ex_row = extra[n][extra[n].index <= dates[i]]
                    if len(ex_row) > 0 and 'open' in ex_row.columns:
                        opens_today[n] = float(ex_row['open'].iloc[-1])
            try:
                # ── 切片: 仅用截至T日的数据 ──
                p_slice = price.iloc[:i + 1]
                ex_slice = {n: df[df.index <= dates[i]]
                            for n, df in extra.items()
                            if len(df[df.index <= dates[i]]) > 0}

                # ── 策略信号 (strategy_engine) ──
                signals_today = {}
                for name in p_slice.columns:
                    try:
                        df_d = pd.DataFrame({
                            'day': [str(d)[:10] for d in p_slice.index],
                            'close': p_slice[name].values,
                            'high': ex_slice[name]['high'].values if name in ex_slice else p_slice[name].values,
                            'low': ex_slice[name]['low'].values if name in ex_slice else p_slice[name].values,
                            'volume': ex_slice[name]['volume'].values if name in ex_slice else [1]*len(p_slice),
                        })
                        # ── 小时线: 截取截至T日的数据 ──
                        df_h = None
                        if name in hourly_by_name:
                            hdf = hourly_by_name[name]
                            h_slice = hdf[hdf['day'].str[:10] <= td]
                            if len(h_slice) >= 20:
                                df_h = h_slice.copy()
                        sig = strat.generate_signal(name, df_d, df_hourly=df_h)
                        if sig['final_action'] != 'HOLD':
                            signals_today[name] = sig['final_action']
                            log.signal(name, sig['day_signal'], sig['hour_signal'],
                                       sig['final_action'], sig['day_score'], sig['reason'])
                    except Exception as e:
                        log.debug(f"信号异常 {name}: {e}")

                # ── T+1 解锁 ──
                executor.unlock_t1()

                # ── 风控检查: 止损 / 止盈 / 保本止损 ──
                for code in list(executor.account.positions.keys()):
                    name = code_to_name(code)
                    p = prices_today.get(name, 0)
                    if p <= 0:
                        continue
                    pos = executor.account.positions.get(code)
                    if not pos or pos.avg_cost <= 0:
                        continue
                    pnl_pct = (p / pos.avg_cost - 1) * 100
                    reason = None
                    if pnl_pct <= cfg.STOP_LOSS_PCT:
                        reason = f'止损({pnl_pct:+.1f}%)'
                    elif pnl_pct >= cfg.TAKE_PROFIT_PCT:
                        reason = f'止盈({pnl_pct:+.1f}%)'
                    elif pnl_pct >= cfg.BREAKEVEN_STOP_PCT and p <= pos.avg_cost:
                        reason = f'保本止损({pnl_pct:+.1f}%)'
                    if reason:
                        avail = executor.account.available_shares(code)
                        if avail >= cfg.LOT_SIZE:
                            order = executor.submit(code, 'SELL', p, avail, tag=reason)
                            if order.status == 'FILLED':
                                signals_today[name] = 'SELL'
                                log.trade(code, 'SELL', order.filled_price,
                                          order.filled_shares, reason=reason)

                # ── 卖出: 信号为SELL (不再对HOLD自动卖出) ──
                for code in list(executor.account.positions.keys()):
                    name = code_to_name(code)
                    action = signals_today.get(name, 'HOLD')
                    if action != 'SELL':
                        continue
                    p = prices_today.get(name, 0)
                    if p <= 0: continue
                    avail = executor.account.available_shares(code)
                    if avail >= cfg.LOT_SIZE:
                        order = executor.submit(code, 'SELL', p, avail, tag=f'信号SELL')
                        if order.status == 'FILLED':
                            log.trade(code, 'SELL', order.filled_price, order.filled_shares,
                                      reason='信号SELL')

                # ── 买入: 按 quality_score 排序, 限制最大持仓数 ──
                current_pos_count = len(executor.account.positions)
                slots = cfg.MAX_POSITIONS - current_pos_count
                # ── 市场温度: HS300实时动量检查 (防弱市买入) ──
                hs300_name = '沪深300ETF'
                hs300_ok = False
                if hs300_name in p_slice.columns and len(p_slice[hs300_name].dropna()) >= 22:
                    hs300_s = p_slice[hs300_name].dropna()
                    hs300_ma20 = float(hs300_s.iloc[-20:].mean())
                    hs300_now = float(hs300_s.iloc[-1])
                    hs300_mom5 = (hs300_now / float(hs300_s.iloc[-6]) - 1) * 100 if len(hs300_s) >= 6 else -99
                    hs300_ok = hs300_now > hs300_ma20  # 仅MA20过滤，避免过度收紧
                if slots > 0 and hs300_ok:
                    # 收集BUY信号候选人
                    buy_candidates = []
                    for name, p in prices_today.items():
                        if signals_today.get(name) != 'BUY':
                            continue
                        if p <= 0:
                            continue
                        # 涨停不追
                        if opens_today.get(name, 0) > 0 and p / opens_today[name] - 1 >= 0.098:
                            continue
                        # 日内实时截面指标 (防未来函数)
                        s_slice = p_slice[name].dropna()
                        if len(s_slice) < cfg.MOM_LONG + 2:
                            continue
                        mom40 = (float(s_slice.iloc[-1]) / float(s_slice.iloc[-cfg.MOM_LONG-1]) - 1) * 100
                        mom5 = (float(s_slice.iloc[-1]) / float(s_slice.iloc[-cfg.MOM_SHORT-1]) - 1) * 100
                        if mom40 <= 3 or mom5 <= 0:  # 长动量>3% + 短动量>0
                            continue
                        # ── 简单质量分: 动量/波动率 ──
                        rets = s_slice.pct_change().dropna().iloc[-60:]
                        vol60 = rets.std() * np.sqrt(252) * 100 if len(rets) >= 20 else 99
                        quality = mom40 / max(vol60, 1.0)
                        if quality < 0.25:  # 动量须显著大于波动率
                            continue
                        daily_chg = (float(s_slice.iloc[-1]) / float(s_slice.iloc[-2]) - 1) * 100
                        if daily_chg >= cfg.MAX_DAILY_RISE:
                            continue
                        buy_candidates.append((name, p, mom40))

                    # 按动量降序, 只取前 slots 只
                    buy_candidates.sort(key=lambda x: x[2], reverse=True)
                    buy_candidates = buy_candidates[:slots]

                    if buy_candidates:
                        budget_per = executor.account.cash * 0.90 / len(buy_candidates)
                        min_budget = cfg.INITIAL_CASH * cfg.MIN_POSITION_PCT
                        for name, p, mom in buy_candidates:
                            if budget_per < min_budget:
                                continue
                            code = name_to_code(name)
                            if not code: continue
                            shares = int(budget_per / p / cfg.LOT_SIZE) * cfg.LOT_SIZE
                            if shares >= cfg.LOT_SIZE:
                                order = executor.submit(code, 'BUY', p, shares, tag=f'mom={mom:+.1f}%')
                                if order.status == 'FILLED':
                                    log.trade(code, 'BUY', order.filled_price, order.filled_shares,
                                              reason=f'mom={mom:+.1f}%')

                # ── 净值快照 ──
                nw = executor.account.net_worth(prices_by_code)
                hv = nw - executor.account.cash
                if i == start_idx:
                    prev_nw = cfg.INITIAL_CASH
                else:
                    prev_prices = {n: float(price.iloc[i-1][n]) for n in price.columns if not pd.isna(price.iloc[i-1][n])}
                    prev_prices_by_code = {}
                    for n, p in prev_prices.items():
                        c = name_to_code(n)
                        if c:
                            prev_prices_by_code[c] = p
                    prev_nw = executor.account.net_worth(prev_prices_by_code)
                daily_ret = (nw / prev_nw - 1) * 100 if prev_nw > 0 else 0
                log.nav_snapshot(td, executor.account.cash, hv,
                                 positions={c: p.shares for c, p in executor.account.positions.items()},
                                 daily_return_pct=round(daily_ret, 4))

            except Exception as e:
                log.warn(f"{td} 回测异常: {e}")
                continue

        # ── 报告 ──
        final_prices = {n: float(price.iloc[-1][n]) for n in price.columns if not pd.isna(price.iloc[-1][n])}
        final_prices_by_code = {}
        for n, p in final_prices.items():
            c = name_to_code(n)
            if c:
                final_prices_by_code[c] = p
        executor.print_summary(prices=final_prices_by_code)
        closed = executor.account.win_count + executor.account.loss_count
        wr = (executor.account.win_count / closed * 100) if closed > 0 else 0
        result = {
            'initial_cash': cfg.INITIAL_CASH,
            'final_nw': round(executor.account.net_worth(final_prices_by_code), 2),
            'trades': executor.account.trade_count,
            'fees': round(executor.account.total_fees, 2),
            'total_pnl': round(executor.account.total_pnl, 2),
            'win_rate': round(wr, 1),
            'wins': executor.account.win_count,
            'losses': executor.account.loss_count,
        }
        log.info(f"回测完成 | 终值{result['final_nw']:,.0f} | "
                 f"交易{result['trades']}笔 | 胜率{result['win_rate']:.1f}% | "
                 f"盈亏{result['total_pnl']:+,.0f}")

    # ── 6. 实盘信号 (保存到 signal.json) ──
    if args.signal or args.live:
        signal = {
            'generated_at': datetime.now(BJ).strftime('%Y-%m-%d %H:%M:%S'),
            'data_date': str(price.index[-1]),
            'position_ratio': ratio,
            'position_text': pos_text,
            'reason': reason,
            'picks': [{'name': n, 'weight': w, 'quality': q, 'momentum': mom}
                      for n, w, q, mom in picks],
        }
        spath = os.path.join(DATA_DIR, 'signal.json')
        with open(spath, 'w', encoding='utf-8') as f:
            json.dump(signal, f, ensure_ascii=False, indent=2)
        print(f"\n  [保存] {spath}")

    elapsed = time.time() - t0
    log.report(f"运行报告 (耗时 {elapsed:.1f}s)")


if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        print("\n[中断] 用户取消")
    except Exception as e:
        print(f"\n[异常] {e}")
        traceback.print_exc()
