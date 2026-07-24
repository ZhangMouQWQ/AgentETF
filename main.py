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
# ║              1. 关键参数配置                                  ║
# ╚══════════════════════════════════════════════════════════════╝

# ── 数据参数 ──
DATA_LEN = 260          # 拉取日线数量
MOM_LONG = 40           # 长周期动量(日)
MOM_SHORT = 5           # 短周期动量(日)
VOL_WINDOW = 60         # 波动率计算窗口

# ── 风控参数 ──
STOP_LOSS_PCT = -4.0    # 单日止损线(%)
MAX_DAILY_RISE = 5.0    # 买入过滤: 当日涨幅上限(%)
MIN_VOL = 5.0           # 最小波动率(防止除零)

# ── 资金参数 ──
INIT_CASH = 8000.0      # 初始本金
FEE_PER_TRADE = 10.0    # 每笔手续费
LOT_SIZE = 100          # 最小交易单位(股)
MIN_TRADE_AMT = 500.0   # 最低交易金额

# ── 执行参数 ──
EXEC_PRICE = 'close'    # 执行价: 'close'(收盘价) | 'vwap'(均价)
T1_LOCK = True          # T+1 锁仓

# ── 路径 ──
DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data')
os.makedirs(DATA_DIR, exist_ok=True)

BJ = timezone(timedelta(hours=8))


# ╔══════════════════════════════════════════════════════════════╗
# ║              2. ETF池管理                                    ║
# ╚══════════════════════════════════════════════════════════════╝

# 39只ETF, 按13个板块分组
ETF_POOL = {
    '510300': ('sh510300', '沪深300ETF', '宽基指数'),
    '510500': ('sh510500', '中证500ETF', '宽基指数'),
    '512100': ('sh512100', '中证1000ETF', '宽基指数'),
    '159915': ('sz159915', '创业板ETF', '宽基指数'),
    '588000': ('sh588000', '科创50ETF', '宽基指数'),
    '512800': ('sh512800', '银行ETF', '金融'),
    '512000': ('sh512000', '券商ETF', '金融'),
    '512200': ('sh512200', '房地产ETF', '金融'),
    '515170': ('sh515170', '食品饮料ETF', '消费'),
    '512690': ('sh512690', '酒ETF', '消费'),
    '159996': ('sz159996', '家电ETF', '消费'),
    '159865': ('sz159865', '养殖ETF', '消费'),
    '512010': ('sh512010', '医药ETF', '医药'),
    '515120': ('sh515120', '创新药ETF', '医药'),
    '512170': ('sh512170', '医疗ETF', '医药'),
    '512480': ('sh512480', '半导体ETF', '电子'),
    '159995': ('sz159995', '芯片ETF', '电子'),
    '159819': ('sz159819', '人工智能ETF', '计算机'),
    '512720': ('sh512720', '计算机ETF', '计算机'),
    '515030': ('sh515030', '新能源车ETF', '电力新能源'),
    '515790': ('sh515790', '光伏ETF', '电力新能源'),
    '159611': ('sz159611', '电力ETF', '电力新能源'),
    '512660': ('sh512660', '军工ETF', '军工'),
    '512670': ('sh512670', '国防ETF', '军工'),
    '512400': ('sh512400', '有色金属ETF', '金属矿产'),
    '516780': ('sh516780', '稀土ETF', '金属矿产'),
    '518880': ('sh518880', '黄金ETF', '金属矿产'),
    '515220': ('sh515220', '煤炭ETF', '能源化工'),
    '159697': ('sz159697', '油气ETF', '能源化工'),
    '159870': ('sz159870', '化工ETF', '能源化工'),
    '515210': ('sh515210', '钢铁ETF', '能源化工'),
    '515880': ('sh515880', '通信ETF', '通信传媒'),
    '512980': ('sh512980', '传媒ETF', '通信传媒'),
    '159869': ('sz159869', '游戏ETF', '通信传媒'),
    '516110': ('sh516110', '汽车ETF', '制造基建'),
    '159530': ('sz159530', '机器人ETF', '制造基建'),
    '516970': ('sh516970', '基建ETF', '制造基建'),
    '510880': ('sh510880', '红利ETF', '红利'),
    '515080': ('sh515080', '中证红利ETF', '红利'),
}

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

    results = _fetcher.fetch_batch(codes, period='daily', datalen=cfg.DATA_LEN, max_workers=max_workers)

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
    valid = [c for c in price.columns if price[c].notna().sum() >= MOM_LONG + 2]
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
    """计算所有ETF的技术指标

    返回: {name: {mom_long, mom_short, vol, quality_score, ...}}
    """
    metrics = {}
    if price is None or len(price) < MOM_LONG + 2:
        return metrics

    for name in price.columns:
        s = price[name].dropna()
        if len(s) < MOM_LONG + 2:
            continue

        # 动量: 严格使用历史数据, 无未来泄露
        latest = s.iloc[-1]
        prev = s.iloc[-2]
        mom_long = (latest / s.iloc[-MOM_LONG - 1] - 1) * 100
        mom_short = (latest / s.iloc[-MOM_SHORT - 1] - 1) * 100
        daily_change = (latest / prev - 1) * 100

        # 波动率 (Parkinson: 用High/Low)
        ex = extra.get(name)
        if ex is not None and 'high' in ex.columns and 'low' in ex.columns and len(ex) >= 20:
            hl = ex[['high', 'low']].dropna().tail(VOL_WINDOW)
            if len(hl) >= 20:
                hl_r = np.log(hl['high'] / hl['low'])
                vol = np.sqrt(1 / (4 * np.log(2)) * (hl_r ** 2).mean()) * np.sqrt(252) * 100
            else:
                vol = 999
        else:
            rets = s.pct_change().dropna().iloc[-VOL_WINDOW:]
            vol = rets.std() * np.sqrt(252) * 100 if len(rets) >= 20 else 999
        vol = max(vol, MIN_VOL)

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
        if change >= MAX_DAILY_RISE:  # 当日涨幅过大不追
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
        bar = '🟢' if avg > 2 else '🟡' if avg > 0 else '🔴'
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

        for i in range(start_idx, len(dates)):
            td = str(dates[i])[:10]
            # ── 当日价格 ──
            prices_today = {n: float(price.iloc[i][n]) for n in price.columns
                           if not pd.isna(price.iloc[i][n])}
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
                        sig = strat.generate_signal(name, df_d)
                        if sig['final_action'] != 'HOLD':
                            signals_today[name] = sig['final_action']
                            log.signal(name, sig['day_signal'], sig['hour_signal'],
                                       sig['final_action'], sig['day_score'], sig['reason'])
                    except Exception as e:
                        log.debug(f"信号异常 {name}: {e}")

                # ── T+1 解锁 ──
                executor.unlock_t1()

                # ── 卖出: 不在信号中或信号为SELL ──
                for code in list(executor.account.positions.keys()):
                    name = code_to_name(code)
                    action = signals_today.get(name, 'SELL')
                    if action == 'BUY':
                        continue
                    p = prices_today.get(name, 0)
                    if p <= 0: continue
                    avail = executor.account.available_shares(code)
                    if avail >= cfg.LOT_SIZE:
                        order = executor.submit(code, 'SELL', p, avail, tag=f'信号{action}')
                        if order.status == 'FILLED':
                            log.trade(code, 'SELL', order.filled_price, order.filled_shares,
                                      reason=f'信号{action}')

                # ── 买入: 信号为BUY ──
                buy_list = [(name, p) for name, p in prices_today.items()
                           if signals_today.get(name) == 'BUY' and p > 0]
                if buy_list:
                    budget = executor.account.cash * 0.95 / len(buy_list)
                    for name, p in buy_list:
                        code = name_to_code(name)
                        if not code: continue
                        shares = int(budget / p / cfg.LOT_SIZE) * cfg.LOT_SIZE
                        if shares >= cfg.LOT_SIZE:
                            order = executor.submit(code, 'BUY', p, shares, tag='策略信号')
                            if order.status == 'FILLED':
                                log.trade(code, 'BUY', order.filled_price, order.filled_shares,
                                          reason='策略信号')

                # ── 净值快照 ──
                nw = executor.account.net_worth(prices_today)
                hv = nw - executor.account.cash
                if i == start_idx:
                    prev_nw = cfg.INITIAL_CASH
                else:
                    prev_prices = {n: float(price.iloc[i-1][n]) for n in price.columns if not pd.isna(price.iloc[i-1][n])}
                    prev_nw = executor.account.net_worth(prev_prices)
                daily_ret = (nw / prev_nw - 1) * 100 if prev_nw > 0 else 0
                log.nav_snapshot(td, executor.account.cash, hv,
                                 positions={c: p.shares for c, p in executor.account.positions.items()},
                                 daily_return_pct=round(daily_ret, 4))

            except Exception as e:
                log.warn(f"{td} 回测异常: {e}")
                continue

        # ── 报告 ──
        final_prices = {n: float(price.iloc[-1][n]) for n in price.columns if not pd.isna(price.iloc[-1][n])}
        executor.print_summary(prices=final_prices)
        result = {
            'initial_cash': cfg.INITIAL_CASH,
            'final_nw': round(executor.account.net_worth(final_prices), 2),
            'trades': executor.account.trade_count,
            'fees': round(executor.account.total_fees, 2),
        }

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
