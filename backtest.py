#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ETF动量轮动策略回测框架
对比 4 种策略变体，选出表现最优的
"""
import json
import time
import numpy as np
import pandas as pd
import requests
from datetime import datetime, timezone, timedelta

# 复用 strategy.py 的数据获取和指标计算
from strategy import (
    SECTOR_ETF_POOL, ETF_POOL, ETF_SECTOR,
    get_etf_sina, get_etf_history_eastmoney, get_etf_history_akshare,
    fetch_daily_data, calc_metrics, calc_flow_signal,
    DATA_LEN, MOM_LONG, MOM_SHORT, VOL_WINDOW, MIN_VOL,
    MAX_DAILY_DROP, MAX_DAILY_RISE, TOP_K
)

# ==================== 策略变体定义 ====================

def market_timing_hs300(metrics):
    """V1/V2: 沪深300绝对动量判断仓位"""
    m = metrics.get('沪深300ETF')
    if m is None:
        return 0.0
    if m['mom_long'] > 2:
        return 1.0
    elif m['mom_long'] > 0:
        return 0.5
    return 0.0


def market_timing_breadth(metrics):
    """V3/V4: 板块宽度判断仓位
    - >=50% 板块有正动量 → 满仓
    - >=30% 板块有正动量 → 半仓
    - <30% → 空仓
    """
    sectors = {}
    for name, m in metrics.items():
        s = m.get('sector', '')
        if s == '宽基指数':
            continue
        sectors.setdefault(s, []).append(m['mom_long'] > 0)

    if not sectors:
        return 0.0

    # 板块动量方向: 板块内超半数ETF动量为正
    positive_sectors = sum(1 for etfs in sectors.values() if sum(etfs) > len(etfs) / 2)
    ratio = positive_sectors / len(sectors)

    if ratio >= 0.5:
        return 1.0
    elif ratio >= 0.3:
        return 0.5
    return 0.0


def build_target_absolute(metrics, position_ratio):
    """V1: 绝对动量选股 (mom_long > 0)"""
    if position_ratio <= 0:
        return []
    candidates = [
        (n, m) for n, m in metrics.items()
        if m['mom_long'] > 0
        and m['mom_short'] > -2
        and m['daily_change'] > MAX_DAILY_DROP
        and m['daily_change'] < MAX_DAILY_RISE
        and m['daily_change'] > 0
    ]
    candidates.sort(key=lambda x: x[1]['score'] + x[1].get('flow_score_norm', 0) * 0.3, reverse=True)
    selected = candidates[:TOP_K]
    if not selected:
        return []
    weight = round(position_ratio / len(selected), 2)
    weights = [weight] * len(selected)
    weights[-1] = round(position_ratio - sum(weights[:-1]), 2)
    return [
        {"name": n, "code": m['code'], "weight": w, "sector": m['sector']}
        for (n, m), w in zip(selected, weights)
    ]


def build_target_relative(metrics, position_ratio, diversify=False):
    """V2/V3/V4: 相对动量选股 (不设绝对值门槛)
    diversify=True: 同一板块最多选1只
    """
    if position_ratio <= 0:
        return []
    candidates = [
        (n, m) for n, m in metrics.items()
        if m['mom_short'] > -3
        and m['daily_change'] > MAX_DAILY_DROP
        and m['daily_change'] < MAX_DAILY_RISE
        and m['daily_change'] > 0
    ]
    candidates.sort(key=lambda x: x[1]['score'] + x[1].get('flow_score_norm', 0) * 0.3, reverse=True)

    if diversify:
        seen_sectors = set()
        diversified = []
        for n, m in candidates:
            if m['sector'] not in seen_sectors:
                diversified.append((n, m))
                seen_sectors.add(m['sector'])
            if len(diversified) >= TOP_K:
                break
        selected = diversified[:TOP_K]
    else:
        selected = candidates[:TOP_K]

    if not selected:
        return []
    weight = round(position_ratio / len(selected), 2)
    weights = [weight] * len(selected)
    weights[-1] = round(position_ratio - sum(weights[:-1]), 2)
    return [
        {"name": n, "code": m['code'], "weight": w, "sector": m['sector']}
        for (n, m), w in zip(selected, weights)
    ]


# ==================== 回测引擎 ====================

def run_backtest(price_df, extra_hist, etf_info, etf_sector,
                 timing_func, target_func, label="V?"):
    """
    日频回测
    price_df: DataFrame, index=date, columns=ETF名称
    返回: (daily_returns, trades_log)
    """
    dates = price_df.index.tolist()
    # 需要至少 MOM_LONG+2 天数据才能开始
    min_idx = MOM_LONG + 5

    daily_returns = []
    trades_log = []
    prev_target = {}  # 上期目标持仓 {name: weight}

    for i in range(min_idx, len(dates) - 1):
        today = dates[i]
        tomorrow = dates[i + 1]

        # 用截至 today 的数据计算指标
        price_slice = price_df.iloc[:i + 1]
        extra_slice = {}
        if extra_hist:
            for name, df in extra_hist.items():
                if name in price_slice.columns:
                    extra_slice[name] = df[df.index <= today]

        metrics = calc_metrics(price_slice, etf_info, etf_sector, extra_slice)
        if not metrics:
            continue

        position_ratio = timing_func(metrics)
        target = target_func(metrics, position_ratio)

        # 计算次日收益
        ret = 0.0
        target_dict = {h['name']: h['weight'] for h in target}
        total_weight = sum(target_dict.values())

        for name, weight in target_dict.items():
            if name in price_df.columns:
                today_price = price_df.loc[today, name]
                tomorrow_price = price_df.loc[tomorrow, name]
                if today_price and tomorrow_price and today_price > 0:
                    stock_ret = (tomorrow_price / today_price - 1) * weight
                    ret += stock_ret

        # 现金部分收益为 0
        daily_returns.append({
            'date': tomorrow,
            'return': ret,
            'position': total_weight,
            'holdings': len(target),
        })

        # 记录调仓
        target_names = set(target_dict.keys())
        prev_names = set(prev_target.keys())
        if target_names != prev_names or any(
            abs(target_dict.get(n, 0) - prev_target.get(n, 0)) > 0.02
            for n in target_names | prev_names
        ):
            trades_log.append({
                'date': today,
                'from': [(n, f"{w*10:.0f}成") for n, w in prev_target.items()],
                'to': [(n, f"{w*10:.0f}成") for n, w in target_dict.items()],
            })
        prev_target = target_dict

    return daily_returns, trades_log


def calc_stats(daily_returns, benchmark_returns=None):
    """计算策略绩效指标"""
    if not daily_returns:
        return {}

    df = pd.DataFrame(daily_returns)
    rets = df['return'].values

    total_return = np.prod(1 + rets) - 1
    n_days = len(rets)
    n_years = n_days / 252
    cagr = (1 + total_return) ** (1 / max(n_years, 0.5)) - 1

    # 年化波动率
    annual_vol = np.std(rets) * np.sqrt(252)

    # Sharpe (假设无风险利率 2%)
    sharpe = (cagr - 0.02) / max(annual_vol, 0.001)

    # 最大回撤
    cum = np.cumprod(1 + rets)
    peak = np.maximum.accumulate(cum)
    drawdown = (cum - peak) / peak
    max_dd = drawdown.min()

    # 胜率
    win_rate = np.mean(rets > 0)

    # 平均持仓数
    avg_holdings = df['holdings'].mean()
    avg_position = df['position'].mean()

    # Calmar
    calmar = cagr / max(abs(max_dd), 0.001)

    return {
        'cagr': cagr,
        'total_return': total_return,
        'annual_vol': annual_vol,
        'sharpe': sharpe,
        'max_drawdown': max_dd,
        'calmar': calmar,
        'win_rate': win_rate,
        'n_trades': len(df),
        'avg_holdings': avg_holdings,
        'avg_position': avg_position,
    }


def print_stats(name, stats):
    print(f"\n{'='*50}")
    print(f"  {name}")
    print(f"{'='*50}")
    print(f"  年化收益: {stats.get('cagr', 0)*100:+.1f}%")
    print(f"  累计收益: {stats.get('total_return', 0)*100:+.1f}%")
    print(f"  年化波动: {stats.get('annual_vol', 0)*100:.1f}%")
    print(f"  夏普比率: {stats.get('sharpe', 0):.2f}")
    print(f"  最大回撤: {stats.get('max_drawdown', 0)*100:.1f}%")
    print(f"  Calmar:   {stats.get('calmar', 0):.2f}")
    print(f"  日胜率:   {stats.get('win_rate', 0)*100:.1f}%")
    print(f"  平均持仓: {stats.get('avg_holdings', 0):.1f}只")
    print(f"  平均仓位: {stats.get('avg_position', 0)*100:.0f}%")


# ==================== 主流程 ====================

def main():
    bj = timezone(timedelta(hours=8))
    print(f"[{datetime.now(bj).strftime('%Y-%m-%d %H:%M')}] 开始回测...")

    # 1. 拉取历史数据（260天约1年）
    print("\n拉取历史数据...")
    price_daily, etf_info, extra_history, data_sources = fetch_daily_data(ETF_POOL)

    if price_daily is None or len(price_daily) < MOM_LONG + 30:
        print("数据不足，无法回测")
        return

    print(f"数据范围: {price_daily.index[0]} ~ {price_daily.index[-1]}, 共 {len(price_daily)} 天")

    # 2. 定义策略变体
    variants = [
        ("V1 绝对动量+沪深300择时",
         market_timing_hs300,
         lambda m, p: build_target_absolute(m, p)),

        ("V2 相对动量+沪深300择时",
         market_timing_hs300,
         lambda m, p: build_target_relative(m, p, diversify=False)),

        ("V3 相对动量+板块宽度择时",
         market_timing_breadth,
         lambda m, p: build_target_relative(m, p, diversify=False)),

        ("V4 相对动量+宽度择时+板块分散",
         market_timing_breadth,
         lambda m, p: build_target_relative(m, p, diversify=True)),
    ]

    # 3. 跑回测
    all_stats = {}
    for label, timing_func, target_func in variants:
        print(f"\n回测 {label}...")
        daily_rets, trades = run_backtest(
            price_daily, extra_history, etf_info, ETF_SECTOR,
            timing_func, target_func, label
        )
        stats = calc_stats(daily_rets)
        all_stats[label] = stats
        print_stats(label, stats)
        print(f"  调仓次数: {len(trades)}")

    # 4. 选出最优
    print(f"\n{'='*50}")
    print("  综合排名 (按 Calmar 排序)")
    print(f"{'='*50}")
    ranked = sorted(all_stats.items(), key=lambda x: x[1].get('calmar', -999), reverse=True)
    for i, (name, s) in enumerate(ranked):
        print(f"  {i+1}. {name}: Calmar={s.get('calmar', 0):.2f}, CAGR={s.get('cagr', 0)*100:+.1f}%, DD={s.get('max_drawdown', 0)*100:.1f}%")

    best_name, best_stats = ranked[0]
    print(f"\n  >>> 最优策略: {best_name}")

    # 5. 保存回测结果
    result = {
        'backtest_time': datetime.now(bj).strftime('%Y-%m-%d %H:%M'),
        'data_range': f"{price_daily.index[0]} ~ {price_daily.index[-1]}",
        'variants': {name: {k: round(v, 4) if isinstance(v, float) else v for k, v in s.items()}
                     for name, s in all_stats.items()},
        'best': best_name,
    }
    with open('backtest_result.json', 'w', encoding='utf-8') as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    print(f"\n结果已保存到 backtest_result.json")


if __name__ == '__main__':
    main()
