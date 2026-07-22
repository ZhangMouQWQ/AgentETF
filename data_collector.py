#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
生产环境数据采集器 (GitHub Actions 中运行)
==========================================
采集 5 只代表性 ETF 的关键数据，提交回 repo 供本地分析。

采集内容:
  A. API 连通性矩阵 — 哪些 API 在生产环境可用
  B. 日线数据样本 — Eastmoney/Tencent 原始返回 (前3条+最后3条)
  C. 盘中快照数据 — fetch_intraday_snapshot 的完整输出
  D. extra_history 合并前后对比 — 核心修复验证数据
  E. 策略指标样本 — 3只 ETF 的完整 metrics

输出: _fixtures/collector_output.json
"""
import json
import sys
import os
import time
from datetime import datetime, timezone, timedelta

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from strategy import (
    ETF_POOL, ETF_SECTOR,
    get_etf_sina, get_etf_extra_sina, get_etf_realtime_eastmoney,
    get_etf_history_eastmoney, get_etf_history_akshare,
    fetch_intraday_snapshot, merge_intraday_price,
    calc_metrics, calc_flow_signal,
    DATA_LEN, MOM_LONG,
)

# 5 只代表性 ETF，覆盖不同板块和交易所
TEST_CODES = ['510300', '512480', '515030', '159915', '512010']
# 沪深300(上), 半导体(上), 新能源车(上), 创业板(深), 医药(上)

bj = timezone(timedelta(hours=8))
now = datetime.now(bj)
today_str = now.strftime('%Y-%m-%d')
run_time = now.strftime('%Y-%m-%d %H:%M:%S')

output = {
    'run_time': run_time,
    'today': today_str,
    'signal_type': 'morning' if 11 <= now.hour < 13 else 'close',
    'test_pool': {},
}

# ═══════════════════════════════════════════════
# A. API 连通性矩阵
# ═══════════════════════════════════════════════
print("A. API 连通性测试...")
api_results = {}
for code in TEST_CODES:
    sina_code, name = ETF_POOL[code]
    output['test_pool'][code] = name

    # Eastmoney K线
    t0 = time.time()
    df = get_etf_history_eastmoney(code, datalen=10)
    t_em = round(time.time() - t0, 2)
    # Tencent K线
    t0 = time.time()
    df2 = get_etf_sina(sina_code, scale=240, datalen=10)
    t_tx = round(time.time() - t0, 2)
    # Sina 实时
    t0 = time.time()
    rt_sina = get_etf_extra_sina(sina_code)
    t_sina = round(time.time() - t0, 2)
    # Eastmoney 实时
    t0 = time.time()
    rt_em = get_etf_realtime_eastmoney(code)
    t_emrt = round(time.time() - t0, 2)

    api_results[code] = {
        'name': name,
        'eastmoney_kline': {'ok': df is not None and len(df) >= 5, 'rows': len(df) if df is not None else 0, 'time_s': t_em},
        'tencent_kline':   {'ok': df2 is not None and len(df2) >= 5, 'rows': len(df2) if df2 is not None else 0, 'time_s': t_tx},
        'sina_realtime':   {'ok': rt_sina is not None, 'keys': list(rt_sina.keys()) if rt_sina else [], 'time_s': t_sina},
        'eastmoney_realtime': {'ok': rt_em is not None, 'keys': list(rt_em.keys()) if rt_em else [], 'time_s': t_emrt},
    }
    status = ''.join([
        'E' if api_results[code]['eastmoney_kline']['ok'] else '-',
        'T' if api_results[code]['tencent_kline']['ok'] else '-',
        'S' if api_results[code]['sina_realtime']['ok'] else '-',
        'R' if api_results[code]['eastmoney_realtime']['ok'] else '-',
    ])
    print(f"  [{status}] {name}({code})")
    time.sleep(0.3)

output['api_matrix'] = api_results

# ═══════════════════════════════════════════════
# B. 日线数据样本
# ═══════════════════════════════════════════════
print("\nB. 日线数据样本...")
test_pool = {k: ETF_POOL[k] for k in TEST_CODES}
price_daily, etf_info, extra_history, sources = {}, {}, {}, set()
# 简化版 fetch_daily_data（只取一只做样本）
for code in TEST_CODES[:2]:
    sina_code, name = ETF_POOL[code]
    df = get_etf_history_eastmoney(code, datalen=DATA_LEN)
    src = 'eastmoney'
    if df is None:
        df = get_etf_sina(sina_code, scale=240, datalen=DATA_LEN)
        src = 'tencent'
    sources.add(src)
    if df is not None:
        price_daily[name] = df.set_index('day')['close']
        etf_info[name] = code
        # 记录样本行
        output.setdefault('kline_samples', {})[name] = {
            'source': src,
            'total_rows': len(df),
            'first_3': df.head(3)[['day', 'close']].to_dict('records') if 'close' in df.columns else [],
            'last_3': df.tail(3)[['day', 'close']].to_dict('records') if 'close' in df.columns else [],
            'columns': [c for c in df.columns],
        }
        if 'amount' in df.columns:
            output['kline_samples'][name]['has_amount'] = True
            output['kline_samples'][name]['last_amount'] = float(df['amount'].iloc[-1]) if len(df) > 0 else None
        if 'turnover' in df.columns:
            output['kline_samples'][name]['has_turnover'] = True
    time.sleep(0.5)
output['kline_sources'] = sorted(sources)

# ═══════════════════════════════════════════════
# C. 盘中快照数据
# ═══════════════════════════════════════════════
print("\nC. 盘中快照数据...")
intraday = fetch_intraday_snapshot(test_pool)
output['intraday_snapshot'] = {}
for name, snap in intraday.items():
    output['intraday_snapshot'][name] = {
        'keys': list(snap.keys()),
        'close': snap.get('close'),
        'has_volume': 'volume' in snap,
        'has_amount': 'amount' in snap,
        'has_turnover': 'turnover' in snap,
    }
    if 'volume' in snap:
        output['intraday_snapshot'][name]['volume'] = snap['volume']
    if 'amount' in snap:
        output['intraday_snapshot'][name]['amount'] = snap['amount']

# ═══════════════════════════════════════════════
# D. extra_history 合并前后对比
# ═══════════════════════════════════════════════
print("\nD. extra_history 合并前后对比...")
# 从日线构建 price DataFrame
import pandas as pd
import numpy as np
if price_daily:
    price_df = pd.DataFrame(price_daily).sort_index()
else:
    price_df = None

extra_before_snap = {}
if extra_history:
    for name, df in extra_history.items():
        extra_before_snap[name] = {
            'latest_date': str(df.index[-1]),
            'has_today': today_str in df.index,
            'columns': [c for c in df.columns],
        }

# 模拟盘中合并
price_after = price_df
extra_after = dict(extra_history) if extra_history else {}
if intraday and price_df is not None:
    price_after, extra_after = merge_intraday_price(price_df, intraday, etf_info, extra_after)

extra_after_snap = {}
for name, df in extra_after.items():
    extra_after_snap[name] = {
        'latest_date': str(df.index[-1]),
        'has_today': today_str in df.index,
    }

output['extra_comparison'] = {
    'before': extra_before_snap,
    'after': extra_after_snap,
    'any_date_advanced': any(
        extra_after_snap.get(n, {}).get('latest_date') != extra_before_snap.get(n, {}).get('latest_date')
        for n in extra_after_snap
    ),
}

# ═══════════════════════════════════════════════
# E. 策略指标样本
# ═══════════════════════════════════════════════
print("\nE. 策略指标样本...")
if price_after is not None and len(price_after.columns) >= 2:
    metrics = calc_metrics(price_after, etf_info, ETF_SECTOR, extra_after)
    output['metrics_sample'] = {}
    for name in sorted(metrics.keys())[:3]:
        m = metrics[name]
        output['metrics_sample'][name] = {
            'sector': m['sector'],
            'daily_change': m['daily_change'],
            'mom_long': m['mom_long'],
            'mom_short': m['mom_short'],
            'vol': m['vol'],
            'score': m['score'],
            'flow_signal': m['flow_signal'],
            'flow_score_norm': m.get('flow_score_norm', 0),
            'amount_yi': m.get('amount'),
            'turnover': m.get('turnover'),
        }

# ═══════════════════════════════════════════════
# 写入文件
# ═══════════════════════════════════════════════
os.makedirs('_fixtures', exist_ok=True)
with open('_fixtures/collector_output.json', 'w', encoding='utf-8') as f:
    json.dump(output, f, ensure_ascii=False, indent=2)

print(f"\n✅ 数据采集完成 → _fixtures/collector_output.json")
print(f"   API矩阵: {len(api_results)} 只 ETF")
print(f"   快照数据: {len(intraday)} 只")
print(f"   指标样本: {len(output.get('metrics_sample', {}))} 只")
