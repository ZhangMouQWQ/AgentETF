#!/usr/bin/env python3
"""数据抓取独立脚本 — 供 GitHub Actions 使用"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from datetime import datetime, timedelta, timezone
import pandas as pd
import numpy as np

from data_fetcher import DataFetcher, clear_all_cache, is_trading_day
from config import Config

BJ = timezone(timedelta(hours=8))

cfg = Config()
pool = cfg.get_etf_pool()
codes = list(pool.keys())

print(f"=== 数据抓取: {len(codes)} 只 ETF ===")

# 清除缓存, 强制重新拉取
clear_all_cache()

fetcher = DataFetcher()
results = fetcher.fetch_batch(codes, datalen=cfg.DATA_LEN, max_workers=3)

# ═══════════════════════════════════════
# 数据完整性校验
# ═══════════════════════════════════════

MIN_ROWS = cfg.MOM_LONG + 5
success = 0
warnings = []
errors = []

for code in codes:
    df = results.get(code)
    name = pool[code][1]

    if df is None:
        errors.append(f"{code}({name}): 无数据")
        continue
    if len(df) < MIN_ROWS:
        errors.append(f"{code}({name}): 仅{len(df)}行(需>{MIN_ROWS})")
        continue

    issues = []

    # ── 1. 日期检查 ──
    df['day_dt'] = pd.to_datetime(df['day'])
    first_date = df['day_dt'].iloc[0]
    last_date = df['day_dt'].iloc[-1]
    today = datetime.now(BJ)

    # 最新数据不应超过今天
    if last_date.date() > today.date():
        issues.append(f"最后日期{last_date.date()}晚于今天")

    # 如果今天是交易日, 最新数据应为今天或昨天
    if is_trading_day(today):
        days_behind = (today.date() - last_date.date()).days
        if days_behind > 2:
            issues.append(f"数据滞后{days_behind}天(最后{last_date.date()})")

    # ── 2. OHLC 有效性 ──
    for col in ['open', 'high', 'low', 'close']:
        if col not in df.columns:
            issues.append(f"缺少{col}列")
        else:
            n_null = df[col].isna().sum()
            n_zero = (df[col] == 0).sum()
            if n_null > len(df) * 0.1:
                issues.append(f"{col}缺失{n_null}行")
            if n_zero > len(df) * 0.05:
                issues.append(f"{col}为0共{n_zero}行")

    # ── 3. OHLC 关系 (high>=max(open,close), low<=min(open,close)) ──
    if all(c in df.columns for c in ['high', 'low', 'open', 'close']):
        valid = df.dropna(subset=['high', 'low', 'open', 'close'])
        if len(valid) > 0:
            h_ok = (valid['high'] >= valid[['open', 'close']].max(axis=1))
            l_ok = (valid['low'] <= valid[['open', 'close']].min(axis=1))
            if not h_ok.all():
                issues.append(f"high<max(O,C)共{(~h_ok).sum()}行")
            if not l_ok.all():
                issues.append(f"low>min(O,C)共{(~l_ok).sum()}行")

    # ── 4. 成交量检查 ──
    if 'volume' in df.columns:
        vol_zero = (df['volume'] == 0).sum()
        if vol_zero > len(df) * 0.3:
            issues.append(f"成交量为0共{vol_zero}行")

    # ── 5. 日期连续性 (间隔>4天, 排除节假日) ──
    gaps = df['day_dt'].diff().dt.days
    big_gaps = gaps[gaps > 4]
    real_gaps = []  # 排除纯节假日的gap
    if len(big_gaps) > 0:
        for i, g in big_gaps.items():
            prev_date = df['day_dt'].iloc[i-1]
            curr_date = df['day_dt'].iloc[i]
            # 检查gap中间的交易日数
            missing_trading_days = 0
            d = prev_date + timedelta(days=1)
            while d < curr_date:
                if is_trading_day(d.strftime('%Y-%m-%d')):
                    missing_trading_days += 1
                d += timedelta(days=1)
            if missing_trading_days > 0:
                real_gaps.append((str(prev_date)[:10], str(curr_date)[:10], int(g), missing_trading_days))
        if real_gaps:
            gap_strs = [f"{s}->{e}({d}天,缺{t}个交易日)" for s, e, d, t in real_gaps]
            issues.append(f"日期gap(缺失交易日): {gap_strs}")

    if issues:
        warnings.append(f"{code}({name}): {'; '.join(issues)}")
    else:
        success += 1

# ═══════════════════════════════════════
# amount 数据质量
# ═══════════════════════════════════════
amt_ok = 0
amt_est = 0
for code, df in results.items():
    if df is not None and '_amount_source' in df.columns:
        src = df['_amount_source'].value_counts().to_dict()
        real = src.get('real', 0) + src.get('realtime', 0)
        est = src.get('estimated', 0)
        if real > len(df) * 0.5:
            amt_ok += 1
        else:
            amt_est += 1

# ═══════════════════════════════════════
# 输出报告
# ═══════════════════════════════════════

print(f"\n{'='*60}")
print(f"数据完整性报告")
print(f"{'='*60}")
print(f"总ETF数: {len(codes)}")
print(f"完全通过: {success}")
print(f"有警告:   {len(warnings)}")
print(f"有错误:   {len(errors)}")
print(f"amount 精确: {amt_ok}只, 估算: {amt_est}只")

if warnings:
    print(f"\n── 警告 ({len(warnings)}只) ──")
    for w in warnings:
        print(f"  [WARN] {w}")

if errors:
    print(f"\n── 错误 ({len(errors)}只) ──")
    for e in errors:
        print(f"  [ERROR] {e}")

print(f"{'='*60}")

# 严格模式: 有错误则退出码 1
if errors:
    print("\n[FAIL] 数据不完整!")
    sys.exit(1)

if warnings:
    print(f"\n[WARN] {len(warnings)}只ETF有警告, 但仍可使用")

print("\n[OK] 数据完整")
