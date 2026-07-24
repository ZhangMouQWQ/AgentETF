#!/usr/bin/env python3
"""数据抓取独立脚本 — 供 GitHub Actions 使用"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from data_fetcher import DataFetcher, clear_all_cache
from config import Config

cfg = Config()
pool = cfg.get_etf_pool()
codes = list(pool.keys())

print(f"=== 数据抓取: {len(codes)} 只 ETF ===")

# 清除缓存, 强制重新拉取
clear_all_cache()

fetcher = DataFetcher()
results = fetcher.fetch_batch(codes, period='daily', datalen=cfg.DATA_LEN, max_workers=3)

success = sum(1 for df in results.values() if df is not None and len(df) >= cfg.MOM_LONG + 5)
failed = [c for c in codes if c not in results or results[c] is None or len(results[c]) < cfg.MOM_LONG + 5]

print(f"\n结果: 成功 {success}/{len(codes)}, 失败 {len(failed)}")
if failed:
    print(f"失败列表: {failed}")

# 检查 amount 数据质量
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

print(f"amount 精确: {amt_ok}只, 估算: {amt_est}只")

# 严格模式: 成功 < 39 则退出码 1
if success < len(codes):
    print("\n[FAIL] 数据不完整!")
    sys.exit(1)

print("\n[OK] 数据完整")
