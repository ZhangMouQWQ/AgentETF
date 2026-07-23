#!/usr/bin/env python3
"""本地数据缓存: 每日收盘后拉取, 测试时直接读取, 避免重复API调用"""
import os
import pickle
import pandas as pd
from datetime import datetime, timezone, timedelta

CACHE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), '_cache')
CACHE_META = os.path.join(CACHE_DIR, 'meta.pkl')
CACHE_PRICE = os.path.join(CACHE_DIR, 'price.pkl')
CACHE_EXTRA = os.path.join(CACHE_DIR, 'extra.pkl')
CACHE_INFO = os.path.join(CACHE_DIR, 'etf_info.pkl')

os.makedirs(CACHE_DIR, exist_ok=True)

bj = timezone(timedelta(hours=8))


def get_cache_status():
    """返回缓存状态: (has_cache, latest_date, is_fresh)"""
    if not os.path.exists(CACHE_META):
        return False, None, False
    try:
        with open(CACHE_META, 'rb') as f:
            meta = pickle.load(f)
        latest = meta.get('latest_date', '')
        today = datetime.now(bj).strftime('%Y-%m-%d')
        is_fresh = (latest == today)
        return True, latest, is_fresh
    except Exception:
        return False, None, False


def load_from_cache():
    """从本地缓存加载数据, 返回 (price, etf_info, extra_history) 或 (None,None,None)"""
    if not os.path.exists(CACHE_PRICE):
        return None, None, None
    try:
        price = pd.read_pickle(CACHE_PRICE)
        with open(CACHE_INFO, 'rb') as f:
            etf_info = pickle.load(f)
        with open(CACHE_EXTRA, 'rb') as f:
            extra_history = pickle.load(f)
        return price, etf_info, extra_history
    except Exception as e:
        print(f"  [缓存] 读取失败: {e}")
        return None, None, None


def save_to_cache(price, etf_info, extra_history):
    """保存数据到本地缓存"""
    try:
        price.to_pickle(CACHE_PRICE)
        with open(CACHE_INFO, 'wb') as f:
            pickle.dump(etf_info, f)
        with open(CACHE_EXTRA, 'wb') as f:
            pickle.dump(extra_history, f)

        today = datetime.now(bj).strftime('%Y-%m-%d')
        latest = str(price.index[-1]) if price is not None and len(price) > 0 else today
        meta = {
            'cached_at': datetime.now(bj).strftime('%Y-%m-%d %H:%M:%S'),
            'latest_date': latest,
            'shape': list(price.shape) if price is not None else [0, 0],
            'etf_count': len(etf_info),
        }
        with open(CACHE_META, 'wb') as f:
            pickle.dump(meta, f)

        print(f"  [缓存] 已保存: {meta['shape'][0]}行×{meta['shape'][1]}列, 最新={latest}")
        return True
    except Exception as e:
        print(f"  [缓存] 保存失败: {e}")
        return False


def should_refresh():
    """判断是否需要刷新数据: 无缓存 或 缓存日期≠今日"""
    has, latest, fresh = get_cache_status()
    if not has:
        print("  [缓存] 无缓存, 需要拉取数据")
        return True
    if not fresh:
        print(f"  [缓存] 过期 (最新={latest}), 需要拉取今日数据")
        return True
    print(f"  [缓存] 数据已是最新 ({latest}), 直接加载")
    return False


def invalidate_cache():
    """强制清除缓存"""
    for f in [CACHE_PRICE, CACHE_EXTRA, CACHE_INFO, CACHE_META]:
        if os.path.exists(f):
            os.remove(f)
    print("  [缓存] 已清除")
