#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ETF 数据获取模块
================
职责: 拉取历史K线, 处理复权/停牌/缓存/交易日判断

数据源: Tencent(日线, 前复权) + Sina(实时成交额 + 60分钟K线)

特性:
  - 日线: Tencent qfq 前复权 + Sina 实时 amount 补充
  - 60分钟线: Sina K线 (非复权, 用于小时线定时信号)
  - 本地 pickle 缓存, 避免重复请求
  - 交易日判断 (周末 + 简单假日检测)
  - 停牌检测 (连续缺失数据)
  - amount 智能补充: 最后一行用 Sina 实时, 历史行用成交量估算

用法:
    from data_fetcher import DataFetcher

    fetcher = DataFetcher()
    df_daily = fetcher.fetch('510300', datalen=260)
    df_60min = fetcher.fetch_60min('510300', datalen=200)
    df_all = fetcher.fetch_batch(['510300','159915'])
"""

import os
import pickle
import time
import hashlib
from datetime import datetime, timezone, timedelta
from concurrent.futures import ThreadPoolExecutor, as_completed

import requests
import pandas as pd
import numpy as np

# ═══════════════════════════════════════
# 配置
# ═══════════════════════════════════════

BJ = timezone(timedelta(hours=8))
CACHE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), '_cache')
os.makedirs(CACHE_DIR, exist_ok=True)


# ═══════════════════════════════════════
# 交易日历 (简易版: 周末 + 已知假日)
# ═══════════════════════════════════════

# A股主要假日 (只需覆盖近2年主要休市日)
_TRADING_HOLIDAYS = {
    # 2025
    '2025-01-01', '2025-01-02',  # 元旦
    '2025-01-28', '2025-01-29', '2025-01-30', '2025-01-31',  # 春节
    '2025-02-03', '2025-02-04',
    '2025-04-04', '2025-04-07',  # 清明
    '2025-05-01', '2025-05-02', '2025-05-05',  # 劳动节
    '2025-05-30',  # 端午
    '2025-10-01', '2025-10-02', '2025-10-03', '2025-10-06', '2025-10-07', '2025-10-08',  # 国庆+中秋
    # 2026
    '2026-01-01', '2026-01-02',
    '2026-02-16', '2026-02-17', '2026-02-18', '2026-02-19', '2026-02-20', '2026-02-23',  # 春节
    '2026-04-06',  # 清明
    '2026-05-01', '2026-05-04', '2026-05-05',  # 劳动节
    '2026-06-19',  # 端午
    '2026-10-01', '2026-10-02', '2026-10-05', '2026-10-06', '2026-10-07', '2026-10-08',  # 国庆
}
_HOLIDAY_SET = set(_TRADING_HOLIDAYS)


def is_trading_day(date=None):
    """判断是否为A股交易日

    Args:
        date: str 'YYYY-MM-DD' 或 datetime, 默认今天

    Returns:
        bool
    """
    if date is None:
        date = datetime.now(BJ)
    if isinstance(date, datetime):
        date = date.strftime('%Y-%m-%d')

    dt = datetime.strptime(date, '%Y-%m-%d')
    # 周末
    if dt.weekday() >= 5:
        return False
    # 假日
    if date in _HOLIDAY_SET:
        return False
    return True


def next_trading_day(date_str=None):
    """获取下一个交易日

    Args:
        date_str: str, 起始日期

    Returns:
        str 'YYYY-MM-DD'
    """
    if date_str is None:
        d = datetime.now(BJ)
    else:
        d = datetime.strptime(date_str[:10], '%Y-%m-%d')
    d += timedelta(days=1)
    while not is_trading_day(d.strftime('%Y-%m-%d')):
        d += timedelta(days=1)
    return d.strftime('%Y-%m-%d')


# ═══════════════════════════════════════
# 缓存管理
# ═══════════════════════════════════════

def _cache_key(code, period, datalen):
    """生成缓存键"""
    raw = f"{code}_{period}_{datalen}"
    return hashlib.md5(raw.encode()).hexdigest()[:16]


def _cache_path(code, period, datalen):
    """缓存文件路径"""
    key = _cache_key(code, period, datalen)
    return os.path.join(CACHE_DIR, f"kl_{key}.pkl")


def _cache_meta_path(code, period, datalen):
    """缓存元信息路径"""
    key = _cache_key(code, period, datalen)
    return os.path.join(CACHE_DIR, f"kl_{key}_meta.pkl")


def load_cache(code, period, datalen, max_age_hours=6):
    """从缓存加载数据

    Args:
        code: ETF代码
        period: 'daily' | '60min'
        datalen: 请求的数据长度
        max_age_hours: 缓存有效期(小时), 超过则需刷新

    Returns:
        DataFrame 或 None
    """
    path = _cache_path(code, period, datalen)
    meta_path = _cache_meta_path(code, period, datalen)
    if not os.path.exists(path) or not os.path.exists(meta_path):
        return None

    try:
        with open(meta_path, 'rb') as f:
            meta = pickle.load(f)
        age = (datetime.now(BJ) - meta.get('cached_at', datetime(2000, 1, 1, tzinfo=BJ))).total_seconds() / 3600

        # 日线: 如果缓存最新日期 == 今天, 不限时; 否则看是否超时
        if meta.get('latest_date') == datetime.now(BJ).strftime('%Y-%m-%d'):
            pass  # 今日数据已缓存, 可用
        elif age > max_age_hours:
            return None

        df = pd.read_pickle(path)
        if df is not None and len(df) >= 10:
            return df
    except Exception:
        pass
    return None


def save_cache(code, period, datalen, df):
    """保存数据到缓存"""
    path = _cache_path(code, period, datalen)
    meta_path = _cache_meta_path(code, period, datalen)
    try:
        df.to_pickle(path)
        meta = {
            'cached_at': datetime.now(BJ),
            'code': code,
            'period': period,
            'rows': len(df),
            'latest_date': str(df['day'].iloc[-1])[:10] if 'day' in df.columns else None,
        }
        with open(meta_path, 'wb') as f:
            pickle.dump(meta, f)
    except Exception as e:
        print(f"  [缓存] 保存失败 {code}: {e}")


def clear_all_cache():
    """清除所有缓存"""
    for f in os.listdir(CACHE_DIR):
        if f.startswith('kl_'):
            os.remove(os.path.join(CACHE_DIR, f))


# ═══════════════════════════════════════
# 数据源: Tencent 日线
# ═══════════════════════════════════════

def _fetch_tencent(sina_code, datalen=260):
    """腾讯财经日线K线 (前复权)"""
    url = (f'http://web.ifzq.gtimg.cn/appstock/app/fqkline/get'
           f'?param={sina_code},day,,,{datalen},qfq')
    headers = {'User-Agent': 'Mozilla/5.0', 'Referer': 'https://finance.qq.com/'}

    try:
        r = requests.get(url, timeout=15, headers=headers)
        data = r.json()
        if data.get('code') != 0:
            print(f"  [Tencent FAIL] {sina_code} 返回码异常: code={data.get('code')}")
            return None
        stock = data.get('data', {}).get(sina_code, {})
        klines = stock.get('qfqday', []) or stock.get('day', [])
        if not klines or len(klines) < 10:
            print(f"  [Tencent FAIL] {sina_code} K线不足: {len(klines) if klines else 0}条")
            return None

        rows = []
        for k in klines:
            if len(k) >= 6:
                rows.append({
                    'day': k[0],
                    'open': float(k[1]) if k[1] else None,
                    'close': float(k[2]) if k[2] else None,
                    'high': float(k[3]) if k[3] else None,
                    'low': float(k[4]) if k[4] else None,
                    'volume': float(k[5]) if k[5] else None,  # 腾讯返回手
                    'amount': None,
                    'turnover': None,
                })

        df = pd.DataFrame(rows).dropna(subset=['close'])
        if len(df) < 10:
            print(f"  [Tencent FAIL] {sina_code} 有效行不足: {len(df)}行")
            return None

        # ── 补充最后一行 amount/turnover (从 qt 实时字段) ──
        today_str = datetime.now(BJ).strftime('%Y-%m-%d')
        qt = stock.get('qt', {}).get(sina_code, [])
        if len(qt) > 38 and df['day'].iloc[-1] <= today_str:
            try:
                amt = float(qt[37]) if qt[37] else None  # 万元
                to = float(qt[38]) if qt[38] else None
                if amt is not None:
                    df.loc[df.index[-1], 'amount'] = amt * 10000
                if to is not None:
                    df.loc[df.index[-1], 'turnover'] = to
            except (ValueError, IndexError):
                pass

        cols = ['day', 'open', 'high', 'low', 'close', 'volume']
        if df['amount'].notna().any():
            cols.append('amount')
        if df['turnover'].notna().any():
            cols.append('turnover')
        result = df[cols]
        date_range = f"{result['day'].iloc[0]}~{result['day'].iloc[-1]}"
        print(f"  [Tencent OK] {sina_code} 日线 {len(result)}行 {date_range}")
        return result

    except Exception as e:
        print(f"  [Tencent FAIL] {sina_code} 异常: {e}")
        return None


# ═══════════════════════════════════════
# Sina 实时行情 (补充当日 amount)
# ═══════════════════════════════════════

def _fetch_sina_realtime(sina_code):
    """新浪实时行情 (close, volume, amount)"""
    url = f'http://hq.sinajs.cn/list={sina_code}'
    headers = {'User-Agent': 'Mozilla/5.0', 'Referer': 'https://finance.sina.com.cn/'}
    try:
        r = requests.get(url, timeout=10, headers=headers)
        text = r.text
        if '=' not in text:
            return None
        parts = text.split('"')[1].split(',') if '"' in text else text.split('=')[1].split(',')
        if len(parts) < 10:
            return None
        result = {}
        if parts[3] and parts[3] != '0.000':
            result['close'] = float(parts[3])
        if parts[8] and parts[8] != '0.000':
            result['volume'] = float(parts[8]) / 100.0
        if parts[9] and parts[9] != '0.000':
            result['amount'] = float(parts[9])
        return result if result else None
    except Exception as e:
        print(f"  [Sina FAIL] {sina_code} 实时行情失败: {e}")
        return None


# ═══════════════════════════════════════
# Sina 60分钟K线 (小时线定时)
# ═══════════════════════════════════════

def _fetch_sina_60min(sina_code, datalen=200):
    """新浪 60分钟K线 (非复权, 用于小时线定时信号)

    返回字段: day(含时分秒), open, high, low, close, volume(股)
    注意: 无 amount/turnover, 非前复权价格
    """
    url = (f'http://money.finance.sina.com.cn/quotes_service/api/json_v2.php/'
           f'CN_MarketData.getKLineData?symbol={sina_code}&scale=60&datalen={datalen}')
    headers = {'User-Agent': 'Mozilla/5.0', 'Referer': 'https://finance.sina.com.cn/'}

    try:
        r = requests.get(url, timeout=15, headers=headers)
        data = r.json()
        if not data or not isinstance(data, list):
            print(f"  [Sina 60min FAIL] {sina_code} 返回空或格式异常")
            return None

        rows = []
        for k in data:
            try:
                rows.append({
                    'day': k['day'],
                    'open': float(k['open']),
                    'close': float(k['close']),
                    'high': float(k['high']),
                    'low': float(k['low']),
                    'volume': float(k['volume']) / 100.0,  # 股→手
                })
            except (KeyError, ValueError):
                continue

        if len(rows) < 10:
            print(f"  [Sina 60min FAIL] {sina_code} 有效行不足: {len(rows)}")
            return None

        df = pd.DataFrame(rows).dropna(subset=['close'])
        # 统一时间戳: YYYY-MM-DD HH:MM
        df['day'] = df['day'].astype(str).str[:16]
        date_range = f"{df['day'].iloc[0]}~{df['day'].iloc[-1]}"
        print(f"  [Sina 60min OK] {sina_code} {len(df)}条 {date_range}")
        return df[['day', 'open', 'high', 'low', 'close', 'volume']]

    except Exception as e:
        print(f"  [Sina 60min FAIL] {sina_code} 失败: {e}")
        return None


# ═══════════════════════════════════════
# 核心类: DataFetcher
# ═══════════════════════════════════════

class DataFetcher:
    """ETF 数据获取器

    用法:
        fetcher = DataFetcher()
        df = fetcher.fetch('510300', datalen=260)
    """

    # ETF代码 → 新浪代码 映射表 (从 ETF_POOL 构建)
    # 子类或外部使用时可注入
    CODE_TO_SINA = {}

    def __init__(self, code_to_sina=None):
        """
        Args:
            code_to_sina: dict, {ETF代码: 新浪代码}, 如 {'510300': 'sh510300'}
                          若不传则使用默认39只ETF池
        """
        if code_to_sina:
            self.CODE_TO_SINA = code_to_sina
        else:
            # 默认加载39只ETF池
            self._load_default_pool()

    def _load_default_pool(self):
        """加载默认ETF映射表 (从 config.py)"""
        try:
            from config import Config
            pool = Config.get_etf_pool()
            self.CODE_TO_SINA = {
                c: info[0] for c, info in pool.items()
            }
            return
        except Exception:
            pass
        # 回退: 硬编码常用映射
        self.CODE_TO_SINA = {
            '510300': 'sh510300', '510500': 'sh510500', '512100': 'sh512100',
            '159915': 'sz159915', '588000': 'sh588000',
        }

    def _get_sina_code(self, code):
        """ETF代码 → 新浪代码"""
        sina = self.CODE_TO_SINA.get(code)
        if sina:
            return sina
        # 自动推断: 5/6开头→sh, 1/0开头→sz
        if code.startswith(('5', '6')):
            return f'sh{code}'
        else:
            return f'sz{code}'

    def fetch(self, code, datalen=260, use_cache=True, force_refresh=False):
        """拉取单只ETF的日线K线数据

        Args:
            code: ETF代码, 如 '510300'
            datalen: 拉取条数
            use_cache: 是否使用缓存
            force_refresh: 强制刷新

        Returns:
            DataFrame (day, open, high, low, close, volume, amount, turnover)
            已前复权, 或 None
        """
        # ── 检查缓存 ──
        if use_cache and not force_refresh:
            cached = load_cache(code, 'daily', datalen)
            if cached is not None:
                return cached

        sina = self._get_sina_code(code)

        # ── 拉取 Tencent 日线 ──
        df = self._fetch_daily(code, sina, datalen)

        if df is None:
            return None

        # ── 数据清洗 ──
        df = self._clean_data(df, 'daily')

        # ── 保存缓存 ──
        if use_cache and df is not None:
            save_cache(code, 'daily', datalen, df)

        return df

    def _fetch_daily(self, code, sina, datalen):
        """日线: Tencent 前复权 + Sina 实时 amount 补充"""
        df = _fetch_tencent(sina, datalen=datalen)
        if df is not None:
            # 补充 amount: 最后一行用 Sina 实时, 历史行用成交量估算
            self._supplement_amount(df, code, sina)
        else:
            print(f"  [Tencent FAIL] {code}({sina}) 获取失败")
        return df

    def fetch_60min(self, code, datalen=2000, use_cache=True, force_refresh=False):
        """拉取单只ETF的60分钟K线

        Args:
            code: ETF代码
            datalen: 拉取条数
            use_cache: 是否使用缓存
            force_refresh: 强制刷新

        Returns:
            DataFrame (day, open, high, low, close, volume) 或 None
        """
        if use_cache and not force_refresh:
            cached = load_cache(code, '60min', datalen)
            if cached is not None:
                return cached

        sina = self._get_sina_code(code)
        df = _fetch_sina_60min(sina, datalen=datalen)

        if df is not None and use_cache:
            save_cache(code, '60min', datalen, df)

        return df

    def fetch_batch_60min(self, codes, datalen=2000, max_workers=3, delay=0.3):
        """批量拉取60分钟K线

        Returns:
            dict: {code: DataFrame}
        """
        results = {}
        total = len(codes)
        success = 0
        failed_codes = []

        with ThreadPoolExecutor(max_workers=max_workers) as ex:
            futures = {ex.submit(self.fetch_60min, c, datalen, True, False): c for c in codes}
            for f in as_completed(futures):
                code = futures[f]
                try:
                    df = f.result()
                    if df is not None and len(df) >= 10:
                        results[code] = df
                        success += 1
                    else:
                        failed_codes.append(code)
                        rows = len(df) if df is not None else 0
                        print(f"  [批量60min FAIL] {code} 数据不足: {rows}条")
                except Exception as e:
                    failed_codes.append(code)
                    print(f"  [批量60min FAIL] {code} 异常: {e}")
                if delay > 0:
                    time.sleep(delay)

        print(f"[批量60min] 完成: {success}/{total} 成功", end='')
        if failed_codes:
            print(f", 失败({len(failed_codes)}): {failed_codes}")
        else:
            print()
        return results

    def _supplement_amount(self, df, code, sina):
        """补充 amount 字段 + 标注数据来源

        数据来源标记 (存入 _amount_source 列):
          'realtime'  — Sina实时行情 (100%精确, 仅最后一行)
          'estimated' — 成交量×均价估算 (误差 ~0.2%)
        """
        if df is None or len(df) == 0:
            return

        # 初始化来源列
        if '_amount_source' not in df.columns:
            df['_amount_source'] = None

        # ── 1. 最后一行: Sina实时行情 (100%精确) ──
        today_str = datetime.now(BJ).strftime('%Y-%m-%d')
        if str(df['day'].iloc[-1])[:10] <= today_str:
            rt = _fetch_sina_realtime(sina)
            if rt and 'amount' in rt:
                if 'amount' not in df.columns:
                    df['amount'] = None
                df.loc[df.index[-1], 'amount'] = rt['amount']
                df.loc[df.index[-1], '_amount_source'] = 'realtime'

        # ── 2. 历史行: 成交量×均价估算 ──
        need_est = ('amount' not in df.columns or df['amount'].isna().any())
        if need_est and all(c in df.columns for c in ['volume', 'high', 'low', 'close']):
            if 'amount' not in df.columns:
                df['amount'] = None
            mask = df['amount'].isna()
            df.loc[mask, 'amount'] = (
                df.loc[mask, 'volume'] * 100 *
                (df.loc[mask, 'high'] + df.loc[mask, 'low'] + df.loc[mask, 'close']) / 3
            )
            # 标注为估算 (保留已有 realtime 标记)
            df.loc[mask & df['_amount_source'].isna(), '_amount_source'] = 'estimated'

    def _clean_data(self, df, period):
        """数据清洗: 停牌检测 + 日期格式统一"""
        if df is None or len(df) == 0:
            return df

        df = df.copy()

        # ── 停牌检测: 标记连续缺失 ──
        df['day'] = pd.to_datetime(df['day'])
        df = df.sort_values('day')
        gaps = df['day'].diff().dt.days
        df = df.reset_index(drop=True)
        df['_suspended'] = gaps > 4  # 间隔超过4天(含周末)标记
        df['day'] = df['day'].dt.strftime('%Y-%m-%d')

        # ── 最小行数检查 ──
        if len(df) < 20:
            return None

        return df

    def fetch_batch(self, codes, datalen=260, max_workers=5, delay=0.3):
        """批量拉取多只ETF日线

        Args:
            codes: ETF 代码列表
            datalen: 拉取条数
            max_workers: 并发数
            delay: 请求间隔(秒), 避免限频

        Returns:
            dict: {code: DataFrame}
        """
        results = {}
        total = len(codes)
        success = 0
        failed_codes = []

        fetch_fn = self.fetch

        with ThreadPoolExecutor(max_workers=max_workers) as ex:
            futures = {ex.submit(fetch_fn, c, datalen, True, False): c for c in codes}
            for i, f in enumerate(as_completed(futures), 1):
                code = futures[f]
                try:
                    df = f.result()
                    if df is not None and len(df) >= 10:
                        results[code] = df
                        success += 1
                    else:
                        failed_codes.append(code)
                        rows = len(df) if df is not None else 0
                        print(f"  [批量 FAIL] {code} 数据不足: {rows}行")
                except Exception as e:
                    failed_codes.append(code)
                    print(f"  [批量 FAIL] {code} 异常: {e}")
                if delay > 0:
                    time.sleep(delay)

        print(f"\n[批量] 完成: {success}/{total} 成功", end='')
        if failed_codes:
            print(f", 失败({len(failed_codes)}): {failed_codes}")
        else:
            print()
        return results

    def detect_suspension(self, code, datalen=260):
        """检测 ETF 是否当前停牌

        Returns:
            dict: {is_suspended: bool, last_trade_date: str, days_since_trade: int}
        """
        df = self.fetch(code, datalen=datalen)
        if df is None or len(df) == 0:
            return {'is_suspended': True, 'last_trade_date': None, 'days_since_trade': 999}

        last_date = df['day'].iloc[-1]
        last_dt = datetime.strptime(str(last_date)[:10], '%Y-%m-%d')
        today = datetime.now(BJ)
        days_since = (today.date() - last_dt.date()).days

        # 如果超过2个交易日没数据, 判定为停牌
        is_suspended = days_since >= 2 and is_trading_day(today)

        return {
            'is_suspended': is_suspended,
            'last_trade_date': str(last_date)[:10],
            'days_since_trade': days_since,
        }

    def get_data_quality(self, code, datalen=260):
        """获取数据质量报告

        Returns:
            dict: {total_rows, amount_real, amount_realtime, amount_estimated, pct_accurate}
        """
        df = self.fetch(code, datalen=datalen)
        if df is None or '_amount_source' not in df.columns:
            return {'total_rows': 0, 'amount_realtime': 0, 'amount_estimated': 0, 'amount_accurate_pct': 0}
        src = df['_amount_source'].value_counts().to_dict()
        total = len(df)
        accurate = src.get('realtime', 0)
        return {
            'total_rows': total,
            'amount_realtime': src.get('realtime', 0),
            'amount_estimated': src.get('estimated', 0),
            'amount_accurate_pct': round(accurate / total * 100, 1) if total > 0 else 0,
        }


# ═══════════════════════════════════════
# 快捷函数
# ═══════════════════════════════════════

_default_fetcher = None


def get_fetcher():
    """获取默认 DataFetcher 单例"""
    global _default_fetcher
    if _default_fetcher is None:
        _default_fetcher = DataFetcher()
    return _default_fetcher


def fetch(code, datalen=260):
    """快捷拉取单只ETF日线"""
    return get_fetcher().fetch(code, datalen=datalen)


def fetch_batch(codes, datalen=260):
    """快捷批量拉取"""
    return get_fetcher().fetch_batch(codes, datalen=datalen)


# ═══════════════════════════════════════
# 自测
# ═══════════════════════════════════════

if __name__ == '__main__':
    print("=== DataFetcher 自测 ===\n")

    fetcher = DataFetcher()

    # 日线
    print("[1] 日线拉取: 510300 (沪深300ETF)")
    df = fetcher.fetch('510300', datalen=20)
    if df is not None:
        print(f"  行数: {len(df)}, 列: {list(df.columns)}")
        print(f"  区间: {df['day'].iloc[0]} ~ {df['day'].iloc[-1]}")
        print(f"  最新: close={df['close'].iloc[-1]:.3f}, vol={df['volume'].iloc[-1]:.0f}")
        if 'amount' in df.columns:
            amt_ok = df['amount'].notna().sum()
            print(f"  amount有效行: {amt_ok}/{len(df)}")

    # 停牌检测
    print("\n[2] 停牌检测: 510300")
    susp = fetcher.detect_suspension('510300')
    print(f"  停牌: {susp['is_suspended']}, 最后交易: {susp['last_trade_date']}")

    # 4. 交易日
    print("\n[4] 交易日判断")
    today = datetime.now(BJ).strftime('%Y-%m-%d')
    print(f"  今天({today})是交易日: {is_trading_day(today)}")
    print(f"  下一个交易日: {next_trading_day(today)}")

    # 5. 数据质量
    print("\n[5] 数据质量报告: 510300")
    q = fetcher.get_data_quality('510300')
    print(f"  总行数: {q['total_rows']}")
    print(f"  精确amount: {q['amount_accurate_pct']}% (实时{q['amount_realtime']}行)")
    print(f"  估算amount: {q['amount_estimated']}行")

    # 6. 缓存
    print("\n[6] 缓存测试 (再次拉取应走缓存)")
    t0 = time.time()
    df2 = fetcher.fetch('510300', datalen=20)
    elapsed = (time.time() - t0) * 1000
    print(f"  耗时: {elapsed:.0f}ms {'(缓存命中)' if elapsed < 100 else '(API拉取)'}")

    print("\n=== 自测完成 ===")
