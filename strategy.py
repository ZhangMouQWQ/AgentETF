#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
行业ETF动量轮动策略 - 概念板块分组版(双层架构)
========================================
改进内容:
1. ETF按热门概念板块分组,每个板块放最具代表性的ETF
2. 40日动量基于昨日收盘计算(稳定层),不受盘中未完成数据干扰
3. 当日涨跌/5日动量独立用于风控和买入确认(灵活层)
4. 买入增加"当日涨幅<5%"过滤,避免盘中追高
5. 60分钟K线仅用于合成当日最新价,不用于历史动量计算
6. 波动率基于历史日K线(不含今天盘中噪声)
"""
import requests
import pandas as pd
import numpy as np
import time
import json
import os
from datetime import datetime, timezone, timedelta
from concurrent.futures import ThreadPoolExecutor, as_completed

try:
    import akshare as ak
except Exception:
    ak = None

# ==================== 配置 ====================
SECTOR_ETF_POOL = {
    '宽基指数': {
        '510300': ('sh510300', '沪深300ETF'),
        '510500': ('sh510500', '中证500ETF'),
        '512100': ('sh512100', '中证1000ETF'),
        '159915': ('sz159915', '创业板ETF'),
        '588000': ('sh588000', '科创50ETF'),
    },
    '金融': {
        '512800': ('sh512800', '银行ETF'),
        '512000': ('sh512000', '券商ETF'),
        '512200': ('sh512200', '房地产ETF'),
    },
    '消费': {
        '515170': ('sh515170', '食品饮料ETF'),
        '512690': ('sh512690', '酒ETF'),
        '159996': ('sz159996', '家电ETF'),
        '159865': ('sz159865', '养殖ETF'),
    },
    '医药': {
        '512010': ('sh512010', '医药ETF'),
        '515120': ('sh515120', '创新药ETF'),
        '512170': ('sh512170', '医疗ETF'),
    },
    '电子': {
        '512480': ('sh512480', '半导体ETF'),
        '159995': ('sz159995', '芯片ETF'),
    },
    '计算机': {
        '159819': ('sz159819', '人工智能ETF'),
        '512720': ('sh512720', '计算机ETF'),
    },
    '电力新能源': {
        '515030': ('sh515030', '新能源车ETF'),
        '515790': ('sh515790', '光伏ETF'),
        '159611': ('sz159611', '电力ETF'),
    },
    '军工': {
        '512660': ('sh512660', '军工ETF'),
        '512670': ('sh512670', '国防ETF'),
    },
    '金属矿产': {
        '512400': ('sh512400', '有色金属ETF'),
        '516780': ('sh516780', '稀土ETF'),
        '518880': ('sh518880', '黄金ETF'),
    },
    '能源化工': {
        '515220': ('sh515220', '煤炭ETF'),
        '159697': ('sz159697', '油气ETF'),
        '159870': ('sz159870', '化工ETF'),
        '515210': ('sh515210', '钢铁ETF'),
    },
    '通信传媒': {
        '515880': ('sh515880', '通信ETF'),
        '512980': ('sh512980', '传媒ETF'),
        '159869': ('sz159869', '游戏ETF'),
    },
    '制造基建': {
        '516110': ('sh516110', '汽车ETF'),
        '159530': ('sz159530', '机器人ETF'),
        '516970': ('sh516970', '基建ETF'),
    },
    '红利': {
        '510880': ('sh510880', '红利ETF'),
        '515080': ('sh515080', '中证红利ETF'),
    },
}


def flatten_pool(sector_pool):
    """将分组ETF池扁平化,同时记录每个ETF所属板块"""
    flat = {}
    etf_sector = {}
    for sector, etfs in sector_pool.items():
        for code, (sina, name) in etfs.items():
            flat[code] = (sina, name)
            etf_sector[name] = sector
    return flat, etf_sector


ETF_POOL, ETF_SECTOR = flatten_pool(SECTOR_ETF_POOL)

DATA_LEN = 260
MOM_LONG = 40
MOM_SHORT = 5
STOP_LOSS = -4.0
MAX_DAILY_DROP = -4.0
MAX_DAILY_RISE = 5.0
VOL_WINDOW = 60
MIN_VOL = 5.0
TOP_K = 3
# ==============================================


def get_etf_sina(sina_code, scale=240, datalen=DATA_LEN, retry=3):
    """通过腾讯财经API获取K线数据（替代已废弃的新浪旧K线接口）"""
    # sina_code 格式如 sh510300 / sz159915，直接用于腾讯API
    # scale: 240=日线, 60=60分钟
    ktype_map = {240: 'day', 60: '60'}
    ktype = ktype_map.get(scale, 'day')
    # 腾讯API的K线key带 qfq 前缀（前复权）
    key = f'qfq{ktype}'
    url = (f'http://web.ifzq.gtimg.cn/appstock/app/fqkline/get'
           f'?param={sina_code},{ktype},,,{datalen},qfq')
    headers = {'User-Agent': 'Mozilla/5.0', 'Referer': 'https://finance.qq.com/'}
    for attempt in range(retry):
        try:
            r = requests.get(url, timeout=15, headers=headers)
            data = r.json()
            if data.get('code') != 0:
                raise ValueError(f"腾讯API返回code={data.get('code')}")
            stock_data = data.get('data', {}).get(sina_code, {})
            if not stock_data:
                raise ValueError(f"未找到标的 {sina_code}")
            # 尝试多个可能的键名: qfqday (前复权), day (不复权), qfq60 (60分钟前复权)
            klines = stock_data.get(key, [])
            if not klines and key == 'qfqday':
                klines = stock_data.get('day', [])
            if not klines and key == 'qfq60':
                klines = stock_data.get('60', [])
            if not klines or len(klines) < 10:
                raise ValueError(f"K线数据不足: {len(klines) if klines else 0}条 key={key}")
            rows = []
            for k in klines:
                # 腾讯K线格式: [日期, 开盘, 收盘, 最高, 最低, 成交量(股)]
                if len(k) >= 6:
                    day_str = k[0]
                    rows.append({
                        'day': day_str,
                        'open': float(k[1]) if k[1] else None,
                        'close': float(k[2]) if k[2] else None,
                        'high': float(k[3]) if k[3] else None,
                        'low': float(k[4]) if k[4] else None,
                        'volume': float(k[5]) / 100.0 if k[5] else None,  # 股→手
                        'amount': None,
                        'turnover': None,
                    })
            df = pd.DataFrame(rows).dropna(subset=['close'])
            if len(df) < 10:
                raise ValueError(f"有效K线不足: {len(df)}条")

            # 从 qt 字段提取实时成交额/换手率,覆盖最后一行
            # ≤ today_str: 允许凌晨/盘前填充昨日数据(此时qt来自上个交易日)
            today_str = datetime.now(timezone(timedelta(hours=8))).strftime('%Y-%m-%d')
            qt_list = stock_data.get('qt', {}).get(sina_code, [])
            if len(qt_list) > 38 and df['day'].iloc[-1] <= today_str:
                try:
                    amt_wan = float(qt_list[37]) if qt_list[37] else None  # 万元
                    turnover_pct = float(qt_list[38]) if qt_list[38] else None  # %
                    if amt_wan is not None:
                        df.loc[df.index[-1], 'amount'] = amt_wan * 10000  # 万元→元
                    if turnover_pct is not None:
                        df.loc[df.index[-1], 'turnover'] = turnover_pct
                except (ValueError, IndexError):
                    pass
            return_cols = ['day', 'open', 'high', 'low', 'close', 'volume']
            if df['amount'].notna().any():
                return_cols.append('amount')
            if df['turnover'].notna().any():
                return_cols.append('turnover')
            return df[return_cols]
        except Exception as e:
            print(f"  [警告] {sina_code} scale={scale} 第{attempt+1}次失败: {e}")
            time.sleep(1)
    return None


def get_etf_extra_sina(sina_code):
    """从新浪实时行情获取当日累计数据（价格/成交量/成交额），比60分钟K线更可靠"""
    url = f'http://hq.sinajs.cn/list={sina_code}'
    headers = {'User-Agent': 'Mozilla/5.0', 'Referer': 'https://finance.sina.com.cn/'}
    try:
        r = requests.get(url, timeout=10, headers=headers)
        text = r.text
        if '=' not in text:
            return None
        # 格式: var hq_str_sh510300="名称,今开,昨收,当前,最高,最低,买一,卖一,...,成交量(股),成交额(元),..."
        data = text.split('"')[1] if '"' in text else text.split('=')[1]
        parts = data.split(',')
        if len(parts) < 10:
            return None
        result = {}
        # parts[1]=今开, parts[2]=昨收, parts[3]=当前价, parts[4]=最高, parts[5]=最低
        close_str = parts[3] if len(parts) > 3 else ''
        if close_str and close_str != '0.000' and close_str != '':
            result['close'] = float(close_str)
        # parts[8]=成交量(股,需/100转为手), parts[9]=成交额(元)
        vol_str = parts[8] if len(parts) > 8 else ''
        amt_str = parts[9] if len(parts) > 9 else ''
        if vol_str and vol_str != '0.000' and vol_str != '':
            result['volume'] = float(vol_str) / 100.0  # 股→手
        if amt_str and amt_str != '0.000' and amt_str != '':
            result['amount'] = float(amt_str)  # 已是元,直接使用
        return result if result else None
    except Exception as e:
        print(f"  [提示] 新浪实时行情 {sina_code} 失败: {e}")
    return None


def get_etf_realtime_eastmoney(code):
    """从东方财富实时行情获取当日数据（含换手率），作为新浪的补充"""
    if code.startswith('5') or code.startswith('6'):
        secid = f'1.{code}'
    else:
        secid = f'0.{code}'
    # f43=最新价(/1000), f47=成交量(股), f48=成交额(元), f168=换手率(%/100), f50=量比
    params = {'secid': secid, 'fields': 'f43,f47,f48,f50,f168,f170',
              'ut': 'fa5fd1943c7b386f172d6893dbfba10b'}
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36',
        'Referer': 'https://quote.eastmoney.com/',
        'Accept': '*/*',
        'Connection': 'keep-alive',
    }
    try:
        r = requests.get('https://push2.eastmoney.com/api/qt/stock/get',
                        params=params, headers=headers,
                        proxies={'http': None, 'https': None}, timeout=10)
        data = r.json()
        if data.get('rc') != 0:
            return None
        d = data.get('data', {})
        if not d:
            return None
        result = {}
        if d.get('f43') is not None and d['f43'] != '-':
            result['close'] = float(d['f43']) / 1000.0
        if d.get('f47') is not None and d['f47'] != '-':
            result['volume'] = float(d['f47']) / 100.0  # 股→手
        if d.get('f48') is not None and d['f48'] != '-':
            result['amount'] = float(d['f48'])
        if d.get('f168') is not None and d['f168'] != '-':
            result['turnover'] = float(d['f168']) / 100.0  # /100 → %
        return result if result else None
    except Exception as e:
        print(f"  [提示] 东方财富实时行情 {code} 失败: {e}")
    return None


def get_etf_history_eastmoney(code, datalen=DATA_LEN):
    """直接从东方财富K线接口获取历史数据（含成交额/换手率）"""
    if code.startswith('5') or code.startswith('6'):
        secid = f'1.{code}'
    else:
        secid = f'0.{code}'
    params = {
        'secid': secid,
        'ut': 'fa5fd1943c7b386f172d6893dbfba10b',
        'fields1': 'f1,f2,f3,f4,f5,f6',
        'fields2': 'f51,f52,f53,f54,f55,f56,f57,f58,f59,f60,f61',
        'klt': '101', 'fqt': '1', 'end': '20500101', 'lmt': str(datalen + 20),
    }
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36',
        'Referer': 'https://quote.eastmoney.com/',
        'Accept': '*/*',
        'Connection': 'keep-alive',
    }
    try:
        session = requests.Session()
        r = session.get('https://push2his.eastmoney.com/api/qt/stock/kline/get',
                        params=params, headers=headers,
                        proxies={'http': None, 'https': None}, timeout=15)
        session.close()
        data = r.json()
        if not data or not data.get('data') or not data['data'].get('klines'):
            return None
        klines = data['data']['klines']
        if len(klines) < MOM_LONG + 5:
            return None
        rows = []
        for k in klines:
            # 格式: 日期,开盘,收盘,最高,最低,成交量,成交额,振幅,涨跌幅,涨跌额,换手率
            parts = k.split(',')
            if len(parts) >= 11:
                rows.append({
                    'day': parts[0],
                    'open': float(parts[1]) if parts[1] and parts[1] != '-' else None,
                    'close': float(parts[2]) if parts[2] and parts[2] != '-' else None,
                    'high': float(parts[3]) if parts[3] and parts[3] != '-' else None,
                    'low': float(parts[4]) if parts[4] and parts[4] != '-' else None,
                    'volume': float(parts[5]) if parts[5] and parts[5] != '-' else None,
                    'amount': float(parts[6]) if parts[6] and parts[6] != '-' else None,
                    'turnover': float(parts[10]) if parts[10] and parts[10] != '-' else None,
                })
        df = pd.DataFrame(rows).dropna(subset=['close'])
        if len(df) < MOM_LONG + 5:
            return None
        available = [c for c in ['day', 'open', 'high', 'low', 'close', 'volume', 'amount', 'turnover'] if c in df.columns]
        return df[available]
    except Exception as e:
        print(f"  [警告] 东方财富K线 {code} 失败: {e}")
        return None


def get_etf_history_akshare(code, datalen=DATA_LEN):
    if ak is None:
        print(f"  [提示] akshare 未安装或不可用, {code} 将回退到新浪数据")
        return None
    try:
        end_date = datetime.now(timezone(timedelta(hours=8))).strftime('%Y%m%d')
        start_date = (datetime.now(timezone(timedelta(hours=8))) - timedelta(days=datalen + 60)).strftime('%Y%m%d')
        df = ak.fund_etf_hist_em(symbol=code, start_date=start_date, end_date=end_date, adjust='')
        if df is None or df.empty:
            return None
        df = df.copy()
        rename_map = {
            '日期': 'day', 'date': 'day', '交易日期': 'day',
            '开盘': 'open', 'open': 'open',
            '收盘': 'close', 'close': 'close',
            '最高': 'high', 'high': 'high',
            '最低': 'low', 'low': 'low',
            '成交量': 'volume', 'volume': 'volume', 'vol': 'volume',
            '成交额': 'amount', 'amount': 'amount', '额': 'amount',
            '换手率': 'turnover', 'turnover': 'turnover', '换手': 'turnover',
        }
        df = df.rename(columns={k: v for k, v in rename_map.items() if k in df.columns})
        if 'day' not in df.columns:
            return None
        df['day'] = pd.to_datetime(df['day']).dt.strftime('%Y-%m-%d')
        for col in ['open', 'close', 'high', 'low', 'volume', 'amount', 'turnover']:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
        available = [c for c in ['day', 'open', 'high', 'low', 'close', 'volume', 'amount', 'turnover'] if c in df.columns]
        return df[available]
    except Exception as e:
        print(f"  [警告] akshare 获取 {code} 失败: {e}")
        return None


def _merge_amount_turnover(existing, supplement_df):
    """将 supplement_df 中的 amount/turnover 按日期合并到 existing DataFrame

    existing: 来自 Tencent 的 DataFrame (index=day, columns可能缺amount/turnover)
    supplement_df: 来自 Eastmoney/AKShare 的 DataFrame (index=day, 含amount/turnover)

    返回: 合并后的 DataFrame, 或 None(无可合并数据)
    """
    if supplement_df is None or len(supplement_df) == 0:
        return None

    need_cols = []
    if 'amount' not in existing.columns or existing['amount'].isna().all():
        if 'amount' in supplement_df.columns:
            need_cols.append('amount')
    if 'turnover' not in existing.columns or existing['turnover'].isna().all():
        if 'turnover' in supplement_df.columns:
            need_cols.append('turnover')

    if not need_cols:
        return existing  # 不需要补充

    # 只取需要的列
    supp = supplement_df[need_cols].copy()

    # 对齐日期: 取existing中缺失amount/turnover的行
    existing_aligned = existing.copy()
    for col in need_cols:
        if col not in existing_aligned.columns:
            existing_aligned[col] = float('nan')

    # 按公共日期合并
    common_dates = existing_aligned.index.intersection(supp.index)
    if len(common_dates) == 0:
        return None

    for col in need_cols:
        # 只填充 existing 中为 NaN 的值
        mask = existing_aligned[col].isna()
        fill_dates = common_dates.intersection(existing_aligned.index[mask])
        if len(fill_dates) > 0:
            existing_aligned.loc[fill_dates, col] = supp.loc[fill_dates, col]

    filled_count = sum(
        (existing_aligned[col].notna() & existing[col].isna()).sum()
        if col in existing.columns
        else existing_aligned[col].notna().sum()
        for col in need_cols
    ) if len(need_cols) > 0 else 0

    if filled_count > 0:
        return existing_aligned
    return None


def fetch_daily_data(pool, datalen=DATA_LEN, max_workers=5):
    all_close = {}
    etf_info = {}
    extra_history = {}
    data_sources = set()
    tencent_etfs = {}  # 记录腾讯源的ETF: {name: (code, sina)}
    fail_count = 0
    success_count = 0
    total = len(pool)

    def _fetch_one(item):
        """单只ETF拉取: AKShare(8字段) → Eastmoney(8字段最全) → Tencent(6字段)

        AKShare 优先提供自然错峰: 大部分ETF在AKShare快速失败后,
        Eastmoney 请求被分散到不同时间点, 避免并发限频。
        Tencent 仅 OHLCV, amount/turnover 由后续二次补充获取。
        """
        code, (sina, name) = item
        # 1. AKShare: 完整8字段, 同时为Eastmoney提供错峰
        df = get_etf_history_akshare(code, datalen=datalen)
        source = 'akshare'
        if df is None:
            # 2. Eastmoney: 字段最全 (OHLCV + amount + turnover)
            df = get_etf_history_eastmoney(code, datalen=datalen)
            source = 'eastmoney'
        if df is None:
            # 3. Tencent: 仅 OHLCV, 缺 amount/turnover
            df = get_etf_sina(sina, scale=240, datalen=datalen)
            source = 'tencent'
        return name, code, source, df, sina

    print(f"拉取历史日K线 (共{total}只, {max_workers}线程并行)...")
    items = list(pool.items())
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(_fetch_one, item): item for item in items}
        for i, future in enumerate(as_completed(futures), 1):
            name, code, source, df, sina = future.result()
            data_sources.add(source)
            if df is not None and len(df) >= MOM_LONG + 5:
                all_close[name] = df.set_index('day')['close']
                etf_info[name] = code
                available_extra = [c for c in ['open', 'volume', 'amount', 'turnover', 'high', 'low'] if c in df.columns]
                if available_extra:
                    extra_df = df[['day'] + available_extra].copy()
                    extra_df = extra_df.dropna(how='all')
                    extra_df = extra_df.set_index('day')
                    extra_history[name] = extra_df
                extra_info = ''
                # 标记腾讯源: 缺 amount 或 turnover
                has_amount = 'amount' in (df.columns if df is not None else []) and df['amount'].notna().any()
                has_turnover = 'turnover' in (df.columns if df is not None else []) and df['turnover'].notna().any()
                if source == 'tencent' or (not has_amount and not has_turnover):
                    tencent_etfs[name] = (code, sina)
                    extra_info += ' (缺成交额/换手率,待补充)'
                print(f"  [{i}/{total}] ok {name}({code}): {len(df)} 条, 最新 {df['day'].iloc[-1]} 来源={source}{extra_info}")
                success_count += 1
            else:
                print(f"  [{i}/{total}] FAIL {name}({code}): 数据不足")
                fail_count += 1
    print(f"\n数据拉取完成: 成功={success_count}/{total}, 失败={fail_count}/{total}")
    print(f"  数据来源: Eastmoney={sum(1 for s in data_sources if s=='eastmoney')} "
          f"AKShare={sum(1 for s in data_sources if s=='akshare')} "
          f"Tencent={sum(1 for s in data_sources if s=='tencent')}")

    # ── 二次补充: 对腾讯源ETF, 尝试从 Eastmoney/AKShare 获取完整 amount/turnover ──
    if tencent_etfs:
        print(f"\n二次补充: {len(tencent_etfs)}只ETF缺成交额/换手率, 尝试从其他API获取...")
        supplemented = 0

        def _supplement_one(name_code_sina):
            """对单只ETF尝试补充 amount/turnover (带延迟避免限频)"""
            name, (code, sina) = name_code_sina
            existing = extra_history.get(name)
            if existing is None:
                return name, code, False, "无已有数据"

            # 尝试 Eastmoney (最全), 先加短暂延迟避免并发冲击
            time.sleep(0.5)
            df_em = get_etf_history_eastmoney(code, datalen=datalen)
            if df_em is not None and len(df_em) >= MOM_LONG // 2:
                df_em = df_em.set_index('day')
                merged = _merge_amount_turnover(existing, df_em)
                if merged is not None:
                    extra_history[name] = merged
                    return name, code, True, "eastmoney"

            # 尝试 AKShare
            time.sleep(0.3)
            df_ak = get_etf_history_akshare(code, datalen=datalen)
            if df_ak is not None and len(df_ak) >= MOM_LONG // 2:
                df_ak = df_ak.set_index('day')
                merged = _merge_amount_turnover(existing, df_ak)
                if merged is not None:
                    extra_history[name] = merged
                    return name, code, True, "akshare"

            return name, code, False, "无可补充源"

        with ThreadPoolExecutor(max_workers=2) as executor:
            supp_futures = {
                executor.submit(_supplement_one, item): item[0]
                for item in tencent_etfs.items()
            }
            for future in as_completed(supp_futures):
                name, code, ok, supp_src = future.result()
                if ok:
                    supplemented += 1
                    print(f"  [补充] {name}({code}): amount/turnover 已通过 {supp_src} 补全")
                else:
                    print(f"  [警告] {name}({code}): 补充失败({supp_src}), 将用成交量估算")

        print(f"  二次补充完成: {supplemented}/{len(tencent_etfs)} 成功")

    # ── 最终回退: 对仍缺 amount 的ETF, 用 成交量 × 均价 估算 ──
    estimated = 0
    for name, df in extra_history.items():
        if 'amount' not in df.columns or df['amount'].isna().all():
            if 'volume' in df.columns and 'high' in df.columns and 'low' in df.columns:
                close_s = all_close.get(name, pd.Series(dtype=float))
                df['amount'] = df['volume'] * (df['high'] + df['low'] + close_s) / 3
                estimated += 1
    if estimated:
        print(f"  [估算] {estimated}只ETF的成交额通过 成交量×均价 估算(精确度较低)")

    if not all_close:
        return None, {}, {}, data_sources
    price = pd.DataFrame(all_close).sort_index()
    # 先 ffill 填充各列内部缺失，再只删除全空列（而非有任何NaN就删）
    price = price.ffill().dropna(how='all', axis=1)
    # 丢弃数据量不足 MOM_LONG+2 的列（在 calc_metrics 中也会检查，这里提前过滤）
    valid_cols = [c for c in price.columns if price[c].notna().sum() >= MOM_LONG + 2]
    dropped = set(price.columns) - set(valid_cols)
    if dropped:
        print(f"  [过滤] 数据量不足({MOM_LONG+2}条)的标的: {', '.join(sorted(dropped))}")
    price = price[valid_cols]
    print(f"日K线维度: {price.shape}, 最新日期: {price.index[-1]}")

    # 补充成交额: 腾讯API仅最后一行有amount(历史行为NaN)
    # 检查今日行是否缺amount,若缺则用新浪实时补全
    print("\n补充成交额(新浪实时行情)...")
    latest_day = price.index[-1]
    needs_supp = {}
    for code, (sina, name) in pool.items():
        if name not in etf_info:
            continue
        existing = extra_history.get(name)
        # 仅检查今日行(iloc[-1])是否缺amount,新浪只能补今日
        missing_today = (existing is not None and 'amount' in existing.columns
                         and len(existing) >= 1
                         and pd.isna(existing['amount'].iloc[-1]))
        if missing_today or (existing is None or 'amount' not in existing.columns):
            needs_supp[sina] = (code, name, existing)

    if needs_supp:
        # 批量请求: 新浪支持逗号分隔多个代码
        batch_codes = ','.join(needs_supp.keys())
        url = f'http://hq.sinajs.cn/list={batch_codes}'
        headers = {'User-Agent': 'Mozilla/5.0', 'Referer': 'https://finance.sina.com.cn/'}
        try:
            r = requests.get(url, timeout=15, headers=headers)
            lines = r.text.strip().split('\n')
            for line in lines:
                if '=' not in line:
                    continue
                parts = line.split('"')
                if len(parts) < 2:
                    continue
                var_part = parts[0]
                sina_code = var_part.split('_str_')[1] if '_str_' in var_part else ''
                if sina_code not in needs_supp:
                    continue
                data = parts[1].split(',')
                if len(data) < 10:
                    continue
                code, name, existing = needs_supp[sina_code]
                row = {}
                amt_str = data[9] if len(data) > 9 else ''
                if amt_str and amt_str != '0.000':
                    row['amount'] = float(amt_str)
                vol_str = data[8] if len(data) > 8 else ''
                if vol_str and vol_str != '0.000' and (existing is None or 'volume' not in existing.columns):
                    row['volume'] = float(vol_str) / 100.0
                if row:
                    supp_df = pd.DataFrame(row, index=[latest_day])
                    supp_df.index.name = 'day'
                    if existing is not None:
                        if latest_day in existing.index:
                            for col, val in row.items():
                                existing.loc[latest_day, col] = val
                        else:
                            extra_history[name] = pd.concat([existing, supp_df])
                    else:
                        extra_history[name] = supp_df
                    print(f"  ok {name}({code}): 已补充 {list(row.keys())}")
        except Exception as e:
            print(f"  [警告] 新浪批量补充失败: {e}，回退串行")
            for sina_code, (code, name, existing) in needs_supp.items():
                extra = get_etf_extra_sina(sina_code)
                if extra:
                    row = {}
                    if 'amount' in extra: row['amount'] = extra['amount']
                    if 'volume' in extra and (existing is None or 'volume' not in existing.columns):
                        row['volume'] = extra['volume']
                    if row:
                        supp_df = pd.DataFrame(row, index=[latest_day])
                        supp_df.index.name = 'day'
                        if existing is not None:
                            if latest_day in existing.index:
                                for col, val in row.items():
                                    existing.loc[latest_day, col] = val
                            else:
                                extra_history[name] = pd.concat([existing, supp_df])
                        else:
                            extra_history[name] = supp_df
                        print(f"  ok {name}({code}): 已补充 {list(row.keys())}")
                time.sleep(0.15)

    return price, etf_info, extra_history, data_sources


def fetch_60min_for_execution(pool, max_workers=5):
    """拉取60分钟K线 (最近40根≈10天), 用于执行层优化买卖点
    
    返回: {etf_name: DataFrame}
    """
    intraday = {}
    total = len(pool)
    
    def _fetch_one(item):
        code, (sina, name) = item
        df = get_etf_sina(sina, scale=60, datalen=40)
        return name, df
    
    print(f"\n拉取60分钟K线(执行层, {max_workers}线程)...")
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(_fetch_one, item): item for item in pool.items()}
        for i, future in enumerate(as_completed(futures), 1):
            name, df = future.result()
            if df is not None and len(df) >= 4:
                intraday[name] = df
                print(f"  [{i}/{total}] ok {name}: {len(df)}根60min K线")
            else:
                print(f"  [{i}/{total}] fail {name}")
    return intraday


def fetch_daily_data_cached(pool, datalen=DATA_LEN, max_workers=5, force_refresh=False):
    """带缓存的 fetch_daily_data: 优先读本地, 无变化则跳过API调用
    
    force_refresh=True 强制重新拉取(用于收盘后更新)
    """
    try:
        from data_cache import should_refresh, load_from_cache, save_to_cache
    except ImportError:
        return fetch_daily_data(pool, datalen, max_workers)

    if not force_refresh and not should_refresh():
        price, etf_info, extra_history = load_from_cache()
        if price is not None:
            print(f"  [OK] 从缓存加载: {price.shape[0]}行x{price.shape[1]}列")
            return price, etf_info, extra_history, {'cached'}
        print("  [缓存] 读取失败, 回退API拉取")

    # 需要拉取
    result = fetch_daily_data(pool, datalen, max_workers)
    if result[0] is not None:
        save_to_cache(result[0], result[1], result[2])
    return result


def fetch_60min_for_execution(etf_code, sina_code, target_date, datalen=10):
    """拉取单只ETF的60分钟K线, 用于当日执行层优化买卖点
    
    返回: {date_str: [(time, open, high, low, close), ...]} 或 None
    仅拉取最近datalen根K线(约2.5天)
    """
    df = get_etf_sina(sina_code, scale=60, datalen=datalen)
    if df is None or len(df) < 2:
        return None
    
    # 按日期分组
    result = {}
    for _, row in df.iterrows():
        day_str = row['day'][:10]  # "2026-07-23 10:30" → "2026-07-23"
        time_str = row['day'][11:16] if len(row['day']) > 10 else ''
        bar = (time_str, float(row['open']), float(row['high']),
               float(row['low']), float(row['close']))
        result.setdefault(day_str, []).append(bar)
    return result


def find_best_entry(day_bars):
    """从60分钟K线找最优买入价: 优先等回调, 无回调则用开盘
    
    逻辑: 在上午的2根K线(9:30-11:30)中找最低点
    策略: 开盘后如果第2根K线比第1根低 → 用第2根低点, 否则用开盘
    """
    if not day_bars or len(day_bars) < 1:
        return None
    
    # 第一根K线(9:30-10:30)的开盘价
    open_price = day_bars[0][1]
    
    # 上午的K线(前2根: 9:30-11:30)
    morning_bars = day_bars[:2]
    lows = [b[3] for b in morning_bars]  # 最低价
    
    if len(lows) >= 2 and lows[1] < open_price:
        # 第2根K线低于开盘 → 等到了回调
        return round(lows[1], 3)
    
    # 无回调 → 用开盘价
    return round(open_price, 3)


def find_best_exit(day_bars):
    """从60分钟K线找最优卖出价: 全天最高点
    
    逻辑: 在4根K线(全天)中找最高点
    """
    if not day_bars:
        return None
    
    highs = [b[2] for b in day_bars]  # 最高价
    return round(max(highs), 3) if highs else None


def fetch_intraday_snapshot(pool, max_workers=5):
    """使用实时行情API获取当日盘中数据（价格+成交量+成交额+换手率）
    数据源优先级: 新浪实时行情(可靠,5/5) → 东方财富实时行情(补充换手率,不稳定) → 60分钟K线兜底"""
    today = datetime.now(timezone(timedelta(hours=8))).strftime('%Y-%m-%d')
    snapshot = {}

    def _fetch_one(code, sina, name):
        snap = {'code': code}
        # 1) 新浪实时行情: 价格 + 成交量 + 成交额（可靠，200ms级）
        sina_data = get_etf_extra_sina(sina)
        if sina_data:
            for k in ['close', 'volume', 'amount']:
                if k in sina_data:
                    snap[k] = sina_data[k]
        # 2) 东方财富实时行情: 补充换手率 + 量比
        em_data = get_etf_realtime_eastmoney(code)
        if em_data:
            if 'close' not in snap and 'close' in em_data:
                snap['close'] = em_data['close']
            if 'volume' not in snap and 'volume' in em_data:
                snap['volume'] = em_data['volume']
            if 'amount' not in snap and 'amount' in em_data:
                snap['amount'] = em_data['amount']
            if 'turnover' in em_data:
                snap['turnover'] = em_data['turnover']
        # 3) 兜底: 60分钟K线
        if 'close' not in snap:
            df = get_etf_sina(sina, scale=60, datalen=20)
            if df is not None and len(df) >= 1:
                df['date'] = df['day'].str[:10]
                df_today = df[df['date'] == today].sort_values('day')
                if len(df_today) >= 1:
                    last = df_today.iloc[-1]
                    snap['close'] = float(last['close'])
                    if 'volume' in df_today.columns:
                        snap['volume'] = float(df_today['volume'].sum())
                    if 'amount' in df.columns and pd.notna(last.get('amount')):
                        snap['amount'] = float(last['amount'])
        return name, snap

    print(f"\n拉取实时行情合成当日数据 (今日 {today}, {max_workers}线程并行)...")
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(_fetch_one, code, sina, name): name
                   for code, (sina, name) in pool.items()}
        for future in as_completed(futures):
            name, snap = future.result()
            if 'close' in snap:
                snapshot[name] = snap
                fields = ','.join(k for k in ['close', 'volume', 'amount', 'turnover'] if k in snap)
                print(f"  ok {name}: {snap['close']:.3f} ({fields})")
            else:
                print(f"  fail {name}: 无可用实时数据")
    return snapshot


def merge_intraday_price(price_daily, intraday_snapshot, etf_info, extra_history=None):
    if not intraday_snapshot or price_daily is None:
        return price_daily, extra_history
    today_str = datetime.now(timezone(timedelta(hours=8))).strftime('%Y-%m-%d')
    print(f"\n合并盘中数据到日K线...")

    if price_daily.index[-1] == today_str:
        print(f"日K最新日期已是 {today_str}, 用实时行情覆盖当日数据")
        for name, snap in intraday_snapshot.items():
            if name in price_daily.columns:
                price_daily.loc[today_str, name] = snap['close']
    else:
        print(f"日K最新日期 {price_daily.index[-1]}, 追加今日 {today_str}")
        new_row = pd.Series({n: np.nan for n in price_daily.columns}, name=today_str)
        for name in price_daily.columns:
            if name in intraday_snapshot:
                new_row[name] = intraday_snapshot[name]['close']
            else:
                new_row[name] = price_daily[name].iloc[-1]
        price_daily = pd.concat([price_daily, new_row.to_frame().T])

    # 同步更新 extra_history：将盘中量/额/换手率追加到今日日期
    if extra_history is not None:
        for name, snap in intraday_snapshot.items():
            extra_cols = {k: snap[k] for k in ['volume', 'amount', 'turnover'] if k in snap}
            if not extra_cols:
                continue
            row = pd.DataFrame(extra_cols, index=[today_str])
            row.index.name = 'day'
            existing = extra_history.get(name)
            if existing is not None:
                # 若今日已存在则覆盖，否则追加
                if today_str in existing.index:
                    for col, val in extra_cols.items():
                        existing.loc[today_str, col] = val
                else:
                    extra_history[name] = pd.concat([existing, row])
            else:
                extra_history[name] = row

    price_daily = price_daily.dropna(how='all')
    print(f"合并后维度: {price_daily.shape}, 最新日期: {price_daily.index[-1]}")
    if extra_history is not None:
        updated = sum(1 for snap in intraday_snapshot.values() if any(k in snap for k in ['volume', 'amount', 'turnover']))
        if updated:
            print(f"盘中量价数据已同步: {updated} 只 ETF 的 extra_history 已更新至 {today_str}")
    return price_daily.sort_index(), extra_history


def calc_momentum_change(series, lookback=MOM_LONG, is_stale=False, is_intraday=False):
    """动量变化: 今日vs昨日。intraday时用昨日vs前日(匹配mom_ref基准)"""
    offset = 1 if is_stale else 0
    intra = 1 if is_intraday and not is_stale else 0  # 盘中: 跳过今日未完成数据
    if len(series) < lookback + 3 + offset + intra:
        return None, None, None
    curr_day = series.iloc[-1 - intra]
    prev_day = series.iloc[-2 - intra]
    current_mom = (curr_day / series.iloc[-lookback - 1 - offset - intra] - 1) * 100
    previous_mom = (prev_day / series.iloc[-lookback - 2 - offset - intra] - 1) * 100
    change = current_mom - previous_mom
    return current_mom, previous_mom, change


def format_metric_value(value, show_sign=False):
    if value is None:
        return '0.00'
    if show_sign:
        return f"{value:+.2f}"
    return f"{value:.2f}"


def calc_session_progress(now):
    """计算A股交易时段进度(0.0~1.0)。非交易日/非交易时段返回1.0(全日)"""
    if now.weekday() >= 5:  # 周末
        return 1.0
    minutes = now.hour * 60 + now.minute
    open_min = 9 * 60 + 30    # 570  (9:30)
    morning_end = 11 * 60 + 30  # 690  (11:30)
    afternoon_start = 13 * 60   # 780  (13:00)
    close_min = 15 * 60         # 900  (15:00)
    if minutes < open_min or minutes >= close_min:
        return 1.0
    if minutes <= morning_end:
        elapsed = minutes - open_min
    elif minutes < afternoon_start:
        elapsed = morning_end - open_min   # 午休
    else:
        elapsed = (morning_end - open_min) + (minutes - afternoon_start)
    return min(elapsed / 240.0, 1.0)


def calc_metrics(price, etf_info, etf_sector, extra_history=None, session_progress=None,
                 data_date=None, is_intraday=False):
    metrics = {}
    if price is None or len(price) < MOM_LONG + 2:
        return metrics
    for name in price.columns:
        s = price[name]
        if len(s) < MOM_LONG + 2:
            continue

        # 动量基准: 盘中用昨日(数据不完整), 盘前/假日用最新, 正常收盘用今日
        stale = (data_date is not None and str(s.index[-1]) != data_date)

        if is_intraday and not stale:
            mom_ref = s.iloc[-2]; mom_lag = 2
        elif stale:
            mom_ref = s.iloc[-1]; mom_lag = 2
        else:
            mom_ref = s.iloc[-1]; mom_lag = 1

        latest = s.iloc[-1]
        prev = s.iloc[-2]

        mom_long = (mom_ref / s.iloc[-MOM_LONG - mom_lag] - 1) * 100
        mom_short = (mom_ref / s.iloc[-MOM_SHORT - mom_lag] - 1) * 100
        _, _, mom_long_change = calc_momentum_change(s, lookback=MOM_LONG, is_stale=stale, is_intraday=is_intraday)
        _, _, mom_short_change = calc_momentum_change(s, lookback=MOM_SHORT, is_stale=stale, is_intraday=is_intraday)
        # Parkinson波动率: 利用High/Low捕捉日内振幅,统一截面无偏置
        extra = extra_history.get(name) if extra_history else None
        if extra is not None and 'high' in extra.columns and 'low' in extra.columns and len(extra) >= VOL_WINDOW // 2:
            hl = extra[['high', 'low']].dropna().tail(VOL_WINDOW)
            if len(hl) >= VOL_WINDOW // 2:
                hl_ratio = np.log(hl['high'] / hl['low'])
                parkinson_daily = np.sqrt(1 / (4 * np.log(2)) * (hl_ratio ** 2).mean())
                vol = parkinson_daily * np.sqrt(252) * 100
            else:
                vol = 999.0
        else:
            # 回退: 收盘价标准差(统一截面,保留此能力)
            rets = s.iloc[:-1].pct_change().dropna().iloc[-VOL_WINDOW:]
            vol = rets.std() * np.sqrt(252) * 100 if len(rets) >= VOL_WINDOW // 2 else 999.0
        vol = max(vol, MIN_VOL)
        raw_score = (mom_long / vol) if mom_long > 0 else None

        daily_change = (latest / prev - 1) * 100

        direction = '持平'
        if mom_long_change is not None:
            if mom_long_change > 0:
                direction = '上升'
            elif mom_long_change < 0:
                direction = '下降'

        amount = None
        turnover = None
        volume_ratio = None
        amount_change = None
        amount_chg_pct = None
        turnover_change = None
        if extra is not None and not extra.empty:
            last_row = extra.iloc[-1]
            amount = last_row.get('amount')
            turnover = last_row.get('turnover')
            # 较昨日变化：优先用原始数据，否则基于成交量估算
            if len(extra) >= 2:
                prev_row = extra.iloc[-2]
                prev_amount = prev_row.get('amount')
                prev_turnover = prev_row.get('turnover')
                if amount is not None and prev_amount is not None and not (isinstance(prev_amount, float) and np.isnan(prev_amount)):
                    amount_change = round((amount - prev_amount) / 1e8, 2)
                    if prev_amount > 0:
                        amount_chg_pct = round((amount / prev_amount - 1) * 100, 2)
                elif amount is not None and 'volume' in extra.columns:
                    today_vol = last_row.get('volume')
                    prev_vol = prev_row.get('volume')
                    if today_vol and prev_vol and today_vol > 0:
                        amount_change = round((amount / 1e8) * (1 - prev_vol / today_vol), 2)
                        amount_chg_pct = round((today_vol / prev_vol - 1) * 100, 2)
                if turnover is not None and prev_turnover is not None and not (isinstance(prev_turnover, float) and np.isnan(prev_turnover)):
                    turnover_change = round(turnover - prev_turnover, 2)
                elif turnover is not None and 'volume' in extra.columns:
                    # 仅当日有换手率时，用成交量变化估算（假设总股本不变）
                    today_vol = last_row.get('volume')
                    prev_vol = prev_row.get('volume')
                    if today_vol and prev_vol and today_vol > 0:
                        turnover_change = round(turnover * (1 - prev_vol / today_vol), 2)
            recent_volumes = extra['volume'].dropna()
            # Bug2修复: 用今天之前的5个完整交易日做均量基准,避免盘中量拉低分母
            if len(recent_volumes) >= 6:
                avg_vol = recent_volumes.iloc[-6:-1].mean()
                latest_vol = recent_volumes.iloc[-1]
            elif len(recent_volumes) >= 2:
                avg_vol = recent_volumes.iloc[:-1].mean()
                latest_vol = recent_volumes.iloc[-1]
            else:
                avg_vol = None
                latest_vol = None
            if avg_vol and avg_vol > 0 and latest_vol:
                latest_date = str(extra.index[-1])
                if session_progress and 0.05 < session_progress < 1.0 and latest_date == data_date:
                    projected_vol = min(latest_vol / session_progress, avg_vol * 5)
                    volume_ratio = ((projected_vol / avg_vol) - 1) * 100
                elif session_progress and session_progress <= 0.05:
                    volume_ratio = None  # 开盘5分钟内数据量不足,不计算量比
                else:
                    volume_ratio = ((latest_vol / avg_vol) - 1) * 100

        # 暂存原始资金流指标,稍后统一做截面Z-score
        flow_raw = {
            'turnover': turnover,
            'volume_ratio': volume_ratio,
            'amount_chg_pct': amount_chg_pct,
        }
        metrics[name] = {
            'code': etf_info.get(name, ''),
            'sector': etf_sector.get(name, '其他'),
            'latest': round(latest, 3),
            'prev': round(prev, 3),
            'daily_change': round(daily_change, 2),
            'mom_long': round(mom_long, 2),
            'mom_short': round(mom_short, 2),
            'mom_short_change': round(mom_short_change, 2) if mom_short_change is not None else 0.0,
            'mom_long_change': round(mom_long_change, 2) if mom_long_change is not None else 0.0,
            'mom_long_change_direction': direction,
            'mom_long_change_text': (("↑" if mom_long_change and mom_long_change > 0 else "↓" if mom_long_change and mom_long_change < 0 else "→") + format_metric_value(abs(mom_long_change), show_sign=False)) if mom_long_change is not None else '→0.00',
            'mom_short_change_text': (("↑" if mom_short_change and mom_short_change > 0 else "↓" if mom_short_change and mom_short_change < 0 else "→") + format_metric_value(abs(mom_short_change), show_sign=False)) if mom_short_change is not None else '→0.00',
            'vol': round(vol, 1),
            'amount': round(amount / 1e8, 2) if amount is not None else None,
            'amount_change': amount_change,
            'amount_chg_pct': amount_chg_pct,
            'turnover': round(turnover, 2) if turnover is not None else None,
            'turnover_change': turnover_change,
            'volume_ratio': round(volume_ratio, 2) if volume_ratio is not None else None,
            '_flow_raw': flow_raw,
            'score': raw_score,
        }

    # ── 截面Z-score: 资金流用全市场相对排名替代硬编码阈值 ──
    def _valid(v):
        return v is not None and not (isinstance(v, float) and np.isnan(v))

    to_vals = np.array([m['_flow_raw']['turnover'] for m in metrics.values()
                        if _valid(m['_flow_raw']['turnover'])])
    vr_vals = np.array([m['_flow_raw']['volume_ratio'] for m in metrics.values()
                        if _valid(m['_flow_raw']['volume_ratio'])])
    ac_vals = np.array([m['_flow_raw']['amount_chg_pct'] for m in metrics.values()
                        if _valid(m['_flow_raw']['amount_chg_pct'])])

    def _z(arr, val):
        if not _valid(val) or len(arr) < 3:
            return 0.0
        std = np.std(arr) or 1.0
        return (val - np.mean(arr)) / std

    for m in metrics.values():
        r = m['_flow_raw']
        z_to = _z(to_vals, r['turnover'])
        z_vr = _z(vr_vals, r['volume_ratio'])
        z_ac = _z(ac_vals, r['amount_chg_pct'])
        m['flow_signal'] = round(z_to * 0.5 + z_vr * 0.3 + z_ac * 0.2, 2)
        del m['_flow_raw']  # 清理临时字段

    positives = [m['score'] for m in metrics.values() if m['score'] is not None]
    flow_values = [m['flow_signal'] for m in metrics.values()]
    for m in metrics.values():
        if m['score'] is None or not positives:
            m['score'] = 0.0
        else:
            lo, hi = min(positives), max(positives)
            m['score'] = 100.0 if hi == lo else round((m['score'] - lo) / (hi - lo) * 100, 1)
        if flow_values:
            flow_lo, flow_hi = min(flow_values), max(flow_values)
            if flow_hi == flow_lo:
                m['flow_score_norm'] = 50.0
            else:
                m['flow_score_norm'] = round((m['flow_signal'] - flow_lo) / (flow_hi - flow_lo) * 100, 1)
        else:
            m['flow_score_norm'] = 0.0

    # ── 统一质量评分: 截面Z-score加权,排序与前端展示使用同一公式 ──
    non_broad_mom = np.array([m['mom_long'] for m in metrics.values()
                              if m.get('sector') != '宽基指数'])
    non_broad_score = np.array([m['score'] for m in metrics.values()
                                if m.get('sector') != '宽基指数'])
    non_broad_flow = np.array([m['flow_signal'] for m in metrics.values()
                               if m.get('sector') != '宽基指数'])

    def _safe_z(arr, val):
        std = np.std(arr) or 1.0
        return (val - np.mean(arr)) / std

    for m in metrics.values():
        if m.get('sector') == '宽基指数':
            m['quality_score'] = 0.0
            m['rec_score'] = 5.0
            continue
        z_mom = _safe_z(non_broad_mom, m['mom_long'])
        z_score = _safe_z(non_broad_score, m['score'])
        z_flow = _safe_z(non_broad_flow, m['flow_signal'])
        m['quality_score'] = round(z_mom * 0.4 + z_score * 0.35 + z_flow * 0.25, 2)
        # 推荐值: quality_score 映射到 [0,10], 5=中性, >5=偏多, <5=偏空
        raw = (m['quality_score'] + 2.5) / 5.0 * 10
        m['rec_score'] = round(max(0.0, min(raw, 10.0)), 1)

    return metrics


def market_timing(metrics):
    """双层仓位判定: 沪深300动量(主) + 板块宽度(防单点失效)
    
    避免结构性行情中大盘指数走弱但题材板块暴涨时被强行空仓。
    """
    m = metrics.get('沪深300ETF')

    # ── 板块宽度: 板块内超半数ETF动量为正则算该板块为正 ──
    sectors = {}
    for name, mt in metrics.items():
        s = mt.get('sector', '')
        if s == '宽基指数':
            continue
        sectors.setdefault(s, []).append(mt['mom_long'] > 0)

    breadth_ratio = 0.0
    if sectors:
        positive_sectors = sum(1 for etfs in sectors.values() if sum(etfs) > len(etfs) / 2)
        breadth_ratio = positive_sectors / len(sectors)

    if m is None:
        return 0.0, "0成(空仓)", "沪深300数据缺失,保守观望", "danger"

    hs300_mom = m['mom_long']

    # 沪深300强势 → 正常满仓
    if hs300_mom > 2:
        return 1.0, "10成(满仓)", "大盘强势,积极参与", "ok"

    # 沪深300震荡偏多 → 半仓
    if hs300_mom > 0:
        return 0.5, "5成(半仓)", "大盘震荡,控制仓位", "warn"

    # 沪深300走弱 → 不强行空仓，看板块宽度
    if breadth_ratio >= 0.5:
        return 0.5, "5成(半仓)", f"大盘弱势但{positive_sectors}/{len(sectors)}板块强势,结构性行情参与", "warn"
    if breadth_ratio >= 0.3:
        return 0.3, "3成(轻仓)", f"大盘弱势,仅{positive_sectors}/{len(sectors)}板块活跃,轻仓试探", "warn"

    return 0.0, "0成(空仓)", "大盘弱势且板块全线走弱,观望为主", "danger"


def selection_override(target, position_ratio, signal_type='close',
                       prev_day_return=None, consecutive_losses=0, hs300_mom=None):
    """选股信号覆盖择时 V2: 仓位分级 + 紧止损 + 熔断 + HS300混合

    改进点:
    - P1 仓位分级: quality_score>2.0→5成, >1.0→3成 (而非统一5成)
    - P0 紧止损: 昨日跌幅<-2% → 明日仓位降至3成
    - P1 熔断: 连续2天亏损 → 暂停覆盖3天
    - P2 HS300混合: HS300动量>0时覆盖可给更高仓位

    Returns: (new_ratio, new_text, new_reason, new_cls, overridden, state_updates)
      state_updates: dict with keys 'consecutive_losses', 'last_return', 'paused_until'
    """
    state_updates = {}

    if position_ratio > 0 or signal_type == 'morning':
        return position_ratio, None, None, None, False, state_updates

    # ── P1 熔断: 连续2天亏损 → 暂停覆盖3天 ──
    if consecutive_losses >= 2:
        print(f"\n  🛑 熔断: 连续{consecutive_losses}天亏损, 暂停选股覆盖")
        return position_ratio, None, None, None, False, state_updates

    # ── 选股质量检查 ──
    high_quality = [h for h in target if h.get('quality_score', 0) > 1.0]
    super_quality = [h for h in high_quality if h.get('quality_score', 0) > 2.0]

    if len(high_quality) < 2:
        return position_ratio, None, None, None, False, state_updates

    # ── P1 仓位分级: quality_score > 2.0 → 5成, > 1.0 → 3成 ──
    if len(super_quality) >= 2:
        base_ratio = 0.5  # 5成(半仓)
        level = "半仓"
    else:
        base_ratio = 0.3  # 3成(轻仓)
        level = "轻仓"

    # ── P2 HS300混合: HS300动量>0时可额外加仓 ──
    hs300_bonus = 0
    if hs300_mom is not None and hs300_mom > 0:
        hs300_bonus = 0.1  # HS300偏多, 额外加1成
        base_ratio = min(base_ratio + hs300_bonus, 0.6)

    # ── P0 紧止损: 昨日跌>2%则降仓 ──
    stop_loss_triggered = False
    if prev_day_return is not None and prev_day_return < -0.02:
        base_ratio = min(base_ratio, 0.3)
        stop_loss_triggered = True

    new_ratio = round(base_ratio, 1)

    # 构建理由
    names = ', '.join(f"{h['name']}(质量{h['quality_score']:+.1f})" for h in high_quality[:3])
    parts = [f"{len(high_quality)}只高质量标的({names})"]

    if len(super_quality) >= 2:
        parts.append(f"{len(super_quality)}只超优(quality>2.0)→{level}")
    else:
        parts.append(f"quality>1.0→{level}")

    if hs300_bonus > 0:
        parts.append(f"HS300动量{hs300_mom:+.1f}%偏多+1成")
    if stop_loss_triggered:
        parts.append("⚠️紧止损:昨日跌幅>2%→降仓")

    new_text = f"{int(new_ratio*10)}成({level})"
    new_reason = f"选股覆盖: {'; '.join(parts)} → 大盘弱势但个股强势,{level}参与"
    new_cls = "warn" if new_ratio >= 0.5 else "danger"

    print(f"\n  ⚡ 选股覆盖触发: position_ratio {position_ratio}→{new_ratio}")
    print(f"  ⚡ 分级: {level} | 高质量={len(high_quality)}只 | 超优={len(super_quality)}只")
    if stop_loss_triggered:
        print(f"  ⚡ 紧止损生效: 仓位被限制在{int(new_ratio*10)}成")

    return new_ratio, new_text, new_reason, new_cls, True, state_updates


def calc_adaptive_params(metrics):
    """根据市场近期波动率动态调整阈值,避免单边行情中被固定参数误判
    
    核心思路: 所有百分比阈值用"市场波动率的倍数"表达
    - 高波动市场 → 放宽止损/涨幅上限(不容易被震出去)
    - 低波动市场 → 收紧阈值(更敏感地捕捉异常)
    """
    hs300 = metrics.get('沪深300ETF', {})
    market_vol = hs300.get('vol', 20.0)
    # 防御: vol=999 表示数据异常(calc_metrics中的兜底值),用全市场中位数替代
    if market_vol > 100:
        all_vols = [m.get('vol', 20.0) for m in metrics.values() if m.get('vol', 0) < 100]
        market_vol = np.median(all_vols) if all_vols else 20.0  # 沪深300年化波动率(%)
    # 相对基准 20% 年化波动的缩放因子,限制在 0.5~2.0
    vol_factor = max(0.5, min(market_vol / 20.0, 2.0))

    # 板块宽度: 调整 Top-K
    sectors = {}
    for m in metrics.values():
        s = m.get('sector', '')
        if s == '宽基指数':
            continue
        sectors.setdefault(s, []).append(m['mom_long'] > 0)
    positive_sectors = sum(1 for etfs in sectors.values() if sum(etfs) > len(etfs) / 2)
    breadth_ratio = positive_sectors / len(sectors) if sectors else 0

    # 连续函数替代离散阈值,避免边界震荡导致回撤放大
    # top_k ≈ 1 + breadth × 4, 平滑过渡: 0%→1只, 25%→2只, 50%→3只, 75%→4只, 100%→5只
    top_k = max(1, min(round(1 + breadth_ratio * 4), 5))

    return {
        'stop_loss': round(STOP_LOSS * vol_factor, 1),
        'max_daily_drop': round(MAX_DAILY_DROP * vol_factor, 1),
        'max_daily_rise': round(MAX_DAILY_RISE * vol_factor, 1),
        'top_k': top_k,
        'market_vol': market_vol,
        'vol_factor': round(vol_factor, 2),
    }


def build_target(metrics, position_ratio=0, adaptive=None):
    """纯选股推荐器: 以40日动量+截面资金流综合排序,不因择时空仓而隐藏推荐"""
    ad = adaptive or {}
    max_rise = ad.get('max_daily_rise', MAX_DAILY_RISE)
    max_drop = ad.get('max_daily_drop', -3.0)
    k = ad.get('top_k', TOP_K)

    # 基础门槛: 40日动量为正、5日动量不过度恶化、当日未爆拉(防追高)、排除宽基指数
    candidates = [
        (n, m) for n, m in metrics.items()
        if m['mom_long'] > 0
        and m['mom_short'] > max_drop
        and m['daily_change'] < max_rise
        and m.get('sector', '') != '宽基指数'
    ]

    if not candidates:
        return [], position_ratio

    # 统一使用 quality_score 排序(与前端展示一致)
    candidates.sort(key=lambda x: x[1].get('quality_score', 0), reverse=True)

    # 板块集中度风控: 同板块最多2只
    sector_counts = {}
    selected = []
    for n, m in candidates:
        s = m['sector']
        if sector_counts.get(s, 0) >= 2:
            continue
        selected.append((n, m))
        sector_counts[s] = sector_counts.get(s, 0) + 1
        if len(selected) >= k:
            break
    if not selected:
        selected = candidates[:k]

    # 按仓位比例分配权重 (0仓位时权重为0,仅展示)
    if position_ratio > 0:
        w = round(position_ratio / len(selected), 2)
        weights = [w] * len(selected)
        weights[-1] = round(position_ratio - sum(weights[:-1]), 2)
    else:
        weights = [0.0] * len(selected)

    return [
        {"name": n, "code": m['code'], "weight": w, "mom_long": m['mom_long'],
         "score": m['score'], "sector": m['sector'], "quality_score": m.get('quality_score', 0)}
        for (n, m), w in zip(selected, weights)
    ], position_ratio

def generate_recommendations(target, position_ratio, signal_type, position_reason):
    """纯推荐器: 将选股结果转化为可读推荐列表"""
    recs = []
    if not target:
        recs.append({"type": "WAIT", "name": "空仓", "code": "", "weight": 0,
                     "msg": "【观望】当前无符合条件的标的",
                     "urgency": "low", "reason": "无40日动量为正且满足过滤条件的ETF"})
        return recs

    label = "下午关注" if signal_type == 'morning' else "推荐"
    for h in target:
        recs.append({
            "type": "PICK", "name": h['name'], "code": h['code'],
            "weight": h['weight'],
            "msg": f"【{label}】{h['name']}({h['code']}) "
                   f"质量{h.get('quality_score',0):+.2f} | 40日动量{h['mom_long']:+.2f}%",
            "urgency": "normal",
            "reason": f"质量分{h.get('quality_score',0):+.2f}, "
                      f"40日动量{h['mom_long']:+.2f}%"
        })

    if position_ratio <= 0:
        recs.insert(0, {"type": "NOTE", "name": "", "code": "", "weight": 0,
                        "msg": f"⚠️ 大盘择时建议空仓({position_reason}),以下推荐仅作逆势板块观察",
                        "urgency": "low", "reason": "择时空仓"})
    return recs

def build_html(target, position_text, position_reason, market_cls,
               asof, update_time, signal_type, metrics, data_source_label):

    def buy_picks():
        """潜力推荐：使用统一 quality_score，与排序逻辑一致"""
        out = []
        # 空仓警示
        if not target or all(h.get('weight', 0) == 0 for h in target):
            out.append('<div class="action-card warn-text">'
                       '<div class="action-row"><span class="action-label">⚠️</span>'
                       '<span class="action-value"><b>大盘择时建议空仓'
                       f'({position_reason})</b>，以下推荐仅作逆势板块观察参考</span></div>'
                       '</div>')
        if not target:
            out.append('<div class="action-card hold">'
                       '<div class="action-row"><span class="action-label">建议</span>'
                       '<span class="action-value">【观望】暂无符合40日动量条件的标的</span></div>'
                       '</div>')
            return "\n".join(out)

        sector_items = {}
        for i, h in enumerate(target):
            s = h.get('sector', '')
            sector_items.setdefault(s, []).append((i, h))

        out = []
        for sector, items_in_sector in sector_items.items():
            out.append('<div style="font-size:12px;font-weight:700;color:#2c5364;padding:8px 0 4px 0;border-bottom:1px solid #e8ecf0;margin-top:4px">&#128193; ' + sector + '</div>')
            for rank, h in items_in_sector:
                m = metrics.get(h['name'], {})
                quality = h.get('quality_score', m.get('quality_score', 0))
                mom_str   = f"{h['mom_long']:+.2f}%" if h.get('mom_long') is not None else '-'
                short_str = (format_metric_value(m.get('mom_short', 0), show_sign=True) + '%') if m.get('mom_short') is not None else '-'
                daily_str = (format_metric_value(m.get('daily_change', 0), show_sign=True) + '%') if m.get('daily_change') is not None else '-'
                score_str = f"{h['score']:.1f}" if h.get('score') is not None else '-'
                flow_str  = f"{m.get('flow_signal', 0):.1f}" if m.get('flow_signal') is not None else '-'
                vol_str   = (str(round(m.get('vol', 0), 1)) + '%') if m.get('vol') is not None else '-'
                q_label   = '⭐⭐⭐' if quality > 1.2 else '⭐⭐' if quality > 0.5 else '⭐'

                out.append(
                    '<div class="action-card pos">'
                    '<div class="action-row"><span class="action-label">#' + str(rank + 1) + '</span>'
                    '<span class="action-value"><b>' + h['name'] + '(' + h['code'] + ')</b>'
                    ' <span style="font-size:11px">' + q_label + ' 质量 ' + f'{quality:+.2f}' + '</span></span></div>'
                    '<div class="action-row"><span class="action-label">动量</span>'
                    '<span class="action-value">40日 ' + mom_str
                    + ' | 5日 ' + short_str
                    + ' | 当日 ' + daily_str + '</span></div>'
                    '<div class="action-row"><span class="action-label">评分</span>'
                    '<span class="action-value">得分 ' + score_str
                    + ' | 资金流 ' + flow_str
                    + ' | 波动率 ' + vol_str + '</span></div>'
                    '</div>')
        return "\n".join(out)

    def risk_warnings():
        """风险警示：截面Z-score + 自适应绝对阈值双重验证"""
        items = [(n, m) for n, m in metrics.items() if m.get('sector') != '宽基指数']
        n_all = len(items)
        if n_all < 5:
            return ('<div class="action-card hold">'
                    '<div class="action-row"><span class="action-label">安全</span>'
                    '<span class="action-value">样本不足，无法评估风险</span></div>'
                    '</div>')

        # 自适应阈值: 基于市场波动率动态调节
        hs300_vol = metrics.get('沪深300ETF', {}).get('vol', 20.0)
        if hs300_vol > 100: hs300_vol = 20.0
        vf = max(0.5, min(hs300_vol / 20.0, 2.0))
        abs_daily = round(-2.0 * vf, 1)   # 当日跌幅阈值(高波动放宽)
        abs_short = round(-3.0 * vf, 1)   # 5日动量阈值
        abs_long  = round(-5.0 * vf, 1)   # 40日动量破位阈值

        vals_daily   = np.array([m.get('daily_change', 0) for _, m in items])
        vals_short   = np.array([m.get('mom_short', 0) for _, m in items])
        vals_long_chg = np.array([m.get('mom_long_change', 0) for _, m in items])

        def safe_z(arr, val):
            return (val - np.mean(arr)) / (np.std(arr) or 1.0)

        raw = []
        for name, m in items:
            z_daily   = safe_z(vals_daily,   m.get('daily_change', 0))
            z_short   = safe_z(vals_short,   m.get('mom_short', 0))
            z_long    = safe_z(vals_long_chg, m.get('mom_long_change', 0))
            risk = z_daily * 0.3 + z_short * 0.4 + z_long * 0.3

            has_absolute = (m.get('daily_change', 0) < abs_daily or
                            m.get('mom_short', 0) < abs_short or
                            m.get('mom_long', 0) < abs_long)
            if risk < -1.0 and has_absolute:
                signals = []
                if m.get('daily_change', 0) < abs_daily:
                    signals.append(('stop', f'当日跌幅 {m["daily_change"]:+.1f}%(阈值{abs_daily}%,z={z_daily:.1f}σ)'))
                if m.get('mom_short', 0) < abs_short:
                    signals.append(('deteriorate', f'5日动量 {m["mom_short"]:+.1f}%(阈值{abs_short}%,z={z_short:.1f}σ)'))
                if m.get('mom_long', 0) < abs_long:
                    signals.append(('top', f'40日动量 {m["mom_long"]:+.1f}%(阈值{abs_long}%),趋势破位'))
                if not signals:
                    signals.append(('warn', f'综合风险分 {risk:.1f}σ'))
                raw.append((m['sector'], name, m['code'], signals, -risk))

        if not raw:
            return ('<div class="action-card hold">'
                    '<div class="action-row"><span class="action-label">安全</span>'
                    '<span class="action-value">当前无显著截面风险（所有 ETF 偏离均在 1σ 以内）</span></div>'
                    '</div>')

        # 按板块分组，板块按总风险排序
        sector_order = sorted(set(r[0] for r in raw),
            key=lambda s: sum(r[4] for r in raw if r[0] == s), reverse=True)

        type_label = {'stop': '跌幅', 'top': '破位', 'deteriorate': '恶化', 'warn': '预警'}
        out = []
        last_sector = None
        for sector, name, code, signals, risk_score in sorted(raw,
                key=lambda r: (sector_order.index(r[0]), -r[4])):
            if sector != last_sector:
                out.append('<div style="font-size:12px;font-weight:700;color:#2c5364;padding:8px 0 4px 0;border-bottom:1px solid #e8ecf0;margin-top:4px">&#128193; ' + sector + '</div>')
                last_sector = sector

            # 风险分决定卡片颜色和等级标识
            if risk_score > 2.0:
                card_cls, risk_icon, risk_label = 'neg', '🔴🔴🔴', '高风险'
            elif risk_score > 1.5:
                card_cls, risk_icon, risk_label = 'neg', '🔴🔴', '中高风险'
            elif risk_score > 1.2:
                card_cls, risk_icon, risk_label = 'warn-text', '🔴', '中风险'
            else:
                card_cls, risk_icon, risk_label = 'warn-text', '🟠', '关注'

            rows = []
            for stype, sreason in signals:
                rows.append('<div class="action-row"><span class="action-label">' + type_label.get(stype, '') + '</span><span class="action-value">' + sreason + '</span></div>')
            out.append(
                '<div class="action-card ' + card_cls + '">'
                '<div class="action-row"><span class="action-label">标的</span>'
                '<span class="action-value"><b>' + name + '(' + code + ')</b>'
                ' <span style="font-size:11px">' + risk_icon + ' ' + risk_label + ' ' + f'{risk_score:.1f}' + '</span></span></div>'
                + "\n".join(rows) +
                '</div>')
        return "\n".join(out)

    def monitor_rows():
        sector_groups = {}
        for name, m in metrics.items():
            sector = m['sector']
            sector_groups.setdefault(sector, []).append((name, m))
        sector_order = sorted(sector_groups.keys(),
                              key=lambda s: sum(m.get('amount') or 0 for _, m in sector_groups[s]),
                              reverse=True)
        out = []
        for sector in sector_order:
            items = sorted(sector_groups[sector], key=lambda x: x[1].get('amount') or 0, reverse=True)
            out.append(
                '<tr style="background:#f0f7ff">' +
                '<td style="font-weight:700;color:#2c5364;padding:8px 10px;position:sticky;left:0;z-index:1;background:#f0f7ff;min-width:140px">&#128193; ' + sector + '</td>' +
                '<td colspan="12" style="background:#f0f7ff;padding:8px 10px"></td>' +
                '</tr>')
            for name, m in items:
                mom_cls = 'pos' if m['mom_long'] > 0 else 'neg'
                short_cls = 'pos' if m['mom_short'] > 0 else 'neg'
                short_change_cls = 'pos' if m['mom_short_change'] > 0 else 'neg' if m['mom_short_change'] < 0 else ''
                daily_cls = 'pos' if m['daily_change'] > 0 else 'neg'
                change_cls = 'pos' if m['mom_long_change'] > 0 else 'neg' if m['mom_long_change'] < 0 else ''
                amount_text = '-' if m.get('amount') is None else format_metric_value(m['amount'])
                amount_change_val = m.get('amount_change')
                amount_change_text = '-' if amount_change_val is None else format_metric_value(amount_change_val, show_sign=True)
                amount_change_cls = 'pos' if amount_change_val and amount_change_val > 0 else 'neg' if amount_change_val and amount_change_val < 0 else ''
                turnover_text = '-' if m.get('turnover') is None else format_metric_value(m['turnover'])
                turnover_change_val = m.get('turnover_change')
                turnover_change_text = '-' if turnover_change_val is None else format_metric_value(turnover_change_val, show_sign=True)
                turnover_change_cls = 'pos' if turnover_change_val and turnover_change_val > 0 else 'neg' if turnover_change_val and turnover_change_val < 0 else ''
                vol_text = '-' if m.get('vol') is None else str(round(m['vol'], 1))
                flow_value = m.get('flow_signal', 0)
                flow_cls = 'pos' if flow_value >= 5 else 'neg' if flow_value <= 0 else ''
                flow_text = '-' if flow_value is None else format_metric_value(flow_value)
                rec_val = m.get('rec_score', 5.0)
                rec_cls = 'pos' if rec_val > 5.0 else 'neg'
                rec_text = f"{rec_val:.1f}"
                out.append(
                    '<tr><td class="nm" style="padding-left:24px">' + name + '(' + m["code"] + ')</td>' +
                    '<td class="' + mom_cls + '">' + format_metric_value(m["mom_long"], show_sign=True) + '%</td>' +
                    '<td class="' + change_cls + '">' + m["mom_long_change_text"] + '</td>' +
                    '<td class="' + short_cls + '">' + format_metric_value(m["mom_short"], show_sign=True) + '%</td>' +
                    '<td class="' + short_change_cls + '">' + m["mom_short_change_text"] + '</td>' +
                    '<td class="' + daily_cls + '">' + format_metric_value(m["daily_change"], show_sign=True) + '%</td>' +
                    '<td>' + amount_text + '</td>' +
                    '<td class="' + amount_change_cls + '">' + amount_change_text + '</td>' +
                    '<td>' + (turnover_text if turnover_text == '-' else turnover_text + '%') + '</td>' +
                    '<td class="' + turnover_change_cls + '">' + turnover_change_text + '</td>' +
                    '<td class="' + flow_cls + '">' + flow_text + '</td>' +
                    '<td>' + (vol_text if vol_text == '-' else vol_text + '%') + '</td>' +
                    '<td class="' + rec_cls + '">' + rec_text + '</td></tr>')
        return "\n".join(out)

    if signal_type == 'morning':
        label = "上午风控"
        sublabel = "11:30 盘中 - 下午操作窗口 13:00-15:00"
        tip = "&#9888; T+1 保护:上午只建议卖出/减仓,不建议买入(下午买入后当日无法止损)"
        tag_color = "tag-morning"
    else:
        label = "收盘决策"
        sublabel = "15:00 收盘 - 次日开盘操作或尾盘固定价"
        tip = "&#128161; T+1 提示:今日买入的标的,需待下一个交易日才能卖出;今日卖出资金当日可用"
        tag_color = "tag-close"

    # 市场概览数据
    non_broad = [(n, m) for n, m in metrics.items() if m.get('sector') != '宽基指数']
    up_today = sum(1 for _, m in non_broad if m.get('daily_change', 0) > 0)
    down_today = len(non_broad) - up_today
    sectors_pos = len(set(m['sector'] for _, m in non_broad if m.get('mom_long', 0) > 0))
    sectors_total = len(set(m['sector'] for _, m in non_broad))
    hs300_mom = metrics.get('沪深300ETF', {}).get('mom_long', 0)

    market_summary = (
        f'<b>沪深300 40日动量</b> <span class="{"pos" if hs300_mom > 0 else "neg"}">{hs300_mom:+.1f}%</span>'
        f' &nbsp;|&nbsp; <b>板块动量</b> {sectors_pos}/{sectors_total} 板块为正'
        f' &nbsp;|&nbsp; <b>今日涨跌</b> <span class="pos">{up_today}涨</span> / <span class="neg">{down_today}跌</span>'
    )

    buy_picks_html = buy_picks()
    risk_warnings_html = risk_warnings()
    monitor_rows_html = monitor_rows()

    html = """<!DOCTYPE html>
<html lang="zh-CN"><head><meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>ETF板块轮动 - """ + label + """</title>
<style>
*{margin:0;padding:0;box-sizing:border-box}
body{font-family:-apple-system,"Segoe UI","PingFang SC","Microsoft YaHei",sans-serif;
background:linear-gradient(135deg,#0f2027,#203a43,#2c5364);min-height:100vh;padding:20px;color:#222}
.wrap{max-width:960px;margin:0 auto}
.head{text-align:center;color:#fff;margin-bottom:22px}
.head h1{font-size:24px;letter-spacing:1px}
.head p{opacity:.8;font-size:13px;margin-top:6px}
.card{background:#fff;border-radius:16px;padding:22px;margin-bottom:18px;
box-shadow:0 10px 30px rgba(0,0,0,.25)}
.card h2{font-size:17px;color:#2c5364;border-left:4px solid #2c5364;padding-left:10px;margin-bottom:14px}
table{width:100%;border-collapse:separate;border-spacing:0;font-size:13px;min-width:960px}
th,td{padding:10px 8px;text-align:left;border-bottom:1px solid #e8ecf0;vertical-align:top}
th{background:#f0f3f7;color:#555;font-weight:600;position:sticky;top:0;z-index:2}
th:first-child{left:0;z-index:3}
td.nm{min-width:140px;font-weight:600;position:sticky;left:0;z-index:1;background:#fff}
tr:nth-child(even) td.nm{background:#fafbfc}
tr:hover td.nm{background:#eef2ff}
tr:hover td{background:#eef2ff}
/* 板块分隔行 hover 保持原色 */
tr[style*="f0f7ff"]:hover td{background:#f0f7ff !important}
.table-wrap{overflow:auto;max-height:70vh;border-radius:12px;border:1px solid #e8ecf0}
.pos{color:#d7263d;font-weight:600} .neg{color:#1a936f;font-weight:600}
.warn-text{color:#e67e22;font-weight:600} .hold{color:#666}
td.sc{font-weight:700;color:#2c5364}
.badge-sell,.badge-reduce{display:inline-block;background:#d7263d;color:#fff;font-size:11px;
padding:2px 8px;border-radius:4px;margin-right:6px}
.badge-buy,.badge-add{display:inline-block;background:#27ae60;color:#fff;font-size:11px;
padding:2px 8px;border-radius:4px;margin-right:6px}
.badge-hold,.badge-wait{display:inline-block;background:#95a5a6;color:#fff;font-size:11px;
padding:2px 8px;border-radius:4px;margin-right:6px}
.signal{color:#fff;border-radius:14px;padding:24px;text-align:center;margin-bottom:18px}
.signal.signal-danger{background:linear-gradient(135deg,#c0392b,#e74c3c);box-shadow:0 8px 24px rgba(192,57,43,.4)}
.signal.signal-warn{background:linear-gradient(135deg,#d35400,#e67e22);box-shadow:0 8px 24px rgba(211,84,0,.4)}
.signal.signal-ok{background:linear-gradient(135deg,#1e8449,#27ae60);box-shadow:0 8px 24px rgba(30,132,73,.4)}
.signal .lab{font-size:13px;opacity:.9}
.signal .val{font-size:28px;font-weight:700;margin-top:8px;color:#fff;text-shadow:0 1px 4px rgba(0,0,0,.25)}
.signal .sub{font-size:14px;margin-top:6px;opacity:.9}
.grid{display:grid;grid-template-columns:1fr 1fr;gap:14px;margin-bottom:18px}
.mini{background:#fff;border-radius:14px;padding:18px;box-shadow:0 8px 24px rgba(0,0,0,.2)}
.mini .t{font-size:13px;color:#888} .mini .v{font-size:20px;font-weight:700;color:#2c5364;margin-top:6px}
.hold-item{display:flex;justify-content:space-between;padding:8px 0;border-bottom:1px solid #f5f5f5;font-size:14px}
.info-box{background:#e8f6ff;border-left:4px solid #3498db;padding:14px 18px;border-radius:8px;margin-bottom:18px;color:#1a5276;font-size:14px;line-height:1.6}
.note{background:#fff;border-radius:14px;padding:18px;font-size:13px;color:#555;line-height:1.9}
.note b{color:#2c5364}
.foot{text-align:center;color:#fff;opacity:.7;font-size:12px;margin-top:14px}
.tag{display:inline-block;padding:2px 10px;border-radius:4px;font-size:12px;margin-left:8px;vertical-align:middle}
.tag-morning{background:#f39c12;color:#fff} .tag-close{background:#27ae60;color:#fff}
.sector-tag{display:inline-block;background:#e8f6ff;color:#1a5276;font-size:11px;padding:1px 6px;border-radius:3px;margin-left:6px}
.action-list{display:flex;flex-direction:column;gap:10px}
.action-card{background:#fafbfc;border-radius:10px;padding:14px 16px;border-left:4px solid #ccc;font-size:13px}
.action-card.neg{border-left-color:#d7263d;background:#fff5f5}
.action-card.warn-text{border-left-color:#e67e22;background:#fffaf0}
.action-card.pos{border-left-color:#27ae60;background:#f0fff4}
.action-card.hold{border-left-color:#95a5a6;background:#f8f9fa}
.action-row{display:flex;align-items:flex-start;padding:5px 0;border-bottom:1px solid #eee}
.action-row:last-child{border-bottom:none}
.action-label{min-width:56px;color:#888;font-size:12px;font-weight:600;flex-shrink:0;margin-top:1px}
.action-value{flex:1;color:#333;line-height:1.6;word-break:break-all}
.action-card.neg .action-value{color:#d7263d}
.action-card.warn-text .action-value{color:#e67e22}
.action-card.pos .action-value{color:#27ae60}
@media(max-width:700px){.grid{grid-template-columns:1fr} table{font-size:12px} .nm{min-width:100px}}
</style></head><body><div class="wrap">
<div class="head"><h1>&#128202; ETF 板块轮动 - """ + label + """ <span class='tag """ + tag_color + """'>""" + label + """</span></h1>
<p>""" + sublabel + """ - 概念板块分组 &#183; 仓位管理 &#183; T+1安全</p></div>

<div class="signal signal-""" + market_cls + """"><div class="lab">当前建议总仓位</div>
<div class='val'>""" + position_text + """</div>
<div class="sub">""" + position_reason + """ - 数据截至 """ + asof + """</div></div>

<div class="info-box">
<b>&#128203; 市场概览</b><br>
""" + market_summary + """
</div>

<div class="card"><h2>&#127919; 潜力推荐</h2>
<div class="action-list">
""" + buy_picks_html + """
</div></div>

<div class="card"><h2>&#9888; 风险警示</h2>
<div class="action-list">
""" + risk_warnings_html + """
</div></div>

<div class="card"><h2>&#128202; 板块与标的监控(共 """ + str(len(metrics)) + """ 只,分""" + str(len(set(m['sector'] for m in metrics.values()))) + """个板块)</h2>
<div class="table-wrap"><table><thead><tr><th class="nm">ETF</th><th>40日动量(%)</th><th>较昨日</th><th>5日动量(%)</th><th>5日较昨日</th><th>当日涨跌(%)</th><th>成交额(亿)</th><th>成交额较昨日</th><th>换手率(%)</th><th>换手率较昨日</th><th>资金流评分</th><th>波动率(%)</th><th>推荐值</th></tr></thead>
<tbody>""" + monitor_rows_html + """</tbody></table></div></div>

<div class="note"><b>策略逻辑</b><br>
<b>仓位</b>:沪深300 40日动量&gt;2%&#8594;满仓10成;0~2%&#8594;半仓5成;&#8804;0%时看板块宽度(&#8805;50%板块动量正→半仓5成,&#8805;30%→3成轻仓,否则空仓).<br>
<b>选股</b>:40日风险调整动量排名前""" + str(TOP_K) + """(不设绝对值门槛) + 5日动量&gt;-3% + 当日跌幅&gt;""" + str(MAX_DAILY_DROP) + """% + 当日涨幅&lt;""" + str(MAX_DAILY_RISE) + """% + 资金流评分优先.<br>
<b>上午</b>:只风控(卖出/减仓),不买入(T+1保护).<br><br>
<b>T+1 制度说明</b><br>
A股 ETF 实行T+1交易:今日买入的份额,需待下一个交易日才能卖出;今日卖出资金当日可用(可继续买入其他ETF),但不可取现至银行卡.</div>

<div class="foot">更新于 """ + update_time + """ - 数据源: """ + data_source_label + """ - 策略状态自动维护</div>
</div></body></html>"""
    return html


def detect_market_regime(metrics):
    """市场状态检测器: 综合HS300动量+板块宽度+动量改善比"""
    hs300 = metrics.get('沪深300ETF', {})
    hs300_mom = hs300.get('mom_long', -999)
    hs300_vol = hs300.get('vol', 20.0)
    if hs300_vol > 100: hs300_vol = 20.0

    sectors = {}
    for m in metrics.values():
        s = m.get('sector', '')
        if s == '宽基指数': continue
        sectors.setdefault(s, []).append(m['mom_long'] > 0)
    pos_sectors = sum(1 for etfs in sectors.values() if sum(etfs) > len(etfs) / 2)
    breadth = pos_sectors / max(len(sectors), 1)

    improving = sum(1 for m in metrics.values()
                    if m.get('mom_long_change', 0) > 0.3
                    and m.get('sector', '') != '宽基指数')
    worsening = sum(1 for m in metrics.values()
                   if m.get('mom_long_change', 0) < -0.3
                   and m.get('sector', '') != '宽基指数')
    improve_ratio = improving / max(improving + worsening, 1)

    up_today = sum(1 for m in metrics.values()
                   if m.get('daily_change', 0) > 0
                   and m.get('sector', '') != '宽基指数')
    total = max(sum(1 for m in metrics.values() if m.get('sector', '') != '宽基指数'), 1)

    signals = {'hs300_mom': hs300_mom, 'breadth': breadth,
               'improve_ratio': improve_ratio, 'up_ratio': up_today / total,
               'volatility': hs300_vol}

    if hs300_mom > 2 and breadth >= 0.5:
        return 'bull', 0.9, signals
    if hs300_mom > 0 and breadth >= 0.3:
        return 'sideways', 0.7, signals
    if hs300_mom <= 0 and breadth >= 0.3 and improve_ratio > 0.55:
        return 'structural', 0.6, signals
    if hs300_mom <= 0 and improve_ratio > 0.6 and up_today / total > 0.4:
        return 'recovery', 0.5, signals
    return 'bear', 0.8, signals


def meta_strategy(metrics, signal_type='close'):
    """元策略: 根据市场状态自动选择最优策略变体

    策略映射:
      bull      -> V2 HS300择时  (满仓进攻)
      sideways  -> V8 宽度+V2覆盖+分散 (半仓精选)
      structural-> V8 宽度+V2覆盖+分散 (结构行情精选)
      recovery  -> V10 HS300+V3+分散 (智能反弹捕捉)
      bear      -> V3 空仓避险
    非熊市且quality>2.0标的>=3只 -> 额外加仓
    """
    regime, confidence, signals = detect_market_regime(metrics)
    hs300_mom = signals['hs300_mom']
    breadth = signals['breadth']
    improve_ratio = signals['improve_ratio']

    regime_map = {
        'bull':       (None, "🐂 牛市(V2): 顺势进攻"),
        'sideways':   (None, "📊 震荡(V8): 精选参与"),
        'structural': (None, "🏗️ 结构(V8): 精选强势板块"),
        'recovery':   (None, "🔄 反弹(V10): 试探参与"),
        'bear':       (None, "🐻 熊市(V3): 避险观望"),
    }

    # 基础仓位
    if regime == 'bull':
        ratio = 1.0 if hs300_mom > 2 else 0.5
    elif regime in ('sideways', 'structural'):
        ratio = 0.5
    elif regime == 'recovery':
        ratio = 0.3
    else:
        ratio = 0.0

    position_text = f"{int(ratio*10)}成"
    market_cls = {'bull': 'ok', 'sideways': 'warn', 'structural': 'warn',
                  'recovery': 'warn', 'bear': 'danger'}[regime]
    base_reason = regime_map[regime][1]

    # 超优标的增强
    super_q = sum(1 for m in metrics.values()
                  if m.get('quality_score', 0) > 2.0
                  and m.get('sector', '') != '宽基指数')

    if regime != 'bull' and super_q >= 3 and improve_ratio > 0.5 and ratio > 0:
        bonus = 0.1
        ratio = min(ratio + bonus, 0.7)
        position_text = f"{int(ratio*10)}成"
        base_reason += f" | ⚡{super_q}只超优增强+{int(bonus*10)}成"

    reason = (f"{base_reason} | HS300={hs300_mom:+.1f}% "
              f"宽度={breadth*100:.0f}% 改善={improve_ratio*100:.0f}%")

    print(f"\n  🧠 元策略: regime={regime}({confidence:.0%}) ratio={ratio}")
    print(f"  🧠 {reason}")

    return ratio, position_text, reason, market_cls, regime


def main():
    bj = timezone(timedelta(hours=8))
    now = datetime.now(bj)
    update_time = now.strftime('%Y-%m-%d %H:%M:%S')
    today_str = now.strftime('%Y-%m-%d')

    print(f"\n{'='*60}")
    print(f"[{update_time}] 策略启动")
    print(f"{'='*60}")

    price_daily, etf_info, extra_history, data_sources = fetch_daily_data_cached(ETF_POOL)
    if price_daily is None:
        print("日K线获取失败")
        return

    trade_date = str(price_daily.index[-1])
    days_behind = (now.date() - datetime.strptime(trade_date, '%Y-%m-%d').date()).days
    is_weekday = now.weekday() < 5  # 周一=0, 周日=6

    # ── 鲁棒信号判定: 以数据状态为准,而非时钟时间 ──
    # 盘中合成条件: 工作日 + 盘中时段 + 数据日期≠今日 + 间隔≤4天(覆盖周末和短假)
    morning = ((now.hour == 9 and now.minute >= 30) or now.hour == 10 or
               (now.hour == 11 and now.minute <= 30))
    afternoon = (now.hour == 13 or now.hour == 14)
    actual_trading = (morning or afternoon) and is_weekday

    intraday_trigger = False
    if trade_date != today_str and actual_trading and days_behind <= 4:
        signal_type = 'morning' if morning else 'afternoon'
        intraday_trigger = True
        label = "上午风控" if morning else "下午盘中"
        print(f"时段: {label} (盘中合成今日数据, 最新交易日={trade_date})")
    else:
        signal_type = 'close'
        if days_behind > 1:
            print(f"时段: 收盘决策 (休市/假日,最新交易日={trade_date},距今日{days_behind}天)")
        elif trade_date == today_str:
            print(f"时段: 收盘决策 (日K已含今日数据)")
        else:
            print(f"时段: 收盘决策 (非盘中,最新交易日={trade_date})")

    print(f"信号类型: {signal_type} | 盘中触发: {intraday_trigger}")

    # 快照合并前状态
    extra_before = {}
    for name, df in extra_history.items():
        extra_before[name] = {
            'latest_date': str(df.index[-1]),
            'has_today': today_str in df.index,
            'amount_latest': float(df['amount'].iloc[-1]) if 'amount' in df.columns and len(df) > 0 else None,
        }

    if intraday_trigger:
        print(f"\n日K最新 {trade_date}，盘中时段，拉取实时行情合成今日数据...")
        intraday_snapshot = fetch_intraday_snapshot(ETF_POOL)
        # 假日防御: 若实时行情中成交量全为0或极低,说明今日休市
        valid_count = sum(1 for s in intraday_snapshot.values()
                          if s.get('volume', 0) > 0 and s.get('close', 0) > 0.01)
        if valid_count < len(ETF_POOL) * 0.3:
            print(f"  假日检测: 有效快照仅{valid_count}/{len(ETF_POOL)}只(阈值30%),判定休市→退回收盘模式")
            intraday_trigger = False
            signal_type = 'close'
            price = price_daily
        else:
            price, extra_history = merge_intraday_price(price_daily, intraday_snapshot, etf_info, extra_history)
    else:
        if trade_date != today_str:
            print(f"\n日K最新 {trade_date}（非盘中或假日），直接使用")
        else:
            print(f"\n日K已包含今天数据，直接使用")
        price = price_daily

    session_progress = calc_session_progress(now)
    if session_progress < 1.0:
        print(f"盘中交易进度: {session_progress*100:.0f}%")
    if str(price.index[-1]) != today_str:
        print(f"数据偏移检测: 最新日期={price.index[-1]} ≠ 今日={today_str}, 动量窗口已自动对齐")
    metrics = calc_metrics(price, etf_info, ETF_SECTOR, extra_history, session_progress,
                           today_str, intraday_trigger)
    if not metrics:
        print("指标计算失败")
        return

    # ── 诊断：记录盘中数据管道状态，用于验证时间轴修复 ──
    diagnostic = {
        'run_time': update_time,
        'signal_type': signal_type,
        'data_sources': sorted(data_sources),
        'intraday_triggered': intraday_trigger,
        'price_latest_date': str(price.index[-1]),
        'price_has_today': str(price.index[-1]) == today_str,
        'extra_before_merge': extra_before,
        'extra_after_merge': {},
        'fix_verdict': 'PASS' if str(price.index[-1]) == today_str else 'PENDING',
    }
    for name in sorted(extra_history.keys())[:8]:
        df = extra_history[name]
        after_date = str(df.index[-1])
        before_date = extra_before.get(name, {}).get('latest_date', '?')
        diagnostic['extra_after_merge'][name] = {
            'latest_date': after_date,
            'has_today': today_str in df.index,
            'date_advanced': after_date != before_date,
        }
    with open('diagnostic.json', 'w', encoding='utf-8') as f:
        json.dump(diagnostic, f, ensure_ascii=False, indent=2)

    asof = price.index[-1]
    print(f"\n统一数据基准日期: {asof}")

    # ── 元策略: 根据市场状态自动选择最优策略 ──
    position_ratio, position_text, position_reason, market_cls, regime = meta_strategy(
        metrics, signal_type)

    adaptive = calc_adaptive_params(metrics)
    print(f"自适应: vol_factor={adaptive['vol_factor']} stop_loss={adaptive['stop_loss']}% "
          f"max_rise={adaptive['max_daily_rise']}% top_k={adaptive['top_k']}")

    target, _ = build_target(metrics, position_ratio, adaptive)
    actions = generate_recommendations(target, position_ratio, signal_type, position_reason)
    for a in actions:
        print(f"  {a['type']}: {a['msg']}")

    # 统一数据源名称显示
    source_names = {'eastmoney': '东方财富', 'akshare': 'akshare', 'tencent': '腾讯财经', 'sina': '新浪'}
    data_source_label = ' + '.join(source_names.get(s, s) for s in sorted(data_sources)) if data_sources else '未知'
    html = build_html(target, position_text, position_reason,
                      market_cls, asof, update_time, signal_type, metrics, data_source_label)
    with open('index.html', 'w', encoding='utf-8') as f:
        f.write(html)

    data = {
        'update_time': update_time, 'asof': asof, 'signal_type': signal_type,
        'data_source': data_source_label,
        'market': {'position_text': position_text, 'reason': position_reason, 'ratio': position_ratio},
        'actions': actions, 'metrics': metrics
    }
    with open('data.json', 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

    print(f"\n已生成 index.html, data.json")
    print(f"{'='*60}\n")


def build_target_v5(metrics, position_ratio=0):
    """V5 单ETF精选: 价格过滤 + 质量门槛 + 选最优1只

    筛选条件:
      - 价格 ¥0.5-3.0 (排除低价/高价ETF)
      - quality_score > 0.5
      - 40日动量 > 0
      - 排除宽基指数
    选股: quality_score 最高的一只
    """
    candidates = []
    for name, m in metrics.items():
        price = m.get('latest', 0)
        quality = m.get('quality_score', 0)
        mom = m.get('mom_long', -999)

        if price < 0.5 or price > 3.0:
            continue
        if quality <= 0.5:
            continue
        if mom <= 0:
            continue
        if m.get('sector', '') == '宽基指数':
            continue

        candidates.append((name, m))

    if not candidates:
        return [], position_ratio

    # 按 quality_score 降序, 选最优 1 只
    candidates.sort(key=lambda x: x[1].get('quality_score', 0), reverse=True)
    best_name, best_m = candidates[0]

    w = round(position_ratio, 2)
    return [{
        'name': best_name,
        'code': best_m['code'],
        'weight': w,
        'mom_long': best_m['mom_long'],
        'score': best_m['score'],
        'sector': best_m.get('sector', ''),
        'quality_score': best_m.get('quality_score', 0),
    }], position_ratio


def market_timing_v5(metrics):
    """V5 择时: 比 HS300 标准择时更宽松, 单 ETF 策略可适度参与弱势市场

    逻辑:
      - HS300 40日动量 > 2    → 满仓 (1.0)
      - HS300 40日动量 > 0    → 7成
      - HS300 40日动量 > -2   → 半仓 (弱市中单ETF集中兵力)
      - HS300 40日动量 ≤ -2   → 空仓
    额外: 若质优标的 quality_score > 2.0 且 ≥ 2 只, 即使空仓也给 3 成
    """
    hs300 = metrics.get('沪深300ETF', {})
    hs300_mom = hs300.get('mom_long', -999)

    if hs300_mom > 2:
        return 1.0
    elif hs300_mom > 0:
        return 0.7
    elif hs300_mom > -2:
        # 弱市半仓: V5 单 ETF 集中兵力, 止损更快
        return 0.5

    # ── 极端弱势: 检查是否有逆势高质量标的 ──
    super_quality = sum(1 for m in metrics.values()
                        if m.get('quality_score', 0) > 2.0
                        and m.get('sector', '') != '宽基指数'
                        and m.get('mom_long', 0) > 0)
    if super_quality >= 2:
        return 0.3  # 轻仓试探

    return 0.0


if __name__ == '__main__':
    main()
