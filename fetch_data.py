#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ETF 数据抓取脚本 — GitHub Actions 专用
====================================
功能:
  1. 拉取 39 只 ETF 的日线 OHLCV + 成交额 + 换手率
  2. 拉取实时行情（新浪 + 东方财富）
  3. 输出 JSON 到 data/ 目录
  4. 三级回退: AKShare → Eastmoney → Tencent

输出文件:
  data/history.json    — 历史日线矩阵 (close + full OHLCV)
  data/realtime.json   — 实时行情快照
  data/meta.json       — 拉取元信息 (时间、成功数、来源)

用法:
  python fetch_data.py              # 全部拉取
  python fetch_data.py --realtime   # 仅实时行情
  python fetch_data.py --history    # 仅历史日线
"""

import json
import os
import sys
import time
import argparse
from datetime import datetime, timezone, timedelta
from concurrent.futures import ThreadPoolExecutor, as_completed

import pandas as pd
import numpy as np

# ── 导入策略模块中的 ETF 池和 API 函数 ──
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from strategy import (
    SECTOR_ETF_POOL, ETF_POOL, ETF_SECTOR,
    get_etf_history_akshare, get_etf_history_eastmoney, get_etf_sina,
    get_etf_extra_sina, get_etf_realtime_eastmoney,
    DATA_LEN, MOM_LONG,
)

BJ = timezone(timedelta(hours=8))
DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data')
os.makedirs(DATA_DIR, exist_ok=True)


# ═══════════════════════════════════════════════
# 数据拉取
# ═══════════════════════════════════════════════

def fetch_history_single(code, sina, name, datalen=DATA_LEN):
    """单只 ETF 历史日线 — 三级回退 (按数据完整性排序)

    优先级: Eastmoney(8字段) → AKShare(8字段) → Tencent(6字段)
    Eastmoney 和 AKShare 都有完整的 OHLCV+amount+turnover,
    Tencent 仅 OHLCV, amount/turnover 需后续估算。
    """
    result = {'code': code, 'sina': sina, 'name': name, 'success': False, 'source': None, 'rows': 0}

    # 1. Eastmoney K-line (字段最全: OHLCV + amount + turnover)
    df = get_etf_history_eastmoney(code, datalen=datalen)
    if df is not None and len(df) >= MOM_LONG + 5:
        result.update({'success': True, 'source': 'eastmoney', 'rows': len(df), 'data': df})
        return result

    # 2. AKShare (同样完整, 但更重的包装层)
    df = get_etf_history_akshare(code, datalen=datalen)
    if df is not None and len(df) >= MOM_LONG + 5:
        result.update({'success': True, 'source': 'akshare', 'rows': len(df), 'data': df})
        return result

    # 3. Tencent K-line (缺 amount/turnover, 仅 OHLCV)
    df = get_etf_sina(sina, scale=240, datalen=datalen)
    if df is not None and len(df) >= MOM_LONG + 5:
        result.update({'success': True, 'source': 'tencent', 'rows': len(df), 'data': df})
        return result

    result['rows'] = len(df) if df is not None else 0
    return result


def _supplement_amount_turnover(result, datalen=DATA_LEN):
    """对 Tencent 源 ETF 补充 amount/turnover

    尝试从 Eastmoney → AKShare 获取完整数据，按日期合并。
    """
    if not result['success'] or result.get('source') != 'tencent':
        return result

    code = result['code']
    tencent_df = result['data']

    # 尝试 Eastmoney
    df_em = get_etf_history_eastmoney(code, datalen=datalen)
    if df_em is not None and len(df_em) >= MOM_LONG // 2:
        merged = _merge_supplement(tencent_df, df_em)
        if merged is not None:
            result['data'] = merged
            result['source'] = 'tencent+eastmoney'
            result['rows'] = len(merged)
            return result

    # 尝试 AKShare
    df_ak = get_etf_history_akshare(code, datalen=datalen)
    if df_ak is not None and len(df_ak) >= MOM_LONG // 2:
        merged = _merge_supplement(tencent_df, df_ak)
        if merged is not None:
            result['data'] = merged
            result['source'] = 'tencent+akshare'
            result['rows'] = len(merged)
            return result

    return result


def _merge_supplement(main_df, supp_df):
    """将 supp_df 中的 amount/turnover 按 day 列合并到 main_df"""
    if supp_df is None or len(supp_df) == 0:
        return None

    supp = supp_df.set_index('day') if 'day' in supp_df.columns else supp_df
    main = main_df.set_index('day') if 'day' in main_df.columns else main_df.copy()

    need_cols = []
    for col in ['amount', 'turnover']:
        if col not in main.columns or main[col].isna().all():
            if col in supp.columns:
                need_cols.append(col)

    if not need_cols:
        return None

    for col in need_cols:
        if col not in main.columns:
            main[col] = float('nan')

    common = main.index.intersection(supp.index)
    if len(common) == 0:
        return None

    filled = 0
    for col in need_cols:
        mask = main[col].isna()
        fill_dates = common.intersection(main.index[mask])
        if len(fill_dates) > 0:
            main.loc[fill_dates, col] = supp.loc[fill_dates, col]
            filled += len(fill_dates)

    if filled > 0:
        main = main.reset_index().rename(columns={'index': 'day'})
        return main
    return None


def fetch_all_history(pool, max_workers=5, datalen=DATA_LEN):
    """并行拉取全部 ETF 历史日线"""
    total = len(pool)
    results = []
    success = 0

    print(f"\n{'='*55}")
    print(f"  历史日线拉取 ({total} 只 ETF, {max_workers} 线程)")
    print(f"{'='*55}")

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(fetch_history_single, code, sina, name, datalen): name
            for code, (sina, name) in pool.items()
        }
        for i, future in enumerate(as_completed(futures), 1):
            r = future.result()
            results.append(r)
            status = '✅' if r['success'] else '❌'
            extra = f" 来源={r['source']}" if r['source'] else ''
            print(f"  [{i:02d}/{total}] {status} {r['name']}({r['code']}) {r['rows']}行{extra}")
            if r['success']:
                success += 1

    print(f"\n  完成: {success}/{total} 成功")

    # ── 二次补充: 对 Tencent 源 ETF 补充 amount/turnover ──
    tencent_results = [r for r in results if r['success'] and r.get('source') == 'tencent']
    if tencent_results:
        print(f"\n  二次补充: {len(tencent_results)}只ETF缺成交额/换手率...")
        supplemented = 0
        for r in tencent_results:
            updated = _supplement_amount_turnover(r, datalen)
            if updated.get('source') != 'tencent':
                supplemented += 1
                print(f"    [补充] {r['name']}({r['code']}): → {updated['source']} {updated['rows']}行")
            else:
                print(f"    [警告] {r['name']}({r['code']}): 补充失败, 将估算")
        print(f"  二次补充完成: {supplemented}/{len(tencent_results)} 成功")

    return results, success


def fetch_all_realtime(pool, max_workers=5):
    """并行拉取实时行情（新浪 + 东方财富）"""
    total = len(pool)
    results = []

    print(f"\n{'='*55}")
    print(f"  实时行情拉取 ({total} 只 ETF, {max_workers} 线程)")
    print(f"{'='*55}")

    def _fetch_one(item):
        code, (sina, name) = item
        r = {'code': code, 'sina': sina, 'name': name, 'sina_ok': False, 'em_ok': False, 'data': {}}
        # 新浪实时
        sn = get_etf_extra_sina(sina)
        if sn:
            r['sina_ok'] = True
            r['data'].update(sn)
        # 东方财富实时（补充换手率）
        em = get_etf_realtime_eastmoney(code)
        if em:
            r['em_ok'] = True
            r['data'].update({k: v for k, v in em.items() if k not in r['data']})
        time.sleep(0.3)  # 避免限频
        return r

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(_fetch_one, item): item[1][1] for item in pool.items()}
        for i, future in enumerate(as_completed(futures), 1):
            r = future.result()
            results.append(r)
            sina = '✅' if r['sina_ok'] else '❌'
            em = '✅' if r['em_ok'] else '❌'
            price = r['data'].get('close', '?')
            print(f"  [{i:02d}/{total}] Sina={sina} EM={em} {r['name']} ¥{price}")

    return results


# ═══════════════════════════════════════════════
# JSON 序列化
# ═══════════════════════════════════════════════

class NpEncoder(json.JSONEncoder):
    """处理 numpy 类型的 JSON 编码"""
    def default(self, obj):
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            if np.isnan(obj) or np.isinf(obj):
                return None
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, pd.Timestamp):
            return str(obj)
        return super().default(obj)


def build_close_matrix(history_results):
    """从拉取结果构建 close 价格矩阵 (已对齐 + 填充)"""
    all_close = {}
    etf_info = {}

    for r in history_results:
        if not r['success'] or 'data' not in r:
            continue
        df = r['data']
        all_close[r['name']] = df.set_index('day')['close']
        etf_info[r['name']] = r['code']

    if not all_close:
        return None, {}

    price = pd.DataFrame(all_close).sort_index()
    price = price.ffill().dropna(how='all', axis=1)

    # 过滤数据量不足的
    valid = [c for c in price.columns if price[c].notna().sum() >= MOM_LONG + 2]
    price = price[valid]

    return price, etf_info


def build_extra_matrix(history_results, etf_info):
    """构建 extra 数据矩阵 (open/high/low/volume/amount/turnover)"""
    extra_history = {}
    for r in history_results:
        if not r['success'] or 'data' not in r:
            continue
        name = r['name']
        if name not in etf_info:
            continue
        df = r['data']
        available = [c for c in ['open', 'high', 'low', 'volume', 'amount', 'turnover']
                     if c in df.columns and df[c].notna().any()]
        if not available:
            continue
        extra_df = df[['day'] + available].copy()
        extra_df = extra_df.set_index('day')
        extra_history[name] = extra_df

    # 腾讯源补充成交额估算
    for name, df in extra_history.items():
        price_series = etf_info.get(name)  # just check existence
        if 'amount' not in df.columns or df['amount'].isna().all():
            if 'volume' in df.columns and 'high' in df.columns and 'low' in df.columns:
                df['amount'] = df['volume'] * (df['high'] + df['low'] + df.index.map(
                    lambda _: float(df.loc[_, '__placeholder__']) if '__placeholder__' in df.columns else 0
                )) / 3
                # Simpler: use (H+L)/2 approximation without close
                pass  # handled below via close matrix

    return extra_history


def save_history_json(history_results, price, etf_info, extra_history):
    """保存历史数据为 JSON"""
    # ── history.json: close 矩阵 + 元信息 ──
    history_out = {
        'generated_at': datetime.now(BJ).strftime('%Y-%m-%d %H:%M:%S'),
        'data_start': str(price.index[0]) if price is not None else None,
        'data_end': str(price.index[-1]) if price is not None else None,
        'etf_count': len(price.columns) if price is not None else 0,
        'dates': price.index.tolist() if price is not None else [],
        'etfs': {},
        'sources': {},
    }

    if price is not None:
        for name in price.columns:
            etf_data = {
                'code': etf_info.get(name, ''),
                'sector': ETF_SECTOR.get(name, '其他'),
                'close': [round(float(v), 3) if not pd.isna(v) else None
                         for v in price[name].values],
            }
            # 附加 extra 数据 (open/high/low/volume/amount/turnover)
            if name in extra_history:
                edf = extra_history[name]
                for col in ['open', 'high', 'low', 'volume', 'amount', 'turnover']:
                    if col in edf.columns:
                        vals = []
                        for d in price.index:
                            if d in edf.index:
                                v = edf.loc[d, col]
                                vals.append(round(float(v), 2) if not pd.isna(v) else None)
                            else:
                                vals.append(None)
                        etf_data[col] = vals
            history_out['etfs'][name] = etf_data

    # 数据来源
    for r in history_results:
        if r['success']:
            history_out['sources'][r['name']] = r['source']

    path = os.path.join(DATA_DIR, 'history.json')
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(history_out, f, ensure_ascii=False, cls=NpEncoder)
    print(f"\n  💾 已保存: {path} ({os.path.getsize(path)/1024:.0f} KB)")


def save_realtime_json(realtime_results):
    """保存实时行情为 JSON"""
    out = {
        'generated_at': datetime.now(BJ).strftime('%Y-%m-%d %H:%M:%S'),
        'etfs': [],
    }
    for r in realtime_results:
        out['etfs'].append({
            'code': r['code'],
            'name': r['name'],
            'sector': ETF_SECTOR.get(r['name'], '其他'),
            'sina_ok': r['sina_ok'],
            'em_ok': r['em_ok'],
            'price': r['data'].get('close'),
            'volume': r['data'].get('volume'),
            'amount': r['data'].get('amount'),
            'turnover': r['data'].get('turnover'),
        })

    path = os.path.join(DATA_DIR, 'realtime.json')
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(out, f, ensure_ascii=False, cls=NpEncoder)
    print(f"  💾 已保存: {path} ({os.path.getsize(path)/1024:.0f} KB)")


def save_meta_json(history_count, realtime_count, total, sources, elapsed):
    """保存拉取元信息"""
    meta = {
        'generated_at': datetime.now(BJ).strftime('%Y-%m-%d %H:%M:%S'),
        'elapsed_seconds': round(elapsed, 1),
        'history': {
            'success': history_count,
            'total': total,
            'rate': f'{history_count}/{total}',
            'sources': sources,
        },
        'realtime': {
            'fetched': bool(realtime_count),
            'count': realtime_count,
        },
    }
    path = os.path.join(DATA_DIR, 'meta.json')
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(meta, f, ensure_ascii=False)
    print(f"  💾 已保存: {path}")


# ═══════════════════════════════════════════════
# 主入口
# ═══════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description='ETF 数据抓取')
    parser.add_argument('--history', action='store_true', default=False, help='仅拉取历史日线')
    parser.add_argument('--realtime', action='store_true', default=False, help='仅拉取实时行情')
    parser.add_argument('--workers', type=int, default=3, help='并行线程数 (默认3, GitHub Actions 限制)')
    parser.add_argument('--datalen', type=int, default=DATA_LEN, help=f'历史数据长度 (默认{DATA_LEN})')
    args = parser.parse_args()

    # 默认: 全部拉取
    do_history = not args.realtime or args.history
    do_realtime = not args.history or args.realtime

    t0 = time.time()
    print(f"[{datetime.now(BJ):%Y-%m-%d %H:%M:%S}] ETF 数据抓取启动")
    print(f"  ETF 池: {len(ETF_POOL)} 只, {len(SECTOR_ETF_POOL)} 个板块")
    print(f"  线程数: {args.workers}")

    history_count = 0
    realtime_count = 0
    sources = {}

    # ── 历史日线 ──
    if do_history:
        results, history_count = fetch_all_history(ETF_POOL, max_workers=args.workers, datalen=args.datalen)

        # 统计来源
        for r in results:
            if r['success'] and r['source']:
                sources[r['source']] = sources.get(r['source'], 0) + 1

        # 构建矩阵并保存
        if history_count > 0:
            price, etf_info = build_close_matrix(results)
            extra_history = build_extra_matrix(results, etf_info)

            # 用 close 矩阵修正 extra_history 中的 amount 估算
            if price is not None:
                for name, df in extra_history.items():
                    if 'amount' not in df.columns or df['amount'].isna().all():
                        if 'volume' in df.columns and 'high' in df.columns and 'low' in df.columns:
                            close_s = price[name] if name in price.columns else None
                            if close_s is not None:
                                for d in df.index:
                                    if d in close_s.index:
                                        c = close_s.loc[d]
                                        h = df.loc[d, 'high'] if not pd.isna(df.loc[d, 'high']) else c
                                        l = df.loc[d, 'low'] if not pd.isna(df.loc[d, 'low']) else c
                                        df.loc[d, 'amount'] = df.loc[d, 'volume'] * (h + l + c) / 3

            save_history_json(results, price, etf_info, extra_history)
        else:
            print("\n  ⚠️ 历史数据拉取完全失败，跳过 history.json")
            # 仍然创建空文件标记失败
            path = os.path.join(DATA_DIR, 'history.json')
            with open(path, 'w') as f:
                json.dump({'error': 'all_failed', 'generated_at': datetime.now(BJ).isoformat()}, f)

    # ── 实时行情 ──
    if do_realtime:
        realtime_results = fetch_all_realtime(ETF_POOL, max_workers=args.workers)
        realtime_count = sum(1 for r in realtime_results if r['sina_ok'] or r['em_ok'])
        save_realtime_json(realtime_results)

    # ── 元信息 ──
    elapsed = time.time() - t0
    save_meta_json(history_count, realtime_count, len(ETF_POOL), sources, elapsed)

    print(f"\n{'='*55}")
    print(f"  抓取完成! 耗时 {elapsed:.0f}s")
    if do_history:
        print(f"  历史: {history_count}/{len(ETF_POOL)} 成功  来源: {sources}")
    if do_realtime:
        print(f"  实时: {realtime_count}/{len(ETF_POOL)} 成功")
    print(f"{'='*55}")

    # 返回值: 历史拉取至少 50% 成功才算通过
    if do_history and history_count < len(ETF_POOL) * 0.5:
        print("\n  ❌ 历史数据成功率不足 50%, 退出码 1")
        sys.exit(1)

    print("\n  ✅ 完成")


if __name__ == '__main__':
    main()
