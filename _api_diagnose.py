#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
API 独立诊断脚本 — 逐个测试每个数据源
=====================================
对 39 只 ETF 分别调用 Eastmoney / AKShare / Tencent / Sina 实时 / Eastmoney 实时，
记录每只 ETF 在每个 API 下的表现。

输出: data/api_diagnose.json — 结构化诊断报告

用途:
  - GitHub Actions 工作流中运行，了解各 API 在不同环境的表现
  - 本地可单独运行，帮助选择最优数据组合策略

用法:
  python _api_diagnose.py              # 全部测试
  python _api_diagnose.py --quick      # 仅测试前 5 只 (快速诊断)
  python _api_diagnose.py --api eastmoney  # 仅测试指定 API
"""

import os, sys, json, time, argparse
from datetime import datetime, timezone, timedelta

import pandas as pd
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import Config
from data_fetcher import (
    DataFetcher, _fetch_tencent, _fetch_akshare_daily,
    _fetch_akshare_60min, _fetch_sina_realtime,
)

# 映射到旧 API 名以保持兼容
cfg = Config()


def get_etf_history_eastmoney(code, datalen=None):
    """暂不可用: Eastmoney API 被限频""" 
    return None

get_etf_history_akshare = _fetch_akshare_daily
get_etf_sina = _fetch_tencent

def get_etf_extra_sina(sina_code):
    return _fetch_sina_realtime(sina_code)

def get_etf_realtime_eastmoney(code):
    return None  # 暂不可用

ETF_POOL = {c: (s, n, sec) for c, (s, n, sec) in cfg.get_etf_pool().items()}
DATA_LEN = cfg.DATA_LEN
MOM_LONG = cfg.MOM_LONG

BJ = timezone(timedelta(hours=8))
DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data')
REPORT_FILE = os.path.join(DATA_DIR, 'api_diagnose.json')
os.makedirs(DATA_DIR, exist_ok=True)

# ═══════════════════════════════════════════════
# API 测试定义
# ═══════════════════════════════════════════════

API_DEFS = {
    'eastmoney_hist': {
        'name': 'Eastmoney K-line (历史日线)',
        'call': lambda code, sina: get_etf_history_eastmoney(code),
        'key': 'eastmoney_hist',
        'type': 'history',
        'expected_fields': ['day', 'open', 'high', 'low', 'close', 'volume', 'amount', 'turnover'],
        'min_rows': MOM_LONG + 5,
    },
    'akshare_hist': {
        'name': 'AKShare (历史日线)',
        'call': lambda code, sina: get_etf_history_akshare(code),
        'key': 'akshare_hist',
        'type': 'history',
        'expected_fields': ['day', 'open', 'high', 'low', 'close', 'volume', 'amount', 'turnover'],
        'min_rows': MOM_LONG + 5,
    },
    'tencent_hist': {
        'name': 'Tencent K-line (历史日线)',
        'call': lambda code, sina: get_etf_sina(sina, scale=240, datalen=DATA_LEN),
        'key': 'tencent_hist',
        'type': 'history',
        'expected_fields': ['day', 'open', 'high', 'low', 'close', 'volume'],
        'min_rows': MOM_LONG + 5,
    },
    'sina_realtime': {
        'name': 'Sina 实时行情',
        'call': lambda code, sina: get_etf_extra_sina(sina),
        'key': 'sina_realtime',
        'type': 'realtime',
        'expected_fields': ['close', 'volume', 'amount'],
        'min_rows': 1,
    },
    'eastmoney_realtime': {
        'name': 'Eastmoney 实时行情',
        'call': lambda code, sina: get_etf_realtime_eastmoney(code),
        'key': 'eastmoney_realtime',
        'type': 'realtime',
        'expected_fields': ['close', 'volume', 'amount', 'turnover'],
        'min_rows': 1,
    },
}


# ═══════════════════════════════════════════════
# 单 API 单 ETF 测试
# ═══════════════════════════════════════════════

def test_one(api_key, api_def, code, sina, name, delay=0):
    """对单只 ETF 调用单个 API，返回诊断结果"""
    if delay > 0:
        time.sleep(delay)

    result = {
        'code': code,
        'sina': sina,
        'name': name,
        'api': api_key,
        'success': False,
        'error': None,
        'elapsed_ms': 0,
        'rows': 0,
        'fields': [],
        'missing_fields': [],
        'empty_fields': [],
        'date_start': None,
        'date_end': None,
        'sample': None,
    }

    t0 = time.time()
    try:
        df = api_def['call'](code, sina)
    except Exception as e:
        result['elapsed_ms'] = round((time.time() - t0) * 1000)
        result['error'] = f"{type(e).__name__}: {str(e)[:200]}"
        return result

    result['elapsed_ms'] = round((time.time() - t0) * 1000)

    if df is None:
        result['error'] = '返回 None (无数据或请求失败)'
        return result

    if api_def['type'] == 'history':
        if not isinstance(df, pd.DataFrame) or len(df) < api_def['min_rows']:
            result['error'] = f"行数不足 ({len(df) if isinstance(df, pd.DataFrame) else 0} < {api_def['min_rows']})"
            result['rows'] = len(df) if isinstance(df, pd.DataFrame) else 0
            return result

        result['rows'] = len(df)
        result['fields'] = list(df.columns)
        result['date_start'] = str(df['day'].iloc[0]) if 'day' in df.columns else None
        result['date_end'] = str(df['day'].iloc[-1]) if 'day' in df.columns else None

        # 字段覆盖检查
        for f in api_def['expected_fields']:
            if f not in df.columns:
                result['missing_fields'].append(f)
            elif f in df.columns and df[f].isna().all():
                result['empty_fields'].append(f)

        # 关键字段有效行数 (amount/turnover 可能仅最后一行有值)
        for f in ['amount', 'turnover']:
            if f in df.columns:
                valid_rows = int(df[f].notna().sum())
                result[f'{f}_valid_rows'] = valid_rows

        # 采样最新一行
        try:
            last = df.iloc[-1]
            result['sample'] = {
                c: (round(float(last[c]), 4) if not pd.isna(last[c]) else None)
                for c in df.columns if c != 'day'
            }
            if 'day' in df.columns:
                result['sample']['day'] = str(last['day'])
        except Exception:
            pass

    elif api_def['type'] == 'realtime':
        if not isinstance(df, dict) or len(df) == 0:
            result['error'] = '返回空 dict'
            return result

        result['rows'] = 1
        result['fields'] = list(df.keys())
        result['sample'] = {k: (round(float(v), 4) if isinstance(v, (int, float)) else v) for k, v in df.items()}

        for f in api_def['expected_fields']:
            if f not in df:
                result['missing_fields'].append(f)

    result['success'] = True
    return result


# ═══════════════════════════════════════════════
# 批量测试
# ═══════════════════════════════════════════════

def run_api_test(api_key, api_def, etf_list, max_workers=3, delay_between=0.3):
    """对指定 API 批量测试所有 ETF"""
    total = len(etf_list)
    results = []
    success = 0
    fail = 0

    print(f"\n{'='*65}")
    print(f"  {api_def['name']} ({total} 只 ETF, {max_workers} 线程)")
    print(f"{'='*65}")

    # 串行调用更安全，避免限频
    for i, (code, sina, name) in enumerate(etf_list, 1):
        r = test_one(api_key, api_def, code, sina, name, delay=delay_between)
        results.append(r)

        status = '[OK]' if r['success'] else '[FAIL]'
        extra = ''
        if r['success']:
            extra = f" {r['rows']}rows fields={r['fields']}"
            if r['missing_fields']:
                extra += f" MISSING={r['missing_fields']}"
            if r['empty_fields']:
                extra += f" EMPTY={r['empty_fields']}"
            # 标注 amount/turnover 有效行数
            for f in ['amount', 'turnover']:
                vk = f'{f}_valid_rows'
                if vk in r and r[vk] < r['rows']:
                    extra += f" {f}={r[vk]}/{r['rows']}"
            if r['elapsed_ms'] > 3000:
                extra += f" SLOW({r['elapsed_ms']}ms)"
            success += 1
        else:
            extra = f" {r['error'][:80]}"
            fail += 1

        print(f"  [{i:02d}/{total}] {status} {name}({code}){extra}")

    elapsed_times = [r['elapsed_ms'] for r in results if r['elapsed_ms'] > 0]
    summary = {
        'api': api_key,
        'name': api_def['name'],
        'total': total,
        'success': success,
        'fail': fail,
        'success_rate': round(success / total * 100, 1),
        'avg_elapsed_ms': round(np.mean(elapsed_times)) if elapsed_times else 0,
        'max_elapsed_ms': round(max(elapsed_times)) if elapsed_times else 0,
    }

    # 字段覆盖统计
    field_coverage = {}
    amt_full = 0  # amount 覆盖率 >50% 的 ETF 数
    to_full = 0   # turnover 覆盖率 >50% 的 ETF 数
    for r in results:
        if r['success']:
            for f in r['fields']:
                field_coverage[f] = field_coverage.get(f, 0) + 1
            if r.get('amount_valid_rows', 0) > r['rows'] * 0.5:
                amt_full += 1
            if r.get('turnover_valid_rows', 0) > r['rows'] * 0.5:
                to_full += 1

    summary['field_coverage'] = {
        f: {'count': c, 'pct': round(c / total * 100, 1)}
        for f, c in sorted(field_coverage.items())
    }
    summary['amount_full_pct'] = round(amt_full / max(success, 1) * 100, 1)
    summary['turnover_full_pct'] = round(to_full / max(success, 1) * 100, 1)

    print(f"\n  Summary: {success}/{total} success ({summary['success_rate']}%), "
          f"avg {summary['avg_elapsed_ms']}ms")

    return results, summary


def run_all(etf_list, apis_to_test=None, quick=False, max_workers=3, delay=0.3):
    """运行全部 API 测试"""
    if apis_to_test is None:
        apis_to_test = list(API_DEFS.keys())

    all_results = {}
    all_summaries = {}

    for api_key in apis_to_test:
        api_def = API_DEFS[api_key]
        results, summary = run_api_test(api_key, api_def, etf_list,
                                        max_workers=1, delay_between=delay)
        all_results[api_key] = results
        all_summaries[api_key] = summary

    return all_results, all_summaries


# ═══════════════════════════════════════════════
# 报告生成
# ═══════════════════════════════════════════════

def print_comparison_table(all_summaries, etf_list, all_results):
    """打印 API 对比汇总表"""
    print(f"\n{'='*80}")
    print(f"  API 对比汇总")
    print(f"{'='*80}")

    header = f"  {'API':25s} | {'成功':>5s} | {'成功率':>7s} | {'耗时':>6s} | {'amount':>8s} | {'turnover':>9s} | {'关键缺失字段'}"
    print(header)
    print(f"  {'-'*25} | {'-'*5} | {'-'*7} | {'-'*6} | {'-'*8} | {'-'*9} | {'-'*30}")

    for api_key in API_DEFS:
        if api_key not in all_summaries:
            continue
        s = all_summaries[api_key]
        name = API_DEFS[api_key]['name']

        amt_str = f"{s.get('amount_full_pct', 0):.0f}%" if 'amount_full_pct' in s else '--'
        to_str = f"{s.get('turnover_full_pct', 0):.0f}%" if 'turnover_full_pct' in s else '--'

        # 找出经常缺失的字段
        field_cov = s.get('field_coverage', {})
        weak_fields = []
        for f, info in field_cov.items():
            if info['pct'] < 100:
                weak_fields.append(f"{f}({info['pct']:.0f}%)")

        missing_str = ', '.join(weak_fields[:4]) if weak_fields else '--'
        if len(weak_fields) > 4:
            missing_str += f' +{len(weak_fields)-4} more'

        print(f"  {name:25s} | {s['success']:5d} | {s['success_rate']:6.1f}% | "
              f"{s['avg_elapsed_ms']:4d}ms | {amt_str:>8s} | {to_str:>9s} | {missing_str}")

    # 每只 ETF 在各 API 的表现
    print(f"\n{'='*80}")
    print(f"  逐 ETF 对比 (前 10 只)")
    print(f"{'='*80}")

    api_keys = [k for k in all_summaries]
    header2 = f"  {'ETF':14s}"
    for k in api_keys:
        header2 += f" | {k:12s}"
    print(header2)
    print(f"  {'-'*14}" + f"{'|'+'-'*13}" * len(api_keys))

    for code, sina, name in etf_list[:10]:
        row = f"  {name:14s}"
        for k in api_keys:
            r = next((r for r in all_results.get(k, []) if r['name'] == name), None)
            if r and r['success']:
                fields = len(r['fields'])
                missing = len(r['missing_fields']) + len(r['empty_fields'])
                cell = f"[OK]{fields}f"
                if missing > 0:
                    cell += f"-{missing}"
            else:
                cell = "[FAIL]"
            row += f" | {cell:12s}"
        print(row)


def save_report(all_results, all_summaries, etf_list):
    """保存诊断报告到 JSON"""
    report = {
        'generated_at': datetime.now(BJ).strftime('%Y-%m-%d %H:%M:%S'),
        'etf_count': len(etf_list),
        'apis_tested': list(all_summaries.keys()),
        'summaries': all_summaries,
        'details': {},
    }

    # 每只 ETF 的详细结果
    for code, sina, name in etf_list:
        etf_detail = {'code': code, 'sina': sina, 'name': name, 'apis': {}}
        for api_key, results in all_results.items():
            r = next((x for x in results if x['name'] == name), None)
            if r:
                etf_detail['apis'][api_key] = {
                    'success': r['success'],
                    'error': r['error'],
                    'elapsed_ms': r['elapsed_ms'],
                    'rows': r['rows'],
                    'fields': r['fields'],
                    'missing_fields': r['missing_fields'],
                    'empty_fields': r['empty_fields'],
                    'amount_valid_rows': r.get('amount_valid_rows'),
                    'turnover_valid_rows': r.get('turnover_valid_rows'),
                    'date_start': r['date_start'],
                    'date_end': r['date_end'],
                    'sample': r['sample'],
                }
        report['details'][name] = etf_detail

    with open(REPORT_FILE, 'w', encoding='utf-8') as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    print(f"\n  Report saved: {REPORT_FILE} ({os.path.getsize(REPORT_FILE)/1024:.0f} KB)")


# ═══════════════════════════════════════════════
# 主入口
# ═══════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description='API 独立诊断')
    parser.add_argument('--quick', action='store_true', help='仅测试前 5 只 ETF')
    parser.add_argument('--api', type=str, help='仅测试指定 API (eastmoney_hist|akshare_hist|tencent_hist|sina_realtime|eastmoney_realtime)')
    parser.add_argument('--delay', type=float, default=0.5, help='API 调用间隔(秒), 默认 0.5')
    args = parser.parse_args()

    # ETF 列表
    etf_list = [(code, sina, name) for code, (sina, name, sec) in ETF_POOL.items()]
    if args.quick:
        etf_list = etf_list[:5]

    # 选择 API
    if args.api:
        if args.api not in API_DEFS:
            print(f"Error: unknown API '{args.api}'. Options: {list(API_DEFS.keys())}")
            sys.exit(1)
        apis_to_test = [args.api]
    else:
        apis_to_test = list(API_DEFS.keys())

    t0 = time.time()
    print(f"\n{'#'*65}")
    print(f"# ETF API 独立诊断")
    print(f"# 时间: {datetime.now(BJ).strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"# ETF: {len(etf_list)} 只 | API: {len(apis_to_test)} 个 | 间隔: {args.delay}s")
    print(f"{'#'*65}")

    all_results, all_summaries = run_all(
        etf_list, apis_to_test,
        quick=args.quick,
        delay=args.delay
    )

    print_comparison_table(all_summaries, etf_list, all_results)
    save_report(all_results, all_summaries, etf_list)

    elapsed = time.time() - t0
    print(f"\n  Total elapsed: {elapsed:.1f}s")
    print(f"{'='*65}\n")

    # 退出码: 任一历史 API 成功率为 0 则返回 1
    for api_key in ['eastmoney_hist', 'akshare_hist', 'tencent_hist']:
        if api_key in all_summaries and all_summaries[api_key]['success_rate'] == 0:
            print(f"  WARNING: {api_key} 全部失败!")
            sys.exit(1)

    sys.exit(0)


if __name__ == '__main__':
    main()
