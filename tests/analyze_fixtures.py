#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
本地 Fixture 分析器
==================
读取 GitHub Actions 回传的 _fixtures/collector_output.json，
生成可读的验证报告。

用法: python tests/analyze_fixtures.py
"""
import json
import os
import sys
from datetime import datetime


def load_fixture(path='_fixtures/collector_output.json'):
    if not os.path.exists(path):
        print(f"❌ 文件不存在: {path}")
        print("   请先触发 GitHub Actions 工作流，然后 git pull 同步")
        sys.exit(1)
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)


def analyze_api_matrix(matrix):
    """分析 API 连通性"""
    print("\n" + "=" * 55)
    print("  📡 API 连通性矩阵 (生产环境)")
    print("=" * 55)
    print(f"  {'ETF':<16} {'EM-K线':<8} {'TX-K线':<8} {'Sina实时':<8} {'EM实时':<8}")
    print(f"  {'─'*16} {'─'*8} {'─'*8} {'─'*8} {'─'*8}")

    em_ok = tx_ok = sina_ok = emrt_ok = 0
    total = len(matrix)
    for code, info in matrix.items():
        ek = '✅' if info['eastmoney_kline']['ok'] else '❌'
        tk = '✅' if info['tencent_kline']['ok'] else '❌'
        sr = '✅' if info['sina_realtime']['ok'] else '❌'
        er = '✅' if info['eastmoney_realtime']['ok'] else '❌'
        if info['eastmoney_kline']['ok']: em_ok += 1
        if info['tencent_kline']['ok']: tx_ok += 1
        if info['sina_realtime']['ok']: sina_ok += 1
        if info['eastmoney_realtime']['ok']: emrt_ok += 1
        print(f"  {info['name']:<16} {ek:<8} {tk:<8} {sr:<8} {er:<8}")

    print(f"\n  汇总: Eastmoney K线 {em_ok}/{total} | Tencent K线 {tx_ok}/{total} | Sina实时 {sina_ok}/{total} | EM实时 {emrt_ok}/{total}")

    # 判断
    print()
    if em_ok == total:
        print("  ✅ Eastmoney K线全线畅通 — 可获得完整成交额/换手率历史数据")
    elif em_ok > 0:
        print(f"  ⚠️ Eastmoney K线部分可用 ({em_ok}/{total})")
    else:
        print("  ⚠️ Eastmoney K线不可用 — 将降级到 Tencent + Sina 补充")
    if sina_ok == total:
        print("  ✅ 新浪实时行情全线畅通 — 盘中量价数据有保障")
    return em_ok, tx_ok, sina_ok


def analyze_kline_samples(samples):
    """分析 K 线数据样本"""
    print("\n" + "=" * 55)
    print("  📊 K线数据样本")
    print("=" * 55)
    for name, info in samples.items():
        src = info['source']
        rows = info['total_rows']
        cols = info.get('columns', [])
        has_amt = info.get('has_amount', False)
        has_to = info.get('has_turnover', False)
        print(f"  {name} ({src}): {rows}行, 列={cols}")
        if has_amt:
            print(f"    ✅ 含成交额 (最新={info.get('last_amount', '?'):.0f}元)" if isinstance(info.get('last_amount'), (int, float)) else f"    ✅ 含成交额")
        if has_to:
            print(f"    ✅ 含换手率")
        if not has_amt and not has_to:
            print(f"    ⚠️ 无成交额/换手率 (腾讯源限制)")


def analyze_intraday(snapshot):
    """分析盘中快照"""
    print("\n" + "=" * 55)
    print("  ⚡ 盘中快照 (fetch_intraday_snapshot)")
    print("=" * 55)
    for name, snap in snapshot.items():
        keys = snap.get('keys', [])
        has_all = snap.get('has_volume') and snap.get('has_amount')
        print(f"  {name}: fields={keys}")
        print(f"    close={snap.get('close')}, volume={'✅' if snap.get('has_volume') else '❌'}, amount={'✅' if snap.get('has_amount') else '❌'}, turnover={'✅' if snap.get('has_turnover') else '❌'}")
        if not has_all:
            print(f"    ⚠️ 缺少量/额 — flow_signal 可能使用旧数据")


def analyze_extra_comparison(comp):
    """分析 extra_history 合并前后"""
    print("\n" + "=" * 55)
    print("  🔄 extra_history 合并前后对比 (核心修复)")
    print("=" * 55)
    before = comp.get('before', {})
    after = comp.get('after', {})
    advanced = comp.get('any_date_advanced', False)

    for name in after:
        b = before.get(name, {})
        a = after.get(name, {})
        b_date = b.get('latest_date', '?')
        a_date = a.get('latest_date', '?')
        b_today = b.get('has_today', False)
        a_today = a.get('has_today', False)
        date_chg = '🔄' if b_date != a_date else '='
        today_chg = '✅' if a_today and not b_today else ('✅' if a_today else '❌')
        print(f"  {date_chg} {name}: {b_date} → {a_date} | today: {b_today}→{a_today} {today_chg}")

    print()
    if advanced:
        print("  ✅ 至少一个 ETF 的 extra_history 日期推进 — 修复生效")
    else:
        print("  ⚠️ extra_history 日期无推进（可能日K已含今日数据，盘中路径未触发）")


def analyze_metrics(metrics):
    """分析策略指标"""
    print("\n" + "=" * 55)
    print("  📈 策略指标样本")
    print("=" * 55)
    print(f"  {'ETF':<12} {'板块':<10} {'日涨跌':<8} {'40日动量':<8} {'5日动量':<8} {'得分':<6} {'资金流':<6}")
    for name, m in metrics.items():
        print(f"  {name:<12} {m['sector']:<10} {m['daily_change']:>+7.2f}% {m['mom_long']:>+7.2f}% {m['mom_short']:>+7.2f}% {m['score']:>5.1f} {m['flow_signal']:>5.2f}")


def main():
    print("=" * 55)
    print("  🔍 Fixture 分析报告")
    print(f"  分析时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 55)

    data = load_fixture()
    print(f"  采集时间: {data.get('run_time', '?')}")
    print(f"  基准日期: {data.get('today', '?')}")
    print(f"  信号类型: {data.get('signal_type', '?')}")
    print(f"  测试标的: {list(data.get('test_pool', {}).values())}")

    # A
    em_ok, tx_ok, sina_ok = analyze_api_matrix(data.get('api_matrix', {}))

    # B
    if data.get('kline_samples'):
        analyze_kline_samples(data['kline_samples'])
    else:
        print("\n  ⚠️ 无 K 线样本数据")

    # C
    if data.get('intraday_snapshot'):
        analyze_intraday(data['intraday_snapshot'])
    else:
        print("\n  ⚠️ 无盘中快照数据")

    # D
    if data.get('extra_comparison'):
        analyze_extra_comparison(data['extra_comparison'])

    # E
    if data.get('metrics_sample'):
        analyze_metrics(data['metrics_sample'])

    # 总结
    print("\n" + "=" * 55)
    print("  📋 总结")
    print("=" * 55)
    checks = []
    if em_ok >= 3:
        checks.append(('✅', 'Eastmoney K线可用 — 历史数据含成交额/换手率'))
    else:
        checks.append(('⚠️', f'Eastmoney K线仅 {em_ok}/5 可用 — 降级到Tencent'))

    if sina_ok >= 5:
        checks.append(('✅', '新浪实时行情可用 — 盘中量价数据可靠'))

    extra = data.get('extra_comparison', {})
    if extra.get('any_date_advanced'):
        checks.append(('✅', 'extra_history 日期推进 — 盘中修复在生产环境生效'))
    else:
        checks.append(('⚠️', 'extra_history 无日期推进 — 可能是收盘运行或日K已更新'))

    for icon, msg in checks:
        print(f"  {icon} {msg}")


if __name__ == '__main__':
    main()
