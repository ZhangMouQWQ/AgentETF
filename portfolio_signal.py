#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ETF 动量轮动 · 模拟持仓信号（GitHub Actions 专用）
============================================
基于策略指标 + T+1 持仓规则，生成合法的买入/卖出指令。

规则:
  T+1:   当日买入的份额次日才能卖出
  买入价: 次日 (开+低)/2 (有回调) 或 开盘价 (无回调)
  卖出价: 次日 (高+收)/2 (强势位卖出)
  跳过:   金额 < ¥500 或同ETF权重变化 < 20%
  费用:   ¥10/笔
  手数:   100股整数倍
  本金:   ¥8,000

输出:
  data/signal.json — 完整信号 (持仓 + 订单 + 板块 + 风险)
"""

import json
import os
import sys
import time
from datetime import datetime, timezone, timedelta

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from strategy import (
    ETF_POOL, ETF_SECTOR, SECTOR_ETF_POOL,
    fetch_daily_data_cached, calc_metrics,
    market_timing_v5, build_target_v5,
    MOM_LONG, DATA_LEN,
)

BJ = timezone(timedelta(hours=8))
DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data')
POSITIONS_FILE = os.path.join(DATA_DIR, 'positions.json')
SIGNAL_FILE = os.path.join(DATA_DIR, 'signal.json')

INIT_CASH = 8000.0
FEE = 10.0
LOT = 100
MIN_TRADE = 500.0       # 最低交易金额
MIN_WEIGHT_DELTA = 0.20  # 同ETF最小权重变化


# ═══════════════════════════════════════════════
# 持仓状态管理
# ═══════════════════════════════════════════════

def load_positions():
    """加载模拟持仓状态"""
    if os.path.exists(POSITIONS_FILE):
        with open(POSITIONS_FILE, 'r', encoding='utf-8') as f:
            return json.load(f)
    return _default_positions()


def _default_positions():
    return {
        'cash': INIT_CASH,
        'holdings': {},       # {code: {shares, avg_cost, locked_shares, locked_until_date}}
        'fees_paid': 0,
        'trades_count': 0,
        'history': [],        # 每日净值快照
        'last_update': None,
        'consecutive_losses': 0,
        'prev_day_return': None,
    }


def save_positions(state):
    """保存持仓状态"""
    os.makedirs(DATA_DIR, exist_ok=True)
    with open(POSITIONS_FILE, 'w', encoding='utf-8') as f:
        json.dump(state, f, ensure_ascii=False, indent=2)


# ═══════════════════════════════════════════════
# 净值计算
# ═══════════════════════════════════════════════

def calc_net_worth(state, prices):
    """计算当前净值 (现金 + 持仓市值)"""
    hv = 0.0
    for code, h in state['holdings'].items():
        name = _code_to_name(code)
        if name in prices:
            hv += h['shares'] * prices[name]
    return state['cash'] + hv


def _code_to_name(code):
    """ETF代码 → 名称"""
    for c, (sina, n) in ETF_POOL.items():
        if c == code:
            return n
    return None


def _name_to_code(name):
    """ETF名称 → 代码"""
    for c, (sina, n) in ETF_POOL.items():
        if n == name:
            return c
    return None


# ═══════════════════════════════════════════════
# 订单生成引擎
# ═══════════════════════════════════════════════

def generate_orders(state, target, prices, extra_history):
    """
    对比当前持仓与目标持仓，生成买卖订单。

    参数:
      state:   持仓状态 dict
      target:  build_target() 返回的目标列表 [{'name','code','weight',...}]
      prices:  {etf_name: close_price} (最新收盘价)
      extra_history: {etf_name: DataFrame} (含 open/high/low)

    返回:
      orders:  [{'action':'BUY'/'SELL', 'code', 'name', 'shares', 'price_est', 'amount', 'reason'}]
      alerts:  [{'level':'info'/'warn'/'danger', 'msg'}]
    """
    orders = []
    alerts = []

    # ── 解锁 T+1 ──
    today_str = datetime.now(BJ).strftime('%Y-%m-%d')
    for code in list(state['holdings'].keys()):
        h = state['holdings'][code]
        if h.get('locked_until_date') and h['locked_until_date'] <= today_str:
            h['locked_shares'] = 0
            h['locked_until_date'] = None

    nw = calc_net_worth(state, prices)

    # ── 目标权重 → 目标股数 ──
    target_map = {}
    for t in target:
        code = _name_to_code(t['name'])
        if code and t['weight'] > 0:
            target_map[code] = t

    # ── 卖出: 不在目标中 或 超配 ──
    for code in list(state['holdings'].keys()):
        h = state['holdings'][code]
        sellable = h['shares'] - h.get('locked_shares', 0)
        if sellable < LOT:
            continue

        name = _code_to_name(code)
        price = prices.get(name, 0)
        if price <= 0:
            continue

        if code not in target_map:
            # ── 清仓 ──
            sell_shares = (sellable // LOT) * LOT
            if sell_shares < LOT:
                continue
            revenue = sell_shares * price - FEE
            if revenue < MIN_TRADE:
                alerts.append({'level': 'info',
                               'msg': f'{name} 清仓跳过: 金额 ¥{revenue:.0f} < ¥{MIN_TRADE}'})
                continue
            # 执行价估算: (今日高+收)/2 (实际次日执行)
            hi = _get_extra(extra_history, name, 'high', price)
            sell_price_est = round((hi + price) / 2, 3)
            orders.append({
                'action': 'SELL', 'code': code, 'name': name,
                'shares': sell_shares,
                'price_est': sell_price_est,
                'amount': round(sell_shares * sell_price_est - FEE, 2),
                'reason': '调出组合 → 清仓',
            })
            # 更新状态
            remaining = h['shares'] - sell_shares
            if remaining < LOT:
                state['cash'] += sell_shares * sell_price_est - FEE
                state['fees_paid'] += FEE
                state['trades_count'] += 1
                del state['holdings'][code]
            else:
                state['cash'] += sell_shares * sell_price_est - FEE
                state['fees_paid'] += FEE
                state['trades_count'] += 1
                h['shares'] = remaining
            nw = calc_net_worth(state, prices)
        else:
            # ── 减仓 (超配) ──
            tw = target_map[code]['weight']
            target_shares = int(nw * tw / price / LOT) * LOT
            current_shares = h['shares']
            if current_shares > target_shares + LOT:
                sell_shares = min(((current_shares - target_shares) // LOT) * LOT, sellable)
                if sell_shares < LOT:
                    continue
                revenue = sell_shares * price - FEE
                if revenue < MIN_TRADE:
                    continue
                hi = _get_extra(extra_history, name, 'high', price)
                sell_price_est = round((hi + price) / 2, 3)
                orders.append({
                    'action': 'SELL', 'code': code, 'name': name,
                    'shares': sell_shares,
                    'price_est': sell_price_est,
                    'amount': round(sell_shares * sell_price_est - FEE, 2),
                    'reason': f'减仓 {tw*100:.0f}% ← {current_shares*price/nw*100:.0f}%',
                })
                state['cash'] += sell_shares * sell_price_est - FEE
                state['fees_paid'] += FEE
                state['trades_count'] += 1
                h['shares'] -= sell_shares
                nw = calc_net_worth(state, prices)

    # ── 买入: 在目标中但未持有或低配 ──
    buy_candidates = []
    for code, t in target_map.items():
        name = t['name']
        price = prices.get(name, 0)
        if price <= 0:
            continue
        target_shares = int(nw * t['weight'] / price / LOT) * LOT
        if target_shares < LOT:
            continue
        current_shares = state['holdings'].get(code, {}).get('shares', 0)
        buy_shares = target_shares - current_shares
        if buy_shares < LOT:
            continue
        buy_candidates.append((code, name, buy_shares, t['weight'], price))

    # 按权重降序 → 优先买大仓位
    buy_candidates.sort(key=lambda x: x[3], reverse=True)

    for code, name, buy_shares, weight, price in buy_candidates:
        # 估算买入价: (今日开+低)/2 (有回调) 或 开盘价
        op = _get_extra(extra_history, name, 'open', price)
        lo = _get_extra(extra_history, name, 'low', price)
        if lo < op:
            buy_price_est = round((op + lo) / 2, 3)
            tag = '回调买入'
        else:
            buy_price_est = round(op, 3)
            tag = '开盘买入'

        cost = buy_shares * buy_price_est + FEE
        nw = calc_net_worth(state, prices)

        # 金额检查
        if cost < MIN_TRADE:
            continue

        # 同ETF权重变化检查 (仅对已有持仓)
        if code in state['holdings']:
            current_shares = state['holdings'][code]['shares']
            current_val = current_shares * price
            current_weight = current_val / nw if nw > 0 else 0
            if abs(weight - current_weight) < MIN_WEIGHT_DELTA:
                continue

        # 资金检查
        if cost > state['cash'] - FEE:
            affordable = int((state['cash'] - FEE * 2) / buy_price_est / LOT) * LOT
            if affordable < LOT:
                alerts.append({'level': 'warn',
                               'msg': f'{name} 资金不足: 需 ¥{cost:.0f}, 现金 ¥{state["cash"]:.0f}'})
                continue
            buy_shares = affordable
            cost = buy_shares * buy_price_est + FEE
            tag = '资金受限' + tag

        order = {
            'action': 'BUY', 'code': code, 'name': name,
            'shares': buy_shares,
            'price_est': buy_price_est,
            'amount': round(cost, 2),
            'reason': f'新建 {weight*100:.0f}% ({tag})',
        }
        orders.append(order)

        # 更新状态
        state['cash'] -= cost
        state['fees_paid'] += FEE
        state['trades_count'] += 1
        if code in state['holdings']:
            h = state['holdings'][code]
            total_shares = h['shares'] + buy_shares
            h['shares'] = total_shares
        else:
            state['holdings'][code] = {
                'shares': buy_shares,
                'avg_cost': buy_price_est,
                'locked_shares': buy_shares,
                'locked_until_date': _next_trading_day(today_str),
            }

    return orders, alerts


def _get_extra(extra_history, name, field, fallback):
    """从 extra_history 获取最新字段值"""
    if name not in extra_history:
        return fallback
    df = extra_history[name]
    if field not in df.columns:
        return fallback
    val = df[field].iloc[-1]
    if pd.isna(val):
        return fallback
    return float(val)


def _next_trading_day(date_str):
    """估算下一个交易日 (跳过周末, 不考虑节假日)"""
    from datetime import datetime as dt
    d = dt.strptime(date_str, '%Y-%m-%d')
    d += timedelta(days=1)
    while d.weekday() >= 5:  # 周六=5, 周日=6
        d += timedelta(days=1)
    return d.strftime('%Y-%m-%d')


# ═══════════════════════════════════════════════
# 板块监控
# ═══════════════════════════════════════════════

def build_sector_report(metrics, target):
    """构建板块监控报告"""
    sectors = {}
    for name, m in metrics.items():
        s = m.get('sector', '其他')
        if s == '宽基指数':
            continue
        sectors.setdefault(s, []).append((name, m))

    target_names = {t['name'] for t in target}

    report = []
    for sname, etfs in sorted(sectors.items()):
        avg_mom = np.mean([m['mom_long'] for _, m in etfs])
        positive = sum(1 for _, m in etfs if m['mom_long'] > 0)
        count = len(etfs)
        # 板块中是否有推荐标的
        in_target = [name for name, _ in etfs if name in target_names]

        report.append({
            'sector': sname,
            'etf_count': count,
            'positive_ratio': round(positive / count, 2),
            'avg_momentum': round(float(avg_mom), 2),
            'picks_in_sector': in_target,
            'status': '强势' if avg_mom > 2 else '中性' if avg_mom > 0 else '弱势',
        })

    return report


# ═══════════════════════════════════════════════
# 主入口
# ═══════════════════════════════════════════════

def main():
    t0 = time.time()
    today_str = datetime.now(BJ).strftime('%Y-%m-%d')
    now_str = datetime.now(BJ).strftime('%Y-%m-%d %H:%M:%S')

    print(f"\n{'='*60}")
    print(f"[{now_str}] ETF 模拟持仓信号生成")
    print(f"{'='*60}")

    # ── 1. 加载数据 ──
    price, etf_info, extra_history, _ = fetch_daily_data_cached(ETF_POOL)
    if price is None:
        print("❌ 数据拉取失败")
        return

    asof = str(price.index[-1])
    print(f"数据日期: {asof} | ETF 数量: {len(price.columns)}")

    # ── 2. 计算指标 ──
    metrics = calc_metrics(price, etf_info, ETF_SECTOR, extra_history)
    if not metrics:
        print("❌ 指标计算失败")
        return

    # ── 3. 择时 (V5 单ETF) ──
    position_ratio = market_timing_v5(metrics)
    # 仓位文本
    ratio_int = int(position_ratio * 10)
    pos_text = f"{ratio_int}成" if ratio_int > 0 else "空仓"
    hs300 = metrics.get('沪深300ETF', {})
    pos_reason = f"V5单ETF | HS300动量={hs300.get('mom_long', 0):+.1f}%"
    market_cls = 'ok' if position_ratio >= 0.7 else 'warn' if position_ratio >= 0.3 else 'danger'
    regime = 'bull' if position_ratio >= 0.7 else 'sideways' if position_ratio >= 0.3 else 'bear'

    # ── 4. 选股 (V5 单ETF) ──
    target, _ = build_target_v5(metrics, position_ratio)

    # ── 5. 加载持仓 ──
    state = load_positions()
    state['last_update'] = now_str

    # ── 6. 生成订单 ──
    close_prices = {name: metrics[name]['latest'] for name in metrics}
    orders, alerts = generate_orders(state, target, close_prices, extra_history)

    # ── 7. 板块监控 ──
    sector_report = build_sector_report(metrics, target)

    # ── 8. 计算净值 ──
    nw = calc_net_worth(state, close_prices)
    hv = nw - state['cash']
    state['history'].append({
        'date': asof,
        'cash': round(state['cash'], 2),
        'holdings_value': round(hv, 2),
        'total': round(nw, 2),
        'position_ratio': position_ratio,
    })
    # 保留最近 365 条
    if len(state['history']) > 365:
        state['history'] = state['history'][-365:]

    # ── 9. 保存状态 ──
    save_positions(state)

    # ── 10. 构建信号 JSON ──

    # 持仓明细
    holdings_detail = []
    for code, h in state['holdings'].items():
        name = _code_to_name(code)
        p = close_prices.get(name, 0)
        val = h['shares'] * p
        holdings_detail.append({
            'code': code,
            'name': name,
            'shares': h['shares'],
            'avg_cost': round(h['avg_cost'], 3),
            'price': round(p, 3),
            'value': round(val, 2),
            'weight': round(val / nw, 3) if nw > 0 else 0,
            'locked_shares': h.get('locked_shares', 0),
            'pnl_pct': round((p / h['avg_cost'] - 1) * 100, 2) if h['avg_cost'] > 0 else 0,
        })

    # 目标持仓
    target_detail = []
    for t in target:
        target_detail.append({
            'name': t['name'],
            'code': _name_to_code(t['name']),
            'weight': t['weight'],
            'sector': t.get('sector', ''),
            'mom_long': t.get('mom_long', 0),
            'quality_score': t.get('quality_score', 0),
        })

    signal = {
        'generated_at': now_str,
        'data_date': asof,

        # 市场状态
        'market': {
            'strategy': 'V5',
            'position_ratio': position_ratio,
            'position_text': pos_text,
            'position_reason': pos_reason,
            'market_class': market_cls,
        },

        # 投资组合
        'portfolio': {
            'net_worth': round(nw, 2),
            'cash': round(state['cash'], 2),
            'holdings_value': round(hv, 2),
            'total_return_pct': round((nw / INIT_CASH - 1) * 100, 2),
            'fees_paid': round(state['fees_paid'], 2),
            'trades_count': state['trades_count'],
            'holdings': holdings_detail,
            'target': target_detail,
        },

        # 买卖订单
        'orders': orders,

        # 板块监控
        'sectors': sector_report,

        # 提醒
        'alerts': alerts,

        # 风控
        'risk': {
            'consecutive_losses': state.get('consecutive_losses', 0),
            'prev_day_return': state.get('prev_day_return'),
        },
    }

    # 保存
    os.makedirs(DATA_DIR, exist_ok=True)
    with open(SIGNAL_FILE, 'w', encoding='utf-8') as f:
        json.dump(signal, f, ensure_ascii=False, indent=2)

    # ── 11. 终端摘要 ──
    elapsed = time.time() - t0
    print(f"\n{'─'*55}")
    print(f"  🎯 V5单ETF | {pos_text} | {pos_reason}")
    print(f"  💰 净值: ¥{nw:,.2f} (回报 {(nw/INIT_CASH-1)*100:+.1f}%)")
    print(f"  📦 持仓: {len(holdings_detail)} 只 | 现金 ¥{state['cash']:,.0f}")
    print(f"{'─'*55}")

    if orders:
        print(f"\n  📋 订单 ({len(orders)} 笔):")
        for o in orders:
            icon = '🟢' if o['action'] == 'BUY' else '🔴'
            print(f"    {icon} {o['action']:4s} {o['name']} {o['shares']}股 "
                  f"≈¥{o['amount']:,.0f} @{o['price_est']:.3f}")
            print(f"       理由: {o['reason']}")
    else:
        print(f"\n  ✅ 无需调仓 — 持仓与目标一致")

    if alerts:
        print(f"\n  ⚠️ 提醒 ({len(alerts)} 条):")
        for a in alerts:
            print(f"    [{a['level']}] {a['msg']}")

    print(f"\n  📁 signal.json 已保存 ({os.path.getsize(SIGNAL_FILE)/1024:.0f} KB)")
    print(f"  📁 positions.json 已更新")
    print(f"{'='*60}")
    print(f"  完成! 耗时 {elapsed:.1f}s")


if __name__ == '__main__':
    main()
