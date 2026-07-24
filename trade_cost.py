#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
交易成本计算模块
================
职责: 精确计算 ETF 交易佣金/过户费, 返回扣除费用后的实际成交金额

A股 ETF 费率 (2024年):
  佣金:    万分之2.5 (0.025%), 最低5元/笔
  印花税:  0 (ETF 暂免征收)
  过户费:  沪市(5/6开头) 万分之0.1, 深市(0/1/3开头) 免收

用法:
    from trade_cost import calc_cost, format_cost

    cost = calc_cost('510300', 'BUY', price=2.0, shares=4000)
    print(format_cost(cost))
    # 买入 4000股 @2.000 = ¥8000.00
    #   佣金: ¥5.00 (最低5元, 万分之2.5 = ¥2.00)
    #   过户费: ¥0.08
    #   实际支出: ¥8005.08
"""

# ═══════════════════════════════════
# 费率配置
# ═══════════════════════════════════

COMMISSION_RATE = 0.00025    # 佣金费率: 万分之2.5
COMMISSION_MIN = 5.0          # 佣金最低: 5元/笔
TRANSFER_RATE = 0.00001       # 过户费: 万分之0.1 (仅沪市)
STAMP_DUTY_RATE = 0.0         # 印花税: ETF暂免


# ═══════════════════════════════════
# 交易所判断
# ═══════════════════════════════════

def _is_shanghai(code):
    """判断是否沪市 ETF (5/6开头)"""
    if not code:
        return False
    return str(code)[0] in ('5', '6')


def _is_shenzhen(code):
    """判断是否深市 ETF (0/1/3开头)"""
    if not code:
        return False
    return str(code)[0] in ('0', '1', '3')


# ═══════════════════════════════════
# 成本计算
# ═══════════════════════════════════

def calc_cost(code, action, price, shares):
    """计算单笔交易成本

    Args:
        code:      ETF代码, 如 '510300'
        action:    'BUY' 或 'SELL'
        price:     成交单价
        shares:    成交股数

    Returns:
        dict: {
            'code': str,
            'action': str,
            'price': float,
            'shares': int,
            'turnover': float,        # 成交金额 (不含费)
            'commission': float,       # 佣金
            'commission_detail': str,  # 佣金明细
            'stamp_duty': float,       # 印花税 (ETF为0)
            'transfer_fee': float,     # 过户费
            'total_fee': float,        # 总费用
            'net_amount': float,       # 实际支出(+)或收入(-)
            'fee_rate_pct': float,     # 费率占成交额百分比
        }
    """
    turnover = price * shares

    # ── 1. 佣金 ──
    commission_raw = turnover * COMMISSION_RATE
    commission = max(commission_raw, COMMISSION_MIN)
    if commission_raw < COMMISSION_MIN:
        commission_detail = f"最低5元 (万分之2.5 = ¥{commission_raw:.2f})"
    else:
        commission_detail = f"万分之2.5 = ¥{commission_raw:.2f}"

    # ── 2. 印花税 (仅卖出, ETF暂免) ──
    stamp_duty = 0.0
    if action.upper() == 'SELL':
        stamp_duty = turnover * STAMP_DUTY_RATE

    # ── 3. 过户费 (仅沪市) ──
    transfer_fee = 0.0
    if _is_shanghai(code):
        transfer_fee = turnover * TRANSFER_RATE
    # 深市免过户费

    # ── 4. 汇总 ──
    total_fee = commission + stamp_duty + transfer_fee

    if action.upper() == 'BUY':
        net_amount = turnover + total_fee  # 实际支出
    else:
        net_amount = turnover - total_fee  # 实际收入

    fee_rate = (total_fee / turnover * 100) if turnover > 0 else 0

    return {
        'code': code,
        'action': action.upper(),
        'price': price,
        'shares': shares,
        'turnover': round(turnover, 2),
        'commission': round(commission, 2),
        'commission_detail': commission_detail,
        'stamp_duty': round(stamp_duty, 2),
        'transfer_fee': round(transfer_fee, 2),
        'total_fee': round(total_fee, 2),
        'net_amount': round(net_amount, 2),
        'fee_rate_pct': round(fee_rate, 2),
    }


def calc_cost_simple(action, price, shares, is_shanghai=False):
    """简化版成本计算 (不需要代码, 直接指定是否沪市)

    Args:
        action:      'BUY' | 'SELL'
        price:       成交单价
        shares:      成交股数
        is_shanghai: 是否沪市

    Returns:
        (net_amount, total_fee) 元组
    """
    code = '510000' if is_shanghai else '159900'
    result = calc_cost(code, action, price, shares)
    return result['net_amount'], result['total_fee']


# ═══════════════════════════════════
# 格式化输出
# ═══════════════════════════════════

def format_cost(result):
    """格式化输出交易成本明细

    Args:
        result: calc_cost() 返回的 dict

    Returns:
        str: 多行格式化字符串
    """
    action_cn = '买入' if result['action'] == 'BUY' else '卖出'
    direction = '支出' if result['action'] == 'BUY' else '收入'

    lines = [
        f"{action_cn} {result['shares']}股 @{result['price']:.3f} = ¥{result['turnover']:,.2f}",
        f"  佣金:      ¥{result['commission']:.2f} ({result['commission_detail']})",
    ]

    if result['stamp_duty'] > 0:
        lines.append(f"  印花税:    ¥{result['stamp_duty']:.2f} ({STAMP_DUTY_RATE*100:.2f}%)")

    if result['transfer_fee'] > 0:
        lines.append(f"  过户费:    ¥{result['transfer_fee']:.2f} (沪市 {_is_shanghai(result['code'])})")
    else:
        is_sh = _is_shanghai(result['code'])
        market = '沪市' if is_sh else '深市(免)'
        lines.append(f"  过户费:    ¥0.00 ({market})")

    lines.append(f"  总费用:    ¥{result['total_fee']:.2f} ({result['fee_rate_pct']:.2f}% of 成交额)")
    lines.append(f"  实际{direction}: ¥{result['net_amount']:,.2f}")

    return '\n'.join(lines)


# ═══════════════════════════════════
# 成本分析工具
# ═══════════════════════════════════

def analyze_trade_efficiency(initial_cash, trade_count, avg_turnover):
    """分析交易成本对策略的影响

    Args:
        initial_cash: 初始资金
        trade_count:  年交易次数
        avg_turnover: 平均每笔成交额

    Returns:
        dict: 成本分析结果
    """
    # 简化: 假设每笔均为买入或卖出, 佣金最低5元
    avg_commission = max(avg_turnover * COMMISSION_RATE, COMMISSION_MIN)
    avg_fee = avg_commission  # ETF无印花税, 简化忽略过户费

    annual_fee = avg_fee * trade_count * 2  # 买入+卖出各一次
    fee_ratio = annual_fee / initial_cash * 100

    # 需要多少超额收益才能覆盖成本
    required_alpha = fee_ratio

    return {
        'initial_cash': initial_cash,
        'trade_count_per_year': trade_count,
        'avg_turnover': avg_turnover,
        'avg_fee_per_trade': round(avg_fee, 2),
        'annual_total_fee': round(annual_fee, 2),
        'fee_pct_of_capital': round(fee_ratio, 2),
        'required_alpha_pct': round(required_alpha, 2),
        'verdict': (
            '费用合理' if fee_ratio < 5 else
            '费用偏高, 需策略有足够超额收益' if fee_ratio < 20 else
            '费用过高! 降低交易频率或提高单笔金额'
        ),
    }


def recommend_min_trade_amount():
    """推荐最低交易金额 (使佣金不低于最低5元的阈值)"""
    min_for_commission = COMMISSION_MIN / COMMISSION_RATE
    return {
        'commission_min': COMMISSION_MIN,
        'commission_rate_pct': COMMISSION_RATE * 100,
        'min_trade_for_efficient_commission': round(min_for_commission, 2),
        'advice': (
            f"单笔成交额应 ≥ ¥{min_for_commission:,.0f} "
            f"才能使佣金(万分之{COMMISSION_RATE*10000:.1f})超过最低{COMMISSION_MIN}元"
        ),
    }


# ═══════════════════════════════════
# 自测
# ═══════════════════════════════════

if __name__ == '__main__':
    print("=== 交易成本计算模块 自测 ===\n")

    # 1. 沪市买入
    print("[1] 沪市 ETF 买入 (510300 沪深300ETF)")
    r1 = calc_cost('510300', 'BUY', price=4.70, shares=1700)
    print(format_cost(r1))

    # 2. 沪市卖出
    print(f"\n[2] 沪市 ETF 卖出")
    r2 = calc_cost('510300', 'SELL', price=4.80, shares=1700)
    print(format_cost(r2))

    # 3. 深市买入
    print(f"\n[3] 深市 ETF 买入 (159915 创业板ETF)")
    r3 = calc_cost('159915', 'BUY', price=2.50, shares=3200)
    print(format_cost(r3))

    # 4. 小额定投 (佣金最低5元效果)
    print(f"\n[4] 小额定投 (佣金最低5元吞噬收益)")
    r4 = calc_cost('510300', 'BUY', price=4.70, shares=100)
    print(format_cost(r4))
    print(f"  ⚠️ 成交额仅 ¥{100*4.70:.0f}, 佣金 ¥5 占 {5/(100*4.70)*100:.1f}%!")

    # 5. 成本分析
    print(f"\n[5] 策略成本分析 (8000本金, 年200笔, 均额¥4000)")
    analysis = analyze_trade_efficiency(8000, 200, 4000)
    for k, v in analysis.items():
        print(f"  {k}: {v}")

    # 6. 最低交易金额建议
    print(f"\n[6] 最低交易金额建议")
    rec = recommend_min_trade_amount()
    for k, v in rec.items():
        print(f"  {k}: {v}")

    # 7. 对比: 大额 vs 小额
    print(f"\n[7] 费率对比")
    for shares in [100, 500, 1000, 2000, 5000, 10000]:
        r = calc_cost('510300', 'BUY', 4.70, shares)
        print(f"  {shares:5d}股 ¥{r['turnover']:,.0f} → 费 ¥{r['total_fee']:.2f} ({r['fee_rate_pct']:.2f}%)")

    print("\n=== 自测完成 ===")
