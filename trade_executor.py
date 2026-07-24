#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
交易执行模块
============
职责: 接收策略信号 → 生成委托单 → 校验 → 模拟成交

模式:
  模拟模式 (SIM): 记录委托 + 假设成交 (本程序推荐)
  实盘模式 (LIVE): 调用券商API (需自行接入, 不建议回测使用)

用法:
    from trade_executor import TradeExecutor

    executor = TradeExecutor(mode='SIM', initial_cash=8000)
    result = executor.submit(code='510300', action='BUY', price=4.70, shares=1700)
    print(executor.summary())
"""

from dataclasses import dataclass
from datetime import datetime, timezone, timedelta
from typing import Optional

BJ = timezone(timedelta(hours=8))

# ═══════════════════════════════════
# 数据结构
# ═══════════════════════════════════

@dataclass
class Order:
    """委托单"""
    order_id: int
    code: str
    action: str            # 'BUY' | 'SELL'
    price_type: str        # 'LIMIT' | 'MARKET'
    price: float           # 委托价格 (市价单填0)
    shares: int            # 委托数量 (100的整数倍)
    status: str = 'PENDING'  # PENDING | FILLED | PARTIAL | CANCELLED | REJECTED
    filled_shares: int = 0
    filled_price: float = 0.0
    submit_time: str = ''
    fill_time: str = ''
    reject_reason: str = ''
    tag: str = ''          # 用户自定义标签


@dataclass
class Trade:
    """成交记录"""
    trade_id: int
    order_id: int
    code: str
    action: str
    price: float           # 实际成交价
    shares: int
    turnover: float        # 成交金额
    commission: float       # 佣金
    transfer_fee: float     # 过户费
    stamp_duty: float       # 印花税 (ETF为0)
    total_fee: float        # 总费用
    net_amount: float       # 净支出/收入
    time: str


@dataclass
class Position:
    """持仓"""
    code: str
    shares: int
    locked_shares: int = 0     # T+1 锁定
    avg_cost: float = 0.0
    buy_date: str = ''


class Account:
    """虚拟账户"""
    def __init__(self, initial_cash: float):
        self.initial_cash = initial_cash
        self.cash = initial_cash
        self.positions: dict[str, Position] = {}
        self.trade_count = 0
        self.total_fees = 0.0

    @property
    def holdings_value(self, prices: dict) -> float:
        """持仓市值"""
        v = 0.0
        for code, pos in self.positions.items():
            p = prices.get(code, 0)
            v += pos.shares * p
        return v

    def net_worth(self, prices: dict) -> float:
        return self.cash + self.holdings_value(prices)

    def available_cash(self, reserved: float = 0) -> float:
        """可用资金 (预留部分)"""
        return max(0, self.cash - reserved)

    def available_shares(self, code: str) -> int:
        """可卖股数 (排除T+1锁定)"""
        pos = self.positions.get(code)
        if not pos:
            return 0
        return pos.shares - pos.locked_shares


# ═══════════════════════════════════
# 交易执行器
# ═══════════════════════════════════

class TradeExecutor:
    """交易执行器

    mode='SIM':  模拟模式 (记录委托, 假设按指定价格成交)
    mode='LIVE': 实盘模式 (需接入券商API, 当前仅做桩)
    """

    def __init__(self, mode='SIM', initial_cash: float = 8000, lot_size: int = 100):
        """
        Args:
            mode: 'SIM' | 'LIVE'
            initial_cash: 初始资金
            lot_size: 最小交易单位(股), 默认100
        """
        if mode not in ('SIM', 'LIVE'):
            raise ValueError(f"mode must be 'SIM' or 'LIVE', got '{mode}'")

        self.mode = mode
        self.lot_size = lot_size
        self.account = Account(initial_cash)
        self.orders: list[Order] = []
        self.trades: list[Trade] = []
        self._order_id = 0
        self._trade_id = 0

    # ── 委托提交 ──

    def submit(self, code: str, action: str, price: float, shares: int,
               price_type: str = 'LIMIT', tag: str = '') -> Order:
        """提交委托单

        Args:
            code:       ETF代码
            action:     'BUY' | 'SELL'
            price:      委托价格 (限价单)
            shares:     委托数量 (会自动取整到100的倍数)
            price_type: 'LIMIT' (限价) | 'MARKET' (市价, 不推荐)
            tag:        用户标签

        Returns:
            Order: 委托单 (含状态)
        """
        # ── 数量取整 ──
        shares = (shares // self.lot_size) * self.lot_size
        if shares < self.lot_size:
            return self._reject(f"委托数量不足 ({shares} < {self.lot_size}股)")

        action = action.upper()
        if action not in ('BUY', 'SELL'):
            return self._reject(f"无效方向: {action}")

        self._order_id += 1
        order = Order(
            order_id=self._order_id,
            code=code,
            action=action,
            price_type=price_type,
            price=price,
            shares=shares,
            submit_time=datetime.now(BJ).strftime('%Y-%m-%d %H:%M:%S'),
            tag=tag,
        )

        # ── 预校验 ──
        rejection = self._validate(order)
        if rejection:
            order.status = 'REJECTED'
            order.reject_reason = rejection
            self.orders.append(order)
            return order

        # ── 执行 ──
        if self.mode == 'SIM':
            self._execute_sim(order)
        elif self.mode == 'LIVE':
            self._execute_live(order)

        self.orders.append(order)
        return order

    def _reject(self, reason: str) -> Order:
        """生成被拒委托"""
        self._order_id += 1
        return Order(
            order_id=self._order_id, code='', action='BUY',
            price_type='LIMIT', price=0, shares=0,
            status='REJECTED', reject_reason=reason,
        )

    # ── 校验 ──

    def _validate(self, order: Order) -> Optional[str]:
        """委托校验

        Returns:
            str: 拒绝原因, None=通过
        """
        # 1. 数量倍数
        if order.shares % self.lot_size != 0:
            return f"数量必须是{self.lot_size}的整数倍"

        # 2. 买入: 资金检查
        if order.action == 'BUY':
            est_cost = order.shares * order.price
            est_fee = self._estimate_fee(order.code, 'BUY', order.price, order.shares)
            total = est_cost + est_fee
            if total > self.account.cash:
                return (f"资金不足: 需 {total:,.0f}, "
                        f"可用 {self.account.cash:,.0f} "
                        f"(成交额{est_cost:,.0f} + 预估费{est_fee:.2f})")

        # 3. 卖出: 持仓 + T+1检查
        if order.action == 'SELL':
            avail = self.account.available_shares(order.code)
            if avail < order.shares:
                pos = self.account.positions.get(order.code)
                locked = pos.locked_shares if pos else 0
                return (f"可卖不足: 需{order.shares}股, "
                        f"可用{avail}股 (总{pos.shares if pos else 0}股, "
                        f"T+1锁定{locked}股)")

        return None  # 通过

    def _estimate_fee(self, code: str, action: str, price: float, shares: int) -> float:
        """预估手续费 (用于资金校验)"""
        try:
            from trade_cost import calc_cost
            return calc_cost(code, action, price, shares)['total_fee']
        except ImportError:
            # 回退: 简单估算
            t = price * shares
            comm = max(t * 0.00025, 5.0)
            trans = t * 0.00001 if str(code)[0] in ('5', '6') else 0
            return comm + trans

    # ── 模拟成交 ──

    def _execute_sim(self, order: Order):
        """模拟模式: 直接成交"""
        from trade_cost import calc_cost

        cost = calc_cost(order.code, order.action, order.price, order.shares)

        if order.action == 'BUY':
            self.account.cash -= cost['net_amount']
            pos = self.account.positions.get(order.code)
            if pos:
                old_total = pos.avg_cost * pos.shares
                pos.shares += order.shares
                pos.avg_cost = (old_total + cost['net_amount']) / pos.shares  # 用净支出(含手续费)
            else:
                self.account.positions[order.code] = Position(
                    code=order.code,
                    shares=order.shares,
                    locked_shares=order.shares,  # T+1
                    avg_cost=order.price,
                    buy_date=datetime.now(BJ).strftime('%Y-%m-%d'),
                )
        else:  # SELL
            self.account.cash += cost['net_amount']
            pos = self.account.positions[order.code]
            pos.shares -= order.shares
            pos.locked_shares = max(0, pos.locked_shares - order.shares)
            if pos.shares < self.lot_size:
                del self.account.positions[order.code]

        self.account.trade_count += 1
        self.account.total_fees += cost['total_fee']

        # 成交记录
        order.status = 'FILLED'
        order.filled_shares = order.shares
        order.filled_price = order.price
        order.fill_time = datetime.now(BJ).strftime('%Y-%m-%d %H:%M:%S')

        self._trade_id += 1
        trade = Trade(
            trade_id=self._trade_id,
            order_id=order.order_id,
            code=order.code,
            action=order.action,
            price=order.price,
            shares=order.shares,
            turnover=cost['turnover'],
            commission=cost['commission'],
            transfer_fee=cost['transfer_fee'],
            stamp_duty=cost['stamp_duty'],
            total_fee=cost['total_fee'],
            net_amount=cost['net_amount'],
            time=order.fill_time,
        )
        self.trades.append(trade)

    # ── 实盘桩 ──

    def _execute_live(self, order: Order):
        """实盘模式桩 (需自行接入券商API)

        ⚠️ 警告: 回测代码绝不能调用此方法!
        """
        order.status = 'PENDING'
        order.reject_reason = '实盘模式需接入券商API, 当前为桩'

    # ── T+1 解锁 ──

    def unlock_t1(self):
        """解锁所有T+1持仓 (每天调用一次)"""
        for pos in self.account.positions.values():
            pos.locked_shares = 0

    # ── 批量委托 ──

    def submit_batch(self, signals: list[dict]) -> list[Order]:
        """批量提交委托

        Args:
            signals: [{'code': '510300', 'action': 'BUY', 'price': 4.70, 'shares': 1700}, ...]

        Returns:
            list[Order]: 委托结果
        """
        results = []
        for sig in signals:
            order = self.submit(
                code=sig.get('code', ''),
                action=sig.get('action', 'BUY'),
                price=sig.get('price', 0),
                shares=sig.get('shares', 0),
                price_type=sig.get('price_type', 'LIMIT'),
                tag=sig.get('tag', ''),
            )
            results.append(order)
        return results

    # ── 摘要 ──

    def summary(self) -> dict:
        """账户摘要"""
        return {
            'mode': self.mode,
            'initial_cash': self.account.initial_cash,
            'cash': round(self.account.cash, 2),
            'positions': {
                code: {
                    'shares': pos.shares,
                    'locked': pos.locked_shares,
                    'avg_cost': round(pos.avg_cost, 3),
                }
                for code, pos in self.account.positions.items()
            },
            'trade_count': self.account.trade_count,
            'total_fees': round(self.account.total_fees, 2),
            'pending_orders': sum(1 for o in self.orders if o.status == 'PENDING'),
            'filled_orders': sum(1 for o in self.orders if o.status == 'FILLED'),
            'rejected_orders': sum(1 for o in self.orders if o.status == 'REJECTED'),
        }

    def print_summary(self, prices: dict = None):
        """打印账户摘要 (中文)"""
        s = self.summary()
        print(f"\n{'='*50}")
        print(f"  交易账户摘要 [{self.mode}模式]")
        print(f"{'='*50}")
        print(f"  初始资金:   {s['initial_cash']:,.0f}")
        print(f"  当前现金:   {s['cash']:,.2f}")

        if prices and s['positions']:
            hv = sum(pos['shares'] * prices.get(code, 0)
                     for code, pos in s['positions'].items())
            nw = s['cash'] + hv
            ret = (nw / s['initial_cash'] - 1) * 100
            print(f"  持仓市值:   {hv:,.2f}")
            print(f"  当前净值:   {nw:,.2f} ({ret:+.2f}%)")

        print(f"  交易次数:   {s['trade_count']}")
        print(f"  累计费用:   {s['total_fees']:.2f}")
        print(f"  成交/拒绝:  {s['filled_orders']}/{s['rejected_orders']}")

        if s['positions']:
            print(f"\n  [当前持仓]")
            for code, pos in s['positions'].items():
                p = prices.get(code, 0) if prices else 0
                val = pos['shares'] * p
                pnl = (p / pos['avg_cost'] - 1) * 100 if pos['avg_cost'] > 0 else 0
                lock = f" (T+1锁定{pos['locked']}股)" if pos['locked'] > 0 else ''
                print(f"    {code}: {pos['shares']}股 "
                      f"成本{pos['avg_cost']:.3f} "
                      f"现价{p:.3f} "
                      f"市值{val:,.0f} "
                      f"盈亏{pnl:+.1f}%{lock}")
        else:
            print(f"\n  [空仓]")


# ═══════════════════════════════════════
# 自测
# ═══════════════════════════════════════

if __name__ == '__main__':
    print("=== TradeExecutor 自测 ===\n")

    # ── 初始化 ──
    executor = TradeExecutor(mode='SIM', initial_cash=8000)

    # 1. 正常买入
    print("[1] 正常买入: 沪深300ETF 1700股 @4.70")
    r1 = executor.submit('510300', 'BUY', 4.70, 1700, tag='信号-日线金叉')
    print(f"  状态: {r1.status}")
    print(f"  成交: {r1.filled_shares}股 @{r1.filled_price}")

    # 2. 资金不足
    print("\n[2] 资金不足: 试图买5000股 (需 ¥23,500)")
    r2 = executor.submit('510300', 'BUY', 4.70, 5000)
    print(f"  状态: {r2.status}")
    print(f"  原因: {r2.reject_reason}")

    # 3. 卖出 (T+1锁定)
    print("\n[3] 卖出T+1锁定股: 1700股 (刚买入, 应被拒绝)")
    r3 = executor.submit('510300', 'SELL', 4.80, 1700)
    print(f"  状态: {r3.status}")
    print(f"  原因: {r3.reject_reason}")

    # 4. 解锁T+1后再卖
    print("\n[4] 解锁T+1后卖出")
    executor.unlock_t1()
    r4 = executor.submit('510300', 'SELL', 4.80, 1700)
    print(f"  状态: {r4.status}")
    print(f"  成交: {r4.filled_shares}股 @{r4.filled_price}")

    # 5. 批量委托
    print("\n[5] 批量委托")
    signals = [
        {'code': '159915', 'action': 'BUY', 'price': 2.50, 'shares': 3200, 'tag': '创业板信号'},
        {'code': '510300', 'action': 'BUY', 'price': 4.65, 'shares': 100, 'tag': '加仓'},
    ]
    results = executor.submit_batch(signals)
    for r in results:
        print(f"  {r.code}: {r.status} {r.filled_shares}股" +
              (f" 拒绝: {r.reject_reason}" if r.status == 'REJECTED' else ''))

    # 6. 数量自动取整
    print("\n[6] 数量取整: 105股 → 100股")
    r6 = executor.submit('159915', 'BUY', 2.50, 105)
    print(f"  委托: {r6.shares}股 (原始105股取整)")

    # 7. 摘要
    executor.print_summary(prices={'510300': 4.70, '159915': 2.50})

    # 8. 成交记录
    print(f"\n[8] 成交记录 ({len(executor.trades)}笔)")
    for t in executor.trades:
        print(f"  #{t.trade_id} {t.action} {t.code} {t.shares}股 @{t.price:.3f} "
              f"净额{t.net_amount:,.0f} 费{t.total_fee:.2f}")

    print("\n=== 自测完成 ===")
