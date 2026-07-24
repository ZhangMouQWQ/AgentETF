#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
回测引擎模块
============
职责: 逐日模拟撮合、维护虚拟账户、计算净值曲线和绩效指标

特性:
  - T+1 锁仓: 当日买入次日才能卖出
  - 涨跌停保护: 涨停不买、跌停不卖
  - 滑点模拟: 0.15% 买卖双向滑点
  - 手续费: 固定 ¥10/笔
  - 严格无未来函数: 每步仅用当前及之前数据
  - 前复权一致: 净值计算与复权逻辑对齐

用法:
    from backtest_engine import BacktestEngine
    
    engine = BacktestEngine(initial_cash=8000)
    result = engine.run(daily_data, signals)
    engine.report()
"""

import numpy as np
import pandas as pd
from datetime import datetime, timezone, timedelta

try:
    from trade_cost import calc_cost as _calc_trade_cost
except ImportError:
    _calc_trade_cost = None

BJ = timezone(timedelta(hours=8))

# ═══════════════════════════════════
# 默认参数
# ═══════════════════════════════════

DEFAULT_CONFIG = {
    'initial_cash': 8000.0,
    'fee_per_trade': 10.0,        # 每笔固定手续费
    'slippage_pct': 0.0015,       # 滑点 0.15%
    'lot_size': 100,              # 最小交易单位(股)
    'min_trade_amt': 500.0,       # 最低交易金额
    'limit_up_threshold': 0.098,  # 涨停阈值 (9.8% 以上视为涨停)
    'limit_down_threshold': -0.098, # 跌停阈值
    'max_position_pct': 1.0,      # 单只最大仓位比
}


# ═══════════════════════════════════
# 回测引擎
# ═══════════════════════════════════

class BacktestEngine:
    """严格无未来函数的逐日回测引擎

    用法:
        engine = BacktestEngine(initial_cash=8000)
        engine.run(daily_data, signals)
        engine.report()
    """

    def __init__(self, config=None):
        """
        Args:
            config: dict, 覆盖默认参数
        """
        self.cfg = {**DEFAULT_CONFIG, **(config or {})}
        self.reset()

    def reset(self):
        """重置回测状态"""
        self.cash = self.cfg['initial_cash']
        self.holdings = {}        # {code: {shares, locked_shares, avg_cost, buy_date}}
        self.trades = []          # 交易记录
        self.daily_nav = []       # 每日净值
        self.fees_total = 0.0
        self.trade_count = 0
        self._date_idx = 0

    # ── 账户查询 ──

    def net_worth(self, prices):
        """当前净值 = 现金 + 持仓市值"""
        hv = 0.0
        for code, h in self.holdings.items():
            p = prices.get(code, 0)
            if p > 0:
                hv += h['shares'] * p
        return self.cash + hv

    def available_shares(self, code):
        """可卖股数 (排除 T+1 锁定)"""
        h = self.holdings.get(code)
        if not h:
            return 0
        return h['shares'] - h.get('locked_shares', 0)

    # ── T+1 管理 ──

    def _lock_shares(self, code, shares, date):
        """锁定买入股份 (T+1)"""
        if code not in self.holdings:
            return
        self.holdings[code]['locked_shares'] = self.holdings[code].get('locked_shares', 0) + shares
        self.holdings[code]['buy_date'] = date

    def _unlock_all(self, current_date):
        """解锁所有到期持仓"""
        for h in self.holdings.values():
            buy_date = h.get('buy_date', '')
            if buy_date and str(current_date) > str(buy_date):
                h['locked_shares'] = 0

    # ── 涨跌停检测 ──

    def _is_limit_up(self, row):
        """检测涨停 (基于当日数据)"""
        o = row.get('open', 0)
        c = row.get('close', 0)
        if o <= 0 or c <= 0:
            return False
        return (c / o - 1) >= self.cfg['limit_up_threshold']

    def _is_limit_down(self, row):
        """检测跌停"""
        o = row.get('open', 0)
        c = row.get('close', 0)
        if o <= 0 or c <= 0:
            return False
        return (c / o - 1) <= self.cfg['limit_down_threshold']

    # ── 滑点 ──

    def _apply_slippage(self, price, is_buy):
        """应用滑点: 买入略高, 卖出略低"""
        if is_buy:
            return price * (1 + self.cfg['slippage_pct'])
        else:
            return price * (1 - self.cfg['slippage_pct'])

    def _get_fee(self, code, action, price, shares):
        """计算单笔交易费用 (优先真实费率, 回退固定费率)"""
        if _calc_trade_cost:
            r = _calc_trade_cost(code, action, price, shares)
            return r['total_fee'], r['net_amount']
        else:
            t = price * shares
            f = self.cfg['fee_per_trade']
            return (f, t + f) if action.upper() == 'BUY' else (f, t - f)

    # ── 撮合执行 ──

    def _execute_buy(self, code, price, shares, date, reason=''):
        """执行买入 (含滑点+真实手续费+资金检查)"""
        price = self._apply_slippage(price, is_buy=True)
        total_fee, net_amount = self._get_fee(code, 'BUY', price, shares)
        cost = net_amount

        # 手数约束
        shares = (shares // self.cfg['lot_size']) * self.cfg['lot_size']
        if shares < self.cfg['lot_size']:
            return None, f"手数不足 (<{self.cfg['lot_size']}股)"

        # 金额约束
        if cost < self.cfg['min_trade_amt']:
            return None, f"金额不足 (¥{cost:.0f} < ¥{self.cfg['min_trade_amt']})"

        # 资金约束
        if cost > self.cash:
            max_shares = int((self.cash - total_fee) / price / self.cfg['lot_size']) * self.cfg['lot_size']
            if max_shares < self.cfg['lot_size']:
                return None, f"资金不足 (需{cost:.0f}, 现金{self.cash:.0f})"
            shares = max_shares
            total_fee, net_amount = self._get_fee(code, 'BUY', price, shares)
            cost = net_amount

        # 执行
        self.cash -= cost
        self.fees_total += total_fee
        self.trade_count += 1

        if code in self.holdings:
            h = self.holdings[code]
            total_sh = h['shares'] + shares
            old_cost = h['avg_cost'] * h['shares']
            h['shares'] = total_sh
            h['avg_cost'] = (old_cost + shares * price) / total_sh if total_sh > 0 else 0
        else:
            self.holdings[code] = {
                'shares': shares,
                'avg_cost': price,
                'locked_shares': 0,
                'buy_date': date,
            }

        # T+1 锁定
        self._lock_shares(code, shares, date)

        record = {
            'date': str(date)[:10],
            'code': code,
            'action': 'BUY',
            'shares': shares,
            'price': round(price, 3),
            'cost': round(cost, 2),
            'fee': round(total_fee, 2),
            'reason': reason,
        }
        self.trades.append(record)
        return record, f"BUY {shares}股 @{price:.3f}"

    def _execute_sell(self, code, price, shares, date, reason=''):
        """执行卖出 (含滑点+真实手续费+可卖检查+涨跌停检查)"""
        price = self._apply_slippage(price, is_buy=False)
        total_fee, net_amount = self._get_fee(code, 'SELL', price, shares)
        revenue = net_amount

        # 可卖检查
        available = self.available_shares(code)
        if shares > available:
            shares = (available // self.cfg['lot_size']) * self.cfg['lot_size']
            if shares < self.cfg['lot_size']:
                return None, f"可卖不足 ({available}股可用)"

        shares = (shares // self.cfg['lot_size']) * self.cfg['lot_size']
        if shares < self.cfg['lot_size']:
            return None, "手数不足"

        # 金额约束 (非清仓时检查)
        if revenue < self.cfg['min_trade_amt'] and self.holdings[code]['shares'] > shares:
            return None, f"金额不足 (¥{revenue:.0f})"

        # 执行
        self.cash += revenue
        self.fees_total += total_fee
        self.trade_count += 1

        h = self.holdings[code]
        h['shares'] -= shares
        h['locked_shares'] = max(0, h.get('locked_shares', 0) - shares)
        if h['shares'] < self.cfg['lot_size']:
            del self.holdings[code]

        record = {
            'date': str(date)[:10],
            'code': code,
            'action': 'SELL',
            'shares': shares,
            'price': round(price, 3),
            'revenue': round(revenue, 2),
            'fee': round(total_fee, 2),
            'reason': reason,
        }
        self.trades.append(record)
        return record, f"SELL {shares}股 @{price:.3f}"

    # ── 主运行循环 ──

    def run(self, price_matrix, signals, extra_data=None, verbose=True):
        """逐日运行回测

        Args:
            price_matrix: DataFrame, index=date, columns=ETF名称, values=收盘价(前复权)
            signals: dict, {date_str: {code: signal_dict}}
                     或 list of {date, code, action}
            extra_data: dict, {ETF名称: DataFrame} 含 open/high/low (用于涨跌停检测)
            verbose: 是否打印进度

        Returns:
            dict: 回测结果汇总
        """
        self.reset()
        dates = price_matrix.index.tolist()
        etf_names = price_matrix.columns.tolist()
        total_days = len(dates)

        # 信号标准化: 统一为 {date_str: {code: action}}
        signal_map = self._normalize_signals(signals)

        for i, date in enumerate(dates):
            self._date_idx = i
            date_str = str(date)[:10]

            # ── T+1 解锁 ──
            self._unlock_all(date_str)

            # ── 获取当日价格 ──
            row = price_matrix.iloc[i]
            prices = {n: float(row[n]) for n in etf_names if not pd.isna(row[n])}

            # ── 获取当日信号 ──
            day_signals = signal_map.get(date_str, {})

            # ── 获取当日 extra 数据 (用于涨跌停检测) ──
            extra_row = {}
            if extra_data:
                for name, df in extra_data.items():
                    if date in df.index:
                        extra_row[name] = df.loc[date]

            # ── 卖出: 不在信号中 或 信号为 SELL ──
            for code in list(self.holdings.keys()):
                action = day_signals.get(code, 'SELL')  # 无信号默认清仓
                if action == 'BUY':
                    continue  # 继续持有

                name = self._code_to_name(code, etf_names, price_matrix)
                price = prices.get(name, 0)
                if price <= 0:
                    continue

                # 跌停检查
                ex = extra_row.get(name)
                if ex is not None and self._is_limit_down(ex):
                    continue  # 跌停不卖

                shares = self.available_shares(code)
                if shares >= self.cfg['lot_size']:
                    result, msg = self._execute_sell(code, price, shares, date_str, f'信号{action}')
                    if verbose and result:
                        print(f"  [{date_str}] SELL {name} {msg}")

            # ── 买入: 信号为 BUY ──
            buy_candidates = []
            for code, action in day_signals.items():
                if action != 'BUY':
                    continue
                name = self._code_to_name(code, etf_names, price_matrix)
                price = prices.get(name, 0)
                if price <= 0:
                    continue

                # 涨停检查
                ex = extra_row.get(name)
                if ex is not None and self._is_limit_up(ex):
                    continue  # 涨停不买

                buy_candidates.append((code, name, price))

            # 等权分配资金
            if buy_candidates:
                nw = self.net_worth(prices)
                budget_per = (self.cash * 0.95) / len(buy_candidates)  # 留5%缓冲

                for code, name, price in buy_candidates:
                    # 检查是否已持有
                    if code in self.holdings:
                        cur_val = self.holdings[code]['shares'] * price
                        if cur_val >= budget_per * 0.8:
                            continue  # 已接近目标

                    target_shares = int(budget_per / price / self.cfg['lot_size']) * self.cfg['lot_size']
                    if target_shares >= self.cfg['lot_size']:
                        result, msg = self._execute_buy(code, price, target_shares, date_str, '信号BUY')
                        if verbose and result:
                            print(f"  [{date_str}] BUY  {name} {msg}")

            # ── 记录每日净值 ──
            nw = self.net_worth(prices)
            hv = nw - self.cash
            self.daily_nav.append({
                'date': date_str,
                'cash': round(self.cash, 2),
                'holdings_value': round(hv, 2),
                'net_worth': round(nw, 2),
                'positions': len(self.holdings),
                'daily_return': 0.0,  # 稍后计算
            })

        # ── 计算每日收益率 ──
        for i in range(len(self.daily_nav)):
            if i == 0:
                self.daily_nav[i]['daily_return'] = round(
                    (self.daily_nav[i]['net_worth'] / self.cfg['initial_cash'] - 1) * 100, 2)
            else:
                prev = self.daily_nav[i - 1]['net_worth']
                if prev > 0:
                    self.daily_nav[i]['daily_return'] = round(
                        (self.daily_nav[i]['net_worth'] / prev - 1) * 100, 2)

        return self._summary()

    def _normalize_signals(self, signals):
        """标准化信号格式

        支持输入:
          - dict {date: {code: action}}
          - dict {date: {code: {'final_action': 'BUY'}}}
          - list [{date, code, action}]
        """
        result = {}
        if isinstance(signals, dict):
            for date_str, day_data in signals.items():
                result[date_str] = {}
                for code, val in day_data.items():
                    if isinstance(val, dict):
                        result[date_str][code] = val.get('final_action', 'HOLD')
                    else:
                        result[date_str][code] = val
        elif isinstance(signals, list):
            for s in signals:
                d = s.get('date', '')
                result.setdefault(d, {})[s.get('code', '')] = s.get('action', 'HOLD')
        return result

    def _code_to_name(self, code, etf_names, price_matrix):
        """代码 → 名称 (简单映射)"""
        # ETF名称列表可能不含代码, 尝试匹配
        for name in etf_names:
            if code in name or name in code:
                return name
        return code

    # ── 绩效统计 ──

    def _summary(self):
        """计算回测绩效指标"""
        if not self.daily_nav:
            return {'error': '无回测数据'}

        initial = self.cfg['initial_cash']
        final_nw = self.daily_nav[-1]['net_worth']
        days = len(self.daily_nav)

        returns = np.array([d['daily_return'] for d in self.daily_nav]) / 100.0

        # 总回报
        total_ret = (final_nw / initial - 1) * 100

        # 年化回报
        ann_ret = ((final_nw / initial) ** (252 / max(days, 1)) - 1) * 100

        # 最大回撤
        peak = initial
        max_dd = 0.0
        max_dd_date = ''
        for d in self.daily_nav:
            nw = d['net_worth']
            peak = max(peak, nw)
            dd = (peak - nw) / peak * 100
            if dd > max_dd:
                max_dd = dd
                max_dd_date = d['date']

        # Sharpe
        avg_daily = np.mean(returns) if len(returns) > 1 else 0
        std_daily = np.std(returns) if len(returns) > 1 else 1e-9
        sharpe = (avg_daily / std_daily * np.sqrt(252)) if std_daily > 0 else 0

        # 胜率
        win_days = sum(1 for r in returns if r > 0)
        win_rate = win_days / max(len(returns), 1) * 100

        # 最大连续亏损
        max_consec_loss = 0
        consec = 0
        for r in returns:
            if r < 0:
                consec += 1
                max_consec_loss = max(max_consec_loss, consec)
            else:
                consec = 0

        # 盈亏比
        win_returns = [r for r in returns if r > 0]
        loss_returns = [abs(r) for r in returns if r < 0]
        avg_win = np.mean(win_returns) * 100 if win_returns else 0
        avg_loss = np.mean(loss_returns) * 100 if loss_returns else 0
        profit_factor = avg_win / avg_loss if avg_loss > 0 else 999

        return {
            'initial_cash': initial,
            'final_net_worth': round(final_nw, 2),
            'total_return_pct': round(total_ret, 2),
            'annual_return_pct': round(ann_ret, 2),
            'max_drawdown_pct': round(max_dd, 2),
            'max_drawdown_date': max_dd_date,
            'sharpe_ratio': round(sharpe, 2),
            'win_rate_pct': round(win_rate, 1),
            'max_consecutive_losses': max_consec_loss,
            'avg_win_pct': round(avg_win, 2),
            'avg_loss_pct': round(avg_loss, 2),
            'profit_factor': round(profit_factor, 2),
            'total_trades': self.trade_count,
            'total_fees': round(self.fees_total, 2),
            'trading_days': days,
        }

    # ── 报告输出 ──

    def report(self):
        """打印回测报告"""
        s = self._summary()
        if 'error' in s:
            print(f"  {s['error']}")
            return

        print(f"\n{'='*55}")
        print(f"  回测绩效报告")
        print(f"{'='*55}")
        print(f"  初始资金:    ¥{s['initial_cash']:,.0f}")
        print(f"  最终净值:    ¥{s['final_net_worth']:,.2f}")
        print(f"  总回报:      {s['total_return_pct']:+.2f}%")
        print(f"  年化回报:    {s['annual_return_pct']:+.2f}%")
        print(f"  最大回撤:    {s['max_drawdown_pct']:.2f}% (日期: {s['max_drawdown_date']})")
        print(f"  Sharpe:      {s['sharpe_ratio']:.2f}")
        print(f"  胜率:        {s['win_rate_pct']:.1f}%")
        print(f"  盈亏比:      {s['profit_factor']:.2f}")
        print(f"  最大连亏:    {s['max_consecutive_losses']}天")
        print(f"  交易次数:    {s['total_trades']}")
        print(f"  手续费:      ¥{s['total_fees']:,.0f}")
        print(f"  交易天数:    {s['trading_days']}")

    def get_trades_df(self):
        """获取交易记录 DataFrame"""
        return pd.DataFrame(self.trades) if self.trades else pd.DataFrame()

    def get_nav_df(self):
        """获取每日净值 DataFrame"""
        return pd.DataFrame(self.daily_nav) if self.daily_nav else pd.DataFrame()


# ═══════════════════════════════════
# 自测
# ═══════════════════════════════════

if __name__ == '__main__':
    print("=== BacktestEngine 自测 ===\n")

    # 生成模拟数据
    np.random.seed(42)
    dates = pd.date_range('2025-06-01', periods=60, freq='B')
    date_strs = [d.strftime('%Y-%m-%d') for d in dates]

    # 模拟两只ETF的价格
    price1 = pd.Series(2.0 + np.cumsum(np.random.randn(60) * 0.03), index=dates)
    price2 = pd.Series(1.5 + np.cumsum(np.random.randn(60) * 0.02), index=dates)
    price_matrix = pd.DataFrame({
        'ETF_A': price1,
        'ETF_B': price2,
    })

    # 模拟信号: 每10天切换一次
    signals = {}
    for i, ds in enumerate(date_strs):
        if i < 10:
            signals[ds] = {'ETF_A': 'BUY'}
        elif i < 20:
            signals[ds] = {'ETF_A': 'SELL', 'ETF_B': 'BUY'}
        elif i < 30:
            signals[ds] = {'ETF_B': 'SELL'}
        elif i < 40:
            signals[ds] = {'ETF_A': 'BUY'}
        elif i < 50:
            signals[ds] = {'ETF_A': 'SELL', 'ETF_B': 'BUY'}
        else:
            signals[ds] = {'ETF_B': 'SELL'}

    # ── 回测 ──
    engine = BacktestEngine({'initial_cash': 8000, 'slippage_pct': 0.001})
    result = engine.run(price_matrix, signals, verbose=False)
    engine.report()

    # ── 最近交易 ──
    print(f"\n  [最近5笔交易]")
    for t in engine.trades[-5:]:
        print(f"    {t['date']} {t['action']:4s} {t['code']} {t['shares']}股 @{t['price']:.3f} ¥{t.get('cost',t.get('revenue',0)):.0f}")

    # ── 净值曲线前5天 ──
    print(f"\n  [净值曲线前5天]")
    for d in engine.daily_nav[:5]:
        print(f"    {d['date']} 净值¥{d['net_worth']:,.2f} 日收益{d['daily_return']:+.2f}% 持仓{d['positions']}只")

    print("\n=== 自测完成 ===")
