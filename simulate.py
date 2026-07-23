#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
V5 单ETF策略 · 260日实盘模拟（严格无数据泄露）
=========================================
规则:
  - 每天仅使用截至当日的已知数据（无未来函数）
  - T+1 锁仓: T日买入 → T+1日解锁
  - 买入价: T+1日 (开盘+最低)/2 (回调) 或 开盘价 (无回调)
  - 卖出价: T+1日 (最高+收盘)/2
  - ¥10/笔手续费, 100股整手, 最低 ¥500 交易
  - ¥8,000 初始本金
"""

import os
import sys
import json
import time
from datetime import datetime, timezone, timedelta

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from strategy import (
    ETF_POOL, ETF_SECTOR,
    fetch_daily_data_cached, calc_metrics,
    market_timing_v5, build_target_v5,
    MOM_LONG,
)
from data_cache import load_from_cache

BJ = timezone(timedelta(hours=8))
DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data')
RESULT_FILE = os.path.join(DATA_DIR, 'sim_result.json')

INIT_CASH = 8000.0
FEE = 10.0
LOT = 100
MIN_TRADE = 500.0
MIN_WEIGHT_DELTA = 0.20


# ═══════════════════════════════════════════════
# 帮助函数
# ═══════════════════════════════════════════════

def _name_to_code(name):
    for c, (s, n) in ETF_POOL.items():
        if n == name:
            return c
    return None


def _code_to_name(code):
    for c, (s, n) in ETF_POOL.items():
        if c == code:
            return n
    return None


# ═══════════════════════════════════════════════
# 核心: 逐日模拟器
# ═══════════════════════════════════════════════

class Simulator:
    def __init__(self, price, etf_info, extra_history):
        self.price = price            # close DataFrame
        self.etf_info = etf_info      # {name: code}
        self.extra = extra_history    # {name: DataFrame with open/high/low/volume/amount}

        # 交易状态
        self.cash = INIT_CASH
        self.holdings = {}            # {code: {shares, avg_cost, locked_shares}}
        self.fees = 0
        self.trades = 0

        # 记录
        self.daily_log = []

        # 获取所有日期
        self.dates = price.index.tolist()
        self.start_idx = MOM_LONG + 5   # 需要足够历史数据

    def _slice_data(self, end_idx):
        """切片: 截至 end_idx (含) 的数据, 无未来泄露"""
        return self.price.iloc[:end_idx + 1]

    def _slice_extra(self, end_date):
        """切片 extra 数据: 截至 end_date"""
        sliced = {}
        for name, df in self.extra.items():
            mask = df.index <= end_date
            sub = df.loc[mask]
            if len(sub) > 0:
                sliced[name] = sub
        return sliced

    def net_worth(self, close_prices):
        """当前净值"""
        hv = sum(h['shares'] * close_prices.get(_code_to_name(c), 0)
                 for c, h in self.holdings.items())
        return self.cash + hv

    def unlock(self):
        """解锁所有 T+1 持仓"""
        for h in self.holdings.values():
            h['locked_shares'] = 0

    def get_close_map(self, date_idx):
        """获取指定日期的收盘价映射 {name: close}"""
        row = self.price.iloc[date_idx]
        return {name: float(row[name]) for name in row.index
                if not pd.isna(row[name])}

    def get_ohlc(self, name, date_idx):
        """获取指定日期的 OHLC 元组 (open, high, low, close)"""
        close = float(self.price.iloc[date_idx][name]) if name in self.price.columns else 0
        o = h = l = close
        if name in self.extra:
            df = self.extra[name]
            d = self.price.index[date_idx]
            if d in df.index:
                row = df.loc[d]
                o = float(row['open']) if 'open' in row and not pd.isna(row['open']) else close
                h = float(row['high']) if 'high' in row and not pd.isna(row['high']) else close
                l = float(row['low']) if 'low' in row and not pd.isna(row['low']) else close
        return (o, h, l, close)

    def execute_buy(self, code, name, buy_shares, ohlc_next, weight, date_str):
        """T+1 执行买入 (使用下一日 OHLC)"""
        op, hi, lo, cl = ohlc_next
        if lo < op:
            buy_price = round((op + lo) / 2, 3)
            tag = '回调'
        else:
            buy_price = round(op, 3)
            tag = '开盘'

        cost = buy_shares * buy_price + FEE
        nw = self.net_worth(self.get_close_map(self.current_day))

        # 金额检查
        if cost < MIN_TRADE:
            return False, f"¥{cost:.0f}<¥{MIN_TRADE}"

        # 权重变化检查
        if code in self.holdings:
            cur_shares = self.holdings[code]['shares']
            cur_val = cur_shares * cl
            cur_w = cur_val / nw if nw > 0 else 0
            if abs(weight - cur_w) < MIN_WEIGHT_DELTA:
                return False, f"权重微调 {cur_w:.0%}→{weight:.0%}"

        # 资金检查
        if cost > self.cash - FEE:
            affordable = int((self.cash - FEE * 2) / buy_price / LOT) * LOT
            if affordable < LOT:
                return False, f"资金不足(需¥{cost:.0f}, 有¥{self.cash:.0f})"
            buy_shares = affordable
            cost = buy_shares * buy_price + FEE
            tag += '(缩)'

        self.cash -= cost
        if code in self.holdings:
            h = self.holdings[code]
            total_sh = h['shares'] + buy_shares
            h['shares'] = total_sh
        else:
            self.holdings[code] = {'shares': buy_shares, 'avg_cost': buy_price, 'locked_shares': buy_shares}
        self.fees += FEE
        self.trades += 1
        return True, f"BUY {buy_shares}股 @{buy_price:.3f} ¥{cost:.0f} ({tag})"

    def execute_sell(self, code, name, sell_shares, ohlc_next, is_exit=False):
        """T+1 执行卖出"""
        op, hi, lo, cl = ohlc_next
        sell_price = round((hi + cl) / 2, 3)
        revenue = sell_shares * sell_price - FEE

        if not is_exit and revenue < MIN_TRADE:
            return False, f"¥{revenue:.0f}<¥{MIN_TRADE}"
        if revenue <= 0:
            return False, "收入≤0"

        self.cash += revenue
        remaining = self.holdings[code]['shares'] - sell_shares
        if remaining < LOT:
            del self.holdings[code]
        else:
            self.holdings[code]['shares'] = remaining
        self.fees += FEE
        self.trades += 1
        return True, f"SELL {sell_shares}股 @{sell_price:.3f} ¥{revenue:.0f}"

    def run(self):
        """主循环: 逐日模拟"""
        total_days = len(self.dates) - self.start_idx

        for i in range(self.start_idx, len(self.dates) - 1):
            self.current_day = i
            td = self.dates[i]      # T日 (决策日)
            tm = self.dates[i + 1]  # T+1日 (执行日)

            # ── 1. 解锁 T+1 ──
            self.unlock()

            # ── 2. 计算指标 (仅用截至T日的数据) ──
            sliced_price = self._slice_data(i)
            sliced_extra = self._slice_extra(td)
            sliced_etf_info = {n: c for n, c in self.etf_info.items()
                               if n in sliced_price.columns}

            metrics = calc_metrics(sliced_price, sliced_etf_info, ETF_SECTOR,
                                   sliced_extra)

            if not metrics:
                self._log_day(td, tm, 0, [], [], '无指标')
                continue

            # ── 3. V5 择时 ──
            ratio = market_timing_v5(metrics)
            hs300_mom = metrics.get('沪深300ETF', {}).get('mom_long', -999)

            # ── 4. V5 选股 ──
            target, _ = build_target_v5(metrics, ratio)

            # ── 5. 执行交易 (T+1日价格) ──
            close_next = self.get_close_map(i + 1)
            nw = self.net_worth(close_next)
            orders = []

            # 目标映射 (仅 weight > 0)
            target_map = {}
            for t in target:
                code = _name_to_code(t['name'])
                if code and t['weight'] > 0:
                    target_map[code] = t

            # 卖出
            for code in list(self.holdings.keys()):
                h = self.holdings[code]
                sellable = h['shares'] - h.get('locked_shares', 0)
                if sellable < LOT:
                    continue
                name = _code_to_name(code)
                ohlc = self.get_ohlc(name, i + 1)

                if code not in target_map:
                    ok, reason = self.execute_sell(code, name, sellable, ohlc, is_exit=True)
                    if ok:
                        orders.append(f"[{tm}] 🔴 {reason}")
                else:
                    tw = target_map[code]['weight']
                    target_sh = int(nw * tw / ohlc[3] / LOT) * LOT
                    cur_sh = h['shares']
                    if cur_sh > target_sh + LOT:
                        sell_sh = min(((cur_sh - target_sh) // LOT) * LOT, sellable)
                        ok, reason = self.execute_sell(code, name, sell_sh, ohlc)
                        if ok:
                            orders.append(f"[{tm}] 🔴 {reason}")

            # 买入
            nw = self.net_worth(close_next)
            buy_list = []
            for code, t in target_map.items():
                name = t['name']
                ohlc = self.get_ohlc(name, i + 1)
                cl = ohlc[3]
                if cl <= 0:
                    continue
                target_sh = int(nw * t['weight'] / cl / LOT) * LOT
                cur_sh = self.holdings.get(code, {}).get('shares', 0)
                buy_sh = target_sh - cur_sh
                if buy_sh >= LOT:
                    buy_list.append((code, name, buy_sh, t['weight'], ohlc))

            buy_list.sort(key=lambda x: x[3], reverse=True)
            for code, name, buy_sh, weight, ohlc in buy_list:
                ok, reason = self.execute_buy(code, name, buy_sh, ohlc, weight, tm)
                if ok:
                    orders.append(f"[{tm}] 🟢 {reason}")

            self._log_day(td, tm, ratio, target, orders,
                          f"HS300={hs300_mom:+.1f}% ratio={ratio}")

            # 进度
            day_num = i - self.start_idx + 1
            if day_num % 50 == 0 or day_num == total_days:
                nw = self.net_worth(close_next)
                print(f"  [{day_num:3d}/{total_days}] {td} → {tm} "
                      f"净值 ¥{nw:,.0f} ({(nw/INIT_CASH-1)*100:+.1f}%) "
                      f"持仓 {len(self.holdings)}只 交易{self.trades}笔")

        return self._finalize()

    def _log_day(self, td, tm, ratio, target, orders, reason):
        close = self.get_close_map(self.current_day)
        nw = self.net_worth(close)
        holding_names = [_code_to_name(c) for c in self.holdings]
        holding_details = {}
        for c, h in self.holdings.items():
            name = _code_to_name(c)
            val = h['shares'] * close.get(name, 0)
            holding_details[name] = {
                'shares': h['shares'],
                'value': round(val, 2),
                'locked': h.get('locked_shares', 0),
            }
        target_names = [t['name'] for t in target if t['weight'] > 0]
        self.daily_log.append({
            'decision_date': str(td),
            'exec_date': str(tm),
            'ratio': ratio,
            'net_worth': round(nw, 2),
            'cash': round(self.cash, 2),
            'holdings': holding_details,
            'holding_names': list(holding_names),
            'target': list(target_names),
            'orders': orders,
            'reason': reason,
            'trades': self.trades,
            'fees': round(self.fees, 2),
        })

    def _finalize(self):
        """最终统计"""
        final_nw = self.net_worth(self.get_close_map(len(self.dates) - 1))
        days = len(self.daily_log)
        returns = []
        for i, log in enumerate(self.daily_log):
            if i == 0:
                returns.append(log['net_worth'] / INIT_CASH - 1)
            else:
                prev = self.daily_log[i - 1]['net_worth']
                returns.append(log['net_worth'] / prev - 1 if prev > 0 else 0)

        returns = np.array(returns)
        total_ret = (final_nw / INIT_CASH - 1) * 100
        ann_ret = ((final_nw / INIT_CASH) ** (252 / max(days, 1)) - 1) * 100

        # 最大回撤
        peak = INIT_CASH
        max_dd = 0
        for log in self.daily_log:
            nw = log['net_worth']
            peak = max(peak, nw)
            dd = (peak - nw) / peak * 100
            max_dd = max(max_dd, dd)

        # Sharpe
        avg_daily = np.mean(returns)
        std_daily = np.std(returns) if len(returns) > 1 else 1
        sharpe = (avg_daily / std_daily * np.sqrt(252)) if std_daily > 0 else 0

        # 胜率
        win_days = sum(1 for r in returns if r > 0)
        win_rate = win_days / max(len(returns), 1) * 100

        result = {
            'strategy': 'V5 单ETF',
            'initial_cash': INIT_CASH,
            'final_net_worth': round(final_nw, 2),
            'total_return_pct': round(total_ret, 2),
            'annual_return_pct': round(ann_ret, 2),
            'max_drawdown_pct': round(max_dd, 2),
            'sharpe_ratio': round(sharpe, 2),
            'win_rate_pct': round(win_rate, 1),
            'total_trades': self.trades,
            'total_fees': round(self.fees, 2),
            'trading_days': days,
            'date_range': f"{self.dates[self.start_idx]} → {self.dates[-1]}",
        }

        return result


# ═══════════════════════════════════════════════
# 主入口
# ═══════════════════════════════════════════════

def main():
    t0 = time.time()
    print(f"\n{'='*60}")
    print(f"  V5 单ETF策略 · 260日严格实盘模拟")
    print(f"{'='*60}")
    print(f"  本金 ¥{INIT_CASH:,.0f} | 费 ¥{FEE}/笔 | 手数 {LOT}股")
    print(f"  T+1 锁仓 | 买=(开+低)/2 | 卖=(高+收)/2")

    # ── 加载数据 ──
    price, etf_info, extra_history = load_from_cache()
    if price is None:
        price, etf_info, extra_history, _ = fetch_daily_data_cached(ETF_POOL)
    if price is None:
        print("❌ 无可用数据")
        return

    print(f"\n  数据: {price.shape[0]}行 × {price.shape[1]}列")
    print(f"  区间: {price.index[0]} → {price.index[-1]}")
    print(f"  开始模拟...\n")

    # ── 运行 ──
    sim = Simulator(price, etf_info, extra_history)
    result = sim.run()

    # ── 输出 ──
    print(f"\n{'='*60}")
    print(f"  模拟结果")
    print(f"{'='*60}")
    print(f"  区间:         {result['date_range']}")
    print(f"  交易天数:     {result['trading_days']}")
    print(f"  最终净值:     ¥{result['final_net_worth']:,.2f}")
    print(f"  总回报:       {result['total_return_pct']:+.2f}%")
    print(f"  年化回报:     {result['annual_return_pct']:+.2f}%")
    print(f"  最大回撤:     {result['max_drawdown_pct']:.2f}%")
    print(f"  Sharpe:       {result['sharpe_ratio']:.2f}")
    print(f"  日胜率:       {result['win_rate_pct']:.1f}%")
    print(f"  交易次数:     {result['total_trades']}")
    print(f"  手续费合计:   ¥{result['total_fees']:.2f}")

    # ── 保存 ──
    os.makedirs(DATA_DIR, exist_ok=True)

    # ── 每日记录 CSV ──
    csv_path = os.path.join(DATA_DIR, 'sim_daily.csv')
    csv_rows = []
    for log in sim.daily_log:
        holding_str = ', '.join(f"{n}({d['shares']}股)" for n, d in log['holdings'].items())
        orders_str = '; '.join(log['orders']) if log['orders'] else '—'
        csv_rows.append({
            '决策日': log['decision_date'],
            '执行日': log['exec_date'],
            '仓位比': log['ratio'],
            '净值': log['net_worth'],
            '现金': log['cash'],
            '持仓': holding_str,
            '目标': ', '.join(log['target']),
            '交易': orders_str,
            '累计交易': log['trades'],
            '累计费用': log['fees'],
        })
    pd.DataFrame(csv_rows).to_csv(csv_path, index=False, encoding='utf-8-sig')
    print(f"  💾 每日记录: {csv_path} ({len(csv_rows)}行)")

    # ── JSON 结果 ──
    log_out = {
        'result': result,
        'daily': sim.daily_log[-250:],
    }
    with open(RESULT_FILE, 'w', encoding='utf-8') as f:
        json.dump(log_out, f, ensure_ascii=False, indent=2)
    print(f"  💾 总结果: {RESULT_FILE}")

    elapsed = time.time() - t0
    print(f"  耗时: {elapsed:.1f}s")
    print(f"{'='*60}\n")


if __name__ == '__main__':
    main()
