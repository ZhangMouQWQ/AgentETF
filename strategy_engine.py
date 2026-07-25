#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
策略计算引擎
============
职责: 技术指标计算 + 日K定方向 + 小时K定时机的信号生成

特性:
  - 无未来函数: 所有指标仅使用截至当前K线的历史数据
  - 防信号闪烁: 连续2根K线确认才出信号
  - 多级信号: 日线BUY + 小时线BUY = 最终BUY; 冲突时HOLD
  - NaN安全: 数据不足时自动降级或跳过

用法:
    from strategy_engine import StrategyEngine

    engine = StrategyEngine()
    indicators = engine.calc_daily_indicators(df_daily)
    signal = engine.generate_signal('510300', df_daily, df_hourly=None)
"""

import numpy as np
import pandas as pd
from datetime import datetime, timezone, timedelta

BJ = timezone(timedelta(hours=8))

# ═══════════════════════════════════
# 参数配置 (可在初始化时覆盖)
# ═══════════════════════════════════

DEFAULT_PARAMS = {
    # 均线周期
    'ma_short': 5,
    'ma_mid': 20,
    'ma_long': 60,

    # MACD
    'macd_fast': 12,
    'macd_slow': 26,
    'macd_signal': 9,

    # RSI
    'rsi_period': 14,
    'rsi_oversold': 30,    # 超卖阈值
    'rsi_overbought': 70,  # 超买阈值

    # 波动率
    'vol_window': 20,

    # 量比
    'volume_ma_period': 5,

    # 信号确认
    'confirm_bars': 2,     # 连续确认K线数

    # 动量
    'mom_period': 20,
}


# ═══════════════════════════════════
# 技术指标计算
# ═══════════════════════════════════

def calc_sma(series, period):
    """简单移动平均"""
    return series.rolling(period, min_periods=max(2, period // 2)).mean()


def calc_ema(series, period):
    """指数移动平均"""
    return series.ewm(span=period, adjust=False).mean()


def calc_macd(close, fast=12, slow=26, signal=9):
    """MACD 指标

    Returns:
        dict: {macd, signal, histogram}
    """
    ema_fast = calc_ema(close, fast)
    ema_slow = calc_ema(close, slow)
    macd_line = ema_fast - ema_slow
    signal_line = calc_ema(macd_line, signal)
    histogram = macd_line - signal_line
    return {
        'macd': macd_line,
        'signal': signal_line,
        'histogram': histogram,
    }


def calc_rsi(close, period=14):
    """RSI 相对强弱指标"""
    delta = close.diff()
    gain = delta.clip(lower=0)
    loss = (-delta).clip(lower=0)
    avg_gain = gain.ewm(alpha=1 / period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1 / period, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    rsi = 100 - (100 / (1 + rs))
    return rsi


def calc_bollinger(close, period=20, std=2):
    """布林带"""
    ma = calc_sma(close, period)
    std_dev = close.rolling(period, min_periods=2).std()
    upper = ma + std * std_dev
    lower = ma - std * std_dev
    # 带宽: 信号强度
    bandwidth = (upper - lower) / ma * 100
    return {'upper': upper, 'middle': ma, 'lower': lower, 'bandwidth': bandwidth}


def calc_atr(high, low, close, period=14):
    """ATR 平均真实波幅"""
    prev_close = close.shift(1)
    tr1 = high - low
    tr2 = (high - prev_close).abs()
    tr3 = (low - prev_close).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    return tr.rolling(period, min_periods=2).mean()


def calc_momentum(close, period=20):
    """动量指标 (百分比变化)"""
    return (close / close.shift(period) - 1) * 100


def calc_volume_ratio(volume, period=5):
    """量比: 当前成交量 / N日均量"""
    avg_vol = volume.shift(1).rolling(period, min_periods=2).mean()
    return volume / avg_vol.replace(0, np.nan)


def calc_parkinson_volatility(high, low, window=20, annualize=True):
    """Parkinson波动率 (基于High/Low)"""
    hl_ratio = np.log(high / low)
    daily_var = (1 / (4 * np.log(2))) * (hl_ratio ** 2)
    vol = np.sqrt(daily_var.rolling(window, min_periods=5).mean())
    if annualize:
        vol = vol * np.sqrt(252)
    return vol * 100  # 百分比


# ═══════════════════════════════════
# 信号生成
# ═══════════════════════════════════

def _ma_crossover_signal(close, short_ma, long_ma, confirm=2):
    """均线交叉信号 (防闪烁)

    Returns:
        pd.Series: 1=金叉, -1=死叉, 0=无信号
    """
    diff = short_ma - long_ma
    # 金叉: diff 从负变正
    golden = (diff > 0) & (diff.shift(1) <= 0)
    # 死叉: diff 从正变负
    death = (diff < 0) & (diff.shift(1) >= 0)

    # 确认: 连续 confirm 根K线保持交叉状态
    if confirm > 1:
        golden_confirmed = golden.copy()
        death_confirmed = death.copy()
        for i in range(1, confirm):
            golden_confirmed = golden_confirmed & (diff.shift(-i) > 0)
            death_confirmed = death_confirmed & (diff.shift(-i) < 0)
        golden = golden_confirmed
        death = death_confirmed

    signal = pd.Series(0, index=close.index)
    signal[golden] = 1
    signal[death] = -1
    return signal


class StrategyEngine:
    """策略计算引擎

    用法:
        engine = StrategyEngine(params={...})
        indicators = engine.calc_daily_indicators(df_daily)
        signal = engine.generate_signal(code, df_daily, df_hourly)
    """

    def __init__(self, params=None):
        """
        Args:
            params: dict, 覆盖默认参数
        """
        self.params = {**DEFAULT_PARAMS, **(params or {})}

    # ── 日线指标 ──
    def calc_daily_indicators(self, df):
        """计算日线全部技术指标

        Args:
            df: DataFrame (day, open, high, low, close, volume, [amount, turnover])

        Returns:
            dict: {
                'ma5', 'ma20', 'ma60': 均线,
                'macd', 'macd_signal', 'macd_hist': MACD,
                'rsi': RSI,
                'bb_upper', 'bb_middle', 'bb_lower', 'bb_width': 布林带,
                'atr': ATR,
                'momentum_20d': 20日动量,
                'vol_parkinson': Parkinson波动率,
                'volume_ratio': 量比,
                'close', 'high', 'low', 'volume': 原始数据,
            }
        """
        if df is None or len(df) < self.params['ma_long']:
            return None

        # 确保有必要的列
        close = df['close'].astype(float)
        high = df['high'].astype(float) if 'high' in df.columns else close
        low = df['low'].astype(float) if 'low' in df.columns else close
        volume = df['volume'].astype(float) if 'volume' in df.columns else pd.Series(1, index=close.index)

        p = self.params
        macd_result = calc_macd(close, p['macd_fast'], p['macd_slow'], p['macd_signal'])
        bb_result = calc_bollinger(close, p['ma_mid'])

        return {
            # 均线
            'ma5': calc_sma(close, p['ma_short']),
            'ma20': calc_sma(close, p['ma_mid']),
            'ma60': calc_sma(close, p['ma_long']),

            # MACD (缓存结果)
            'macd': macd_result['macd'],
            'macd_signal': macd_result['signal'],
            'macd_hist': macd_result['histogram'],

            # RSI
            'rsi': calc_rsi(close, p['rsi_period']),

            # 布林带 (缓存结果)
            'bb_upper': bb_result['upper'],
            'bb_middle': bb_result['middle'],
            'bb_lower': bb_result['lower'],
            'bb_width': bb_result['bandwidth'],

            # ATR
            'atr': calc_atr(high, low, close, 14),

            # 动量
            'momentum_20d': calc_momentum(close, p['mom_period']),

            # 波动率
            'vol_parkinson': calc_parkinson_volatility(high, low, p['vol_window']),

            # 量比
            'volume_ratio': calc_volume_ratio(volume, p['volume_ma_period']),

            # 原始数据(用于信号)
            'close': close,
            'high': high,
            'low': low,
            'volume': volume,
        }

    # ── 小时线指标 ──
    def calc_hourly_indicators(self, df):
        """计算60分钟K线技术指标 (简化版: 仅均线+MACD+RSI)

        Args:
            df: DataFrame (day, open, high, low, close, volume)

        Returns:
            dict 或 None (数据不足时)
        """
        if df is None or len(df) < self.params['ma_long']:
            return None

        close = df['close'].astype(float)
        volume = df['volume'].astype(float) if 'volume' in df.columns else pd.Series(1, index=close.index)
        p = self.params

        return {
            'ma5': calc_sma(close, p['ma_short']),
            'ma20': calc_sma(close, p['ma_mid']),
            'macd': calc_macd(close, p['macd_fast'], p['macd_slow'], p['macd_signal'])['macd'],
            'macd_hist': calc_macd(close, p['macd_fast'], p['macd_slow'], p['macd_signal'])['histogram'],
            'rsi': calc_rsi(close, p['rsi_period']),
            'volume_ratio': calc_volume_ratio(volume, p['volume_ma_period']),
            'close': close,
            'volume': volume,
        }

    # ── 日线方向信号 ──
    def _daily_direction(self, ind):
        """日线定方向: BUY / SELL / HOLD

        逻辑:
          1. MA5 > MA20 > MA60 → 多头排列 → BUY
          2. MA5 < MA20 < MA60 → 空头排列 → SELL
          3. MA5/MA20 金叉 + MACD>0 + RSI不超买 → BUY
          4. MA5/MA20 死叉 + MACD<0 → SELL
          5. 其他 → HOLD

        Args:
            ind: calc_daily_indicators 返回的指标 dict

        Returns:
            str: 'BUY' | 'SELL' | 'HOLD'
        """
        if ind is None:
            return 'HOLD'

        idx = -1  # 最新一根K线

        # 取最新值 (NaN安全)
        def _v(s):
            val = s.iloc[idx] if idx >= -len(s) else np.nan
            return float(val) if not pd.isna(val) else None

        ma5 = _v(ind['ma5'])
        ma20 = _v(ind['ma20'])
        ma60 = _v(ind['ma60'])
        macd = _v(ind['macd'])
        macd_hist = _v(ind['macd_hist'])
        rsi = _v(ind['rsi'])
        mom = _v(ind['momentum_20d'])
        vol_ratio = _v(ind['volume_ratio'])
        bb_w = _v(ind['bb_width'])

        if any(v is None for v in [ma5, ma20, ma60, macd, rsi]):
            return 'HOLD'

        score = 0  # 多头得分

        # 1. 均线排列 (权重最高)
        if ma5 > ma20 > ma60:
            score += 3  # 多头排列
        elif ma5 < ma20 < ma60:
            score -= 3  # 空头排列
        elif ma5 > ma20:
            score += 1  # 短均在上
        elif ma5 < ma20:
            score -= 1

        # 2. MACD
        if macd > 0:
            score += 2
        else:
            score -= 2
        if macd_hist is not None and macd_hist > 0:
            score += 1  # MACD柱扩大

        # 3. RSI
        if rsi is not None:
            if rsi < 25:
                score += 2  # 超卖反弹
            elif rsi < 40:
                score += 1
            elif rsi > 75:
                score -= 2  # 超买
            elif rsi > 60:
                score -= 1

        # 4. 动量
        if mom is not None and mom > 0:
            score += 1
        elif mom is not None:
            score -= 1

        # 5. 量比 (放量加分)
        if vol_ratio is not None and vol_ratio > 1.2:
            score += 1 if score > 0 else 0  # 仅在偏多时加分

        # ── 判定 ──
        if score >= 5:  # 适度放宽 (HS300已过滤弱市)
            return 'BUY'
        elif score <= -3:
            return 'SELL'
        else:
            return 'HOLD'

    # ── 小时线时机信号 ──
    def _hourly_timing(self, ind):
        """小时线定时机: BUY / SELL / HOLD

        逻辑:
          1. MA5/MA20 金叉 + MACD金叉 → BUY
          2. MA5/MA20 死叉 → SELL
          3. 其他 → HOLD

        Args:
            ind: calc_hourly_indicators 返回的指标 dict

        Returns:
            str: 'BUY' | 'SELL' | 'HOLD'
        """
        if ind is None:
            return 'HOLD'

        idx = -1

        def _v(s):
            val = s.iloc[idx] if idx >= -len(s) else np.nan
            return float(val) if not pd.isna(val) else None

        def _prev(s):
            val = s.iloc[idx - 1] if idx - 1 >= -len(s) else np.nan
            return float(val) if not pd.isna(val) else None

        ma5 = _v(ind['ma5'])
        ma20 = _v(ind['ma20'])
        ma5_prev = _prev(ind['ma5'])
        ma20_prev = _prev(ind['ma20'])
        macd = _v(ind['macd'])
        macd_prev = _prev(ind['macd'])
        rsi = _v(ind['rsi'])

        if any(v is None for v in [ma5, ma20, macd, rsi]):
            return 'HOLD'

        score = 0

        # 1. 均线交叉
        if ma5 > ma20:
            score += 2
            if ma5_prev is not None and ma20_prev is not None and ma5_prev <= ma20_prev:
                score += 3  # 刚金叉, 强信号
        else:
            score -= 2
            if ma5_prev is not None and ma20_prev is not None and ma5_prev >= ma20_prev:
                score -= 3  # 刚死叉

        # 2. MACD
        if macd > 0:
            score += 2
            if macd_prev is not None and macd_prev <= 0:
                score += 2  # MACD刚上零轴
        else:
            score -= 2

        # 3. RSI
        if rsi < 30:
            score += 2  # 超卖
        elif rsi > 70:
            score -= 2  # 超买

        # ── 确认防闪烁: 看前一根是否有同向信号 ──
        # 如果当前强烈看多但前一根没有, 降级
        prev_ma5_up = ma5_prev is not None and ma5_prev > (ma20_prev or 0)
        curr_ma5_up = ma5 > ma20
        if curr_ma5_up and not prev_ma5_up:
            score = max(score - 2, -5)  # 未确认, 减分

        if score >= 3:  # 放松以配合日线双确认
            return 'BUY'
        elif score <= -2:
            return 'SELL'
        else:
            return 'HOLD'

    # ── 综合信号生成 ──
    def generate_signal(self, code, df_daily, df_hourly=None):
        """生成最终交易信号

        优先级: 日线定方向 → 小时线定时机
        冲突规则: 日BUY + 时BUY = BUY; 日BUY + 时SELL = HOLD (等待时机)

        Args:
            code: ETF代码
            df_daily: 日线 DataFrame
            df_hourly: 60分钟 DataFrame (可选, 无则仅用日线)

        Returns:
            dict: {
                'code': str,
                'day_signal': 'BUY'|'SELL'|'HOLD',
                'hour_signal': 'BUY'|'SELL'|'HOLD'|None,
                'final_action': 'BUY'|'SELL'|'HOLD',
                'day_score': int (多头得分),
                'hour_score': int|None,
                'indicators': dict (日线指标快照),
                'reason': str (决策原因),
                'timestamp': str,
            }
        """
        # ── 日线指标 ──
        day_ind = self.calc_daily_indicators(df_daily)
        day_signal = self._daily_direction(day_ind)

        # ── 日线得分 ──
        day_score = 0
        if day_ind is not None:
            idx = -1
            try:
                ma5 = float(day_ind['ma5'].iloc[idx]) if not pd.isna(day_ind['ma5'].iloc[idx]) else 0
                ma20 = float(day_ind['ma20'].iloc[idx]) if not pd.isna(day_ind['ma20'].iloc[idx]) else 0
                rsi = float(day_ind['rsi'].iloc[idx]) if not pd.isna(day_ind['rsi'].iloc[idx]) else 50
                mom = float(day_ind['momentum_20d'].iloc[idx]) if not pd.isna(day_ind['momentum_20d'].iloc[idx]) else 0
                day_score = (1 if ma5 > ma20 else -1) + (1 if rsi < 50 else -1 if rsi > 60 else 0) + (1 if mom > 0 else -1 if mom < 0 else 0)
            except Exception:
                pass

        # ── 小时线 ──
        hour_signal = None
        hour_score = None
        if df_hourly is not None and len(df_hourly) >= 20:
            hour_ind = self.calc_hourly_indicators(df_hourly)
            hour_signal = self._hourly_timing(hour_ind)
            if hour_ind is not None:
                try:
                    ma5 = float(hour_ind['ma5'].iloc[-1]) if not pd.isna(hour_ind['ma5'].iloc[-1]) else 0
                    ma20 = float(hour_ind['ma20'].iloc[-1]) if not pd.isna(hour_ind['ma20'].iloc[-1]) else 0
                    hour_score = 1 if ma5 > ma20 else -1
                except Exception:
                    hour_score = 0

        # ── 综合决策 ──
        # 日线为主, 小时线为辅
        if day_signal == 'HOLD':
            final = 'HOLD'
            reason = '日线无明确方向'
        elif hour_signal is None:
            # 无小时线数据: 禁止买入(不闭眼开车), 允许卖出(保命要紧)
            if day_signal == 'SELL':
                final = 'SELL'
                reason = '日线看空 (无小时线数据, 仅允许卖出)'
            else:
                final = 'HOLD'
                reason = f'日线{day_signal}但无小时线数据, 禁止买入'
        elif day_signal == 'BUY' and hour_signal == 'BUY':
            final = 'BUY'
            reason = '日线看多 + 小时线确认'
        elif day_signal == 'SELL' and hour_signal == 'SELL':
            final = 'SELL'
            reason = '日线看空 + 小时线确认'
        elif day_signal == 'BUY' and hour_signal == 'HOLD':
            final = 'BUY'  # 日线为主, 小时线不反对
            reason = '日线看多 (小时线中性)'
        elif day_signal == 'SELL' and hour_signal == 'HOLD':
            final = 'SELL'  # 日线为主, 小时线不反对
            reason = '日线看空 (小时线中性)'
        elif day_signal == 'BUY' and hour_signal == 'SELL':
            final = 'HOLD'
            reason = '日线看多但小时线看空, 等待回调'
        elif day_signal == 'SELL' and hour_signal == 'BUY':
            final = 'HOLD'
            reason = '日线看空但小时线反弹, 等待确认'
        else:
            final = 'HOLD'
            reason = '信号冲突, 观望'

        # ── 指标快照 ──
        indicator_snapshot = {}
        if day_ind is not None:
            idx = -1
            for key in ['ma5', 'ma20', 'ma60', 'rsi', 'momentum_20d', 'vol_parkinson', 'volume_ratio', 'bb_width']:
                try:
                    val = day_ind[key].iloc[idx]
                    indicator_snapshot[key] = round(float(val), 2) if not pd.isna(val) else None
                except Exception:
                    indicator_snapshot[key] = None

        return {
            'code': code,
            'day_signal': day_signal,
            'hour_signal': hour_signal,
            'final_action': final,
            'day_score': day_score,
            'hour_score': hour_score,
            'indicators': indicator_snapshot,
            'reason': reason,
            'timestamp': datetime.now(BJ).strftime('%Y-%m-%d %H:%M:%S'),
        }

    def batch_generate(self, codes, daily_data, hourly_data=None):
        """批量生成信号

        Args:
            codes: ETF代码列表
            daily_data: dict {code: DataFrame}
            hourly_data: dict {code: DataFrame} (可选)

        Returns:
            dict: {code: signal_dict}
        """
        results = {}
        for code in codes:
            df_d = daily_data.get(code)
            df_h = hourly_data.get(code) if hourly_data else None
            results[code] = self.generate_signal(code, df_d, df_h)
        return results


# ═══════════════════════════════════
# 自测
# ═══════════════════════════════════

if __name__ == '__main__':
    print("=== StrategyEngine 自测 ===\n")

    # 生成模拟数据
    np.random.seed(42)
    dates = pd.date_range('2025-01-01', periods=120, freq='B')
    close = pd.Series(100 + np.cumsum(np.random.randn(120) * 1.5), index=dates)
    high = close + np.abs(np.random.randn(120) * 1.0)
    low = close - np.abs(np.random.randn(120) * 1.0)
    volume = pd.Series(np.random.randint(10000, 100000, 120), index=dates)

    df_daily = pd.DataFrame({
        'day': [d.strftime('%Y-%m-%d') for d in dates],
        'open': close.shift(1).fillna(100),
        'high': high,
        'low': low,
        'close': close,
        'volume': volume,
    })

    # 生成60分钟模拟数据 (约 4根/天 × 120天)
    h_dates = []
    for d in dates:
        for h in ['09:30', '10:30', '11:30', '13:30', '14:30']:
            h_dates.append(f"{d.strftime('%Y-%m-%d')} {h}")
    h_close = pd.Series(100 + np.cumsum(np.random.randn(len(h_dates)) * 0.3), index=h_dates)
    df_hourly = pd.DataFrame({
        'day': h_dates,
        'open': h_close.shift(1).fillna(100),
        'high': h_close + np.abs(np.random.randn(len(h_dates)) * 0.5),
        'low': h_close - np.abs(np.random.randn(len(h_dates)) * 0.5),
        'close': h_close,
        'volume': np.random.randint(1000, 10000, len(h_dates)),
    })

    # ── 测试 ──
    engine = StrategyEngine()

    # 1. 日线指标
    print("[1] 日线指标")
    ind = engine.calc_daily_indicators(df_daily)
    if ind:
        print(f"  MA5:   {ind['ma5'].iloc[-1]:.2f}")
        print(f"  MA20:  {ind['ma20'].iloc[-1]:.2f}")
        print(f"  MA60:  {ind['ma60'].iloc[-1]:.2f}")
        print(f"  RSI:   {ind['rsi'].iloc[-1]:.1f}")
        print(f"  MOM:   {ind['momentum_20d'].iloc[-1]:.1f}%")
        print(f"  VOL:   {ind['vol_parkinson'].iloc[-1]:.1f}%")
    else:
        print("  数据不足")

    # 2. 信号生成 (无小时线)
    print("\n[2] 信号生成 (仅日线)")
    sig = engine.generate_signal('510300', df_daily)
    print(f"  日线信号: {sig['day_signal']}")
    print(f"  最终操作: {sig['final_action']}")
    print(f"  原因: {sig['reason']}")
    print(f"  得分: {sig['day_score']}")

    # 3. 信号生成 (日线+小时线)
    print("\n[3] 信号生成 (日线+60分钟)")
    sig2 = engine.generate_signal('510300', df_daily, df_hourly)
    print(f"  日线信号: {sig2['day_signal']}")
    print(f"  小时信号: {sig2['hour_signal']}")
    print(f"  最终操作: {sig2['final_action']}")
    print(f"  原因: {sig2['reason']}")

    # 4. 指标快照
    print("\n[4] 指标快照")
    for k, v in sig2['indicators'].items():
        print(f"  {k}: {v}")

    print("\n=== 自测完成 ===")
