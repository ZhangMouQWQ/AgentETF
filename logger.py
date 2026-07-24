#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
日志与记录模块
==============
职责: 控制台日志 + 文件持久化 + CSV/SQLite 交易记录 + 净值曲线

特性:
  - 双通道输出: 控制台(INFO级别) + 文件(DEBUG级别, 含完整堆栈)
  - 交易记录: CSV 追加模式, 不覆盖历史
  - 净值曲线: 每日快照, 方便画图
  - 时间戳精确到秒
  - INFO/WARN/ERROR 三级日志

用法:
    from logger import TradeLogger

    log = TradeLogger()
    log.info("策略启动")
    log.trade(code='510300', action='BUY', price=4.70, shares=1700)
    log.nav_snapshot(date='2026-07-24', cash=8000, holdings_value=0)
"""

import os, csv, json, sqlite3, logging, logging.handlers
from datetime import datetime, timezone, timedelta

BJ = timezone(timedelta(hours=8))

# ═══════════════════════════════════
# 配置
# ═══════════════════════════════════

LOG_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'logs')
os.makedirs(LOG_DIR, exist_ok=True)

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data')
os.makedirs(DATA_DIR, exist_ok=True)


# ═══════════════════════════════════
# 底层日志配置
# ═══════════════════════════════════

def _setup_file_logger(name: str, filename: str) -> logging.Logger:
    """创建文件 logger (DEBUG级别, 含时间戳)"""
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)
    logger.propagate = False  # 不向父logger传递

    # 避免重复handler
    if logger.handlers:
        return logger

    fh = logging.handlers.RotatingFileHandler(
        filename, maxBytes=10 * 1024 * 1024, backupCount=5, encoding='utf-8'
    )
    fh.setLevel(logging.DEBUG)
    fmt = logging.Formatter(
        '%(asctime)s | %(levelname)-5s | %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    fh.setFormatter(fmt)
    logger.addHandler(fh)
    return logger


# 全局 logger 实例
_trade_logger = _setup_file_logger('trade', os.path.join(LOG_DIR, 'trade.log'))
_system_logger = _setup_file_logger('system', os.path.join(LOG_DIR, 'system.log'))


# ═══════════════════════════════════
# TradeLogger 主类
# ═══════════════════════════════════

class TradeLogger:
    """交易日志记录器

    用法:
        log = TradeLogger(data_dir='data')

        # 系统日志
        log.info("策略启动, 初始资金 ¥8,000")
        log.warn("银行ETF amount为估算值 (MAE 1-3%)")
        log.error("东方财富API拉取失败, 回退腾讯")

        # 交易日志
        log.trade(code='510300', action='BUY', price=4.70, shares=1700, reason='动量信号')

        # 净值快照
        log.nav_snapshot(date='2026-07-24', cash=5000, holdings_value=3000, positions={'510300': 1700})

        # 信号日志
        log.signal(code='510300', day_signal='BUY', hour_signal='HOLD', final='HOLD', reason='小时线未确认')
    """

    def __init__(self, data_dir: str = None):
        """
        Args:
            data_dir: 数据输出目录, 默认 'data/'
        """
        self.data_dir = data_dir or DATA_DIR
        os.makedirs(self.data_dir, exist_ok=True)
        self._trades_csv = os.path.join(self.data_dir, 'trades.csv')
        self._nav_csv = os.path.join(self.data_dir, 'nav_daily.csv')
        self._signals_csv = os.path.join(self.data_dir, 'signals.csv')
        self._db_path = os.path.join(self.data_dir, 'trade.db')

        # 运行标识
        self.run_id = datetime.now(BJ).strftime('%Y%m%d_%H%M%S')
        self.start_time = datetime.now(BJ)

        # 初始化文件 (写表头)
        self._init_files()

    def _now(self) -> str:
        return datetime.now(BJ).strftime('%Y-%m-%d %H:%M:%S')

    def _init_files(self):
        """初始化CSV文件 (仅首次写入表头)"""
        # 交易记录表头
        if not os.path.exists(self._trades_csv):
            with open(self._trades_csv, 'w', newline='', encoding='utf-8-sig') as f:
                w = csv.writer(f)
                w.writerow(['时间', '代码', '方向', '股数', '价格', '成交额',
                            '佣金', '过户费', '总费用', '净额', '原因', '运行ID'])

        # 净值表头
        if not os.path.exists(self._nav_csv):
            with open(self._nav_csv, 'w', newline='', encoding='utf-8-sig') as f:
                w = csv.writer(f)
                w.writerow(['日期', '现金', '持仓市值', '净值', '日收益%', '累计收益%',
                            '持仓数', '最大持仓', '运行ID'])

        # 信号表头
        if not os.path.exists(self._signals_csv):
            with open(self._signals_csv, 'w', newline='', encoding='utf-8-sig') as f:
                w = csv.writer(f)
                w.writerow(['时间', '代码', '日线信号', '小时信号', '最终操作',
                            '日线得分', '原因', '运行ID'])

    # ── 系统日志 ──

    def info(self, msg: str):
        """INFO 级别日志"""
        print(f"  [INFO] {msg}")
        _system_logger.info(f"[{self.run_id}] {msg}")

    def warn(self, msg: str):
        """WARN 级别日志"""
        print(f"  [WARN] {msg}")
        _system_logger.warning(f"[{self.run_id}] {msg}")

    def error(self, msg: str, exc: Exception = None):
        """ERROR 级别日志 (含堆栈)"""
        print(f"  [ERROR] {msg}")
        if exc:
            _system_logger.error(f"[{self.run_id}] {msg}", exc_info=exc)
        else:
            _system_logger.error(f"[{self.run_id}] {msg}")

    def debug(self, msg: str):
        """DEBUG 日志 (仅写入文件, 不打印控制台)"""
        _system_logger.debug(f"[{self.run_id}] {msg}")

    # ── 交易记录 ──

    def trade(self, code: str, action: str, price: float, shares: int,
              turnover: float = 0, commission: float = 0, transfer_fee: float = 0,
              total_fee: float = 0, net_amount: float = 0, reason: str = ''):
        """记录一笔交易 (追加到CSV + 文件日志)

        Args:
            code:        ETF代码
            action:      'BUY' | 'SELL'
            price:       成交价
            shares:      股数
            turnover:    成交金额
            commission:  佣金
            transfer_fee: 过户费
            total_fee:   总费用
            net_amount:  净支出/收入
            reason:      交易原因
        """
        ts = self._now()

        # 控制台
        icon = '+' if action == 'BUY' else '-'
        print(f"  [TRADE] {icon} {code} {action} {shares}股 @{price:.3f} "
              f"净{net_amount:,.0f} 费{total_fee:.2f} | {reason}")

        # 文件日志
        _trade_logger.info(
            f"{code} | {action} | {shares}股 @{price:.3f} | "
            f"成交额{turnover:,.0f} | 费{total_fee:.2f} | 净{net_amount:,.0f} | {reason}"
        )

        # CSV 追加
        with open(self._trades_csv, 'a', newline='', encoding='utf-8-sig') as f:
            w = csv.writer(f)
            w.writerow([ts, code, action, shares, round(price, 3),
                        round(turnover, 2), round(commission, 2),
                        round(transfer_fee, 2), round(total_fee, 2),
                        round(net_amount, 2), reason, self.run_id])

    # ── 净值快照 ──

    def nav_snapshot(self, date: str, cash: float, holdings_value: float,
                     positions: dict = None, daily_return_pct: float = 0,
                     cumulative_return_pct: float = 0):
        """记录每日净值快照

        Args:
            date:              日期 'YYYY-MM-DD'
            cash:              现金
            holdings_value:    持仓市值
            positions:         持仓详情 {code: shares}
            daily_return_pct:  日收益率%
            cumulative_return_pct: 累计收益率%
        """
        nw = cash + holdings_value
        pos_count = len(positions) if positions else 0
        max_holding = max(positions.values()) if positions else ''

        # 文件日志 (简洁)
        _system_logger.info(
            f"NAV {date} | 净值{nw:,.0f} | 现金{cash:,.0f} | "
            f"市值{holdings_value:,.0f} | 日收益{daily_return_pct:+.2f}% | "
            f"持仓{pos_count}只"
        )

        # CSV 追加
        with open(self._nav_csv, 'a', newline='', encoding='utf-8-sig') as f:
            w = csv.writer(f)
            w.writerow([date, round(cash, 2), round(holdings_value, 2),
                        round(nw, 2), round(daily_return_pct, 4),
                        round(cumulative_return_pct, 4),
                        pos_count, str(max_holding), self.run_id])

    # ── 信号日志 ──

    def signal(self, code: str, day_signal: str, hour_signal: str = None,
               final_action: str = 'HOLD', day_score: int = 0, reason: str = ''):
        """记录策略信号

        Args:
            code:         ETF代码
            day_signal:   日线信号 'BUY'|'SELL'|'HOLD'
            hour_signal:  小时线信号 (可选)
            final_action: 最终操作
            day_score:    日线得分
            reason:       决策原因
        """
        ts = self._now()

        # 控制台 (仅非HOLD信号打印)
        if final_action != 'HOLD':
            icon = '+' if final_action == 'BUY' else '-'
            print(f"  [SIGNAL] {icon} {code} {final_action} | "
                  f"日{day_signal} 时{hour_signal or 'N/A'} 得分{day_score} | {reason}")

        # 文件日志
        _trade_logger.info(
            f"SIGNAL {code} | 日{day_signal} 时{hour_signal or 'N/A'} → {final_action} | {reason}"
        )

        # CSV 追加
        with open(self._signals_csv, 'a', newline='', encoding='utf-8-sig') as f:
            w = csv.writer(f)
            w.writerow([ts, code, day_signal, hour_signal or '',
                        final_action, day_score, reason, self.run_id])

    # ── 汇总报告 ──

    def report(self, title: str = '运行报告'):
        """打印运行汇总"""
        elapsed = (datetime.now(BJ) - self.start_time).total_seconds()
        print(f"\n{'='*55}")
        print(f"  {title}")
        print(f"{'='*55}")
        print(f"  运行ID:   {self.run_id}")
        print(f"  开始时间: {self.start_time.strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"  耗时:     {elapsed:.1f}s")

        # 统计交易
        trade_count = 0
        if os.path.exists(self._trades_csv):
            with open(self._trades_csv, 'r', encoding='utf-8-sig') as f:
                trade_count = sum(1 for _ in f) - 1  # 减表头

        print(f"  交易记录: {self._trades_csv} ({trade_count}笔)")
        print(f"  净值曲线: {self._nav_csv}")
        print(f"  信号记录: {self._signals_csv}")
        print(f"  系统日志: {LOG_DIR}/system.log")
        print(f"  交易日志: {LOG_DIR}/trade.log")
        print(f"{'='*55}")

    # ── SQLite 导出 (可选) ──

    def export_sqlite(self):
        """导出交易记录到 SQLite (方便SQL查询)"""
        conn = sqlite3.connect(self._db_path)
        # 交易表
        conn.execute('''CREATE TABLE IF NOT EXISTS trades (
            time TEXT, code TEXT, action TEXT, shares INTEGER,
            price REAL, turnover REAL, commission REAL,
            transfer_fee REAL, total_fee REAL, net_amount REAL,
            reason TEXT, run_id TEXT)''')
        # 净值表
        conn.execute('''CREATE TABLE IF NOT EXISTS nav (
            date TEXT PRIMARY KEY, cash REAL, holdings_value REAL,
            net_worth REAL, daily_return REAL, cumulative_return REAL,
            positions_count INTEGER)''')

        # 导入CSV
        if os.path.exists(self._trades_csv):
            df = __import__('pandas').read_csv(self._trades_csv, encoding='utf-8-sig')
            df.to_sql('trades', conn, if_exists='replace', index=False)
        if os.path.exists(self._nav_csv):
            df = __import__('pandas').read_csv(self._nav_csv, encoding='utf-8-sig')
            df.to_sql('nav', conn, if_exists='replace', index=False)

        conn.commit()
        conn.close()
        self.info(f"SQLite 导出完成: {self._db_path}")

    # ── 查询接口 ──

    def get_trades_df(self):
        """获取交易记录 DataFrame"""
        try:
            import pandas as pd
            if os.path.exists(self._trades_csv):
                return pd.read_csv(self._trades_csv, encoding='utf-8-sig')
        except ImportError:
            pass
        return None

    def get_nav_df(self):
        """获取净值 DataFrame"""
        try:
            import pandas as pd
            if os.path.exists(self._nav_csv):
                return pd.read_csv(self._nav_csv, encoding='utf-8-sig')
        except ImportError:
            pass
        return None


# ═══════════════════════════════════
# 自测
# ═══════════════════════════════════

if __name__ == '__main__':
    print("=== TradeLogger 自测 ===\n")

    # 初始化
    log = TradeLogger()

    # 1. 系统日志
    log.info("策略启动, 初始资金 ¥8,000")
    log.warn("银行ETF amount为估算值 (MAE 1-3%)")

    # 2. 信号日志
    log.signal('510300', 'BUY', 'BUY', 'BUY', 6, '日线多头排列 + 小时线确认')
    log.signal('159915', 'BUY', 'SELL', 'HOLD', -2, '日线看多但小时线看空, 等待')
    log.signal('512800', 'HOLD', None, 'HOLD', 1, '日线无明确方向')

    # 3. 交易日志
    log.trade('510300', 'BUY', 4.70, 1700, turnover=7990, commission=5.0,
              transfer_fee=0.08, total_fee=5.08, net_amount=7995.08, reason='动量信号')
    log.trade('510300', 'SELL', 4.80, 1700, turnover=8160, commission=5.0,
              transfer_fee=0.08, total_fee=5.08, net_amount=8154.92, reason='信号转空')

    # 4. 净值快照
    log.nav_snapshot('2026-07-24', 7995, 0, positions={}, daily_return_pct=0,
                     cumulative_return_pct=0)
    log.nav_snapshot('2026-07-25', 7995, 8050, positions={'510300': 1700},
                     daily_return_pct=0.69, cumulative_return_pct=0.69)

    # 5. 错误日志
    try:
        1 / 0
    except Exception as e:
        log.error("模拟异常: 除零错误", exc=e)

    # 6. 报告
    log.report("自测报告")

    # 7. 读取验证
    print("\n[7] 交易记录验证:")
    df = log.get_trades_df()
    if df is not None:
        print(df.to_string())

    print("\n[8] 净值记录验证:")
    df_nav = log.get_nav_df()
    if df_nav is not None:
        print(df_nav.to_string())

    print("\n=== 自测完成 ===")
