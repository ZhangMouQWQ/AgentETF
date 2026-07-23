# ETF 动量轮动 · V5 单ETF策略（GitHub Actions 自动版）

每个交易日 **自动运行两次**，拉取 39 只行业/宽基 ETF 的历史日线和实时行情，**V5 单 ETF 策略** 生成模拟持仓的买入/卖出指令。完全免费，无需服务器。

> 策略: V5 单 ETF 精选 — 价格 ¥0.5-3.0 + quality_score > 0.5，选最优 1 只集中兵力。

---

## 项目结构

```
AgentETF/
├── strategy.py                  # 核心: ETF池定义 + 5个API接口 + 指标计算
├── fetch_data.py                # 数据抓取脚本
├── portfolio_signal.py          # 持仓信号: V5单ETF模拟持仓 + 买卖订单
├── data_cache.py                # 本地缓存 (pickle, 不在 Actions 中使用)
├── requirements.txt             # Python 依赖
├── data/                        # 自动生成的 JSON 数据 (由 Actions 推送)
│   ├── history.json             #   历史日线 (close + OHLCV + 成交额/换手率)
│   ├── realtime.json            #   实时行情快照
│   ├── meta.json                #   拉取元信息
│   ├── signal.json              #   📊 持仓信号 (市场+订单+板块+风控)
│   └── positions.json           #   模拟持仓状态 (T+1锁仓记录)
└── .github/workflows/fetch.yml  # 定时任务配置
```

## 数据 API（三级回退）

| 优先级 | API | 主机 | 提供字段 |
|:---:|-----|------|---------|
| 1 | AKShare (`fund_etf_hist_em`) | push2his.eastmoney.com | OHLCV + 成交额 + 换手率 |
| 2 | 东方财富 K线 | push2his.eastmoney.com | OHLCV + 成交额 + 换手率 |
| 3 | 腾讯 K线 | web.ifzq.gtimg.cn | OHLC + 量 (缺成交额) |
| + | 新浪实时 | hq.sinajs.cn | 当日成交额补充 |
| + | 东方财富实时 | push2.eastmoney.com | 当日换手率补充 |

## 定时规则（北京时间）

| 时段 | cron (UTC) | 用途 |
|------|-----------|------|
| 收盘后 | `45 7 * * 1-5` | 工作日 15:45，获取完整日线 |
| 盘前 | `0 0 * * 1-5` | 工作日 8:00，拉取盘前实时行情 |

## 手动运行

```bash
# 本地测试
pip install -r requirements.txt
python fetch_data.py

# 仅实时
python fetch_data.py --realtime

# 仅历史
python fetch_data.py --history --datalen 120

# 生成持仓信号
python portfolio_signal.py
```

## V5 单ETF 策略

| 条件 | 阈值 |
|------|------|
| 价格区间 | ¥0.5 – ¥3.0 |
| 质量门槛 | quality_score > 0.5 |
| 动量要求 | 40日动量 > 0 |
| 选股数 | **1 只**（quality_score 最高） |

| HS300动量 | V5仓位 |
|:--------:|:----:|
| > 2% | 满仓 (1.0) |
| 0% ~ 2% | 7成 |
| -2% ~ 0% | 半仓 |
| ≤ -2% + 逆势质优≥2只 | 3成试探 |
| ≤ -2% 且无质优 | 空仓 |

## signal.json 结构

```json
{
  "market": {
    "strategy": "V5",
    "position_ratio": 0.7,
    "position_text": "7成",
    "position_reason": "V5单ETF | HS300动量=+1.2%"
  },
  "portfolio": {
    "net_worth": 9123.45,
    "holdings": [{"name":"半导体ETF", "shares":500, "pnl_pct":3.2}],
    "target": [{"name":"半导体ETF", "weight":0.7, "quality_score":1.8}]
  },
  "orders": [{
    "action": "BUY",
    "name": "半导体ETF", "shares": 100,
    "price_est": 4.690,
    "amount": 479.00,
    "reason": "新建 70% (回调买入)"
  }],
  "sectors": [{"sector":"电子", "avg_momentum":15.2, "status":"强势"}]
}
```

## 交易规则

| 规则 | 说明 |
|------|------|
| T+1 锁仓 | 当日买入份额锁定至下一交易日 |
| 买入价 | 次日 (开盘+最低)/2 (有回调) 或 开盘价 (无回调) |
| 卖出价 | 次日 (最高+收盘)/2 (强势位卖出) |
| 最低交易 | 金额 ≥ ¥500 (否则手续费占比过高) |
| 微调跳过 | 同一 ETF 权重变化 < 20% 不调仓 |
| 手续费 | ¥10/笔 |
| 手数 | 100 股整数倍 |
```

## 部署步骤

1. 上传本仓库到 GitHub
2. Actions 页 → `ETF 数据抓取` → **Run workflow** 手动跑一次
3. 数据自动保存到 `data/` 目录并推送到仓库
4. 之后每个交易日自动运行，无需人工干预
