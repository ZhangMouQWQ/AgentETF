#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
行业ETF动量轮动策略 - 自洽仓位管理版
========================================
特点：
1. 不依赖外部持仓输入，自维护建议状态
2. 上午只风控（卖出/减仓），下午才找机会（买入/加仓）
3. 明确给出总仓位几成 + 每只标的仓位
4. T+1安全：上午绝不建议买入
"""
import requests
import pandas as pd
import numpy as np
import time
import json
import os
from datetime import datetime, timezone, timedelta

# ==================== 配置 ====================
ETF_POOL = {
    '510300': ('sh510300', '沪深300ETF'), '510500': ('sh510500', '中证500ETF'),
    '159915': ('sz159915', '创业板ETF'),  '588000': ('sh588000', '科创50ETF'),
    '512400': ('sh512400', '有色金属ETF'), '516780': ('sh516780', '稀土ETF'),
    '515220': ('sh515220', '煤炭ETF'),    '512000': ('sh512000', '券商ETF'),
    '512800': ('sh512800', '银行ETF'),    '159995': ('sz159995', '芯片ETF'),
    '515000': ('sh515000', '科技ETF'),    '512660': ('sh512660', '军工ETF'),
    '512010': ('sh512010', '医药ETF'),    '512690': ('sh512690', '酒ETF'),
    '159928': ('sz159928', '消费ETF'),    '515120': ('sh515120', '创新药ETF'),
    '515030': ('sh515030', '新能源车ETF'), '515790': ('sh515790', '光伏ETF'),
    '516110': ('sh516110', '汽车ETF'),    '518880': ('sh518880', '黄金ETF'),
}
DATA_LEN = 260
MOM_LONG = 40      # 长周期
MOM_SHORT = 5      # 短周期
STOP_LOSS = -4.0   # 当日止损线
MAX_DAILY_DROP = -4.0  # 买入跌幅上限
TOP_K = 3
PORTFOLIO_FILE = 'portfolio.json'
# ==============================================


def get_etf_sina(sina_code, datalen=DATA_LEN, retry=3):
    url = ("http://money.finance.sina.com.cn/quotes_service/api/json_v2.php/"
           f"CN_MarketData.getKLineData?symbol={sina_code}&scale=240&ma=no&datalen={datalen}")
    headers = {'User-Agent': 'Mozilla/5.0'}
    for attempt in range(retry):
        try:
            r = requests.get(url, timeout=15, headers=headers)
            text = r.text
            if text.startswith('var'):
                text = text.split('=', 1)[-1].strip().rstrip(';')
            j = json.loads(text)
            if j and isinstance(j, list):
                df = pd.DataFrame(j)
                for c in ['open', 'high', 'low', 'close', 'volume']:
                    df[c] = df[c].astype(float)
                df['day'] = pd.to_datetime(df['day']).dt.strftime('%Y-%m-%d')
                return df[['day', 'close']].rename(columns={'day': 'date'})
        except Exception as e:
            print(f"  [警告] {sina_code} 第{attempt+1}次失败: {e}")
            time.sleep(1)
    return None


def fetch_all_data(pool, datalen=DATA_LEN):
    all_close = {}
    etf_info = {}
    etf_last_date = {}
    today_str = datetime.now(timezone(timedelta(hours=8))).strftime('%Y-%m-%d')
    
    for code, (sina, name) in pool.items():
        df = get_etf_sina(sina, datalen)
        if df is not None and len(df) >= MOM_LONG + 5:
            all_close[name] = df.set_index('date')['close']
            etf_info[name] = code
            etf_last_date[name] = df['date'].iloc[-1]
        time.sleep(0.15)
    
    if not all_close:
        return None, {}, {}
    
    price = pd.DataFrame(all_close).sort_index()
    price_filled = price.ffill().dropna(how='any', axis=1).dropna(how='all')
    sync_asof = price_filled.index[-1]
    
    metrics = {}
    for name in price_filled.columns:
        s = price_filled[name]
        if len(s) < MOM_LONG + 2:
            continue
        latest, prev = s.iloc[-1], s.iloc[-2]
        mom_long = (latest / s.iloc[-MOM_LONG-1] - 1) * 100
        mom_short = (latest / s.iloc[-MOM_SHORT-1] - 1) * 100
        daily_change = (latest / prev - 1) * 100
        rets = s.pct_change().iloc[-MOM_LONG:]
        vol = rets.std() * np.sqrt(252) * 100
        score = (mom_long / vol) if vol > 0 else -999
        
        metrics[name] = {
            'code': etf_info[name],
            'latest': round(latest, 3),
            'prev': round(prev, 3),
            'daily_change': round(daily_change, 2),
            'mom_long': round(mom_long, 2),
            'mom_short': round(mom_short, 2),
            'vol': round(vol, 1),
            'score': round(score, 3),
            'data_date': etf_last_date[name],
        }
    return price_filled, etf_info, metrics, sync_asof


def market_timing(metrics):
    """大盘择时：决定总仓位几成"""
    m = metrics.get('沪深300ETF')
    if m is None:
        return 0.0, "0成（空仓）", "沪深300数据缺失，保守观望", "danger"
    
    if m['mom_long'] > 0 and m['mom_short'] > 0:
        return 1.0, "10成（满仓）", "大盘强势，积极参与", "ok"
    elif m['mom_long'] > 0 and m['mom_short'] <= 0:
        return 0.5, "5成（半仓）", "大盘震荡，控制仓位", "warn"
    else:
        return 0.0, "0成（空仓）", "大盘弱势，观望为主", "danger"


def build_target(metrics, position_ratio):
    """基于最新数据构建目标持仓"""
    if position_ratio <= 0:
        return []
    
    candidates = [
        (n, m) for n, m in metrics.items()
        if m['mom_long'] > 0 and m['mom_short'] > -2 and m['daily_change'] > MAX_DAILY_DROP
    ]
    candidates.sort(key=lambda x: x[1]['score'], reverse=True)
    selected = candidates[:TOP_K]
    
    if not selected:
        return []
    
    weight = round(position_ratio / len(selected), 2)
    # 修正权重确保总和准确
    weights = [weight] * len(selected)
    weights[-1] = round(position_ratio - sum(weights[:-1]), 2)
    
    return [
        {"name": n, "code": m['code'], "weight": w, "mom_long": m['mom_long'], "score": m['score']}
        for (n, m), w in zip(selected, weights)
    ]


def load_portfolio():
    """读取策略当前建议持仓"""
    if os.path.exists(PORTFOLIO_FILE):
        with open(PORTFOLIO_FILE, 'r', encoding='utf-8') as f:
            return json.load(f)
    return {"holdings": [], "cash_ratio": 1.0, "last_update": ""}


def save_portfolio(portfolio):
    with open(PORTFOLIO_FILE, 'w', encoding='utf-8') as f:
        json.dump(portfolio, f, ensure_ascii=False, indent=2)


def generate_actions(current, target, signal_type, metrics):
    """
    对比当前建议持仓 vs 目标持仓，生成操作指令
    上午(signal_type='morning')：只生成 SELL/REDUCE/HOLD（T+1保护，不生成买入）
    下午(signal_type='close')：生成全部类型
    """
    actions = []
    current_dict = {h['name']: h for h in current}
    target_dict = {h['name']: h for h in target}
    
    # ===== 卖出/减仓（上午和下午都可建议） =====
    for name, h in current_dict.items():
        m = metrics.get(name, {})
        # 触发风控条件
        risk_reasons = []
        if m.get('daily_change', 0) <= STOP_LOSS:
            risk_reasons.append(f"当日止损({m['daily_change']:+.2f}%)")
        if m.get('mom_short', 0) <= -2:
            risk_reasons.append(f"短期走弱({m['mom_short']:+.2f}%)")
        if name not in target_dict:
            risk_reasons.append("掉出排名")
        
        if risk_reasons:
            urgency = "high" if m.get('daily_change', 0) <= STOP_LOSS else "medium"
            if signal_type == 'morning':
                actions.append({
                    "type": "SELL", "name": name, "code": h['code'], "weight": h['weight'],
                    "msg": f"【下午清仓】{name}({h['code']}) {h['weight']*10:.0f}成，原因：{'、'.join(risk_reasons)}",
                    "urgency": urgency, "reason": "、".join(risk_reasons)
                })
            else:
                actions.append({
                    "type": "SELL", "name": name, "code": h['code'], "weight": h['weight'],
                    "msg": f"【次日清仓】{name}({h['code']}) {h['weight']*10:.0f}成，原因：{'、'.join(risk_reasons)}",
                    "urgency": urgency, "reason": "、".join(risk_reasons)
                })
        elif target_dict.get(name, {}).get('weight', 0) < h['weight'] - 0.03:
            # 减仓
            tw = target_dict[name]['weight']
            if signal_type == 'morning':
                actions.append({
                    "type": "REDUCE", "name": name, "code": h['code'], "weight": h['weight'] - tw,
                    "msg": f"【下午减仓】{name}({h['code']}) 从{h['weight']*10:.0f}成减至{tw*10:.0f}成",
                    "urgency": "medium", "reason": "大盘降仓或排名下降"
                })
            else:
                actions.append({
                    "type": "REDUCE", "name": name, "code": h['code'], "weight": h['weight'] - tw,
                    "msg": f"【次日减仓】{name}({h['code']}) 从{h['weight']*10:.0f}成减至{tw*10:.0f}成",
                    "urgency": "medium", "reason": "大盘降仓或排名下降"
                })
    
    # ===== 买入/加仓（仅下午建议，T+1保护） =====
    if signal_type == 'close':
        for name, h in target_dict.items():
            if name not in current_dict:
                actions.append({
                    "type": "BUY", "name": name, "code": h['code'], "weight": h['weight'],
                    "msg": f"【次日买入】{name}({h['code']}) {h['weight']*10:.0f}成",
                    "urgency": "normal", "reason": f"40日动量{h['mom_long']:+.2f}%，排名进入前{TOP_K}"
                })
            elif h['weight'] > current_dict[name].get('weight', 0) + 0.03:
                diff = round(h['weight'] - current_dict[name]['weight'], 2)
                actions.append({
                    "type": "ADD", "name": name, "code": h['code'], "weight": diff,
                    "msg": f"【次日加仓】{name}({h['code']}) 加{diff*10:.0f}成至{h['weight']*10:.0f}成",
                    "urgency": "normal", "reason": "大盘加仓或排名上升"
                })
    
    # ===== 持有 =====
    for name, h in target_dict.items():
        if name in current_dict and abs(h['weight'] - current_dict[name].get('weight', 0)) <= 0.03:
            actions.append({
                "type": "HOLD", "name": name, "code": h['code'], "weight": h['weight'],
                "msg": f"【持有】{name}({h['code']}) {h['weight']*10:.0f}成",
                "urgency": "low", "reason": "状态良好，无需操作"
            })
    
    # 空仓观望
    if not target and not current:
        actions.append({
            "type": "WAIT", "name": "空仓", "code": "", "weight": 0,
            "msg": "【观望】空仓等待，无符合条件的标的",
            "urgency": "low", "reason": "大盘弱势或个股无机会"
        })
    
    # 按紧急程度排序
    urgency_order = {"high": 0, "medium": 1, "normal": 2, "low": 3}
    actions.sort(key=lambda x: urgency_order.get(x['urgency'], 99))
    return actions


def build_html(actions, current, target, position_text, position_reason, market_cls,
               asof, update_time, signal_type, metrics):
    def action_rows():
        if not actions:
            return '<tr><td colspan="3" style="text-align:center;color:#999">无操作</td></tr>'
        out = []
        for a in actions:
            cls = {
                "SELL": "neg", "REDUCE": "warn-text",
                "BUY": "pos", "ADD": "pos",
                "HOLD": "hold", "WAIT": "hold"
            }.get(a['type'], "")
            badge = {
                "SELL": "清仓", "REDUCE": "减仓",
                "BUY": "买入", "ADD": "加仓",
                "HOLD": "持有", "WAIT": "观望"
            }.get(a['type'], a['type'])
            out.append(
                f'<tr><td class="nm"><span class="badge-{a["type"].lower()}">{badge}</span> {a["name"]}({a["code"]})</td>'
                f'<td class="{cls}">{a["msg"]}</td>'
                f'<td>{a["reason"]}</td></tr>')
        return '\n'.join(out)
    
    def holding_rows(holdings, title):
        if not holdings:
            return f'<p style="color:#999;text-align:center;padding:10px">{title}：无</p>'
        out = []
        total = sum(h['weight'] for h in holdings)
        for h in holdings:
            m = metrics.get(h['name'], {})
            daily = m.get('daily_change', 0)
            dcls = 'pos' if daily > 0 else ('neg' if daily < 0 else '')
            out.append(
                f'<div class="hold-item"><b>{h["name"]}({h["code"]})</b> '
                f'<span>{h["weight"]*10:.0f}成</span> '
                f'<span class="{dcls}">{daily:+.2f}%</span></div>')
        out.append(f'<div class="hold-item" style="border-top:1px solid #eee;margin-top:8px;padding-top:8px">'
                   f'<b>现金</b> <span>{(1-total)*10:.0f}成</span></div>')
        return '\n'.join(out)
    
    if signal_type == 'morning':
        label = "上午风控"
        sublabel = "11:30 盘中 · 下午操作窗口 13:00-15:00"
        tip = "⚠️ T+1 保护：上午只建议卖出/减仓，不建议买入（下午买入后当日无法止损）"
        tag_color = "tag-morning"
    else:
        label = "收盘决策"
        sublabel = "15:00 收盘 · 次日开盘操作或尾盘固定价"
        tip = "💡 T+1 提示：今日买入的标的，需待下一个交易日才能卖出；今日卖出资金当日可用"
        tag_color = "tag-close"
    
    return f"""<!DOCTYPE html>
<html lang="zh-CN"><head><meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>ETF动量轮动 · {label}</title>
<style>
*{{margin:0;padding:0;box-sizing:border-box}}
body{{font-family:-apple-system,"Segoe UI","PingFang SC","Microsoft YaHei",sans-serif;
background:linear-gradient(135deg,#0f2027,#203a43,#2c5364);min-height:100vh;padding:20px;color:#222}}
.wrap{{max-width:960px;margin:0 auto}}
.head{{text-align:center;color:#fff;margin-bottom:22px}}
.head h1{{font-size:24px;letter-spacing:1px}}
.head p{{opacity:.8;font-size:13px;margin-top:6px}}
.card{{background:#fff;border-radius:16px;padding:22px;margin-bottom:18px;
box-shadow:0 10px 30px rgba(0,0,0,.25)}}
.card h2{{font-size:17px;color:#2c5364;border-left:4px solid #2c5364;padding-left:10px;margin-bottom:14px}}
table{{width:100%;border-collapse:collapse;font-size:13px}}
th,td{{padding:10px 8px;text-align:left;border-bottom:1px solid #f0f0f0;vertical-align:top}}
th{{background:#f7f9fb;color:#666;font-weight:600}}
td.nm{{min-width:140px;font-weight:600}}
.pos{{color:#d7263d;font-weight:600}} .neg{{color:#1a936f;font-weight:600}}
.warn-text{{color:#e67e22;font-weight:600}} .hold{{color:#666}}
td.sc{{font-weight:700;color:#2c5364}}
.badge-sell,.badge-reduce{{display:inline-block;background:#d7263d;color:#fff;font-size:11px;
padding:2px 8px;border-radius:4px;margin-right:6px}}
.badge-buy,.badge-add{{display:inline-block;background:#27ae60;color:#fff;font-size:11px;
padding:2px 8px;border-radius:4px;margin-right:6px}}
.badge-hold,.badge-wait{{display:inline-block;background:#95a5a6;color:#fff;font-size:11px;
padding:2px 8px;border-radius:4px;margin-right:6px}}
.signal{{background:linear-gradient(135deg,#d7263d,#f46036);color:#fff;border-radius:14px;
padding:24px;text-align:center;margin-bottom:18px;box-shadow:0 8px 24px rgba(215,38,61,.35)}}
.signal .lab{{font-size:13px;opacity:.9}}
.signal .val{{font-size:28px;font-weight:700;margin-top:8px}}
.signal .sub{{font-size:14px;margin-top:6px;opacity:.9}}
.grid{{display:grid;grid-template-columns:1fr 1fr;gap:14px;margin-bottom:18px}}
.mini{{background:#fff;border-radius:14px;padding:18px;box-shadow:0 8px 24px rgba(0,0,0,.2)}}
.mini .t{{font-size:13px;color:#888}} .mini .v{{font-size:20px;font-weight:700;color:#2c5364;margin-top:6px}}
.hold-item{{display:flex;justify-content:space-between;padding:8px 0;border-bottom:1px solid #f5f5f5;font-size:14px}}
.market-ok{{color:#27ae60}} .market-warn{{color:#e67e22}} .market-danger{{color:#d7263d}}
.warn-box{{background:#fff3cd;border-left:4px solid #f39c12;padding:14px 18px;border-radius:8px;margin-bottom:18px;color:#856404;font-size:14px;line-height:1.6}}
.info-box{{background:#e8f6ff;border-left:4px solid #3498db;padding:14px 18px;border-radius:8px;margin-bottom:18px;color:#1a5276;font-size:14px;line-height:1.6}}
.note{{background:#fff;border-radius:14px;padding:18px;font-size:13px;color:#555;line-height:1.9}}
.note b{{color:#2c5364}}
.foot{{text-align:center;color:#fff;opacity:.7;font-size:12px;margin-top:14px}}
.tag{{display:inline-block;padding:2px 10px;border-radius:4px;font-size:12px;margin-left:8px;vertical-align:middle}}
.tag-morning{{background:#f39c12;color:#fff}} .tag-close{{background:#27ae60;color:#fff}}
@media(max-width:700px){{.grid{{grid-template-columns:1fr}} table{{font-size:12px}} .nm{{min-width:100px}}}}
</style></head><body><div class="wrap">
<div class="head"><h1>📊 ETF 动量轮动 · {label}<span class="tag {tag_color}">{label}</span></h1>
<p>{sublabel} · 仓位管理 + T+1安全</p></div>

<div class="signal"><div class="lab">当前建议总仓位</div>
<div class="val market-{market_cls}">{position_text}</div>
<div class="sub">{position_reason} · 数据截至 {asof}</div></div>

<div class="info-box">
<b>📋 操作提示</b><br>
{tip}<br>
<small>本次共 {len(actions)} 条指令，按紧急程度排序</small>
</div>

<div class="grid">
<div class="mini"><div class="t">当前建议持仓</div><div class="v" style="font-size:14px;line-height:1.6">
{holding_rows(current, "当前持仓")}
</div></div>
<div class="mini"><div class="t">目标持仓（{label}后）</div><div class="v" style="font-size:14px;line-height:1.6">
{holding_rows(target, "目标持仓")}
</div></div>
</div>

<div class="card"><h2>🎯 操作指令清单</h2>
<table><thead><tr><th class="nm">标的</th><th>操作详情</th><th>触发原因</th></tr></thead>
<tbody>{action_rows()}</tbody></table></div>

<div class="card"><h2>📊 大盘与标的监控</h2>
<table><thead><tr><th class="nm">ETF</th><th>40日动量</th><th>5日动量</th><th>当日涨跌</th><th>波动率</th><th>得分</th></tr></thead>
<tbody>{monitor_rows(metrics)}</tbody></table></div>

<div class="note"><b>策略逻辑</b><br>
<b>仓位</b>：沪深300 40日动量>0且5日动量>0→满仓10成；40日>0但5日≤0→半仓5成；40日≤0→空仓0成。<br>
<b>选股</b>：40日风险调整动量前3 + 40日动量>0 + 5日动量>-2% + 当日跌幅>-4%。<br>
<b>上午</b>：只风控（卖出/减仓），不买入（T+1保护）。<br>
<b>下午</b>：重新评估，给出次日目标持仓，可买可卖。<br><br>
<b>T+1 制度说明</b><br>
A股 ETF 实行T+1交易：今日买入的份额，需待下一个交易日才能卖出；今日卖出资金当日可用（可继续买入其他ETF），但不可取现至银行卡。</div>

<div class="foot">更新于 {update_time} · 数据源: 新浪财经 · 策略状态自动维护</div>
</div></body></html>"""


def monitor_rows(metrics):
    ranked = sorted(metrics.items(), key=lambda x: x[1]['score'], reverse=True)
    out = []
    for name, m in ranked[:8]:
        mom_cls = 'pos' if m['mom_long'] > 0 else 'neg'
        short_cls = 'pos' if m['mom_short'] > 0 else 'neg'
        daily_cls = 'pos' if m['daily_change'] > 0 else 'neg'
        out.append(
            f'<tr><td class="nm">{name}({m["code"]})</td>'
            f'<td class="{mom_cls}">{m["mom_long"]:+.2f}%</td>'
            f'<td class="{short_cls}">{m["mom_short"]:+.2f}%</td>'
            f'<td class="{daily_cls}">{m["daily_change"]:+.2f}%</td>'
            f'<td>{m["vol"]:.1f}%</td><td class="sc">{m["score"]:.3f}</td></tr>')
    return '\n'.join(out)


def main():
    bj = timezone(timedelta(hours=8))
    now = datetime.now(bj)
    update_time = now.strftime('%Y-%m-%d %H:%M:%S')
    signal_type = 'morning' if 11 <= now.hour <= 12 else 'close'
    
    print(f"\n{'='*60}")
    print(f"[{update_time}] {signal_type}")
    print(f"{'='*60}")

    # 拉取数据
    price, etf_info, metrics, asof = fetch_all_data(ETF_POOL)
    if not metrics:
        print("数据失败")
        return
    
    # 大盘择时
    position_ratio, position_text, position_reason, market_cls = market_timing(metrics)
    print(f"大盘: {position_text} ({position_reason})")
    
    # 读取当前建议持仓
    portfolio = load_portfolio()
    current = portfolio.get('holdings', [])
    print(f"当前建议持仓: {[(h['name'], h['weight']) for h in current]}")
    
    # 构建目标持仓
    target = build_target(metrics, position_ratio)
    print(f"目标持仓: {[(h['name'], h['weight']) for h in target]}")
    
    # 生成操作指令
    actions = generate_actions(current, target, signal_type, metrics)
    for a in actions:
        print(f"  {a['type']}: {a['msg']}")
    
    # 更新建议持仓状态
    # 注意：上午只更新卖出部分，下午更新全部（包括买入）
    new_holdings = []
    if signal_type == 'morning':
        # 上午：只移除卖出的，保留持有的，不加入新的
        sold_names = {a['name'] for a in actions if a['type'] in ('SELL',)}
        reduced = {a['name']: a for a in actions if a['type'] == 'REDUCE'}
        for h in current:
            if h['name'] in sold_names:
                continue
            if h['name'] in reduced:
                # 找到目标权重
                for t in target:
                    if t['name'] == h['name']:
                        new_holdings.append({**h, 'weight': t['weight']})
                        break
            else:
                new_holdings.append(h)
        # 现金重新计算
        cash = 1 - sum(h['weight'] for h in new_holdings)
    else:
        # 下午：直接更新为目标持仓
        new_holdings = target
        cash = 1 - sum(h['weight'] for h in target)
    
    save_portfolio({
        "holdings": new_holdings,
        "cash_ratio": round(cash, 2),
        "last_update": update_time,
        "position_ratio": position_ratio
    })
    
    # 生成HTML
    html = build_html(actions, current, target, position_text, position_reason,
                      market_cls, asof, update_time, signal_type, metrics)
    with open('index.html', 'w', encoding='utf-8') as f:
        f.write(html)
    
    # data.json
    data = {
        'update_time': update_time, 'asof': asof, 'signal_type': signal_type,
        'market': {'position_text': position_text, 'reason': position_reason, 'ratio': position_ratio},
        'current_holdings': current, 'target_holdings': target,
        'actions': actions, 'metrics': metrics
    }
    with open('data.json', 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    
    print(f"\n已生成 index.html, data.json, {PORTFOLIO_FILE}")


if __name__ == '__main__':
    main()
