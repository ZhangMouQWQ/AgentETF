#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
行业ETF动量轮动策略 - GitHub Actions 盯盘版
============================================
每个交易日运行两次:
  · 午盘休盘(约11:35) → 给出"今日下午"交易窗口的调仓指令
  · 收盘后(约15:05)   → 给出"下一交易日上午"交易窗口的调仓指令

输出:
  1. 下个交易窗口是否调仓(与上期持仓对比: 卖出/买入/维持)
  2. 买入推荐(两方案交集中最优1只 + 理由)
  3. 风险提醒(市场宽度/板块集中度/高波动/动量衰减)

数据源: 新浪财经公开接口(免费, 无需token)
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
# 板块大类(用于集中度检测)
SECTOR = {
    '沪深300ETF': '宽基', '中证500ETF': '宽基',
    '创业板ETF': '科技成长', '科创50ETF': '科技成长', '芯片ETF': '科技成长',
    '科技ETF': '科技成长', '军工ETF': '科技成长',
    '有色金属ETF': '资源', '稀土ETF': '资源', '煤炭ETF': '资源', '黄金ETF': '资源',
    '券商ETF': '金融', '银行ETF': '金融',
    '医药ETF': '医药消费', '创新药ETF': '医药消费', '酒ETF': '医药消费', '消费ETF': '医药消费',
    '新能源车ETF': '新能源制造', '光伏ETF': '新能源制造', '汽车ETF': '新能源制造',
}
DATA_LEN = 260
MOM_MAIN, TOP_MAIN = 40, 3      # 主推方案
MOM_SAFE, TOP_SAFE = 120, 2     # 高胜率方案
# ==============================================


def get_etf_sina(sina_code, datalen=DATA_LEN, retry=3):
    """新浪ETF日K线(含当日open/high/low, 盘中为实时价)"""
    url = ("http://money.finance.sina.com.cn/quotes_service/api/json_v2.php/"
           f"CN_MarketData.getKLineData?symbol={sina_code}&scale=240&ma=no&datalen={datalen}")
    headers = {'User-Agent': 'Mozilla/5.0'}
    for _ in range(retry):
        try:
            r = requests.get(url, timeout=15, headers=headers)
            j = r.json()
            if j:
                df = pd.DataFrame(j)
                for c in ['open', 'high', 'low', 'close', 'volume']:
                    df[c] = df[c].astype(float)
                return df[['day', 'open', 'high', 'low', 'close', 'volume']].rename(columns={'day': 'date'})
        except Exception:
            time.sleep(1)
    return None


def calc_ranking(pool, mom_window):
    all_close, all_open = {}, {}
    for code, (sina, name) in pool.items():
        df = get_etf_sina(sina)
        if df is not None and len(df) >= mom_window + 5:
            all_close[name] = df.set_index('date')['close']
            all_open[name] = df.set_index('date')['open']
        time.sleep(0.15)
    if not all_close:
        return None, None, None
    price = pd.DataFrame(all_close).sort_index()
    openp = pd.DataFrame(all_open).sort_index()
    # 数据行数不足时放弃(避免iloc越界)
    if len(price) < mom_window + 2:
        return None, None, None
    rets = price.pct_change()
    mom = price.iloc[-1] / price.iloc[-mom_window] - 1
    vol = rets.iloc[-mom_window:].std() * np.sqrt(252)
    risk_adj = (mom / vol).replace([np.inf, -np.inf], np.nan).dropna().sort_values(ascending=False)
    today = ((price.iloc[-1] / openp.iloc[-1] - 1) * 100).round(2)
    result = pd.DataFrame({
        'mom_pct': (mom[risk_adj.index] * 100).round(2),
        'vol_pct': (vol[risk_adj.index] * 100).round(2),
        'score': risk_adj.round(3),
        'today_pct': today.reindex(risk_adj.index),  # 当日涨跌, 对齐同一index
    })
    # 显式按风险调整动量降序(避免DataFrame对齐打乱顺序)
    result = result.sort_values('score', ascending=False)
    return result, price, openp


def pick(result, top_k):
    if result is None:
        return []
    top = result.head(top_k)
    return [n for n in top.index if result.loc[n, 'mom_pct'] > 0]


def detect_session(bj_now):
    """判断当前时段 → 返回(时段名, 下个交易窗口)"""
    hm = bj_now.hour * 100 + bj_now.minute
    if hm < 1300:   # 13:00前 = 午盘休盘时段
        return '午盘休盘', '今日下午(13:00开盘后)'
    else:           # 收盘后
        return '下午收盘', '下一交易日上午(9:30开盘后)'


def compare_holdings(new_picks, prev_picks):
    """对比新旧持仓 → 调仓指令"""
    prev = set(prev_picks or [])
    new = set(new_picks)
    sell = [p for p in (prev_picks or []) if p not in new]  # 保持原顺序
    buy = [p for p in new_picks if p not in prev]
    hold = [p for p in new_picks if p in prev]
    changed = bool(sell or buy)
    return {'changed': changed, 'sell': sell, 'buy': buy, 'hold': hold,
            'prev': prev_picks or [], 'new': new_picks}


def make_top_pick(picks_main, picks_safe, rank_main, rank_safe):
    """买入推荐: 优先两方案交集, 取风险调整动量最高者"""
    common = [p for p in picks_main if p in picks_safe]
    if common:
        # 交集中取主推方案分数最高
        best = max(common, key=lambda n: rank_main.loc[n, 'score'])
        src = '两方案交集(最强信号)'
    elif picks_main:
        best = picks_main[0]
        src = '主推方案第一名'
    else:
        return None
    row = rank_main.loc[best]
    return {
        'name': best, 'source': src,
        'mom_pct': float(row['mom_pct']), 'vol_pct': float(row['vol_pct']),
        'score': float(row['score']), 'today_pct': float(row['today_pct']),
        'sector': SECTOR.get(best, '其他'),
    }


def make_risks(rank_main, picks_main, top_pick, market_pos_ratio):
    """生成风险提醒列表"""
    risks = []
    # 1. 市场宽度
    if market_pos_ratio < 0.3:
        risks.append({'level': 'high', 'text': f'市场宽度极窄: 仅{market_pos_ratio*100:.0f}%板块动量为正, 整体弱势, 建议降低总仓位'})
    elif market_pos_ratio < 0.5:
        risks.append({'level': 'mid', 'text': f'市场宽度偏窄: {market_pos_ratio*100:.0f}%板块动量为正, 热点集中, 注意轮动加快'})
    # 2. 板块集中度
    if len(picks_main) >= 2:
        sectors = [SECTOR.get(p, '其他') for p in picks_main]
        if len(set(sectors)) == 1:
            risks.append({'level': 'mid', 'text': f'持仓集中在[{sectors[0]}]单一板块, 板块回调时回撤会放大, 注意控制仓位'})
    # 3. 推荐标的高波动
    if top_pick and top_pick['vol_pct'] > 50:
        risks.append({'level': 'mid', 'text': f"买入推荐[{top_pick['name']}]年化波动{top_pick['vol_pct']:.0f}%, 属高波动品种, 建议分批建仓"})
    # 4. 推荐标的当日走弱
    if top_pick and top_pick['today_pct'] < -2:
        risks.append({'level': 'mid', 'text': f"买入推荐[{top_pick['name']}]当日下跌{top_pick['today_pct']:.1f}%, 动量短期衰减, 可等企稳再介入"})
    # 5. 无强势板块
    if not picks_main:
        risks.append({'level': 'high', 'text': '全部板块动量为负, 无买入信号, 建议空仓观望'})
    # 兜底: 无风险时给一条中性提示
    if not risks:
        risks.append({'level': 'low', 'text': '当前无重大风险信号, 按计划执行即可'})
    return risks


def rows_html(ranking, picks, limit=10):
    if ranking is None:
        return '<tr><td colspan="6" style="text-align:center;color:#999">数据获取失败</td></tr>'
    out = []
    for i, (name, row) in enumerate(ranking.head(limit).iterrows(), 1):
        is_pick = name in picks
        badge = '<span class="buy">持有</span>' if is_pick else ''
        mom_cls = 'pos' if row['mom_pct'] > 0 else 'neg'
        td_cls = 'pos' if row['today_pct'] > 0 else 'neg'
        rank_cls = 'top' if is_pick else ''
        out.append(
            f'<tr class="{rank_cls}"><td class="rk">{i}</td><td class="nm">{name}{badge}</td>'
            f'<td class="{mom_cls}">{row["mom_pct"]:+.2f}%</td><td class="{td_cls}">{row["today_pct"]:+.2f}%</td>'
            f'<td>{row["vol_pct"]:.0f}%</td><td class="sc">{row["score"]:.3f}</td></tr>')
    return '\n'.join(out)


def action_html(action, next_window):
    """调仓指令卡片"""
    if action['changed']:
        sell_txt = '、'.join(action['sell']) if action['sell'] else '无'
        buy_txt = '、'.join(action['buy']) if action['buy'] else '无'
        body = (f'<div class="act-row"><span class="tag sell">卖出</span><span>{sell_txt}</span></div>'
                f'<div class="act-row"><span class="tag buyb">买入</span><span>{buy_txt}</span></div>'
                f'<div class="act-row"><span class="tag hold">持有</span><span>{"、".join(action["hold"]) or "无"}</span></div>')
        head = f'<div class="act-h change">⚠️ 需要调仓</div>'
    else:
        cur = '、'.join(action['new']) if action['new'] else '空仓'
        body = f'<div class="act-keep">维持当前持仓: <b>{cur}</b></div>'
        head = '<div class="act-h keep">✅ 无需调仓</div>'
    return (f'<div class="card action"><h2>📌 下个交易窗口操作指令'
            f'<span class="win">{next_window}</span></h2>{head}{body}</div>')


def risks_html(risks):
    items = []
    for r in risks:
        cls = {'high': 'r-high', 'mid': 'r-mid', 'low': 'r-low'}[r['level']]
        icon = {'high': '🔴', 'mid': '🟡', 'low': '🟢'}[r['level']]
        items.append(f'<div class="risk {cls}">{icon} {r["text"]}</div>')
    return '<div class="card"><h2>⚠️ 风险提醒</h2>' + ''.join(items) + '</div>'


def build_html(rank_main, picks_main, rank_safe, picks_safe, asof, update_time,
               session, next_window, action, top_pick, risks, data_note=''):
    main_txt = ' + '.join(picks_main) if picks_main else '空仓(全部动量为负)'
    safe_txt = ' + '.join(picks_safe) if picks_safe else '空仓(全部动量为负)'

    # 买入推荐卡片
    if top_pick:
        tp = (f'<div class="pick"><div class="pick-lab">🎯 买入推荐(下个交易窗口)</div>'
              f'<div class="pick-name">{top_pick["name"]}</div>'
              f'<div class="pick-meta"><span>{top_pick["source"]}</span><span>板块: {top_pick["sector"]}</span></div>'
              f'<div class="pick-nums"><div><b>{top_pick["mom_pct"]:+.1f}%</b><i>40日动量</i></div>'
              f'<div><b>{top_pick["vol_pct"]:.0f}%</b><i>年化波动</i></div>'
              f'<div><b>{top_pick["score"]:.2f}</b><i>风险调整动量</i></div></div></div>')
    else:
        tp = '<div class="pick none"><div class="pick-lab">🎯 买入推荐</div><div class="pick-name">无</div><div class="pick-meta">全部板块动量为负, 空仓观望</div></div>'

    return f"""<!DOCTYPE html>
<html lang="zh-CN"><head><meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>ETF动量轮动 · 盯盘信号</title>
<style>
*{{margin:0;padding:0;box-sizing:border-box}}
body{{font-family:-apple-system,"Segoe UI","PingFang SC","Microsoft YaHei",sans-serif;
background:linear-gradient(135deg,#0f2027,#203a43,#2c5364);min-height:100vh;padding:18px;color:#222}}
.wrap{{max-width:880px;margin:0 auto}}
.head{{text-align:center;color:#fff;margin-bottom:18px}}
.head h1{{font-size:25px;letter-spacing:1px}}
.head p{{opacity:.85;font-size:13px;margin-top:6px}}
.head .sess{{display:inline-block;background:rgba(255,255,255,.15);padding:3px 12px;border-radius:12px;margin-top:8px;font-size:13px}}
.card{{background:#fff;border-radius:16px;padding:20px;margin-bottom:16px;box-shadow:0 10px 30px rgba(0,0,0,.25)}}
.card h2{{font-size:16px;color:#2c5364;border-left:4px solid #2c5364;padding-left:10px;margin-bottom:14px;display:flex;align-items:center;gap:10px;flex-wrap:wrap}}
.card h2 .win{{font-size:12px;background:#2c5364;color:#fff;padding:2px 10px;border-radius:10px;border-left:none;font-weight:400}}
.pick{{background:linear-gradient(135deg,#d7263d,#f46036);color:#fff;border-radius:16px;padding:22px;margin-bottom:16px;box-shadow:0 8px 24px rgba(215,38,61,.35);text-align:center}}
.pick.none{{background:linear-gradient(135deg,#6b7280,#9ca3af)}}
.pick-lab{{font-size:13px;opacity:.92}} .pick-name{{font-size:30px;font-weight:800;margin:8px 0;letter-spacing:2px}}
.pick-meta{{display:flex;gap:14px;justify-content:center;font-size:12px;opacity:.9;flex-wrap:wrap}}
.pick-nums{{display:flex;gap:8px;justify-content:center;margin-top:14px;flex-wrap:wrap}}
.pick-nums div{{background:rgba(255,255,255,.18);border-radius:10px;padding:8px 16px;min-width:90px}}
.pick-nums b{{display:block;font-size:18px}} .pick-nums i{{font-size:11px;font-style:normal;opacity:.85}}
.action .act-h{{font-size:17px;font-weight:700;margin-bottom:12px}}
.act-h.change{{color:#d7263d}} .act-h.keep{{color:#1a936f}}
.act-row{{display:flex;align-items:center;gap:12px;padding:9px 0;border-bottom:1px dashed #eee;font-size:15px}}
.tag{{font-size:12px;padding:3px 12px;border-radius:10px;color:#fff;min-width:44px;text-align:center}}
.tag.sell{{background:#1a936f}} .tag.buyb{{background:#d7263d}} .tag.hold{{background:#888}}
.act-keep{{font-size:15px;padding:6px 0}} .act-keep b{{color:#2c5364}}
.grid{{display:grid;grid-template-columns:1fr 1fr;gap:14px;margin-bottom:16px}}
.mini{{background:#fff;border-radius:14px;padding:15px;box-shadow:0 8px 24px rgba(0,0,0,.2)}}
.mini .t{{font-size:12px;color:#888}} .mini .v{{font-size:16px;font-weight:700;color:#2c5364;margin-top:6px}}
table{{width:100%;border-collapse:collapse;font-size:13px}}
th,td{{padding:8px 6px;text-align:right;border-bottom:1px solid #f0f0f0}}
th{{background:#f7f9fb;color:#666;font-weight:600}}
td.rk,th.rk,td.nm,th.nm{{text-align:left}}
tr.top{{background:#fff8e6}}
.pos{{color:#d7263d;font-weight:600}} .neg{{color:#1a936f}}
td.sc{{font-weight:700;color:#2c5364}}
.buy{{display:inline-block;background:#d7263d;color:#fff;font-size:10px;padding:1px 7px;border-radius:9px;margin-left:6px;vertical-align:middle}}
.risk{{padding:10px 12px;border-radius:10px;margin-bottom:8px;font-size:13.5px;line-height:1.5}}
.r-high{{background:#fde8ea;border-left:4px solid #d7263d;color:#8a1225}}
.r-mid{{background:#fff6e0;border-left:4px solid #f0a500;color:#7a5200}}
.r-low{{background:#e8f6ef;border-left:4px solid #1a936f;color:#0d5a3f}}
.note{{background:#fff;border-radius:14px;padding:16px;font-size:12.5px;color:#555;line-height:1.9}}
.note b{{color:#2c5364}}
.foot{{text-align:center;color:#fff;opacity:.7;font-size:12px;margin-top:12px}}
@media(max-width:600px){{.grid{{grid-template-columns:1fr}}}}
</style></head><body><div class="wrap">
<div class="head"><h1>📊 ETF 动量轮动 · 盯盘信号</h1>
<p>不预测,只跟随强者 · 风险调整动量(涨幅÷波动率)排名</p>
<div class="sess">当前: {session} · 数据截至 {asof}</div></div>
{f'<div style="background:#fff6e0;color:#7a5200;border-radius:12px;padding:10px 14px;margin-bottom:16px;font-size:13px;text-align:center">{data_note}</div>' if data_note else ''}

{tp}
{action_html(action, next_window)}

<div class="grid">
<div class="mini"><div class="t">主推方案 应持仓(40日/持3只)</div><div class="v">{main_txt}</div></div>
<div class="mini"><div class="t">高胜率方案 应持仓(120日/持2只)</div><div class="v">{safe_txt}</div></div>
</div>

{risks_html(risks)}

<div class="card"><h2>主推方案 · 风险调整动量{MOM_MAIN}日排名</h2>
<table><thead><tr><th class="rk">#</th><th class="nm">ETF</th><th>40日动量</th><th>当日</th><th>年化波动</th><th>风险调整动量</th></tr></thead>
<tbody>{rows_html(rank_main, picks_main)}</tbody></table></div>

<div class="card"><h2>高胜率方案 · 风险调整动量{MOM_SAFE}日排名</h2>
<table><thead><tr><th class="rk">#</th><th class="nm">ETF</th><th>120日动量</th><th>当日</th><th>年化波动</th><th>风险调整动量</th></tr></thead>
<tbody>{rows_html(rank_safe, picks_safe)}</tbody></table></div>

<div class="note"><b>使用说明</b><br>
1. 每个交易日更新两次: 午盘休盘(约11:35)→提示"今日下午"操作; 收盘后(约15:05)→提示"下一交易日上午"操作<br>
2. 看"操作指令"卡片: 显示卖出/买入/持有, 或"无需调仓"<br>
3. "买入推荐"是两方案交集中的最优标的, 适合作为重点配置<br>
4. 动量为负的板块不买, 该仓位持现金(绝对动量择时)<br><br>
<b>回测表现(2023-06~2026-07)</b>: 主推年化28.8%/月胜率70.3%/最大回撤-23.5%; 高胜率年化21.8%; 同期沪深300年化9.5%<br>
<b>风险提示</b>: 历史回测不代表未来收益, 单年可能跑输大盘, 请结合自身风险承受能力决策。</div>

<div class="foot">更新于 {update_time} · 由 GitHub Actions 自动计算 · 数据源: 新浪财经</div>
</div></body></html>"""


def main():
    bj = timezone(timedelta(hours=8))
    bj_now = datetime.now(bj)
    update_time = bj_now.strftime('%Y-%m-%d %H:%M:%S')
    session, next_window = detect_session(bj_now)
    print(f"[{update_time}] 时段={session}, 下个窗口={next_window}")

    # 计算排名(主推失败则尝试用高胜率, 都失败则保留上次结果)
    rank_main, price, openp = calc_ranking(ETF_POOL, MOM_MAIN)
    rank_safe, _, _ = calc_ranking(ETF_POOL, MOM_SAFE)

    if rank_main is None and rank_safe is None:
        print("⚠️ 数据获取失败(网络抖动), 保留上次结果, 本次不更新")
        if not os.path.exists('index.html'):
            with open('index.html', 'w', encoding='utf-8') as f:
                f.write('<!DOCTYPE html><html lang="zh-CN"><head><meta charset="UTF-8">'
                        '<meta name="viewport" content="width=device-width,initial-scale=1">'
                        '<title>数据获取失败</title></head><body style="font-family:sans-serif;'
                        'text-align:center;padding:60px;color:#555">'
                        '<h2>⚠️ 数据获取失败</h2><p>数据源暂时不可用, 请稍后刷新重试。</p>'
                        f'<p style="color:#999;font-size:12px">{update_time}</p></body></html>')
        return

    picks_main = pick(rank_main, TOP_MAIN)
    picks_safe = pick(rank_safe, TOP_SAFE)
    asof = price.index[-1] if price is not None else '未知'

    # === 细节: 数据新鲜度检测(节假日/数据延迟时给出提示, 避免用旧数据误导) ===
    stale_days = 0
    data_note = ''
    try:
        last_dt = datetime.strptime(str(asof), '%Y-%m-%d').replace(tzinfo=bj)
        stale_days = (bj_now.date() - last_dt.date()).days
        if stale_days >= 3:
            data_note = f'⚠️ 数据为{stale_days}天前(可能遇节假日), 信号基于最近交易日, 请在节后首个交易日复核'
        elif stale_days >= 1:
            data_note = f'数据截至上个交易日({asof})'
    except Exception:
        pass

    # 市场宽度(动量为正占比)
    pos_ratio = float((rank_main['mom_pct'] > 0).mean()) if rank_main is not None else 0.0

    # 读取上期持仓做对比
    prev_picks = []
    if os.path.exists('data.json'):
        try:
            with open('data.json', 'r', encoding='utf-8') as f:
                prev = json.load(f)
            prev_picks = prev.get('main_plan', {}).get('buy', [])
        except Exception:
            prev_picks = []
    action = compare_holdings(picks_main, prev_picks)

    # 买入推荐 + 风险提醒
    top_pick = make_top_pick(picks_main, picks_safe, rank_main, rank_safe)
    risks = make_risks(rank_main, picks_main, top_pick, pos_ratio)

    print(f"主推应持仓: {picks_main}")
    print(f"高胜率应持仓: {picks_safe}")
    print(f"上期持仓: {prev_picks} → 调仓={action['changed']}")
    print(f"买入推荐: {top_pick['name'] if top_pick else '无'}")
    print(f"风险提醒: {len(risks)}条")

    # 生成网页
    html = build_html(rank_main, picks_main, rank_safe, picks_safe, asof, update_time,
                      session, next_window, action, top_pick, risks, data_note)
    with open('index.html', 'w', encoding='utf-8') as f:
        f.write(html)

    # 结构化数据(含本期持仓, 供下期对比)
    data = {
        'update_time': update_time, 'session': session, 'next_window': next_window, 'asof': asof,
        'main_plan': {'window': MOM_MAIN, 'top_k': TOP_MAIN, 'buy': picks_main},
        'safe_plan': {'window': MOM_SAFE, 'top_k': TOP_SAFE, 'buy': picks_safe},
        'action': action,
        'top_pick': top_pick,
        'risks': risks,
        'market_pos_ratio': round(pos_ratio, 3),
        'data_note': data_note,
    }
    with open('data.json', 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

    print("已生成 index.html 和 data.json")


if __name__ == '__main__':
    main()
