#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
行业ETF动量轮动策略 - 概念板块分组版(双层架构)
========================================
改进内容:
1. ETF按热门概念板块分组,每个板块放最具代表性的ETF
2. 40日动量基于昨日收盘计算(稳定层),不受盘中未完成数据干扰
3. 当日涨跌/5日动量独立用于风控和买入确认(灵活层)
4. 买入增加"当日涨幅<5%"过滤,避免盘中追高
5. 60分钟K线仅用于合成当日最新价,不用于历史动量计算
6. 波动率基于历史日K线(不含今天盘中噪声)
"""
import requests
import pandas as pd
import numpy as np
import time
import json
import os
from datetime import datetime, timezone, timedelta

# ==================== 配置 ====================
SECTOR_ETF_POOL = {
    '宽基指数': {
        '510300': ('sh510300', '沪深300ETF'),
        '510500': ('sh510500', '中证500ETF'),
        '159915': ('sz159915', '创业板ETF'),
        '588000': ('sh588000', '科创50ETF'),
    },
    '周期资源': {
        '512400': ('sh512400', '有色金属ETF'),
        '516780': ('sh516780', '稀土ETF'),
        '515220': ('sh515220', '煤炭ETF'),
        '518880': ('sh518880', '黄金ETF'),
    },
    '科技成长': {
        '159995': ('sz159995', '芯片ETF'),
        '515000': ('sh515000', '科技ETF'),
    },
    '新能源': {
        '515030': ('sh515030', '新能源车ETF'),
        '515790': ('sh515790', '光伏ETF'),
    },
    '医药医疗': {
        '512010': ('sh512010', '医药ETF'),
        '515120': ('sh515120', '创新药ETF'),
    },
    '大消费': {
        '512690': ('sh512690', '酒ETF'),
        '159928': ('sz159928', '消费ETF'),
    },
    '金融地产': {
        '512000': ('sh512000', '券商ETF'),
        '512800': ('sh512800', '银行ETF'),
    },
    '先进制造': {
        '512660': ('sh512660', '军工ETF'),
        '516110': ('sh516110', '汽车ETF'),
    },
}


def flatten_pool(sector_pool):
    """将分组ETF池扁平化,同时记录每个ETF所属板块"""
    flat = {}
    etf_sector = {}
    for sector, etfs in sector_pool.items():
        for code, (sina, name) in etfs.items():
            flat[code] = (sina, name)
            etf_sector[name] = sector
    return flat, etf_sector


ETF_POOL, ETF_SECTOR = flatten_pool(SECTOR_ETF_POOL)

DATA_LEN = 260
MOM_LONG = 40
MOM_SHORT = 5
STOP_LOSS = -4.0
MAX_DAILY_DROP = -4.0
MAX_DAILY_RISE = 5.0
VOL_WINDOW = 60
MIN_VOL = 5.0
TOP_K = 3
PORTFOLIO_FILE = 'portfolio.json'
# ==============================================


def get_etf_sina(sina_code, scale=240, datalen=DATA_LEN, retry=3):
    url = ("http://money.finance.sina.com.cn/quotes_service/api/json_v2.php/"
           f"CN_MarketData.getKLineData?symbol={sina_code}&scale={scale}&ma=no&datalen={datalen}")
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
                    if c in df.columns:
                        df[c] = df[c].astype(float)
                if scale == 240:
                    df['day'] = pd.to_datetime(df['day']).dt.strftime('%Y-%m-%d')
                else:
                    df['day'] = pd.to_datetime(df['day']).dt.strftime('%Y-%m-%d %H:%M:%S')
                return df[['day', 'open', 'high', 'low', 'close', 'volume']]
        except Exception as e:
            print(f"  [警告] {sina_code} scale={scale} 第{attempt+1}次失败: {e}")
            time.sleep(1)
    return None


def fetch_daily_data(pool, datalen=DATA_LEN):
    all_close = {}
    etf_info = {}
    print("拉取历史日K线...")
    for code, (sina, name) in pool.items():
        df = get_etf_sina(sina, scale=240, datalen=datalen)
        if df is not None and len(df) >= MOM_LONG + 5:
            all_close[name] = df.set_index('day')['close']
            etf_info[name] = code
            print(f"  ok {name}({code}): {len(df)} 条, 最新 {df['day'].iloc[-1]}")
        else:
            print(f"  fail {name}({code}): 数据不足")
        time.sleep(0.15)
    if not all_close:
        return None, {}
    price = pd.DataFrame(all_close).sort_index()
    price = price.ffill().dropna(how='any', axis=1).dropna(how='all')
    print(f"日K线维度: {price.shape}, 最新日期: {price.index[-1]}")
    return price, etf_info


def fetch_intraday_snapshot(pool):
    today = datetime.now(timezone(timedelta(hours=8))).strftime('%Y-%m-%d')
    snapshot = {}
    print(f"\n拉取60分钟K线合成当日数据 (今日 {today})...")
    for code, (sina, name) in pool.items():
        df = get_etf_sina(sina, scale=60, datalen=20)
        if df is None or len(df) < 1:
            print(f"  fail {name}: 无分钟K线")
            continue
        df['date'] = df['day'].str[:10]
        df_today = df[df['date'] == today].sort_values('day')
        if len(df_today) < 1:
            print(f"  warn {name}: 今日无60分钟K线")
            continue
        last = df_today.iloc[-1]
        snapshot[name] = {
            'code': code,
            'close': float(last['close']),
        }
        print(f"  ok {name}: 最新 {snapshot[name]['close']:.3f} (基于 {df_today['day'].iloc[-1]})")
        time.sleep(0.15)
    return snapshot


def merge_intraday_price(price_daily, intraday_snapshot, etf_info):
    if not intraday_snapshot or price_daily is None:
        return price_daily
    today_str = datetime.now(timezone(timedelta(hours=8))).strftime('%Y-%m-%d')
    print(f"\n合并盘中数据到日K线...")

    if price_daily.index[-1] == today_str:
        print(f"日K最新日期已是 {today_str}, 用60分钟数据覆盖")
        for name, snap in intraday_snapshot.items():
            if name in price_daily.columns:
                price_daily.loc[today_str, name] = snap['close']
    else:
        print(f"日K最新日期 {price_daily.index[-1]}, 追加今日 {today_str}")
        new_row = pd.Series({n: np.nan for n in price_daily.columns}, name=today_str)
        for name in price_daily.columns:
            if name in intraday_snapshot:
                new_row[name] = intraday_snapshot[name]['close']
            else:
                new_row[name] = price_daily[name].iloc[-1]
        price_daily = pd.concat([price_daily, new_row.to_frame().T])

    price_daily = price_daily.dropna(how='all')
    print(f"合并后维度: {price_daily.shape}, 最新日期: {price_daily.index[-1]}")
    return price_daily.sort_index()


def calc_momentum_change(series, lookback=MOM_LONG):
    if len(series) < lookback + 3:
        return None, None, None
    prev_day = series.iloc[-2]
    prev_prev_day = series.iloc[-3]
    current_mom = (prev_day / series.iloc[-lookback-1] - 1) * 100
    previous_mom = (prev_prev_day / series.iloc[-lookback-2] - 1) * 100
    change = current_mom - previous_mom
    return current_mom, previous_mom, change


def calc_metrics(price, etf_info, etf_sector):
    metrics = {}
    if price is None or len(price) < MOM_LONG + 2:
        return metrics
    for name in price.columns:
        s = price[name]
        if len(s) < MOM_LONG + 2:
            continue

        latest = s.iloc[-1]
        prev = s.iloc[-2]

        mom_long = (prev / s.iloc[-MOM_LONG-1] - 1) * 100
        _, _, mom_long_change = calc_momentum_change(s, lookback=MOM_LONG)
        rets = s.iloc[:-1].pct_change().dropna().iloc[-VOL_WINDOW:]
        vol = rets.std() * np.sqrt(252) * 100 if len(rets) >= VOL_WINDOW // 2 else 999.0
        vol = max(vol, MIN_VOL)
        raw_score = (mom_long / vol) if mom_long > 0 else None

        daily_change = (latest / prev - 1) * 100
        mom_short = (prev / s.iloc[-MOM_SHORT-1] - 1) * 100

        direction = '持平'
        if mom_long_change is not None:
            if mom_long_change > 0:
                direction = '上升'
            elif mom_long_change < 0:
                direction = '下降'

        metrics[name] = {
            'code': etf_info.get(name, ''),
            'sector': etf_sector.get(name, '其他'),
            'latest': round(latest, 3),
            'prev': round(prev, 3),
            'daily_change': round(daily_change, 2),
            'mom_long': round(mom_long, 2),
            'mom_short': round(mom_short, 2),
            'mom_long_change': round(mom_long_change, 2) if mom_long_change is not None else 0.0,
            'mom_long_change_direction': direction,
            'mom_long_change_text': (('↑' if mom_long_change and mom_long_change > 0 else '↓' if mom_long_change and mom_long_change < 0 else '→') + f"{abs(mom_long_change):.2f}%") if mom_long_change is not None else '→0.00%',
            'vol': round(vol, 1),
            'score': raw_score,
        }

    positives = [m['score'] for m in metrics.values() if m['score'] is not None]
    for m in metrics.values():
        if m['score'] is None or not positives:
            m['score'] = 0.0
        else:
            lo, hi = min(positives), max(positives)
            m['score'] = 100.0 if hi == lo else round((m['score'] - lo) / (hi - lo) * 100, 1)
    return metrics


def market_timing(metrics):
    m = metrics.get('沪深300ETF')
    if m is None:
        return 0.0, "0成(空仓)", "沪深300数据缺失,保守观望", "danger"
    if m['mom_long'] > 2:
        return 1.0, "10成(满仓)", "大盘强势,积极参与", "ok"
    elif m['mom_long'] > 0:
        return 0.5, "5成(半仓)", "大盘震荡,控制仓位", "warn"
    else:
        return 0.0, "0成(空仓)", "大盘弱势,观望为主", "danger"


def build_target(metrics, position_ratio, signal_type='close'):
    if position_ratio <= 0:
        return []
    candidates = [
        (n, m) for n, m in metrics.items()
        if m['mom_long'] > 0
        and m['mom_short'] > -2
        and m['daily_change'] > MAX_DAILY_DROP
        and m['daily_change'] < MAX_DAILY_RISE
    ]
    if signal_type == 'close':
        candidates = [(n, m) for n, m in candidates if m['daily_change'] > 0]
    candidates.sort(key=lambda x: x[1]['score'], reverse=True)
    selected = candidates[:TOP_K]
    if not selected:
        return []
    weight = round(position_ratio / len(selected), 2)
    weights = [weight] * len(selected)
    weights[-1] = round(position_ratio - sum(weights[:-1]), 2)
    return [
        {"name": n, "code": m['code'], "weight": w, "mom_long": m['mom_long'], "score": m['score'], "sector": m['sector']}
        for (n, m), w in zip(selected, weights)
    ]


def load_portfolio():
    if os.path.exists(PORTFOLIO_FILE):
        with open(PORTFOLIO_FILE, 'r', encoding='utf-8') as f:
            return json.load(f)
    return {"holdings": [], "cash_ratio": 1.0, "last_update": "", "position_ratio": 0}


def save_portfolio(portfolio):
    with open(PORTFOLIO_FILE, 'w', encoding='utf-8') as f:
        json.dump(portfolio, f, ensure_ascii=False, indent=2)


def generate_actions(current, target, signal_type, metrics):
    actions = []
    current_dict = {h['name']: h for h in current}
    target_dict = {h['name']: h for h in target}

    for name, h in current_dict.items():
        m = metrics.get(name, {})
        risk_reasons = []
        if m.get('daily_change', 0) <= STOP_LOSS:
            risk_reasons.append(f"当日止损({m['daily_change']:+.2f}%)")
        if m.get('mom_short', 0) <= -2:
            risk_reasons.append(f"短期走弱({m['mom_short']:+.2f}%)")
        if name not in target_dict:
            risk_reasons.append("掉出排名")

        if risk_reasons:
            urgency = "high" if m.get('daily_change', 0) <= STOP_LOSS else "medium"
            window = "下午" if signal_type == 'morning' else "次日"
            actions.append({
                "type": "SELL", "name": name, "code": h['code'], "weight": h['weight'],
                "msg": f"【{window}清仓】{name}({h['code']}) {h['weight']*10:.0f}成,原因:{'、'.join(risk_reasons)}",
                "urgency": urgency, "reason": "、".join(risk_reasons)
            })
        elif name in target_dict and target_dict[name]['weight'] < h['weight'] - 0.03:
            tw = target_dict[name]['weight']
            window = "下午" if signal_type == 'morning' else "次日"
            actions.append({
                "type": "REDUCE", "name": name, "code": h['code'], "weight": h['weight'] - tw,
                "msg": f"【{window}减仓】{name}({h['code']}) 从{h['weight']*10:.0f}成减至{tw*10:.0f}成",
                "urgency": "medium", "reason": "大盘降仓或排名下降"
            })

    if signal_type == 'close':
        for name, h in target_dict.items():
            if name not in current_dict:
                actions.append({
                    "type": "BUY", "name": name, "code": h['code'], "weight": h['weight'],
                    "msg": f"【次日买入】{name}({h['code']}) {h['weight']*10:.0f}成",
                    "urgency": "normal", "reason": f"40日动量{h['mom_long']:+.2f}%,排名进入前{TOP_K}"
                })
            elif h['weight'] > current_dict[name].get('weight', 0) + 0.03:
                diff = round(h['weight'] - current_dict[name]['weight'], 2)
                actions.append({
                    "type": "ADD", "name": name, "code": h['code'], "weight": diff,
                    "msg": f"【次日加仓】{name}({h['code']}) 加{diff*10:.0f}成至{h['weight']*10:.0f}成",
                    "urgency": "normal", "reason": "大盘加仓或排名上升"
                })

    for name, h in target_dict.items():
        if name in current_dict and abs(h['weight'] - current_dict[name].get('weight', 0)) <= 0.03:
            actions.append({
                "type": "HOLD", "name": name, "code": h['code'], "weight": h['weight'],
                "msg": f"【持有】{name}({h['code']}) {h['weight']*10:.0f}成",
                "urgency": "low", "reason": "状态良好,无需操作"
            })

    if not target and not current:
        actions.append({
            "type": "WAIT", "name": "空仓", "code": "", "weight": 0,
            "msg": "【观望】空仓等待,无符合条件的标的",
            "urgency": "low", "reason": "大盘弱势或板块无机会"
        })

    urgency_order = {"high": 0, "medium": 1, "normal": 2, "low": 3}
    actions.sort(key=lambda x: urgency_order.get(x['urgency'], 99))
    return actions

def build_html(actions, current, target, position_text, position_reason, market_cls,
               asof, update_time, signal_type, metrics):

    def fmt_holdings(holdings, title):
        if not holdings:
            return '<p style="color:#999;text-align:center;padding:10px">' + title + ':无</p>'
        out = []
        total = sum(h['weight'] for h in holdings)
        for h in holdings:
            m = metrics.get(h['name'], {})
            daily = m.get('daily_change', 0)
            dcls = 'pos' if daily > 0 else ('neg' if daily < 0 else '')
            sector = m.get('sector', '')
            sector_tag = ''
            if sector:
                sector_tag = '<span style="background:#e8f6ff;color:#1a5276;font-size:11px;padding:1px 6px;border-radius:3px;margin-left:6px">' + sector + '</span>'
            out.append(
                '<div class="hold-item"><b>' + h["name"] + '(' + h["code"] + ')</b>' + sector_tag +
                ' <span>' + str(int(h["weight"]*10)) + '成</span> ' +
                '<span class="' + dcls + '">' + ('+' if daily >= 0 else '') + str(round(daily, 2)) + '%</span></div>')
        out.append('<div class="hold-item" style="border-top:1px solid #eee;margin-top:8px;padding-top:8px"><b>现金</b> <span>' + str(int((1-total)*10)) + '成</span></div>')
        return "\n".join(out)

    def action_rows():
        if not actions:
            return '<tr><td colspan="3" style="text-align:center;color:#999">无操作</td></tr>'
        out = []
        for a in actions:
            cls_map = {"SELL": "neg", "REDUCE": "warn-text", "BUY": "pos", "ADD": "pos", "HOLD": "hold", "WAIT": "hold"}
            badge_map = {"SELL": "清仓", "REDUCE": "减仓", "BUY": "买入", "ADD": "加仓", "HOLD": "持有", "WAIT": "观望"}
            cls = cls_map.get(a['type'], "")
            badge = badge_map.get(a['type'], a['type'])
            out.append(
                '<tr><td class="nm"><span class="badge-' + a["type"].lower() + '">' + badge + '</span> ' +
                a["name"] + '(' + a["code"] + ')</td>' +
                '<td class="' + cls + '">' + a["msg"] + '</td>' +
                '<td>' + a["reason"] + '</td></tr>')
        return "\n".join(out)

    def monitor_rows():
        sector_groups = {}
        for name, m in metrics.items():
            sector = m['sector']
            sector_groups.setdefault(sector, []).append((name, m))
        sector_order = sorted(sector_groups.keys(),
                              key=lambda s: max(m['score'] for _, m in sector_groups[s]),
                              reverse=True)
        out = []
        for sector in sector_order:
            items = sorted(sector_groups[sector], key=lambda x: x[1]['score'], reverse=True)
            out.append(
                '<tr style="background:#f0f7ff"><td colspan="6" style="font-weight:700;color:#2c5364;padding:8px 10px">&#128193; ' + sector + '</td></tr>')
            for name, m in items:
                mom_cls = 'pos' if m['mom_long'] > 0 else 'neg'
                short_cls = 'pos' if m['mom_short'] > 0 else 'neg'
                daily_cls = 'pos' if m['daily_change'] > 0 else 'neg'
                change_cls = 'pos' if m['mom_long_change'] > 0 else 'neg' if m['mom_long_change'] < 0 else ''
                out.append(
                    '<tr><td class="nm" style="padding-left:24px">' + name + '(' + m["code"] + ')</td>' +
                    '<td class="' + mom_cls + '">' + ('+' if m["mom_long"] >= 0 else '') + str(round(m["mom_long"], 2)) + '%</td>' +
                    '<td class="' + change_cls + '">' + m["mom_long_change_text"] + '</td>' +
                    '<td class="' + short_cls + '">' + ('+' if m["mom_short"] >= 0 else '') + str(round(m["mom_short"], 2)) + '%</td>' +
                    '<td class="' + daily_cls + '">' + ('+' if m["daily_change"] >= 0 else '') + str(round(m["daily_change"], 2)) + '%</td>' +
                    '<td>' + str(round(m["vol"], 1)) + '%</td><td class="sc">' + str(round(m["score"], 1)) + '</td></tr>')
        return "\n".join(out)

    if signal_type == 'morning':
        label = "上午风控"
        sublabel = "11:30 盘中 - 下午操作窗口 13:00-15:00"
        tip = "&#9888; T+1 保护:上午只建议卖出/减仓,不建议买入(下午买入后当日无法止损)"
        tag_color = "tag-morning"
    else:
        label = "收盘决策"
        sublabel = "15:00 收盘 - 次日开盘操作或尾盘固定价"
        tip = "&#128161; T+1 提示:今日买入的标的,需待下一个交易日才能卖出;今日卖出资金当日可用"
        tag_color = "tag-close"

    fmt_holdings_current = fmt_holdings(current, "当前持仓")
    fmt_holdings_target = fmt_holdings(target, "目标持仓")
    action_rows_html = action_rows()
    monitor_rows_html = monitor_rows()

    html = """<!DOCTYPE html>
<html lang="zh-CN"><head><meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>ETF板块轮动 - """ + label + """</title>
<style>
*{margin:0;padding:0;box-sizing:border-box}
body{font-family:-apple-system,"Segoe UI","PingFang SC","Microsoft YaHei",sans-serif;
background:linear-gradient(135deg,#0f2027,#203a43,#2c5364);min-height:100vh;padding:20px;color:#222}
.wrap{max-width:960px;margin:0 auto}
.head{text-align:center;color:#fff;margin-bottom:22px}
.head h1{font-size:24px;letter-spacing:1px}
.head p{opacity:.8;font-size:13px;margin-top:6px}
.card{background:#fff;border-radius:16px;padding:22px;margin-bottom:18px;
box-shadow:0 10px 30px rgba(0,0,0,.25)}
.card h2{font-size:17px;color:#2c5364;border-left:4px solid #2c5364;padding-left:10px;margin-bottom:14px}
table{width:100%;border-collapse:collapse;font-size:13px;min-width:720px}
th,td{padding:10px 8px;text-align:left;border-bottom:1px solid #f0f0f0;vertical-align:top}
th{background:#f7f9fb;color:#666;font-weight:600}
td.nm{min-width:140px;font-weight:600}
.table-wrap{overflow-x:auto;border-radius:12px}
.pos{color:#d7263d;font-weight:600} .neg{color:#1a936f;font-weight:600}
.warn-text{color:#e67e22;font-weight:600} .hold{color:#666}
td.sc{font-weight:700;color:#2c5364}
.badge-sell,.badge-reduce{display:inline-block;background:#d7263d;color:#fff;font-size:11px;
padding:2px 8px;border-radius:4px;margin-right:6px}
.badge-buy,.badge-add{display:inline-block;background:#27ae60;color:#fff;font-size:11px;
padding:2px 8px;border-radius:4px;margin-right:6px}
.badge-hold,.badge-wait{display:inline-block;background:#95a5a6;color:#fff;font-size:11px;
padding:2px 8px;border-radius:4px;margin-right:6px}
.signal{background:linear-gradient(135deg,#d7263d,#f46036);color:#fff;border-radius:14px;
padding:24px;text-align:center;margin-bottom:18px;box-shadow:0 8px 24px rgba(215,38,61,.35)}
.signal .lab{font-size:13px;opacity:.9}
.signal .val{font-size:28px;font-weight:700;margin-top:8px}
.signal .sub{font-size:14px;margin-top:6px;opacity:.9}
.grid{display:grid;grid-template-columns:1fr 1fr;gap:14px;margin-bottom:18px}
.mini{background:#fff;border-radius:14px;padding:18px;box-shadow:0 8px 24px rgba(0,0,0,.2)}
.mini .t{font-size:13px;color:#888} .mini .v{font-size:20px;font-weight:700;color:#2c5364;margin-top:6px}
.hold-item{display:flex;justify-content:space-between;padding:8px 0;border-bottom:1px solid #f5f5f5;font-size:14px}
.market-ok{color:#27ae60} .market-warn{color:#e67e22} .market-danger{color:#d7263d}
.info-box{background:#e8f6ff;border-left:4px solid #3498db;padding:14px 18px;border-radius:8px;margin-bottom:18px;color:#1a5276;font-size:14px;line-height:1.6}
.note{background:#fff;border-radius:14px;padding:18px;font-size:13px;color:#555;line-height:1.9}
.note b{color:#2c5364}
.foot{text-align:center;color:#fff;opacity:.7;font-size:12px;margin-top:14px}
.tag{display:inline-block;padding:2px 10px;border-radius:4px;font-size:12px;margin-left:8px;vertical-align:middle}
.tag-morning{background:#f39c12;color:#fff} .tag-close{background:#27ae60;color:#fff}
.sector-tag{display:inline-block;background:#e8f6ff;color:#1a5276;font-size:11px;padding:1px 6px;border-radius:3px;margin-left:6px}
@media(max-width:700px){.grid{grid-template-columns:1fr} table{font-size:12px} .nm{min-width:100px}}
</style></head><body><div class="wrap">
<div class="head"><h1>&#128202; ETF 板块轮动 - """ + label + """<span class="tag """ + tag_color + """>""" + label + """</span></h1>
<p>""" + sublabel + """ - 概念板块分组 &#183; 仓位管理 &#183; T+1安全</p></div>

<div class="signal"><div class="lab">当前建议总仓位</div>
<div class="val market-""" + market_cls + """>""" + position_text + """</div>
<div class="sub">""" + position_reason + """ - 数据截至 """ + asof + """</div></div>

<div class="info-box">
<b>&#128203; 操作提示</b><br>
""" + tip + """<br>
<small>本次共 """ + str(len(actions)) + """ 条指令,按紧急程度排序</small>
</div>

<div class="grid">
<div class="mini"><div class="t">当前建议持仓</div><div class="v" style="font-size:14px;line-height:1.6">
""" + fmt_holdings_current + """
</div></div>
<div class="mini"><div class="t">目标持仓(""" + label + """后)</div><div class="v" style="font-size:14px;line-height:1.6">
""" + fmt_holdings_target + """
</div></div>
</div>

<div class="card"><h2>&#127919; 操作指令清单</h2>
<table><thead><tr><th class="nm">标的</th><th>操作详情</th><th>触发原因</th></tr></thead>
<tbody>""" + action_rows_html + """</tbody></table></div>

<div class="card"><h2>&#128202; 板块与标的监控(共 """ + str(len(metrics)) + """ 只,分8个板块)</h2>
<div class="table-wrap"><table><thead><tr><th class="nm">ETF</th><th>40日动量</th><th>较昨日</th><th>5日动量</th><th>当日涨跌</th><th>波动率</th><th>得分</th></tr></thead>
<tbody>""" + monitor_rows_html + """</tbody></table></div></div>

<div class="note"><b>策略逻辑</b><br>
<b>板块分组</b>:8大概念板块(宽基指数,周期资源,科技成长,新能源,医药医疗,大消费,金融地产,先进制造),共20只代表性ETF.<br>
<b>仓位</b>:沪深300 40日动量&gt;0且5日动量&gt;0&#8594;满仓10成;40日&gt;0但5日&#8804;0&#8594;半仓5成;40日&#8804;0&#8594;空仓0成.<br>
<b>选股</b>:40日风险调整动量前""" + str(TOP_K) + """ + 40日动量&gt;0 + 5日动量&gt;-2% + 当日跌幅&gt;""" + str(MAX_DAILY_DROP) + """% + 当日涨幅&lt;""" + str(MAX_DAILY_RISE) + """%.<br>
<b>上午</b>:只风控(卖出/减仓),不买入(T+1保护).<br>
<b>下午</b>:重新评估,给出次日目标持仓,可买可卖.<br><br>
<b>T+1 制度说明</b><br>
A股 ETF 实行T+1交易:今日买入的份额,需待下一个交易日才能卖出;今日卖出资金当日可用(可继续买入其他ETF),但不可取现至银行卡.</div>

<div class="foot">更新于 """ + update_time + """ - 数据源: 新浪财经 - 策略状态自动维护</div>
</div></body></html>"""
    return html

def main():
    bj = timezone(timedelta(hours=8))
    now = datetime.now(bj)
    update_time = now.strftime('%Y-%m-%d %H:%M:%S')

    if 11 <= now.hour < 13:
        signal_type = 'morning'
        print("时段: 上午风控 (11:30 盘中,负责下午操作)")
    elif now.hour >= 15:
        signal_type = 'close'
        print("时段: 收盘决策 (15:00 后,负责次日/尾盘操作)")
    else:
        signal_type = 'close'
        print(f"时段: 下午盘中 ({now.hour}:{now.minute:02d} 手动触发,按收盘逻辑处理)")

    print(f"\n{'='*60}")
    print(f"[{update_time}] 信号类型: {signal_type}")
    print(f"{'='*60}")

    price_daily, etf_info = fetch_daily_data(ETF_POOL)
    if price_daily is None:
        print("日K线获取失败")
        return

    today_str = datetime.now(timezone(timedelta(hours=8))).strftime('%Y-%m-%d')
    if price_daily.index[-1] != today_str:
        print(f"\n日K线最新日期 {price_daily.index[-1]} 不是今天,尝试用60分钟K线合成...")
        intraday_snapshot = fetch_intraday_snapshot(ETF_POOL)
        price = merge_intraday_price(price_daily, intraday_snapshot, etf_info)
    else:
        print(f"\n日K线已包含今天数据,直接使用")
        price = price_daily

    metrics = calc_metrics(price, etf_info, ETF_SECTOR)
    if not metrics:
        print("指标计算失败")
        return

    asof = price.index[-1]
    print(f"\n统一数据基准日期: {asof}")

    position_ratio, position_text, position_reason, market_cls = market_timing(metrics)
    print(f"大盘: {position_text} ({position_reason})")

    portfolio = load_portfolio()
    current = portfolio.get('holdings', [])
    print(f"当前建议持仓: {[(h['name'], h['weight']) for h in current]}")

    target = build_target(metrics, position_ratio, signal_type)
    print(f"目标持仓: {[(h['name'], h['weight']) for h in target]}")

    actions = generate_actions(current, target, signal_type, metrics)
    for a in actions:
        print(f"  {a['type']}: {a['msg']}")

    new_holdings = []
    if signal_type == 'morning':
        sold = {a['name'] for a in actions if a['type'] == 'SELL'}
        reduced = {a['name']: a for a in actions if a['type'] == 'REDUCE'}
        for h in current:
            if h['name'] in sold:
                continue
            if h['name'] in reduced:
                for t in target:
                    if t['name'] == h['name']:
                        new_holdings.append({**h, 'weight': t['weight']})
                        break
            else:
                new_holdings.append(h)
        cash = 1 - sum(h['weight'] for h in new_holdings)
    else:
        new_holdings = target
        cash = 1 - sum(h['weight'] for h in target)

    save_portfolio({
        "holdings": new_holdings,
        "cash_ratio": round(cash, 2),
        "last_update": update_time,
        "position_ratio": position_ratio
    })

    html = build_html(actions, current, target, position_text, position_reason,
                      market_cls, asof, update_time, signal_type, metrics)
    with open('index.html', 'w', encoding='utf-8') as f:
        f.write(html)

    data = {
        'update_time': update_time, 'asof': asof, 'signal_type': signal_type,
        'market': {'position_text': position_text, 'reason': position_reason, 'ratio': position_ratio},
        'current_holdings': current, 'target_holdings': target,
        'actions': actions, 'metrics': metrics
    }
    with open('data.json', 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

    print(f"\n已生成 index.html, data.json, {PORTFOLIO_FILE}")
    print(f"{'='*60}\n")


if __name__ == '__main__':
    main()
