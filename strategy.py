#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
行业ETF动量轮动策略 - 带数据验证版
========================================
"""
import requests
import pandas as pd
import numpy as np
import time
import json
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
MOM_MAIN, TOP_MAIN = 40, 3
MOM_SAFE, TOP_SAFE = 120, 2
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
            print(f"  [警告] {sina_code} 第{attempt+1}次请求失败: {e}")
            time.sleep(1)
    print(f"  [错误] {sina_code} 数据获取失败，已跳过")
    return None


def fetch_all_data(pool, datalen=DATA_LEN):
    all_close = {}
    etf_info = {}
    etf_last_date = {}
    etf_has_today = {}
    print("开始拉取 ETF 数据...")
    
    today_str = datetime.now(timezone(timedelta(hours=8))).strftime('%Y-%m-%d')
    
    for code, (sina, name) in pool.items():
        df = get_etf_sina(sina, datalen)
        if df is not None and len(df) >= 5:
            all_close[name] = df.set_index('date')['close']
            etf_info[name] = code
            etf_last_date[name] = df['date'].iloc[-1]
            etf_has_today[name] = (df['date'].iloc[-1] == today_str)
            status = "✓" if etf_has_today[name] else "⚠ 无当日数据"
            print(f"  {name}({code}): {len(df)} 条, 最新 {df['date'].iloc[-1]} {status}")
        else:
            print(f"  ✗ {name}({code}): 数据不足或获取失败，已排除")
        time.sleep(0.15)
    
    if not all_close:
        print("[致命错误] 所有 ETF 数据获取失败")
        return None, {}, {}, {}
    
    price = pd.DataFrame(all_close).sort_index()
    print(f"\n合并后原始维度: {price.shape}, 日期范围 {price.index[0]} ~ {price.index[-1]}")
    
    # 关键修复：对缺失值用前一日填充
    price_filled = price.ffill()
    price_filled = price_filled.dropna(how='any', axis=1)
    price_filled = price_filled.dropna(how='all')
    
    sync_asof = price_filled.index[-1]
    has_today_data = (sync_asof == today_str)
    print(f"处理后维度: {price_filled.shape}, 共同最新日期: {sync_asof}")
    
    # === 新增：生成价格验证快照 ===
    price_verify = {}
    for name in price_filled.columns:
        latest = price_filled[name].iloc[-1]
        prev = price_filled[name].iloc[-2] if len(price_filled) >= 2 else np.nan
        change_pct = (latest / prev - 1) * 100 if not np.isnan(prev) else np.nan
        price_verify[name] = {
            'code': etf_info.get(name, ''),
            'latest': round(latest, 3),
            'prev': round(prev, 3),
            'change_pct': round(change_pct, 2),
            'has_today': etf_has_today.get(name, False),
            'data_date': etf_last_date.get(name, sync_asof),
        }
    
    data_quality = {
        'today': today_str,
        'has_today_data': has_today_data,
        'etf_has_today': etf_has_today,
        'sync_asof': sync_asof,
    }
    
    return price_filled, etf_info, etf_last_date, data_quality, price_verify


def calc_ranking(price, mom_window, etf_info):
    if price is None or len(price) < mom_window + 1:
        print(f"[错误] 数据不足")
        return None
    rets = price.pct_change()
    mom = price.iloc[-1] / price.iloc[-mom_window] - 1
    vol = rets.iloc[-mom_window:].std() * np.sqrt(252)
    risk_adj = (mom / vol).replace([np.inf, -np.inf], np.nan).dropna().sort_values(ascending=False)
    result = pd.DataFrame({
        'code': [etf_info.get(n, '') for n in risk_adj.index],
        'mom_pct': (mom[risk_adj.index] * 100).round(2),
        'vol_pct': (vol[risk_adj.index] * 100).round(2),
        'score': risk_adj.round(3),
    })
    return result


def pick(result, top_k):
    if result is None or result.empty:
        return []
    top = result.head(top_k)
    return [n for n in top.index if result.loc[n, 'mom_pct'] > 0]


def rows_html(ranking, picks, top_k, limit=10):
    if ranking is None or ranking.empty:
        return '<tr><td colspan="6" style="text-align:center;color:#999">数据获取失败</td></tr>'
    out = []
    for i, (name, row) in enumerate(ranking.head(limit).iterrows(), 1):
        is_pick = name in picks
        badge = '<span class="buy">买入</span>' if is_pick else ''
        mom_cls = 'pos' if row['mom_pct'] > 0 else 'neg'
        rank_cls = 'top' if is_pick else ''
        code = row.get('code', '')
        out.append(
            f'<tr class="{rank_cls}"><td class="rk">{i}</td>'
            f'<td class="cd">{code}</td>'
            f'<td class="nm">{name}{badge}</td>'
            f'<td class="{mom_cls}">{row["mom_pct"]:+.2f}%</td>'
            f'<td>{row["vol_pct"]:.1f}%</td>'
            f'<td class="sc">{row["score"]:.3f}</td></tr>')
    return '\n'.join(out)


def verify_rows_html(price_verify, limit=15):
    """生成数据验证表格HTML"""
    items = sorted(price_verify.items(), key=lambda x: x[1]['change_pct'], reverse=True)
    out = []
    for name, v in items[:limit]:
        cls = 'pos' if v['change_pct'] > 0 else ('neg' if v['change_pct'] < 0 else '')
        today_tag = '<span class="tag-today">今日</span>' if v['has_today'] else '<span class="tag-prev">前日</span>'
        out.append(
            f'<tr><td class="nm">{name}({v["code"]}){today_tag}</td>'
            f'<td>{v["latest"]}</td>'
            f'<td>{v["prev"]}</td>'
            f'<td class="{cls}">{v["change_pct"]:+.2f}%</td>'
            f'<td>{v["data_date"]}</td></tr>')
    return '\n'.join(out)


def build_html(rank_main, picks_main, rank_safe, picks_safe, asof, update_time,
               etf_info, data_quality, signal_type, price_verify):
    def fmt(names):
        return '、'.join([f"{n}({etf_info.get(n, '?')})" for n in names]) if names else '无强势板块, 建议观望'
    def fmt_plus(names):
        return ' + '.join([f"{n}({etf_info.get(n, '')})" for n in names]) if names else '空仓(全部动量为负)'
    
    common = sorted(set(picks_main) & set(picks_safe))
    allp = list(dict.fromkeys(picks_main + picks_safe))
    common_txt = fmt_plus(common) if common else '（两方案暂无交集）'
    main_txt = fmt_plus(picks_main)
    safe_txt = fmt_plus(picks_safe)
    signal = fmt(allp)

    if signal_type == 'morning':
        trade_window = "下午开盘 (13:00)"
        trade_tip = "⚠️ A股T+1：今日买入的标的，明日才能卖出"
        signal_label = "上午休市预览"
        data_note = f"数据截至 {asof}（部分ETF上午无成交，使用前日收盘价填充）"
        tag_class = "tag-morning"
    else:
        trade_window = "收盘固定价 (15:05-15:30) 或 次日开盘"
        trade_tip = "⚠️ A股T+1：今日买入的标的，明日才能卖出"
        signal_label = "收盘确认"
        data_note = f"数据截至 {asof}（完整日K数据）"
        tag_class = "tag-close"

    etf_has_today = data_quality.get('etf_has_today', {})
    has_today_count = sum(1 for v in etf_has_today.values() if v)
    total_count = len(etf_has_today)
    coverage_txt = f"{has_today_count}/{total_count} 只ETF有今日数据"

    return f"""<!DOCTYPE html>
<html lang="zh-CN"><head><meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>ETF动量轮动 · {signal_label}</title>
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
table{{width:100%;border-collapse:collapse;font-size:14px}}
th,td{{padding:9px 8px;text-align:right;border-bottom:1px solid #f0f0f0}}
th{{background:#f7f9fb;color:#666;font-weight:600}}
td.rk,th.rk,td.nm,th.nm,td.cd,th.cd{{text-align:left}}
tr.top{{background:#fff8e6}}
.pos{{color:#d7263d;font-weight:600}} .neg{{color:#1a936f}}
td.sc{{font-weight:700;color:#2c5364}}
.buy{{display:inline-block;background:#d7263d;color:#fff;font-size:11px;
padding:1px 7px;border-radius:10px;margin-left:8px;vertical-align:middle}}
.signal{{background:linear-gradient(135deg,#d7263d,#f46036);color:#fff;border-radius:14px;
padding:20px;text-align:center;margin-bottom:18px;box-shadow:0 8px 24px rgba(215,38,61,.35)}}
.signal .lab{{font-size:13px;opacity:.9}}
.signal .val{{font-size:20px;font-weight:700;margin-top:8px;letter-spacing:1px;line-height:1.5}}
.grid{{display:grid;grid-template-columns:1fr 1fr;gap:14px;margin-bottom:18px}}
.mini{{background:#fff;border-radius:14px;padding:16px;box-shadow:0 8px 24px rgba(0,0,0,.2)}}
.mini .t{{font-size:13px;color:#888}} .mini .v{{font-size:16px;font-weight:700;color:#2c5364;margin-top:6px;line-height:1.5}}
.note{{background:#fff;border-radius:14px;padding:18px;font-size:13px;color:#555;line-height:1.9}}
.note b{{color:#2c5364}}
.foot{{text-align:center;color:#fff;opacity:.7;font-size:12px;margin-top:14px}}
.tag{{display:inline-block;padding:2px 8px;border-radius:4px;font-size:11px;margin-left:8px;vertical-align:middle}}
.tag-morning{{background:#f39c12;color:#fff}}
.tag-close{{background:#27ae60;color:#fff}}
.tag-today{{display:inline-block;background:#27ae60;color:#fff;font-size:10px;padding:1px 6px;border-radius:4px;margin-left:6px}}
.tag-prev{{display:inline-block;background:#95a5a6;color:#fff;font-size:10px;padding:1px 6px;border-radius:4px;margin-left:6px}}
.warn{{background:#fff3cd;border-left:4px solid #f39c12;padding:12px 16px;border-radius:8px;margin-bottom:18px;color:#856404;font-size:13px}}
.warn b{{color:#d35400}}
@media(max-width:600px){{.grid{{grid-template-columns:1fr}} table{{font-size:12px}} th,td{{padding:7px 5px}}}}
</style></head><body><div class="wrap">
<div class="head"><h1>📊 ETF 动量轮动 · {signal_label}<span class="tag {tag_class}">{signal_label}</span></h1>
<p>不预测,只跟随强者 · 风险调整动量(涨幅÷波动率)排名</p></div>

<div class="signal"><div class="lab">当前综合信号({data_note})</div>
<div class="val">{signal}</div></div>

<div class="warn">
<b>💡 操作建议</b><br>
建议操作窗口: <b>{trade_window}</b><br>
{coverage_txt} · {trade_tip}<br>
<small>GitHub Actions 通常有 1-5 分钟延迟，请提前预留时间</small>
</div>

<div class="grid">
<div class="mini"><div class="t">主推方案 应买入(40日/持3只)</div><div class="v">{main_txt}</div></div>
<div class="mini"><div class="t">高胜率方案 应买入(120日/持2只)</div><div class="v">{safe_txt}</div></div>
</div>
<div class="mini" style="margin-bottom:18px"><div class="t">两方案交集(最强信号)</div><div class="v">{common_txt}</div></div>

<div class="card"><h2>主推方案 · 风险调整动量{MOM_MAIN}日排名</h2>
<table><thead><tr><th class="rk">#</th><th class="cd">代码</th><th class="nm">ETF</th><th>动量</th><th>年化波动</th><th>风险调整动量</th></tr></thead>
<tbody>{rows_html(rank_main, picks_main, TOP_MAIN)}</tbody></table></div>

<div class="card"><h2>高胜率方案 · 风险调整动量{MOM_SAFE}日排名</h2>
<table><thead><tr><th class="rk">#</th><th class="cd">代码</th><th class="nm">ETF</th><th>动量</th><th>年化波动</th><th>风险调整动量</th></tr></thead>
<tbody>{rows_html(rank_safe, picks_safe, TOP_SAFE)}</tbody></table></div>

<div class="card"><h2>📋 数据验证 · 原始价格快照</h2>
<p style="font-size:12px;color:#888;margin-bottom:10px">可对照东方财富/同花顺核对当日收盘价。绿色标签=今日有数据，灰色标签=使用前日填充。</p>
<table><thead><tr><th class="nm">ETF</th><th>最新价</th><th>前收盘价</th><th>当日涨跌</th><th>数据日期</th></tr></thead>
<tbody>{verify_rows_html(price_verify)}</tbody></table></div>

<div class="note"><b>操作方式</b><br>
1. <b>{signal_label}</b>信号，建议操作窗口: {trade_window}<br>
2. 动量为负的板块不买,该仓位持现金(绝对动量择时)<br>
3. 不预测、只跟随;到点重复,长期坚持方见效<br><br>
<b>回测表现(2023-06~2026-07)</b><br>
主推方案: 年化28.8% · 月胜率70.3% · 夏普1.20 · 最大回撤-23.5%<br>
高胜率方案: 年化21.8% · 月胜率70.3% · 最大回撤-33.3%<br>
同期沪深300: 年化9.5% · 月胜率56.8%<br><br>
<b>风险提示</b>: 历史回测不代表未来收益, 单年可能跑输大盘, 请结合自身风险承受能力决策。<br>
<b>T+1提醒</b>: A股实行T+1交易制度，今日买入的ETF份额，需待下一个交易日才能卖出。请合理安排仓位，避免当日无法止损。</div>

<div class="foot">更新于 {update_time} · 信号类型: {signal_label} · 数据源: 新浪财经</div>
</div></body></html>"""


def main():
    bj = timezone(timedelta(hours=8))
    now = datetime.now(bj)
    update_time = now.strftime('%Y-%m-%d %H:%M:%S')
    
    if 11 <= now.hour <= 12:
        signal_type = 'morning'
    else:
        signal_type = 'close'
    
    print(f"\n{'='*60}")
    print(f"[{update_time}] ETF 动量轮动策略开始计算")
    print(f"信号类型: {signal_type}")
    print(f"{'='*60}")

    # 一次性拉取所有数据，返回价格验证快照
    price, etf_info, etf_last_date, data_quality, price_verify = fetch_all_data(ETF_POOL, DATA_LEN)
    
    if price is None:
        print("数据获取失败，终止")
        return
    
    need_len = max(MOM_MAIN, MOM_SAFE) + 1
    if len(price) < need_len:
        print(f"[致命错误] 同步后数据仅 {len(price)} 行，不满足最长窗口 {need_len} 行")
        return
    
    asof = price.index[-1]
    print(f"\n{'='*60}")
    print(f"统一数据基准日期: {asof}")
    print(f"{'='*60}")
    
    # 打印价格验证摘要（Actions日志中可见）
    print("\n价格验证摘要:")
    for name, v in sorted(price_verify.items(), key=lambda x: x[1]['change_pct'], reverse=True)[:10]:
        tag = "今日" if v['has_today'] else "前日"
        print(f"  {name}({v['code']}): 最新{v['latest']}, 前收{v['prev']}, 涨跌{v['change_pct']:+.2f}% [{tag}]")

    rank_main = calc_ranking(price, MOM_MAIN, etf_info)
    rank_safe = calc_ranking(price, MOM_SAFE, etf_info)
    picks_main = pick(rank_main, TOP_MAIN)
    picks_safe = pick(rank_safe, TOP_SAFE)

    print(f"\n主推方案({MOM_MAIN}日) 应买入: {picks_main}")
    print(f"高胜率方案({MOM_SAFE}日) 应买入: {picks_safe}")

    html = build_html(rank_main, picks_main, rank_safe, picks_safe,
                      asof, update_time, etf_info, data_quality, signal_type, price_verify)
    with open('index.html', 'w', encoding='utf-8') as f:
        f.write(html)

    data = {
        'update_time': update_time,
        'asof': asof,
        'signal_type': signal_type,
        'data_quality': {k: v for k, v in data_quality.items() if k != 'etf_has_today'},
        'etf_has_today': data_quality.get('etf_has_today', {}),
        'price_verify': price_verify,  # 新增：价格验证快照
        'main_plan': {
            'window': MOM_MAIN, 'top_k': TOP_MAIN, 'buy': picks_main,
            'ranking': rank_main.reset_index().rename(columns={'index': 'name'}).to_dict('records') if rank_main is not None else []
        },
        'safe_plan': {
            'window': MOM_SAFE, 'top_k': TOP_SAFE, 'buy': picks_safe,
            'ranking': rank_safe.reset_index().rename(columns={'index': 'name'}).to_dict('records') if rank_safe is not None else []
        },
    }
    with open('data.json', 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

    print(f"\n已生成 index.html 和 data.json")
    print(f"{'='*60}\n")


if __name__ == '__main__':
    main()
