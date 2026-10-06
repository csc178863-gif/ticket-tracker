#!/usr/bin/env python3
"""Panel analysis: separate the booking curve from seasonality.

This is the payoff of daily collection. With ONE snapshot, days-to-departure is
an exact linear function of the departure date, so the two effects cannot be
told apart. Once the same departure date has been quoted from several snapshot
dates, a two-way fixed-effects model identifies them separately:

    log(price) ~ C(departure_date) + f(days_to_departure)

The departure-date fixed effect absorbs ALL seasonality and day-of-week demand,
so the dtd term is a clean within-date booking curve: what waiting actually
costs, holding the flight fixed.

Reports readiness honestly and refuses to act on estimates the data can't yet
support.

Usage:
    python3 analyse_panel.py              # plain text
    python3 analyse_panel.py --markdown   # markdown (for REPORT.md / CI summary)
"""
from __future__ import annotations
import argparse, json
from pathlib import Path
import datetime as dt
import numpy as np
import pandas as pd

HERE = Path(__file__).parent
DATA = HERE / "data" / "quotes.jsonl"
MIN_SNAPSHOTS = 20          # below this the FE model is not worth acting on
ROUTE_LABEL = {"TPE-AOJ": "BR122 台北→青森 (長榮)",
               "HKD-TPE": "JX861 函館→台北 (星宇)"}

MD = False
OUT: list[str] = []


def emit(line: str = "") -> None:
    OUT.append(line)


def h(level: int, text: str) -> None:
    if MD:
        emit(); emit("#" * level + " " + text); emit()
    else:
        emit(); emit("=" * 68); emit(text.upper()); emit("=" * 68)


def table(headers: list[str], rows: list[list[str]]) -> None:
    if MD:
        emit("| " + " | ".join(headers) + " |")
        emit("|" + "|".join("---" for _ in headers) + "|")
        for r in rows:
            emit("| " + " | ".join(str(c) for c in r) + " |")
        emit()
    else:
        for r in rows:
            emit("    " + "  ".join(str(c) for c in r))


def load() -> pd.DataFrame:
    rows = [json.loads(l) for l in DATA.read_text().splitlines() if l.strip()]
    recs = []
    for r in rows:
        if r.get("error") or not r["legs"]:
            continue
        tgt = r["target_carrier"]
        own = [l for l in r["legs"] if l["carrier"] == tgt]
        if not own:
            continue
        leg = min(own, key=lambda x: x["price"])
        comp = [l["price"] for l in r["legs"] if l["carrier"] != tgt]
        recs.append({
            "snapshot": r["snapshot"], "route": r["route"],
            "date": r["date"], "dow": r["dow"], "dtd": r["dtd"],
            "price": leg["price"],
            "comp_price": min(comp) if comp else np.nan,
        })
    return pd.DataFrame(recs)


def readiness(df: pd.DataFrame) -> int:
    h(2, "面板就緒度")
    snaps = sorted(df.snapshot.unique())
    n = len(snaps)
    emit(f"快照數：**{n}**（{snaps[0]} … {snaps[-1]}）" if MD
         else f"snapshots: {n}  ({snaps[0]} .. {snaps[-1]})")
    emit()
    rows = []
    for route, g in df.groupby("route"):
        per_date = g.groupby("date").snapshot.nunique()
        rep = int((per_date >= 2).sum())
        rows.append([ROUTE_LABEL.get(route, route), len(g), g.date.nunique(),
                     f"{per_date.median():.0f}", per_date.max(),
                     f"{rep} ({rep/len(per_date):.0%})"])
    table(["航線", "觀測數", "出發日數", "每出發日報價中位數",
           "最多", "≥2 時點的出發日"], rows)

    if n < MIN_SNAPSHOTS:
        msg = (f"目前 {n} 個快照。訂票曲線模型建議累積 ~{MIN_SNAPSHOTS}+ "
               f"（約 3 週每日執行）後再據此行動；以下估計僅供觀察趨勢。")
        emit(f"> **注意**：{msg}" if MD else f"NOTE: {msg}")
        emit()
    else:
        emit(f"> 快照數已達 {MIN_SNAPSHOTS}+，以下估計可開始採信。" if MD
             else f"snapshots >= {MIN_SNAPSHOTS}: estimates now usable.")
        emit()
    return n


def price_changes(df: pd.DataFrame) -> None:
    h(2, "調價事件")
    found = False
    for route, g in df.groupby("route"):
        g = g.sort_values(["date", "snapshot"])
        g = g.assign(prev=g.groupby("date").price.shift())
        ch = g.dropna(subset=["prev"])
        ch = ch[ch.price != ch.prev]
        if ch.empty:
            continue
        found = True
        ch = ch.assign(delta=ch.price - ch.prev,
                       pct=(ch.price / ch.prev - 1) * 100)
        up, dn = int((ch.delta > 0).sum()), int((ch.delta < 0).sum())
        ratio = f"{up/dn:.1f}:1" if dn else "全漲"
        emit(f"**{ROUTE_LABEL.get(route, route)}**：{len(ch)} 次調價，"
             f"中位數 {ch.delta.median():+,.0f} TWD，"
             f"漲 {up} / 跌 {dn}（{ratio}）" if MD
             else f"\n  {route}: {len(ch)} changes, "
                  f"median {ch.delta.median():+,.0f} TWD, up {up} down {dn}")
        emit()
        top = ch.reindex(ch.delta.abs().sort_values(ascending=False).index)
        table(["出發日", "距起飛", "原價", "新價", "變動"],
              [[r.date, f"{r.dtd}d", f"{r.prev:,.0f}", f"{r.price:,.0f}",
                f"{r.pct:+.1f}%"] for r in top.head(5).itertuples()])
    if not found:
        emit("尚無調價事件（需要 ≥2 個快照覆蓋同一出發日）。")
        emit()


def fit_fe(df: pd.DataFrame, n_snap: int) -> None:
    h(2, "訂票曲線（出發日固定效果）")
    try:
        import statsmodels.formula.api as smf
    except ImportError:
        emit("statsmodels 未安裝，跳過模型。")
        return

    emit("出發日固定效果已吸收所有季節性與星期需求，"
         "因此以下係數是「同一班機、純粹因為等待」造成的價格變化。"
         "負值代表越接近出發越貴。" if MD else
         "(departure-date FE absorbs seasonality; within-date effects below)")
    emit()

    for route, g in df.groupby("route"):
        rep = g.groupby("date").snapshot.nunique()
        usable = g[g.date.isin(rep[rep >= 2].index)].copy()
        label = ROUTE_LABEL.get(route, route)
        if usable.date.nunique() < 10 or len(usable) < 40:
            emit(f"**{label}**：重複觀測不足"
                 f"（{usable.date.nunique()} 個出發日、{len(usable)} 筆），繼續收集。"
                 if MD else
                 f"\n  {route}: not enough repeated dates "
                 f"({usable.date.nunique()} dates, {len(usable)} obs)")
            emit()
            continue

        usable["lp"] = np.log(usable.price)
        for k in (14, 30, 60, 90):
            usable[f"d{k}"] = np.maximum(usable.dtd - k, 0)
        m = smf.ols("lp ~ C(date) + dtd + d14 + d30 + d60 + d90",
                    data=usable).fit()
        emit(f"**{label}** — n={len(usable)}，"
             f"{usable.date.nunique()} 個出發日，R²={m.rsquared:.3f}" if MD
             else f"\n  {route}: n={len(usable)}, R2={m.rsquared:.3f}")
        emit()
        seg_lbl = {"dtd": "0–14 天", "d14": "14–30 天", "d30": "30–60 天",
                   "d60": "60–90 天", "d90": "90 天以上"}
        rows = []
        for term in ["dtd", "d14", "d30", "d60", "d90"]:
            if term not in m.params:
                continue
            b, p = m.params[term], m.pvalues[term]
            star = "**顯著**" if p < 0.05 else ("接近" if p < 0.10 else "不顯著")
            rows.append([seg_lbl[term], f"{b*100:+.3f}% / 天",
                         f"{p:.3f}", star])
        table(["區段（距起飛）", "每天價格變化", "p 值", "判定"], rows)

        sig = [t for t in ["dtd", "d14"]
               if t in m.pvalues and m.pvalues[t] < 0.05]
        if sig and n_snap >= MIN_SNAPSHOTS:
            b = m.params[sig[0]] * 100
            if b < 0:
                emit(f"→ 可行動結論：近期區段每多等一天平均貴 "
                     f"**{abs(b):.2f}%**，宜提早購票。")
            else:
                emit(f"→ 近期區段每多等一天便宜 {b:.2f}%，"
                     f"與一般訂票曲線相反，建議再觀察。")
            emit()


def main() -> None:
    global MD
    ap = argparse.ArgumentParser()
    ap.add_argument("--markdown", action="store_true")
    args = ap.parse_args()
    MD = args.markdown

    if MD:
        emit(f"# 票價面板分析")
        emit()
        emit(f"自動產生於 {dt.datetime.now(dt.timezone.utc):%Y-%m-%d %H:%M} UTC")
        emit()

    if not DATA.exists():
        emit("尚無資料，請先執行 collect.py。")
        print("\n".join(OUT))
        return
    df = load()
    if df.empty:
        emit("尚無可用報價。")
        print("\n".join(OUT))
        return

    n = readiness(df)
    price_changes(df)
    fit_fe(df, n)
    print("\n".join(OUT))


if __name__ == "__main__":
    main()
