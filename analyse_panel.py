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

Reports readiness honestly and refuses to fit what the data cannot support.
"""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
import pandas as pd

HERE = Path(__file__).parent
DATA = HERE / "data" / "quotes.jsonl"
MIN_SNAPSHOTS = 20          # below this the FE model is not worth fitting


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


def readiness(df: pd.DataFrame) -> None:
    print("=" * 68)
    print("PANEL READINESS")
    print("=" * 68)
    snaps = sorted(df.snapshot.unique())
    print(f"snapshots: {len(snaps)}  ({snaps[0]} .. {snaps[-1]})")
    for route, g in df.groupby("route"):
        per_date = g.groupby("date").snapshot.nunique()
        print(f"\n  {route}: {len(g)} obs, {g.date.nunique()} departure dates")
        print(f"    quotes per departure date: median {per_date.median():.0f}, "
              f"max {per_date.max()}")
        rep = int((per_date >= 2).sum())
        print(f"    dates observed from >=2 snapshots: {rep} "
              f"({rep/len(per_date):.0%})")
    print()


def fit_fe(df: pd.DataFrame) -> None:
    """Two-way FE: departure-date dummies + dtd spline."""
    try:
        import statsmodels.formula.api as smf
    except ImportError:
        print("statsmodels not installed; skipping FE model")
        return

    print("=" * 68)
    print("BOOKING CURVE (departure-date fixed effects)")
    print("=" * 68)
    for route, g in df.groupby("route"):
        rep = g.groupby("date").snapshot.nunique()
        usable = g[g.date.isin(rep[rep >= 2].index)].copy()
        if usable.date.nunique() < 10 or len(usable) < 40:
            print(f"\n  {route}: not enough repeated dates yet "
                  f"({usable.date.nunique()} dates, {len(usable)} obs) — "
                  f"keep collecting")
            continue

        usable["lp"] = np.log(usable.price)
        # piecewise-linear in dtd: knots at 14/30/60/90 days
        for k in (14, 30, 60, 90):
            usable[f"d{k}"] = np.maximum(usable.dtd - k, 0)
        m = smf.ols("lp ~ C(date) + dtd + d14 + d30 + d60 + d90",
                    data=usable).fit()
        print(f"\n  {route}: n={len(usable)}, "
              f"{usable.date.nunique()} dates, R2={m.rsquared:.3f}")
        print(f"    (within-date effects; departure-date FE absorbs seasonality)")
        for term in ["dtd", "d14", "d30", "d60", "d90"]:
            if term in m.params:
                b, p = m.params[term], m.pvalues[term]
                sig = "*" if p < 0.05 else " "
                print(f"      {term:<5} {b*100:+7.3f}% per day  "
                      f"(p={p:.3f}){sig}")
        # translate into a practical statement
        seg = m.params.get("dtd", 0) * 100
        if abs(seg) > 0.01:
            direction = "cheaper" if seg > 0 else "more expensive"
            print(f"    => within 14d of departure, each day earlier is "
                  f"{abs(seg):.2f}% {direction}")


def price_changes(df: pd.DataFrame) -> None:
    """Concrete repricing events: same departure date, price moved."""
    print()
    print("=" * 68)
    print("REPRICING EVENTS (same departure date, price changed)")
    print("=" * 68)
    any_found = False
    for route, g in df.groupby("route"):
        g = g.sort_values(["date", "snapshot"])
        g["prev"] = g.groupby("date").price.shift()
        ch = g.dropna(subset=["prev"])
        ch = ch[ch.price != ch.prev]
        if ch.empty:
            continue
        any_found = True
        ch = ch.assign(delta=ch.price - ch.prev,
                       pct=(ch.price / ch.prev - 1) * 100)
        print(f"\n  {route}: {len(ch)} changes, "
              f"median move {ch.delta.median():+,.0f} TWD")
        print(f"    increases {(ch.delta>0).sum()}, decreases {(ch.delta<0).sum()}")
        for r in ch.reindex(ch.delta.abs().sort_values(ascending=False).index).head(5).itertuples():
            print(f"      dep {r.date} @dtd{r.dtd:>3}  "
                  f"{r.prev:>7,.0f} -> {r.price:>7,.0f}  ({r.pct:+.1f}%)")
    if not any_found:
        print("\n  none yet — needs >=2 snapshots covering the same dates")


def main() -> None:
    if not DATA.exists():
        print("no data yet; run collect.py first")
        return
    df = load()
    if df.empty:
        print("no usable quotes yet")
        return
    readiness(df)
    n_snap = df.snapshot.nunique()
    price_changes(df)
    print()
    if n_snap < MIN_SNAPSHOTS:
        print("=" * 68)
        print(f"NOTE: {n_snap} snapshot(s) collected. The booking-curve model "
              f"needs ~{MIN_SNAPSHOTS}+\n(roughly 3 weeks of daily runs) before "
              f"its estimates are worth acting on.\nCollect first; the model "
              f"below will start reporting once data supports it.")
        print("=" * 68)
    fit_fe(df)


if __name__ == "__main__":
    main()
