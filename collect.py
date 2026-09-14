#!/usr/bin/env python3
"""Daily fare collector for BR122 (TPE->AOJ) and JX861 (HKD->TPE).

Run once per day. Each run appends one snapshot: the current quote for every
departure date in a forward window. Over time this builds the (snapshot_date,
departure_date) panel that a single scrape cannot produce -- which is what
finally separates the booking curve from seasonality.

Storage: data/quotes.jsonl, one JSON object per (snapshot, route, date).
Idempotent -- re-running the same day skips rows already collected, so a
crashed or partial run can simply be re-run.

Usage:
    python3 collect.py                 # default 180-day forward window
    python3 collect.py --horizon 240
    python3 collect.py --routes TPE-AOJ
"""
from __future__ import annotations
import argparse, json, random, sys, time
import datetime as dt
from pathlib import Path

import patch_parser
patch_parser.apply()

from fast_flights import FlightQuery, Passengers, create_query, get_flights

HERE = Path(__file__).parent
DATA = HERE / "data" / "quotes.jsonl"
LOGS = HERE / "logs"
ROUTES = {"TPE-AOJ": ("TPE", "AOJ", "BR"), "HKD-TPE": ("HKD", "TPE", "JX")}

MAX_ATTEMPTS = 3
BASE_SLEEP = 0.35          # polite pacing between queries
BACKOFF = 4.0              # seconds, multiplied by attempt number


def log(msg: str) -> None:
    stamp = dt.datetime.now().strftime("%H:%M:%S")
    line = f"[{stamp}] {msg}"
    print(line, flush=True)
    LOGS.mkdir(exist_ok=True)
    with (LOGS / f"{dt.date.today().isoformat()}.log").open("a") as fh:
        fh.write(line + "\n")


def existing_keys(snapshot: str) -> set:
    """Keys already stored for this snapshot, so the run is resumable."""
    keys = set()
    if not DATA.exists():
        return keys
    with DATA.open() as fh:
        for line in fh:
            if not line.strip():
                continue
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue
            if r.get("snapshot") == snapshot:
                keys.add((r["route"], r["date"]))
    return keys


def fetch(origin: str, dest: str, date: str) -> list[dict]:
    q = create_query(
        flights=[FlightQuery(date=date, from_airport=origin,
                             to_airport=dest, max_stops=0)],
        trip="one-way", seat="economy", passengers=Passengers(adults=1),
        currency="TWD", language="zh-TW")
    legs = []
    for f in get_flights(q):
        s = f.flights[0]
        legs.append({
            "carrier": f.type,
            "airline": f.airlines[0] if f.airlines else None,
            "dep": "%02d:%02d" % s.departure.time,
            "arr": "%02d:%02d" % s.arrival.time,
            "plane": s.plane_type,
            "duration": s.duration,
            "price": f.price,
        })
    return legs


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--horizon", type=int, default=180,
                    help="days ahead to scan (default 180)")
    ap.add_argument("--routes", default=",".join(ROUTES),
                    help="comma-separated route keys")
    args = ap.parse_args()

    today = dt.date.today()
    snapshot = today.isoformat()
    routes = [r.strip() for r in args.routes.split(",") if r.strip() in ROUTES]
    if not routes:
        log("no valid routes; nothing to do")
        return 2

    DATA.parent.mkdir(parents=True, exist_ok=True)
    done = existing_keys(snapshot)
    if done:
        log(f"resuming snapshot {snapshot}: {len(done)} rows already present")

    n_new = n_noflight = n_fail = 0
    t0 = time.time()
    with DATA.open("a") as out:
        for offset in range(1, args.horizon + 1):
            day = today + dt.timedelta(days=offset)
            ds = day.isoformat()
            for key in routes:
                origin, dest, target = ROUTES[key]
                if (key, ds) in done:
                    continue

                legs, err = None, None
                for attempt in range(1, MAX_ATTEMPTS + 1):
                    try:
                        legs = fetch(origin, dest, ds)
                        err = None
                        break
                    except Exception as e:
                        err = f"{type(e).__name__}: {e}"[:200]
                        if attempt < MAX_ATTEMPTS:
                            time.sleep(BACKOFF * attempt +
                                       random.uniform(0, 1.5))

                rec = {
                    "snapshot": snapshot,
                    "route": key,
                    "target_carrier": target,
                    "date": ds,
                    "dow": day.isoweekday(),
                    "dtd": offset,
                    "legs": legs if legs is not None else [],
                    "no_flight": bool(legs is not None and not legs),
                    "error": err,
                }
                out.write(json.dumps(rec, ensure_ascii=False) + "\n")
                out.flush()

                if err:
                    n_fail += 1
                elif rec["no_flight"]:
                    n_noflight += 1
                else:
                    n_new += 1
                time.sleep(BASE_SLEEP + random.uniform(0, 0.25))

            if offset % 30 == 0:
                log(f"  ...+{offset}d  quoted={n_new} "
                    f"noflight={n_noflight} failed={n_fail}")

    dur = time.time() - t0
    total = n_new + n_noflight + n_fail
    log(f"snapshot {snapshot} done in {dur/60:.1f}min: "
        f"{n_new} quoted, {n_noflight} no-flight, {n_fail} failed "
        f"({total} rows)")

    if total and n_fail / total > 0.30:
        log(f"WARNING failure rate {n_fail/total:.0%} — likely rate-limited "
            f"or upstream layout change; inspect logs before trusting today")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
