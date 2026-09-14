#!/usr/bin/env python3
"""Retry rows that recorded a hard error, for a given snapshot (default today).

collect.py is resumable by (route, date) key, which means a row that errored
stays errored on re-run. This pass re-probes only those rows and rewrites them
in place, so a transient network failure doesn't leave a permanent hole in the
panel.

Usage:
    python3 retry_failed.py                # today's snapshot
    python3 retry_failed.py 2026-09-08
"""
from __future__ import annotations
import json, sys, time, random
import datetime as dt
from pathlib import Path

import patch_parser
patch_parser.apply()

from collect import fetch, ROUTES, DATA, log

PROBES = 3


def main() -> int:
    snapshot = sys.argv[1] if len(sys.argv) > 1 else dt.date.today().isoformat()
    if not DATA.exists():
        log("no data file yet")
        return 0

    rows = [json.loads(l) for l in DATA.read_text().splitlines() if l.strip()]
    targets = [r for r in rows
               if r.get("snapshot") == snapshot and r.get("error")]
    if not targets:
        log(f"snapshot {snapshot}: no failed rows to retry")
        return 0

    log(f"snapshot {snapshot}: retrying {len(targets)} failed rows")
    fixed = noflight = still = 0

    for r in targets:
        origin, dest, _ = ROUTES[r["route"]]
        legs, err = None, None
        for attempt in range(1, PROBES + 1):
            try:
                legs = fetch(origin, dest, r["date"])
                err = None
                break
            except Exception as e:
                err = f"{type(e).__name__}: {e}"[:200]
                if attempt < PROBES:
                    time.sleep(3.0 * attempt + random.uniform(0, 1.5))

        if err:
            r["error"] = err
            still += 1
        else:
            r["legs"] = legs
            r["error"] = None
            r["no_flight"] = not legs
            if legs:
                fixed += 1
            else:
                noflight += 1
        time.sleep(1.0 + random.uniform(0, 0.5))

    tmp = DATA.with_suffix(".tmp")
    tmp.write_text("\n".join(json.dumps(r, ensure_ascii=False)
                             for r in rows) + "\n")
    tmp.replace(DATA)          # atomic, so an interrupted write can't truncate

    log(f"retry done: {fixed} recovered, {noflight} confirmed no-flight, "
        f"{still} still failing")
    return 0


if __name__ == "__main__":
    sys.exit(main())
