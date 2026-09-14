#!/usr/bin/env bash
# Daily entry point: collect one snapshot, then repair any transient failures.
# Safe to run more than once a day -- collect.py skips rows already stored.
set -uo pipefail

cd "$(dirname "$0")" || exit 1

HORIZON="${HORIZON:-180}"
PY="${PYTHON:-python3}"

echo "=== fare pipeline $(date '+%Y-%m-%d %H:%M:%S') ==="

"$PY" collect.py --horizon "$HORIZON"
rc=$?

# Retry transient failures regardless of collect's exit code: a partial
# snapshot is still worth repairing.
"$PY" retry_failed.py

if [ "$rc" -ne 0 ]; then
  echo "collect.py exited $rc (high failure rate) — check logs/"
fi

# Lightweight rowcount so cron mail shows growth at a glance.
if [ -f data/quotes.jsonl ]; then
  echo "total rows: $(wc -l < data/quotes.jsonl)"
fi

exit "$rc"
