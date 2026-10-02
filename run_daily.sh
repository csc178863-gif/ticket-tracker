#!/usr/bin/env bash
# Daily entry point: collect one snapshot, then repair any transient failures,
# then run analysis and generate a report.
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

# Run analysis and generate report
echo ""
echo "=== Running analysis ==="
mkdir -p reports

# Generate analysis output and save to file
if "$PY" analyse_panel.py > "reports/analysis_$(date '+%Y-%m-%d').txt" 2>&1; then
  echo "✓ Analysis complete. Report saved to reports/analysis_$(date '+%Y-%m-%d').txt"
  
  # Also update a 'latest' symlink for quick access
  ln -sf "analysis_$(date '+%Y-%m-%d').txt" reports/latest.txt
  
  # Display a quick summary (first 50 lines)
  echo ""
  echo "=== Analysis Summary ==="
  head -50 "reports/analysis_$(date '+%Y-%m-%d').txt"
else
  echo "✗ Analysis failed. Check reports/analysis_$(date '+%Y-%m-%d').txt for details."
  # Don't fail the entire pipeline if analysis fails
fi

exit "$rc"
