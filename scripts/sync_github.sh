#!/usr/bin/env bash
# Hourly: refresh analysis tables/plots, commit all run records (metrics, logs, docs), push to GitHub.
# Checkpoints (*.pt), model and data are gitignored. Runs in tmux session "g4sync".
set -u
cd /home/renhe/gemma4_pretrain
PY="conda run --no-capture-output -n gemma3_env python3"
INTERVAL=${1:-3600}
while true; do
  full=$(ls -d runs/E*-full 2>/dev/null | xargs -n1 basename 2>/dev/null | tr '\n' ' ')
  [ -n "$full" ] && $PY scripts/analyze_runs.py $full > logs/analyze.log 2>&1
  git add -A
  if ! git diff --cached --quiet; then
    step=$(tail -n 50 runs/E3-full/metrics.jsonl 2>/dev/null | grep -o '"step": [0-9]*' | tail -1 | grep -o '[0-9]*')
    git commit -q -m "records: $(date '+%F %H:%M') ${full}(E3 step ${step:-?})" \
      && git push -q origin main 2>>logs/sync.log \
      && echo "[sync] $(date '+%F %T') pushed" >> logs/sync.log \
      || echo "[sync] $(date '+%F %T') push FAILED" >> logs/sync.log
  fi
  sleep "$INTERVAL"
done
