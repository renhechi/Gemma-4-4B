#!/usr/bin/env bash
# Sequential E3 → E1 → E2 full runs on GX10 (one GPU → one run at a time).
# E3 first: it is the main result; E1/E2 are the ablation baselines.
# Each run resumes from runs/<exp>/latest.pt, so re-running this script after a
# reboot / crash continues where it stopped. Finished runs (run_end logged) are skipped.
set -u
cd /home/renhe/gemma4_pretrain
PY="conda run --no-capture-output -n gemma3_env python3"
for spec in "E3 E3-full" "E1 E1-full" "E2 E2-full"; do
  set -- $spec
  if grep -q '"event": "run_end"' "runs/$2/metrics.jsonl" 2>/dev/null; then
    echo "[chain] $2 already finished, skip"; continue
  fi
  for try in 1 2 3; do
    echo "[chain] $(date '+%F %T') start $2 (try $try)"
    $PY scripts/train_lgdsrfpa.py --mode "$1" --experiment "$2" >> "logs/$2.log" 2>&1
    grep -q '"event": "run_end"' "runs/$2/metrics.jsonl" && break
    sleep 60
  done
done
$PY scripts/analyze_runs.py E1-full E2-full E3-full >> logs/analyze.log 2>&1
echo "[chain] $(date '+%F %T') all done"
