#!/usr/bin/env bash
# User decision 2026-11-24: end E2-full at step 1550 (= E1 early-stop point).
# 1) stop right after the step-1500 checkpoint  2) add --stop-at-step to the trainer (process stopped)
# 3) resume 1500→1550, stop via the early-stop path (eval@1550, stop_0.2583Dstar.pt, run_end)  4) archive
set -u
cd /home/renhe/gemma4_pretrain
PY="conda run --no-capture-output -n gemma3_env python3"
bash scripts/stop_at_ckpt.sh E2-full 1500
python3 scripts/patch_stop_at_step.py scripts/train_lgdsrfpa.py && python3 -m py_compile scripts/train_lgdsrfpa.py || { echo "[end_E2] patch FAILED"; exit 1; }
echo "[chain] $(date '+%F %T') E2-full: resume 1500→1550 with --stop-at-step 1550 (align with E1 early stop)" >> logs/chain.log
$PY scripts/train_lgdsrfpa.py --mode E2 --experiment E2-full --stop-at-step 1550 >> logs/E2-full.log 2>&1
grep -q '"event": "run_end"' runs/E2-full/metrics.jsonl || { echo "[end_E2] no run_end — check logs/E2-full.log"; exit 1; }
cp -p runs/E2-full/latest.pt runs/E2-full/final_step1550.pt
chmod a-w runs/E2-full/final_step1550.pt runs/E2-full/dose_0.25.pt runs/E2-full/stop_*Dstar.pt 2>/dev/null
{ echo "# E2-full 保留紀錄(結束於 step 1550 = E1 early-stop 點,使用者決定)"; echo; echo "| 檔案 | sha256 |"; echo "|---|---|";
  for f in final_step1550.pt dose_0.25.pt $(cd runs/E2-full && ls stop_*Dstar.pt); do echo "| \`$f\` | $(sha256sum runs/E2-full/$f | cut -d' ' -f1) |"; done; echo;
  echo "step 1500 停止後以 --stop-at-step 1550 自 checkpoint 續跑(lr 排程不變),於 step 1550 eval 後經 early-stop 路徑結束。"; } > runs/E2-full/ARCHIVE.md
python3 scripts/compare_e123.py 1550 > /dev/null; python3 scripts/e5_feasibility.py > /dev/null
echo "[chain] $(date '+%F %T') E2-full ended at step 1550; GPU free for final evaluation" >> logs/chain.log
git add -A && git commit -q -m "E2-full ended at step 1550 (aligned with E1 early stop); --stop-at-step; archive; refresh E1/E2/E3 + E5 tables

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>" && git push -q origin main
DISPLAY=:0 notify-send "GX10" "E2-full 已在 step 1550 結束(對齊 E1),GPU 可用於最終評估" 2>/dev/null
echo "[end_E2] done"
