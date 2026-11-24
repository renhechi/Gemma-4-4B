#!/usr/bin/env bash
# User decision 2026-11-24: end E2-full at step 1500 (0.25 D* dose checkpoint), archive like E3.
set -u
cd /home/renhe/gemma4_pretrain
bash scripts/stop_at_ckpt.sh E2-full 1500
python3 - <<'PY'
import json,time
p='runs/E2-full/metrics.jsonl'
rows=open(p).readlines()
keep=[l for l in rows if not (json.loads(l).get('event') in ('step','eval') and json.loads(l).get('step',0)>1500)]
drop=[l for l in rows if l not in keep]
if drop: open('runs/E2-full/metrics_discarded_after_step1500.jsonl','w').writelines(drop)
keep.append(json.dumps({"experiment":"E2-full","mode":"E2","time":time.time(),"event":"run_end","step":1500,
 "stop_reason":"user_stop_step1500 (plateau ~2.64-2.66; structuring loop, early stop not triggered; aligned with 0.25 D* dose)",
 "final_ckpt":"final_step1500.pt"},ensure_ascii=False)+"\n")
open(p,'w').writelines(keep); print('E2 run_end written; dropped',len(drop))
PY
cp -p runs/E2-full/latest.pt runs/E2-full/final_step1500.pt
chmod a-w runs/E2-full/final_step1500.pt runs/E2-full/dose_0.25.pt 2>/dev/null
{ echo "# E2-full 保留紀錄(2026-11-25 結束於 step 1500,使用者決定)"; echo; echo "| 檔案 | sha256 |"; echo "|---|---|";
  for f in final_step1500.pt dose_0.25.pt; do echo "| \`$f\` | $(sha256sum runs/E2-full/$f | cut -d' ' -f1) |"; done; echo;
  echo "續跑:移除 metrics.jsonl 最後的 run_end 行,確認 latest.pt = final_step1500.pt,執行 \`train_lgdsrfpa.py --mode E2 --experiment E2-full\`。"; } > runs/E2-full/ARCHIVE.md
python3 scripts/compare_e123.py 1500 > /dev/null
echo "[chain] $(date '+%F %T') E2-full ended at step 1500 (user decision); run_end written; GPU free for final evaluation" >> logs/chain.log
git add -A && git commit -q -m "E2-full ended at step 1500 (user decision): run_end, final ckpt, archive; refresh E1/E2/E3 table

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>" && git push -q origin main
DISPLAY=:0 notify-send "GX10" "E2-full 已在 step 1500 結束並封存,GPU 可用於最終評估" 2>/dev/null
echo "[end_E2] done"
