#!/usr/bin/env bash
# Wait until runs/<exp>/latest.pt is written for step <N>, then stop the chain cleanly,
# push final records to GitHub, and notify (desktop + log). Used before the GPU cold drain.
set -u
EXP=${1:-E3-full}; N=${2:-300}
cd /home/renhe/gemma4_pretrain
M=runs/$EXP/metrics.jsonl; CK=runs/$EXP/latest.pt
# 1) wait for the step-N row
until t=$(grep "\"step\": $N," "$M" | grep '"event": "eval"' | tail -1 | grep -o '"time": [0-9.]*' | grep -o '[0-9.]*$') && [ -n "$t" ]; do sleep 30; done
# 2) wait until latest.pt is newer than that row (it is written right after the row)
until [ "$(stat -c %Y "$CK")" -ge "${t%.*}" ]; do sleep 10; done
sleep 30
# 3) stop chain (no restart) and trainer
/usr/bin/tmux kill-session -t g4chain 2>/dev/null
pkill -f "experiment $EXP\$"; sleep 10
echo "[chain] $(date '+%F %T') stopped at checkpoint step $N for GPU cold drain (stop_at_ckpt.sh)" >> logs/chain.log
# 4) final records → GitHub
git add -A && git commit -q -m "records: stopped $EXP at checkpoint step $N for GPU cold drain" && git push -q origin main
MSG="Gemma4 $EXP 已在 step $N checkpoint 停止,可以冷關機(拔電源)了"
DISPLAY=:0 notify-send -u critical "GX10 訓練已停止" "$MSG" 2>/dev/null
echo "$MSG"
