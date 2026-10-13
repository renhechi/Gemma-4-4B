#!/usr/bin/env bash
# Stop E3 right after the step-N checkpoint, switch config dfin_dir → NEW, resume (trainer's data_switch logic).
set -u
N=${1:?step}; NEW=${2:?data dir}
cd /home/renhe/gemma4_pretrain
bash scripts/stop_at_ckpt.sh E3-full "$N" 2>&1 | tee -a logs/stop_at_ckpt.log
sed -i "s|^  dfin_dir: .*|  dfin_dir: $NEW   # $(date '+%F'): switched at step $N (build_dfin_v3.py: broken tables excised, tables complete)|" configs/e4b.yaml
grep -n "dfin_dir" configs/e4b.yaml
echo "[chain] $(date '+%F %T') switch data → $NEW at step $N, resume E3-full" >> logs/chain.log
/usr/bin/tmux new-session -d -s g4chain "bash scripts/run_chain.sh 2>&1 | tee -a logs/chain.log"
DISPLAY=:0 notify-send "GX10" "E3-full 已於 step $N 切換到 $NEW 並續跑" 2>/dev/null
echo "[switch] done"
