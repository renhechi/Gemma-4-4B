#!/usr/bin/env bash
# @reboot: re-measure GPU speed, then resume the Gemma-4 LGDSRFPA chain + hourly GitHub sync in tmux.
set -u
export PATH="/home/renhe/miniconda3/bin:$PATH"
cd /home/renhe/gemma4_pretrain
sleep 90
echo "[boot] $(date '+%F %T') clocks: $(nvidia-smi --query-gpu=clocks.sm,clocks.max.sm --format=csv,noheader)" >> logs/chain.log
/usr/bin/tmux has-session -t g4chain 2>/dev/null || /usr/bin/tmux new-session -d -s g4chain "bash scripts/run_chain.sh 2>&1 | tee -a logs/chain.log"
/usr/bin/tmux has-session -t g4sync 2>/dev/null || /usr/bin/tmux new-session -d -s g4sync "bash scripts/sync_github.sh"
