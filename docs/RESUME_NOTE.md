# 暫停紀錄與續跑說明(2026-10-09 暫停)

## 暫停時狀態

| 項目 | 值 |
|---|---|
| 實驗 | E3-full(Full LGDSRFPA),E1-full / E2-full 尚未開始 |
| 最後執行 step | 1258 / 6000(329.8M tokens,20.97% D*) |
| 可續跑 checkpoint | `runs/E3-full/latest.pt` = **step 1200**(另備份 `ckpt_step1200_pre_instability.pt`) |
| 最佳 L_val | 2.2370(step 1225);step 1200 時 2.2462 |
| p | 15,603(step 1200),原始 10,240 |
| 暫停原因 | 使用者離開一個多月,電腦關機 |

## 暫停前觀察到的不穩定(step 1244 起)

- grad_norm 由平常 10–20 升至 40–103;每步漂移 0.02–0.04(ε 的 4–8 倍)
- L_val 在 step 1250 回升 +0.10(2.2370 → 2.3387)
- 屬 Structuring 自我強化迴圈(protocol §6b)的惡化階段;尚未觸發安全停止(L_val > 5× 最佳)
- step 1201–1258 的紀錄仍在 `metrics.jsonl`;續跑時會自動移到 `metrics_discarded_after_step1200_*.jsonl`

## 回來後待決定

| 選項 | 做法 |
|---|---|
| A | 從 step 1200 續跑 E3,觀察是否自行恢復(步 1244 前後應會重現相同不穩定,因資料與 seed 固定) |
| B(建議) | 以 step 1200 checkpoint 作為 E3 結果結束 E3,記錄不穩定為發現,接著跑 E1-full、E2-full |
| C | 加入新增 channel 增益上限,從 step 1200 續跑(屬實作修改) |

## 開機後步驟

1. **先確認 GPU 時脈**(長時間關機後 PD 狀態可能異常):
   `nvidia-smi --query-gpu=clocks.sm,power.draw --format=csv` — 有負載時應 > 1400 MHz、~85 W;若卡在 ~650 MHz / ~16 W,做冷斷電(拔變壓器兩端 + 等 60 秒以上)。
2. 開機自動續跑**已停用**(crontab 中 `boot_resume_g4.sh` 已註解;備份 `logs/crontab_backup_20261009.txt`)。
3. 依決定執行:
   - A:`tmux new-session -d -s g4chain "bash scripts/run_chain.sh 2>&1 | tee -a logs/chain.log"`
   - B:在 `runs/E3-full/metrics.jsonl` 寫入 run_end(stop_reason=user_stop_step1200),再啟動 chain(會跳過 E3、開始 E1)
   - C:需先修改 `src/lgdsrfpa4` / `train_lgdsrfpa.py`
4. 重新啟動同步:`tmux new-session -d -s g4sync "bash scripts/sync_github.sh"`;若要恢復開機自動續跑,把 crontab 那行取消註解。
