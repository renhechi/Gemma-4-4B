# E3-full 保留紀錄(2026-11-03 暫停,待 E1/E2 結果比較後決定是否續跑)

使用者決定:「紀錄跟資料保留,其他的跑的如果相近,再來看是不是繼續跑」。

## 保留檔案(唯讀保護)

| 檔案 | 內容 | sha256 |
|---|---|---|
| `final_step2500.pt` | **續跑點**:step 2500 完整狀態(MLP p=22,608、AdamW、ε/λ/計數器、RNG、dfin_v3 offset 1600) | 1110d0cb05d3e6b11aee313bc69b7647e88fd89c6ee510cc1fe34bdf74a46bd0 |
| `dose_0.25.pt` | 0.25 D* dose checkpoint(step 1500,v2 資料) | 0615607ba4c58783d8c97a1c61182cb2315cbe91eb386b05d33b93e67fffaa15 |
| `ckpt_step1200_pre_instability.pt` | step 1200(v1 資料,不穩定前) | 72eb17a695f4bf7010096cab0de1f3b77fc78db3dfe8c715dc8ee6e89ed21f62 |
| `latest.pt` | 與 final_step2500.pt 相同(續跑時會被覆寫) | 同上 |
| `metrics.jsonl` | step 0–2500 全紀錄(含 data_switch、config_change、resume、run_end 事件) | — |
| `metrics_discarded_*.jsonl` | 各次續跑前被捨棄的中斷段紀錄(保留備查) | — |

資料(皆保留):`data/dfin`(v1,7.4G)、`data/dfin_v2`(2.7G)、`data/dfin_v3`(3.1G);建置程式 `scripts/build_dfin{,_v2,_v3}.py`。

## 結束時狀態

- 資料歷程:v1 step 0–1200 → v2 1200–1600 → v3 1600–2500
- v3 上 L_val 8-eval 區塊平均:2.7001 → 2.6783 → 2.6691 → 2.6605 → 2.6532(緩慢收斂);最低 2.6481(step 2475)
- p = 22,608(2.21×);累計 Δp⁺ 12,528、Δp⁻ 160、rollback 5,657;速度 ~315 tok/s

## 若要續跑 E3(從 step 2500)

1. 移除 `metrics.jsonl` 最後一行 `"event": "run_end"`(run_chain.sh 以此判斷完成)
2. 確認 `latest.pt` 與 `final_step2500.pt` 相同(sha256 如上),config `dfin_dir` 仍為 dfin_v3
3. `python3 scripts/train_lgdsrfpa.py --mode E3 --experiment E3-full`(trainer 自動從 latest.pt 續跑;資料目錄未變 → 不觸發 data_switch)
