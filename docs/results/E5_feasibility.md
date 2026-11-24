# E5 可行性:ASUS GX10 單機軌道

依 ED §4.3 雙軌設計;**AI-Stack(AMD MI300/ROCm/DeepSpeed)軌道於本研究不採用(2026-11-24 研究決定)**,故僅報告 GX10。

硬體:ASUS Ascent GX10(NVIDIA GB10 Grace Blackwell、128 GB unified memory、4 TB NVMe)。
軟體:PyTorch 2.11+cu128、transformers 5.8、conda `gemma3_env`;純 PyTorch 單 GPU,無 DeepSpeed / 分散式。

## 1. 各實驗可執行性與資源

| 實驗 | can run | 結束方式 | 訓練步數 | tokens | 實際訓練時間 | tok/s(PD 正常期,中位數) | GPU 記憶體峰值 | 續跑次數 |
|---|---|---|---|---|---|---|---|---|
| E1 | yes | early_stop_plateau | 1550 | 406M | 228.3 h (9.5 d) | 494 | 34.2 GiB | 0 |
| E2 | yes | running | 1461 | 383M | 283.4 h (11.8 d) | 386 | 36.0 GiB | 0 |
| E3 | yes | user_stop_step2500 | 2500 | 655M | 579.9 h (24.2 d) | 364 | 37.7 GiB | 4 |

實際訓練時間 = 相鄰步紀錄間隔 < 2 h 者加總(排除停機/等待),含被捨棄重跑的區段。

## 2. 執行環境事件(可行性證據)

| 事件 | 期間 | 影響 | 處置 |
|---|---|---|---|
| USB-C PD 控制器卡死(GPU 鎖在 ~650 MHz / ~16 W) | 2026-09-26 → 10-01 | E3 tok/s 中位數 237(正常期 364) | 冷斷電(拔變壓器兩端 ≥60 s);同步驟實測加速 2.04× |
| 與其他 GPU 工作共用(reasoning_variants,~9 GB) | 2026-11-19 12:00–約 11-20 | E2 降至 ~175 tok/s(約減半) | 自動恢復,無需處置 |
| 網路中斷(Wi-Fi;有線埠 DOWN) | 2026-11-04 → 11-15 間歇 | GitHub 同步失敗 174 次,本機每小時 commit 未遺失 | 網路恢復後一次推送 |
| 結構成長造成減速 | E2/E3 全程 | E1 494 → E2 ~360 → E3 ~315 tok/s(p 增至 1.9–2.2×) | 屬方法本身的計算成本 |

## 3. 單機限制下的設定(GX10 profile)

| 項目 | 設定 | 理由 |
|---|---|---|
| 可訓練範圍 | 僅最後一層 Gated FFN(78.6M);其餘 41 層凍結 | 不需 optimizer offload / ZeRO |
| micro-batch × accum × seq | 4 × 16 × 4096 = 262,144 tok/step | 記憶體峰值 34–37 GiB(128 GiB 中) |
| 精度 / attention | bf16 / SDPA(global 層 head_dim 512 走 math kernel) | Flash attention 不支援 head_dim 512 |
| loss | 分塊(1,024 token)+ checkpoint 的 CE | 避免 262k 詞彙 logits 一次佔用 |
| seq_len 1024 實測 | 僅 +7% tok/s、記憶體 80 GiB | 不採用,維持 4096 |
| 瓶頸 | 時脈 3.7× 僅帶來 2.04× 加速 | 部分步驟受 LPDDR5X 頻寬限制 |

## 4. 雙軌表(ED §4.6 E5 格式)

| 軌道 | can run | tokens/s | peak memory | wall-clock | 備註 |
|---|---|---|---|---|---|
| ASUS GX10 – E1 | yes | 494 | 34.2 GiB | 見 §1 | |
| ASUS GX10 – E2 | yes | 386 | 36.0 GiB | 見 §1 | |
| ASUS GX10 – E3 | yes | 364 | 37.7 GiB | 見 §1 | |
| AI-Stack(AMD MI300) | — | — | — | — | 本研究不採用(研究決定),未執行 |
