# LGDSRFPA × Gemma-4-E4B — 研究參數總表

依據:`configs/e4b.yaml`、`scripts/train_lgdsrfpa.py`、`data/dfin/meta.json`。每次更動的時間與理由見 `EXPERIMENT_PROTOCOL_E4B.md`。

## 1. 模型(第三章 §3.1)

| 參數 | 值 | 對照 12B 原設計 |
|---|---|---|
| 基底模型 | `google/gemma-4-E4B-it` | Gemma-3-TAIDE-12b-Chat |
| 總參數 / 文字模型 | 7.94B / 7.46B(其中 PLE 查表 2.85B,實際運算 transformer 3.95B) | 12B |
| Decoder 層數 | 42 | 48 |
| 可訓練層 | 最後一層(index 41)之 Gated FFN;其餘凍結 | 第 48 層 |
| d_model / intermediate p₀ | 2560 / 10240 | 3840 / 15360 |
| 可訓練參數 N_FFN | 3 × 2560 × 10240 = **78.64M**(另加 gate/up bias 2p) | 177M |
| 激活函數 | GELU-tanh(gated) | GELU(gated) |
| 精度 / attention | bf16 / SDPA | 同 |

## 2. 資料(ED §4.4)

| 參數 | 值 |
|---|---|
| D_fin 文件數 | 940,105(去重、CJK 空白清理、排除 FinKD held-out 130 篇) |
| 訓練 tokens(唯一) | 1.958B;val 9.9M(固定) |
| 組成(token) | 年報 77.6%、新聞 8.3%、EDGAR 6.6%、ESG+OCR 6.8%、學術 0.5%、法規/裁罰 0.2% |
| 上採樣 | 法規 ×4、保險法規 ×4、裁罰 ×2 |
| 序列長度 | 4096 |
| L_val 集 | 固定 128 條序列(≈0.5M tokens) |
| Learning-goal 參考集 R | 固定 64 條 val 序列(與 L_val 集不重疊) |
| 資料順序 | 建檔時固定(seed 42),依序消耗 ⇒ tokens_seen = corpus dose |

## 3. 訓練預算與資料量觀測(ED §4.4)

| 參數 | 值 |
|---|---|
| D*(上限) | **20 × N_FFN = 1.573B tokens**(Chinchilla ratio 20) |
| 每步 tokens | 4(micro-batch)× 16(grad accum)× 4096 = 262,144 |
| 總步數上限 | 6,000 |
| Dose 檢查點 | 0.25 / 0.50 / 0.75 / 1.00 D* |

## 4. Weight-Tuning(式 11–12;ED §4.2)

| 參數 | 值 |
|---|---|
| 目標函數 | L_CLM(E3 於 NT 窗口另加 λ‖w‖²,式 21) |
| Optimizer | AdamW(fused),β = (0.9, 0.95),weight decay 0(正規化以 λ‖w‖² 表示) |
| Learning rate | 2e-5,warmup 100 步,cosine 降至 1e-6 |
| Gradient clipping | 1.0 |
| Seed | 42 |

## 5. Learning goal(式 10、13–15)

| 參數 | 值 |
|---|---|
| 量測位置 z | 最後層 Gated FFN 輸出,序列 token 平均 |
| 判定 | ‖z_c − z*_c‖ / ‖z*_c‖ ≤ ε |
| 參考 z* | **移動參考**:每次檢核回合後以當前網路輸出更新(Algo 7 Step 4 Feedback) |
| ε 初值 / 下限 | 0.015 / 0.005 |
| ε 調整 | 無違反 ⇒ ε × 0.95;有違反 ⇒ 維持(沿用 12B E1-full) |
| 檢核頻率 | **每一次 Weight-Tuning 更新後**(control_interval = 1) |

## 6. Structuring(Algo 1–4;實作差異見 protocol §6b)

| 參數 | 值 |
|---|---|
| Selecting | κ = 違反樣本中 δ 最小者(式 15) |
| Isolating 最大嘗試 T | 32 |
| 每回合最多新增 channel | 8 |
| Node-Adding 修復目標 | 殘差 ≤ 0.5·ε |
| Gate 增益 | β = 4/ζ(實作自加,非計畫書式 17) |
| 新增後判定 | κ 已修復且總違反數下降 ⇒ 接受;否則 rollback |

## 7. Network-Tuning 與 λ(Algo 4–6、式 21–22;ED §4.8)

| 參數 | 值 |
|---|---|
| 執行頻率 | 每 25 步(且當下無違反) |
| 剪枝單位 / 每回合最多接受 | 32 channels / 4 個區塊 |
| 剪枝順序 | saliency = ‖W_down[:,h]‖ · mean\|a_h\| 由小到大 |
| F_max | 5(連續失敗即結束該輪) |
| λ 初值 / 範圍 | 1e-6 / [1e-8, 1e-4] |
| λ 調整(以 25 步窗口判定) | 無違反且 L_val 未惡化 ⇒ ×1.05;違反皆修復 ⇒ 維持;有未修復 ⇒ ×0.70;L_val 惡化 > 0.002 ⇒ ×0.70 |

## 8. 停止規則(ED §4.5、§4.7;沿用 12B E1-full)

| 參數 | 值 |
|---|---|
| L_val 評估頻率 | 每 25 步 |
| Plateau / early stop | \|ΔL_val\| < 1e-3 連續 3 次 ⇒ 停止(step ≥ 100 才生效) |
| 上限 | D*(6,000 步) |
| 安全停止 | NaN;L_train EMA 或 L_val > 5 × 最佳值 |
| Checkpoint | 每 100 步(latest.pt,可續跑);dose 檢查點另存 |

## 9. 執行環境(ED §4.3,GX10 軌道)

| 參數 | 值 |
|---|---|
| 硬體 | ASUS Ascent GX10(NVIDIA GB10,128GB unified memory) |
| 軟體 | conda `gemma3_env`:PyTorch 2.11+cu128、transformers 5.8 |
| GPU 狀態 | 冷斷電後 ~2,390 MHz / 85–92 W(2026-10-01 前 PD 卡死:650 MHz / 16 W) |
| 速度 | ~500 tok/s 基本;含逐步檢核與 Node-Adding 約 390 tok/s |
| GPU 記憶體峰值 | ~35 GiB |
| AI-Stack(AMD MI300)軌道 | 本機未執行 |

## 10. 實驗矩陣(ED §4.6)

| 實驗 | 模組 | 狀態 |
|---|---|---|
| E3-full | Weight-Tuning + Structuring + Network-Tuning | 執行中 |
| E1-full | Weight-Tuning only | 排程(E3 完成後) |
| E2-full | + Structuring | 排程(E1 完成後) |
| E4 | corpus-dose 觀測(由上述 run 產出) | — |
| E5 | GX10 可行性(tokens/s、記憶體、wall-clock) | — |
