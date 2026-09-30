# LGDSRFPA 實驗程序 — Gemma 4 E4B 版(取代 Gemma 3-TAIDE-12B)

依據:`proposal_gemma3.pdf` 第三章(演算法、式 6–22、Algo 1–7)與 `experiment design.pdf` 第四章(ED §4.1–4.10)。
本文件只替換基底模型並把 ED 的每一項觀測落成可執行程式;**不更動方法邏輯**。

## 1. 模型替換對照

| 項目 | 原設計(12B) | 本版(E4B) | 說明 |
|---|---|---|---|
| 基底模型 | Gemma-3-TAIDE-12b-Chat | `google/gemma-4-E4B-it` | Gemma 4 無純「4B」,E4B(effective 4B)為對應尺寸;選 -it 以對齊原設計之 Chat 模型,使 §4.9 專業人士問答評估可行 |
| 層數 / 可訓練層 | 48 / 第 48 層 | 42 / 第 42 層(index 41) | 「最後一層 Gated FFN」語意不變 |
| d_model / p | 3840 / 15360 | 2560 / 10240 | Gemma4TextMLP(GELU-tanh gated) |
| 可訓練參數 N | 177M | 78.64M | 3 × d × p |
| D* = 20 × N | 3.54B | **1.573B tokens** | 同一 Chinchilla 規則 |
| 資料量檢查點 | 0.25/0.50/0.75 D* | **0.25/0.50/0.75/1.00 D*** | ED §4.4 建議版(含 0.75) |

## 2. 資料(以現有資料開始)

`scripts/build_dfin.py` → `data/dfin/`(已完成)

- 來源:gemma3_pretrain_data 新聞、TWSE ESG(文字+OCR)、股東會年報(annual,現已 4.2GB)、邵汶學術、法規×4、保險法規×4、金管會裁罰×2、財經新聞、EDGAR 英文(上限 12%)。
- 清理:CJK 字間空白移除(修正 v5 的「安 能 聚」斷字問題)、全域 sha1 去重、排除 FinKD ppl_heldout(130 篇)。
- **結果:940,105 篇;train 1.958B tokens(≥ D*,不需重複 epoch);val 9.9M tokens(固定)。**
- 組成(token):annual 77.6%、news 8.3%、edgar 6.6%、esg+ocr 6.8%、學術 0.5%、法規/裁罰 0.2%。
  → 年報占比高是「現有資料」的實況;若需平衡,可於 build_dfin 加 annual 上限重建(不影響程式)。
- 訓練順序於建檔時固定(seed 42),訓練依序消耗 ⇒ **tokens_seen 即累積 corpus dose**,E1/E2/E3 看到完全相同的資料序列。

## 3. 實驗矩陣(ED §4.6)

| 實驗 | 指令 | 模組 | 主要輸出 |
|---|---|---|---|
| E1 | `--mode E1` | Weight-Tuning only(learning goal 只記錄不介入) | L_train、L_val、gap、violation count |
| E2 | `--mode E2` | + Structuring(Selecting→Isolating→Node-Adding) | repair success、Δp⁺、violation before/after |
| E3 | `--mode E3` | + Network-Tuning(λ‖w‖²、Node-Pruning、rollback) | Δp⁺、Δp⁻、rollback、Fail、λ 軌跡 |
| E4 | (E1–E3 共同產出) | corpus-dose 觀測 | 各 dose 檢查點之 loss、ΔL、gap、plateau 表 |
| E5 | (同上) | GX10 可行性 | tokens/s、peak memory、wall-clock、can run;AI-Stack(AMD)列為本機未執行 |

三者**同一基底、同一資料序列、同一 seed**,皆從原始權重開始(非串接),使差異只來自模組。

## 4. 演算法實作對應

- **Learning goal(式 10)**:z_c = 最後層 Gated FFN 輸出(序列 token 平均);判定 ‖z_c − z*_c‖/‖z*_c‖ ≤ ε(參考集 R = 64 條固定 val 序列,與 L_val 集不重疊)。
  因前 41 層凍結,x_c 永不改變 ⇒ 檢核只需 MLP 前向。
  - **z* 採移動參考(2026-09-29 使用者選 A)**:每次 control 回合結束後,以該回合模組作用後之網路輸出作為新的 z*(Algo 7 Step 4 Feedback)。
    learning goal 因而約束「每 25 步區間內的表示變化量」。原固定 z* = F_L(x; w_0)(v2)在 step 50–125 違反數卡在 62/64、NT 從未啟動、λ 持續衰減,
    即 CPT 下 loss 大幅下降(4.5→2.8)必然使最後層偏離原始表示,固定參考無法作為可接受性準則。
  - 累積偏離原始模型之量仍記錄為 `dev_base_mean`(不作控制訊號),供論文報告。
- **Weight-Tuning(式 11–12,ED §4.2)**:最佳化目標為 L_CLM;AdamW lr 2e-5 cosine(warmup 100)、batch 64×4096 = 262k tok/step、共 6,001 步。
- **調度(Algo 7)**:**每一次 Weight-Tuning 更新後**檢核 learning goal(2026-10-01 起;第三章「每一輪更新後,系統先檢核 learning goal」):
  有違反 → Structuring;ε 自動調整;參考 z* 更新。Network-Tuning(剪枝)與 λ(§4.8,以 25 步窗口內是否違反/是否修復判定)每 25 步一次。
  原每 25 步檢核一次時,區間內漂移 ≈ 0.005/步 × 25 ≈ 0.12–0.14 ≫ ε(0.014),64/64 違反、每輪只修 2 個、NT 從未執行 → 改為逐步檢核。
- **Selecting(式 13–15)**:κ = 違反樣本中 δ 最小者。
- **Isolating(式 16)**:在 R 的池化輸入 x̄ 空間中搜尋 γ(T=32 次),使 γᵀ(x̄_c − x̄_κ) < −ζ ∀c≠κ。
  *調整:採單側隔離* — 單一 gated channel 沿 γ 為單調,無法實作式 16 的雙側夾擠;單側條件仍保證新 channel 對 κ 以外樣本近乎不激活。
- **Node-Adding(式 17–20)**:gate row = βγ、gate bias = β(ζ − γᵀx̄_κ)(式 18;因 HF MLP 無 bias,本版為 gate/up 加入 bias,原 channel bias=0 ⇒ 初始函數完全不變)、up = 常數 1;
  down column 依式 19 以 (z*_κ − z_κ)/ā_κ 修正偏差最大之分量直到殘差 ≤ 0.5ε,其餘分量為 0(式 20)。
  加入後若 κ 未修復或總違反數未下降 → rollback。
- **Network-Tuning(式 21–22、Algo 5/6、ED §4.8)**:E3 於無違反時對 loss 加 λ‖w‖²;以 saliency(‖W_down[:,h]‖·mean|a_h|)由小到大一次剪 32 個 channel;
  剪後仍滿足 learning goal → 接受(Δp⁻),否則回復 Snapshot(Fail+1);Fail ≥ F_max(5) 結束該輪。後續 Weight-Tuning 即在 L_N 下進行。
- **λ 排程(ED §4.8)**:無違反且 L_val 未惡化 → λ×1.05;違反但修復成功 → 維持;違反持續或 L_val 惡化 → λ×0.7;範圍 [1e-8, 1e-4]。
- **優化器狀態**:結構改變時 AdamW 動量按 channel 重新對應(保留存活 channel 的 m/v),非整體重置。
- **停止規則(2026-09-27 使用者決定,沿用 12B E1-full 模式)**:
  - 上限:**D* = 20 × N_FFN**(Chinchilla scaling law;N_FFN 於啟動時由模型實算 = 3·2560·10240 = 78.64M → 1.573B tokens = 6,001 步)。
  - **Early stop**:L_val 每 25 步評估一次(12B 為 20),|ΔL_val| < 1e-3 連續 3 次 → 停止,另存 `stop_<frac>Dstar.pt`;warmup(100 步)前不停。
  - 安全停止:NaN、L_train EMA > 5× 最佳、L_val > 5× 最佳。
  - 註:此為對 ED §4.7「early stopping 不介入主觀測」之修正 — 在 GX10 算力限制下,以 plateau 決定實際觀測到的 dose,D* 為理論上限;論文中需說明此偏離。
- **ε 自動下降(沿用 12B E1-full)**:ε 由 0.015 起,每次 learning-goal 檢核若無違反則 ε ← max(0.005, 0.95ε),一出現違反即維持。
  避免固定 ε 過寬(v0 用 0.02,前 25 步漂移僅 0.006,E2/E3 永不觸發)或過窄;ε 軌跡記錄於 metrics(`epsilon`)。

## 5. 執行程序(ED §4.10)

```bash
cd ~/gemma4_pretrain
# 0) 元件檢查(已通過:loss 差 3e-5、結構操作前後輸出差 0、optimizer state 對應正確)
conda run -n gemma3_env python3 scripts/test_components.py
# 1) 前導篩選 smoke(ED §4.7:決定 ε;觀察 dev 分佈)
conda run -n gemma3_env python3 scripts/train_lgdsrfpa.py --mode E3 --experiment E3-smoke \
    --max-steps 20 --micro-batch 2 --grad-accum 2 --control-interval 5 --eval-interval 10 --n-val 8 --n-ref 16
# 2) 正式:E3 → E1 → E2 依序(可中斷續跑)
nohup bash scripts/run_chain.sh > logs/chain.log 2>&1 &
# 2b) 紀錄自動推 GitHub(每小時:分析表/圖 + metrics + logs → renhechi/Gemma-4-4B)
tmux new-session -d -s g4sync "bash scripts/sync_github.sh"
# 3) 分析(E4 dose/plateau 表、模組統計、E5 可行性表、曲線圖 → docs/results/)
conda run -n gemma3_env python3 scripts/analyze_runs.py E1-full E2-full E3-full
# 4) 最終評估(一次):PPL、數值 QA、專業人士盲評包
conda run -n gemma3_env python3 scripts/eval_final.py --ckpt runs/E1-full/dose_1.00.pt \
    --ckpt runs/E2-full/dose_1.00.pt --ckpt runs/E3-full/dose_1.00.pt --expert-kit
```

輸出:`runs/<exp>/metrics.jsonl`(每步 + 每次 eval 一列)、`runs/<exp>/dose_{0.25,0.50,0.75,1.00}.pt`(只存最後層 MLP,約 160MB)、
`docs/results/{curves.png, dose_table.md, module_table.md, feasibility.md, final_eval.json, expert_kit/}`。

## 6. 評估(ED §4.9)

- CLM loss:所有 checkpoint(主指標)。
- 自動評估:FinKD held-out PPL(zh/en)、qa_benchmark_350_zh(數值抽取/計算)。注意:QA context 來自 EDGAR/年報,可能與 D_fin 重疊 → 報告時需附 contamination 註記(可沿用 gemma3 track 的 flag_contamination.py)。
- 專業人士:**只一次**,base vs E3 最終模型,30 題(eval/expert_questions.jsonl,6 類,請審閱修改),A/B 隨機匿名 + 5 構面 rubric。
- AI 輔助 rubric 評分:可選,限中間 checkpoint,論文中表述為「AI 輔助判讀」。

## 7. 已知問題 / 待決

**執行時間分析(2026-09-27)**:與 student-4B direct-CPT(246M tok / 5.3 天 / ~539 tok/s)相比,本實驗 token 量 6.4×(20× 以 78.6M FFN 參數計,而非 14M LoRA 參數),
且每 token 需走完 41 層凍結 forward + 262k lm_head(只開最後層不省 forward)。seq 1024 實測僅 +7%(272 vs 255 tok/s @650MHz)→ 保留 4096,GPU 修復後重測(`scripts/bench_seqlen.py`)。
重開機後 `@reboot scripts/boot_resume_g4.sh` 自動續跑 chain + sync;12B E1 的 @reboot 已註解停用(備份 logs/crontab_backup_20260927.txt)。


1. **GX10 GPU 時脈卡在 ~650 MHz(最高 3003)**:bf16 matmul 僅 3.7 TFLOPs,訓練 ~250 tok/s ⇒ 1.573B 需 ~70 天。
   溫度正常、無 lock,SW Power Capping 計數持續累加、開機已 27 天。需 sudo:`sudo nvidia-smi -rgc` 或重開機後重測
   (`scripts/profile_speed.py`)。正常時脈預期提升 10× 以上。12B 軌道的 105 tok/s 很可能也受此影響。
2. z* 由固定改為移動參考(見 §4);v2(固定 z*)step 101–128 紀錄存於 `runs/E3-full/metrics_discarded_steps101-128_fixedref.jsonl`,自 step-100 checkpoint 續跑。
3. ε 改為自動下降(0.015 → 下限 0.005),見 §4。v0(固定 ε=0.02)於 step 25 停止並封存為 `runs/E3-full_v0_eps0.02/`。
3. 年報占 D_fin 77.6%,是否設上限重建,屬研究設計決定。
