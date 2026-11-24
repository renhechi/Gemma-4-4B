#!/usr/bin/env python3
"""E5 feasibility (ED §4.3, §4.6) — ASUS GX10 track only.

User decision 2026-11-24: the AI-Stack (AMD MI300 / ROCm / DeepSpeed) track is not used in this study.
Reads runs/*/metrics.jsonl (kept + discarded rows) and logs/chain.log.
Output: docs/results/E5_feasibility.md
"""
import glob
import json
import statistics
from datetime import datetime, timedelta, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "docs" / "results"
TZ = timezone(timedelta(hours=8))
COLD_DRAIN = datetime(2026, 10, 1, 10, 30, tzinfo=TZ).timestamp()   # PD wedge fixed (cold power drain)
RUNS = [("E1", "E1-full"), ("E2", "E2-full"), ("E3", "E3-full")]


def rows(run):
    out = []
    for f in [ROOT / "runs" / run / "metrics.jsonl"] + sorted(glob.glob(str(ROOT / "runs" / run / "metrics_discarded_*.jsonl"))):
        out += [json.loads(l) for l in open(f)]
    return out


def fmt_h(s):
    return f"{s / 3600:.1f} h ({s / 86400:.1f} d)"


def main():
    md = ["# E5 可行性:ASUS GX10 單機軌道", "",
          "依 ED §4.3 雙軌設計;**AI-Stack(AMD MI300/ROCm/DeepSpeed)軌道於本研究不採用(2026-11-24 研究決定)**,故僅報告 GX10。", "",
          "硬體:ASUS Ascent GX10(NVIDIA GB10 Grace Blackwell、128 GB unified memory、4 TB NVMe)。",
          "軟體:PyTorch 2.11+cu128、transformers 5.8、conda `gemma3_env`;純 PyTorch 單 GPU,無 DeepSpeed / 分散式。", "",
          "## 1. 各實驗可執行性與資源", "",
          "| 實驗 | can run | 結束方式 | 訓練步數 | tokens | 實際訓練時間 | tok/s(PD 正常期,中位數) | GPU 記憶體峰值 | 續跑次數 |",
          "|---|---|---|---|---|---|---|---|---|"]
    detail = []
    for tag, run in RUNS:
        rs = rows(run)
        st = sorted([r for r in rs if r.get("event") in ("step", "eval")], key=lambda r: r["time"])
        ev = [r for r in st if r["event"] == "eval"]
        end = [r for r in rs if r.get("event") == "run_end"]
        resumes = sum(1 for r in rs if r.get("event") == "resume")
        active = sum(b["time"] - a["time"] for a, b in zip(st, st[1:]) if 0 < b["time"] - a["time"] < 7200)
        last_step = max(r["step"] for r in st)
        tps_ok = [r["tokens_per_sec"] for r in ev if r["time"] > COLD_DRAIN]
        tps_wedge = [r["tokens_per_sec"] for r in ev if r["time"] <= COLD_DRAIN]
        peak = max(r.get("peak_alloc_gib", 0) for r in ev)
        how = end[-1].get("stop_reason", "—").split(" ")[0] if end else "running"
        kept_last = max(r["step"] for r in rs if r.get("event") in ("step", "eval") and r in
                        [json.loads(l) for l in open(ROOT / "runs" / run / "metrics.jsonl")])
        md.append(f"| {tag} | yes | {how} | {kept_last} | {kept_last * 262144 / 1e6:.0f}M | {fmt_h(active)} | "
                  f"{statistics.median(tps_ok):.0f} | {peak:.1f} GiB | {resumes} |")
        detail.append((tag, tps_wedge, tps_ok))
    md += ["", "實際訓練時間 = 相鄰步紀錄間隔 < 2 h 者加總(排除停機/等待),含被捨棄重跑的區段。", "",
           "## 2. 執行環境事件(可行性證據)", "",
           "| 事件 | 期間 | 影響 | 處置 |", "|---|---|---|---|"]
    tag, wedge, ok = detail[2]
    if wedge:
        md.append(f"| USB-C PD 控制器卡死(GPU 鎖在 ~650 MHz / ~16 W) | 2026-09-26 → 10-01 | E3 tok/s 中位數 {statistics.median(wedge):.0f}(正常期 {statistics.median(ok):.0f}) | 冷斷電(拔變壓器兩端 ≥60 s);同步驟實測加速 2.04× |")
    md += ["| 與其他 GPU 工作共用(reasoning_variants,~9 GB) | 2026-11-19 12:00–約 11-20 | E2 降至 ~175 tok/s(約減半) | 自動恢復,無需處置 |",
           "| 網路中斷(Wi-Fi;有線埠 DOWN) | 2026-11-04 → 11-15 間歇 | GitHub 同步失敗 174 次,本機每小時 commit 未遺失 | 網路恢復後一次推送 |",
           "| 結構成長造成減速 | E2/E3 全程 | E1 494 → E2 ~360 → E3 ~315 tok/s(p 增至 1.9–2.2×) | 屬方法本身的計算成本 |", "",
           "## 3. 單機限制下的設定(GX10 profile)", "",
           "| 項目 | 設定 | 理由 |", "|---|---|---|",
           "| 可訓練範圍 | 僅最後一層 Gated FFN(78.6M);其餘 41 層凍結 | 不需 optimizer offload / ZeRO |",
           "| micro-batch × accum × seq | 4 × 16 × 4096 = 262,144 tok/step | 記憶體峰值 34–37 GiB(128 GiB 中) |",
           "| 精度 / attention | bf16 / SDPA(global 層 head_dim 512 走 math kernel) | Flash attention 不支援 head_dim 512 |",
           "| loss | 分塊(1,024 token)+ checkpoint 的 CE | 避免 262k 詞彙 logits 一次佔用 |",
           "| seq_len 1024 實測 | 僅 +7% tok/s、記憶體 80 GiB | 不採用,維持 4096 |",
           "| 瓶頸 | 時脈 3.7× 僅帶來 2.04× 加速 | 部分步驟受 LPDDR5X 頻寬限制 |", "",
           "## 4. 雙軌表(ED §4.6 E5 格式)", "",
           "| 軌道 | can run | tokens/s | peak memory | wall-clock | 備註 |", "|---|---|---|---|---|---|"]
    for tag, run in RUNS:
        ev = [r for r in rows(run) if r.get("event") == "eval" and r["time"] > COLD_DRAIN]
        md.append(f"| ASUS GX10 – {tag} | yes | {statistics.median(r['tokens_per_sec'] for r in ev):.0f} | "
                  f"{max(r['peak_alloc_gib'] for r in ev):.1f} GiB | 見 §1 | |")
    md.append("| AI-Stack(AMD MI300) | — | — | — | — | 本研究不採用(研究決定),未執行 |")
    (OUT / "E5_feasibility.md").write_text("\n".join(md) + "\n")
    print("\n".join(md))


if __name__ == "__main__":
    main()
