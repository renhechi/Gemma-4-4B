#!/usr/bin/env python3
"""E4 corpus-dose analysis (ED §4.6 E4), fine-grained version.

Instead of only the 4 planned dose points (0.25/0.5/0.75/1.0 D*), every L_val eval
(every 25 steps = 0.00417 D*) is used as a dose observation.

Caveats handled:
  * E1/E2 ran on dfin_v3 from step 0 (one data version, one val set) -> directly comparable.
  * E3 switched data at step 1200 (v1->v2) and 1600 (v2->v3); each version has its own val set,
    so E3 L_val is only comparable within a segment. E3 is analysed per segment; its v3 segment
    (steps 1600-2500) is compared with E1/E2 by *v3 tokens seen*, not by total dose.

Output: docs/results/E4_dose_response.md, E4_dose_curve.csv, E4_dose_curve.png
"""
import csv
import json
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "docs" / "results"
DSTAR = 1_572_864_000
TOK_STEP = 262_144
COLORS = {"E1": "#2a78d6", "E2": "#eb6834", "E3": "#1baf7a"}   # categorical slots 1-3, fixed order


def evals(run):
    rs = [json.loads(l) for l in open(ROOT / "runs" / run / "metrics.jsonl")]
    sw = [r for r in rs if r.get("event") == "data_switch"]
    ev = sorted([r for r in rs if r.get("event") == "eval"], key=lambda r: r["step"])
    return ev, sw


def segments(ev, sw):
    """Split eval rows at data switches -> [(label, start_step, rows)]."""
    cuts = [0] + [s["at_step"] for s in sw] + [10 ** 9]
    names = ["v1", "v2", "v3"][-len(cuts) + 1:] if sw else ["v3"]
    return [(names[i], cuts[i], [r for r in ev if cuts[i] < r["step"] <= cuts[i + 1]]) for i in range(len(cuts) - 1)]


def analyse(rows, start_step):
    """rows: evals on one data version. Dose measured from the segment start."""
    d = [(r["step"] - start_step) * TOK_STEP / DSTAR for r in rows]
    L = [r["L_val"] for r in rows]
    L0, Lmin = L[0], min(L)
    imp = L0 - Lmin
    res = {"n_eval": len(rows), "L_first": L0, "L_last": L[-1], "L_min": Lmin,
           "step_min": rows[L.index(Lmin)]["step"], "dose_span": d[-1]}
    # dose to reach X% of total improvement (first eval reaching it)
    for q in (0.5, 0.9, 0.95, 0.99):
        tgt = L0 - q * imp
        res[f"d{int(q * 100)}"] = next((di for di, li in zip(d, L) if li <= tgt), None)
    # marginal gain per 0.05 D* bin
    bins = []
    b = 0.0
    while b < d[-1] - 1e-9:
        inb = [li for di, li in zip(d, L) if b <= di < b + 0.05]
        nxt = [li for di, li in zip(d, L) if b + 0.05 <= di < b + 0.10]
        if inb and nxt:
            bins.append((b, b + 0.05, min(inb) - min(nxt)))
        b += 0.05
    res["bins"] = bins
    # |ΔL| decay: log|ΔL| ~ a + k·dose  -> half-life ln2/-k (only Δ>0 points, after warmup step 100)
    pts = [(di, math.log(abs(r["delta_L_t"]))) for di, r in zip(d, rows)
           if r.get("delta_L_t") and abs(r["delta_L_t"]) > 0 and r["step"] - start_step >= 100]
    if len(pts) >= 5:
        mx = sum(p[0] for p in pts) / len(pts)
        my = sum(p[1] for p in pts) / len(pts)
        k = sum((x - mx) * (y - my) for x, y in pts) / sum((x - mx) ** 2 for x, _ in pts)
        res["dL_halflife_Dstar"] = math.log(2) / -k if k < 0 else float("inf")
    # plateau evals (|ΔL|<1e-3) and noise: fraction of evals where L_val rose
    dl = [r.get("delta_L_t") for r in rows if r.get("delta_L_t") is not None]
    res["n_plateau_evals"] = sum(abs(x) < 1e-3 for x in dl)
    res["first_plateau_dose"] = next((di for di, r in zip(d, rows) if r.get("delta_L_t") is not None and abs(r["delta_L_t"]) < 1e-3), None)
    res["frac_Lval_up"] = sum(b > a for a, b in zip(L, L[1:])) / max(1, len(L) - 1)
    # structure vs dose
    res["p_first"], res["p_last"] = rows[0]["p"], rows[-1]["p"]
    res["acc_mean_last10"] = sum(r["acceptability_rate"] for r in rows[-10:]) / len(rows[-10:])
    res["gn_median"] = sorted(r["grad_norm"] for r in rows)[len(rows) // 2]
    return res, d, L


def f(x, n=4):
    return "—" if x is None else (f"{x:.{n}f}" if isinstance(x, float) else str(x))


def main():
    series = {}
    md = ["# E4:corpus-dose 觀測(細粒度劑量反應)", "",
          "依 ED §4.6 E4。原設計 4 個 dose 點(0.25/0.5/0.75/1.0 D*);本研究 E1 於 0.258 D* 早停、E2 同點結束、",
          "E3 於 0.417 D* 結束,0.5 D* 以上無觀測,故改用**每 25 步一次的 L_val**(每點 0.00417 D*)做細粒度劑量反應。", "",
          "D* = 20 × N_FFN = 1.573B tokens;每步 262,144 tokens。", "",
          "**可比性**:E1/E2 全程 dfin_v3(同一 val 集)。E3 於 step 1200(v1→v2)、1600(v2→v3)換資料,各版 val 集不同,",
          "故 E3 依資料版本分段分析;與 E1/E2 比較時使用 E3-v3 段,dose 以「該段 v3 tokens」計(E3 進入 v3 時已累積 0.267 D*,且結構已長到 p≈21k)。", "",
          "## 1. 摘要表", "",
          "| run / 段 | evals | dose 範圍 (D*) | L_val 首 → 末 | 最低 L_val (step) | 50% 改善所需 dose | 90% | 95% | 99% | ΔL 半衰期 (D*) | \\|ΔL\\|<1e-3 次數 | L_val 回升比例 | p 首 → 末 | 末 10 次 acc | gn 中位數 |",
          "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    allrows = []
    for tag in ("E1", "E2", "E3"):
        ev, sw = evals(f"{tag}-full")
        for seg, s0, rows in segments(ev, sw):
            if len(rows) < 3:
                continue
            r, d, L = analyse(rows, s0)
            name = tag if seg == "v3" and s0 == 0 else f"{tag}-{seg}"
            series[name] = (d, L)
            md.append(f"| {name} (step {s0}–{rows[-1]['step']}) | {r['n_eval']} | 0–{r['dose_span']:.3f} | {r['L_first']:.4f} → {r['L_last']:.4f} | "
                      f"{r['L_min']:.4f} ({r['step_min']}) | {f(r['d50'], 3)} | {f(r['d90'], 3)} | {f(r['d95'], 3)} | {f(r['d99'], 3)} | "
                      f"{f(r.get('dL_halflife_Dstar'), 3)} | {r['n_plateau_evals']} | {r['frac_Lval_up']:.2f} | {r['p_first']:,} → {r['p_last']:,} | "
                      f"{r['acc_mean_last10']:.3f} | {r['gn_median']:.2f} |")
            for row, di in zip(rows, d):
                allrows.append([name, row["step"], round(di, 5), round(row["step"] * TOK_STEP / DSTAR, 5), row["L_val"],
                                row.get("delta_L_t"), row["acceptability_rate"], row["p"], row["grad_norm"], row["epsilon"], row.get("lam")])
            series[name + "_r"] = r
    md += ["", "dose 欄位皆自該段起點起算。「X% 改善」= 首次達到 L_first − X%·(L_first − L_min) 的 dose。",
           "ΔL 半衰期:對 log|ΔL_val| 與 dose 做線性迴歸(排除段內前 100 步)。", "",
           "## 2. 每 0.05 D* 的邊際改善(min L_val 差)", "", "| run / 段 | " + " | ".join(f"{b:.2f}→{b + 0.05:.2f}" for b in [i * 0.05 for i in range(8)]) + " |",
           "|---|" + "---|" * 8]
    for name in [k for k in series if not k.endswith("_r")]:
        bins = {round(b0, 2): g for b0, _, g in series[name + "_r"]["bins"]}
        md.append(f"| {name} | " + " | ".join(f"{bins[round(i * 0.05, 2)]:+.4f}" if round(i * 0.05, 2) in bins else "—" for i in range(8)) + " |")
    md += ["", "正值 = 下一個 0.05 D* 區間的最低 L_val 比本區間更低(仍在改善)。", ""]
    # same-dose comparison E1 vs E2 vs E3-v3
    md += ["## 3. 相同 v3 dose 的 L_val 比較", "", "| v3 dose (D*) | E1 | E2 | E3-v3 | E2−E1 | E3v3−E1 |", "|---|---|---|---|---|---|"]

    def at(name, x):
        d, L = series[name]
        c = [li for di, li in zip(d, L) if abs(di - x) < 0.0021]
        return c[0] if c else None
    for x in (0.025, 0.05, 0.10, 0.15, 0.20, 0.25, 0.258):
        e1, e2, e3 = at("E1", x), at("E2", x), at("E3-v3", x)
        md.append(f"| {x:.3f} | {f(e1)} | {f(e2)} | {f(e3)} | {f(e2 - e1 if e1 and e2 else None)} | {f(e3 - e1 if e1 and e3 else None)} |")
    md += ["", "E3-v3 段長 0.150 D*(step 1600–2500,900 步),故 0.15 以上無值。半衰期 inf = |ΔL| 未隨 dose 衰減(迴歸斜率 ≥ 0)。", "",
           "## 4. 解讀", "",
           "- 見上表自動計算值;文字結論寫於 EXPERIMENT_PROTOCOL_E4B.md §2f。", "",
           "圖:`E4_dose_curve.png`;逐點資料:`E4_dose_curve.csv`。"]
    (OUT / "E4_dose_response.md").write_text("\n".join(md) + "\n")
    with open(OUT / "E4_dose_curve.csv", "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["series", "step", "dose_in_segment_Dstar", "dose_total_Dstar", "L_val", "delta_L", "acceptability", "p", "grad_norm", "epsilon", "lambda"])
        w.writerows(allrows)
    plot(series)
    print("\n".join(md))


def plot(series):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.size": 10, "axes.edgecolor": "#c3c2b7", "axes.labelcolor": "#52514e",
                         "xtick.color": "#52514e", "ytick.color": "#52514e"})
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), facecolor="#fcfcfb")
    for ax in axes:
        ax.set_facecolor("#fcfcfb")
        ax.grid(True, color="#e8e7e2", linewidth=0.8)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
    a, b = axes
    for name, style in (("E1", "-"), ("E2", "-"), ("E3-v3", "--")):
        d, L = series[name]
        col = COLORS[name[:2]]
        a.plot(d, L, style, color=col, linewidth=2, label=name)
        a.annotate(name, (d[-1], L[-1]), xytext=(4, 0), textcoords="offset points", color="#0b0b0b", fontsize=9, va="center")
        r = series[name + "_r"]
        dd = [x for x in d][1:]
        pass
    a.set_xlabel("v3 tokens seen in segment (fraction of D*)")
    a.set_ylabel("L_val (dfin_v3 val set)")
    a.set_title("Validation loss vs dose", color="#0b0b0b", loc="left")
    a.legend(frameon=False)
    for name in ("E1", "E2", "E3-v3"):
        ev = [json.loads(l) for l in open(ROOT / "runs" / f"{name[:2]}-full" / "metrics.jsonl")]
        d, _ = series[name]
        s0 = 1600 if name == "E3-v3" else 0
        rows = sorted([r for r in ev if r.get("event") == "eval" and r["step"] > s0], key=lambda r: r["step"])
        xs = [(r["step"] - s0) * TOK_STEP / DSTAR for r in rows if r.get("delta_L_t")]
        ys = [abs(r["delta_L_t"]) for r in rows if r.get("delta_L_t")]
        b.plot(xs, ys, "-" if name != "E3-v3" else "--", color=COLORS[name[:2]], linewidth=2, label=name)
    b.axhline(1e-3, color="#52514e", linewidth=1, linestyle=":")
    b.annotate("early-stop threshold 1e-3", (0.005, 1.1e-3), color="#52514e", fontsize=8)
    b.set_yscale("log")
    b.set_xlabel("v3 tokens seen in segment (fraction of D*)")
    b.set_ylabel("|ΔL_val| per eval (log)")
    b.set_title("Per-eval improvement |ΔL_val|", color="#0b0b0b", loc="left")
    b.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(OUT / "E4_dose_curve.png", dpi=150, facecolor="#fcfcfb")


if __name__ == "__main__":
    main()
