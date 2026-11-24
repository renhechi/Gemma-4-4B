#!/usr/bin/env python3
"""E1 / E2 / E3 side-by-side comparison over steps 1–N (default 1500).

Outputs (docs/results/):
  E123_steps1_<N>.csv  every eval (25 steps), all metrics, long format
  E123_steps1_<N>.md   wide tables every 100 steps + phase summary + caveats
Usage: python3 scripts/compare_e123.py [N]
"""
import csv
import json
import statistics
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "docs" / "results"
N = int(sys.argv[1]) if len(sys.argv) > 1 else 1500
EXPS = [("E1", "E1-full"), ("E2", "E2-full"), ("E3", "E3-full")]


def data_ver(exp, step):
    if exp != "E3":
        return "v3"
    return "v1" if step <= 1200 else ("v2" if step <= 1600 else "v3")


def rows_for(tag, run):
    rs = [json.loads(l) for l in open(ROOT / "runs" / run / "metrics.jsonl")]
    st = {r["step"]: r for r in rs if r.get("event") in ("step", "eval")}
    out = []
    for r in (x for x in rs if x.get("event") == "eval" and x["step"] <= N):
        s = r["step"]
        win = [st[k] for k in range(s - 24, s + 1) if k in st]
        ctrl = [w["ctrl"] for w in win if w.get("ctrl")]
        out.append(dict(
            exp=tag, step=s, data=data_ver(tag, s), tokens_M=round(r["tokens_seen"] / 1e6, 1),
            frac_Dstar=r.get("tokens_frac_Dstar", round(r["tokens_seen"] / 1572864000, 4)),
            L_train_mean25=round(statistics.mean(w["L_train"] for w in win), 4),
            L_val=r["L_val"], delta_L=r["delta_L_t"],
            gap_mean25=round(r["L_val"] - statistics.mean(w["L_train"] for w in win), 4),
            plateau_streak=r["plateau_streak"], acceptability=r["acceptability_rate"],
            viol_before=r.get("violations_before_modules"), viol_after=r["violation_count"],
            dev_mean=r["dev_mean"], dev_base=r.get("dev_base_mean"),
            epsilon=round(r.get("epsilon", 0), 5), lam=r["lam"], p=r["p"],
            dp_plus=r["delta_p_plus"], dp_minus=r["delta_p_minus"], rollbacks=r["rollback_count"],
            struct_triggers=r["structuring_trigger_count"], repair_success=r["repair_success"],
            repair_attempts=r["repair_attempts"],
            adds_25=sum(c.get("adds", 0) for c in ctrl), struct_rb_25=sum(c.get("struct_rollbacks", 0) for c in ctrl),
            gn_median25=round(statistics.median(w["grad_norm"] for w in win), 3),
            gn_max25=round(max(w["grad_norm"] for w in win), 3),
            lr=r["lr"], tok_s=r["tokens_per_sec"], peak_mem_gib=r["peak_alloc_gib"]))
    return out


def fmt(v, nd=4):
    if v is None:
        return "—"
    return f"{v:.{nd}f}" if isinstance(v, float) else str(v)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    allrows = {tag: rows_for(tag, run) for tag, run in EXPS}
    flat = [r for tag in allrows for r in allrows[tag]]
    with open(OUT / f"E123_steps1_{N}.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(flat[0].keys()))
        w.writeheader(); w.writerows(flat)

    idx = {tag: {r["step"]: r for r in rows} for tag, rows in allrows.items()}
    steps = [s for s in range(100, N + 1, 100)]
    md = [f"# E1 / E2 / E3 比較(step 1–{N})", "",
          "資料:E1、E2 全程 dfin_v3;E3 為 v1(step ≤1200)→ v2(1225–1500)— **E3 的 L_val 是不同 val 集,不可與 E1/E2 直接比數值**。",
          "E3 另有設定變更:step 1–100 固定參考 z*、每 25 步檢核;101–200 移動參考、每 25 步;200 起移動參考、每步檢核(E1/E2 全程為後者)。", ""]
    tables = [
        ("L_val", "L_val", 4), ("ΔL_t", "delta_L", 4), ("L_train(25 步平均)", "L_train_mean25", 3),
        ("泛化落差(L_val − L_train 25 步平均)", "gap_mean25", 3),
        ("可接受度", "acceptability", 3), ("違反(修復前→後)", None, 0), ("每步漂移 dev_mean", "dev_mean", 4),
        ("與原始模型漂移 dev_base", "dev_base", 3), ("p(中間通道數)", "p", 0), ("累計 Δp⁺", "dp_plus", 0),
        ("累計 Δp⁻", "dp_minus", 0), ("累計 rollback", "rollbacks", 0), ("近 25 步新增通道", "adds_25", 0),
        ("grad_norm 中位數(近 25 步)", "gn_median25", 2), ("grad_norm 最大(近 25 步)", "gn_max25", 2),
        ("ε", "epsilon", 4), ("λ", "lam", 8), ("tok/s", "tok_s", 0), ("GPU 記憶體峰值 GiB", "peak_mem_gib", 2)]
    for title, key, nd in tables:
        md += [f"## {title}", "", "| step | E1 | E2 | E3 |", "|---|---|---|---|"]
        for s in steps:
            cells = []
            for tag, _ in EXPS:
                r = idx[tag].get(s)
                if r is None:
                    cells.append("—")
                elif key is None:
                    cells.append(f"{fmt(r['viol_before'])}→{r['viol_after']}")
                else:
                    v = r[key]
                    cells.append(fmt(v, nd) if isinstance(v, float) else fmt(v))
                if tag == "E3" and r is not None:
                    cells[-1] += f" ({r['data']})"
            md.append(f"| {s} | " + " | ".join(cells) + " |")
        md.append("")

    # phase summary
    md += ["## 分段摘要(平均)", "",
           "| 區段 | 指標 | E1 | E2 | E3 |", "|---|---|---|---|---|"]
    for lo, hi in [(25, 100), (125, 500), (525, 1000), (1025, N)]:
        for name, key, nd in [("ΔL_t", "delta_L", 4), ("可接受度", "acceptability", 3),
                              ("dev_mean", "dev_mean", 4), ("grad_norm 中位數", "gn_median25", 2),
                              ("tok/s", "tok_s", 0)]:
            vals = []
            for tag, _ in EXPS:
                xs = [r[key] for r in allrows[tag] if lo <= r["step"] <= hi and r[key] is not None]
                vals.append(fmt(round(statistics.mean(xs), nd), nd) if xs else "—")
            md.append(f"| {lo}–{hi} | {name} | " + " | ".join(vals) + " |")
    md.append("")
    last = {tag: allrows[tag][-1]["step"] if allrows[tag] else None for tag, _ in EXPS}
    md.append(f"資料截至:E1 step {last['E1']}、E2 step {last['E2']}、E3 step {last['E3']}(皆 ≤ {N})。")
    (OUT / f"E123_steps1_{N}.md").write_text("\n".join(md) + "\n")
    print("\n".join(md))


if __name__ == "__main__":
    main()
