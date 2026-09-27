#!/usr/bin/env python3
"""ED §4.10 outputs from runs/<exp>/metrics.jsonl.

Produces (docs/results/):
  * curves.png            L_train / L_val / gap, acceptability, violations, Δp±, λ, p  (E1–E3 overlaid)
  * dose_table.csv/.md    E4 corpus-dose plateau table (0.25/0.50/0.75/1.00 D*)
  * module_table.md       E1/E2/E3 module statistics (triggers, repair success, Δp±, rollback, Fail)
  * feasibility.md        E5 GX10 track (tokens/s, peak memory, wall-clock, can-run); AI-Stack row = not run here
Usage: python3 scripts/analyze_runs.py E1-full E2-full E3-full
"""
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "docs" / "results"


def load(exp):
    rows = [json.loads(l) for l in open(ROOT / "runs" / exp / "metrics.jsonl")]
    start = next((r for r in rows if r.get("event") == "run_start"), {})
    return start, [r for r in rows if r.get("event") in ("step", "eval")], \
        [r for r in rows if r.get("event") == "eval"], next((r for r in rows[::-1] if r.get("event") == "run_end"), None)


def main(exps):
    OUT.mkdir(parents=True, exist_ok=True)
    data = {e: load(e) for e in exps}
    fig, ax = plt.subplots(3, 3, figsize=(16, 12))
    ax = ax.ravel()
    for e, (st, steps, evals, _) in data.items():
        tok = lambda rs: [r["tokens_seen"] / 1e9 for r in rs]  # noqa: E731
        ax[0].plot(tok(steps), [r["L_train"] for r in steps], lw=.5, alpha=.5, label=f"{e} L_train")
        ax[0].plot(tok(evals), [r["L_val"] for r in evals], lw=1.5, label=f"{e} L_val")
        ax[1].plot(tok(evals), [r["generalization_gap"] for r in evals], label=e)
        ax[2].plot(tok(evals), [r["acceptability_rate"] for r in evals], label=e)
        ax[3].plot(tok(evals), [r["violation_count"] for r in evals], label=e)
        ax[4].plot(tok(evals), [r["structuring_trigger_count"] for r in evals], label=e)
        ax[5].plot(tok(evals), [r["delta_p_plus"] for r in evals], label=f"{e} Δp+")
        ax[5].plot(tok(evals), [r["delta_p_minus"] for r in evals], "--", label=f"{e} Δp−")
        ax[6].plot(tok(steps), [r["lam"] for r in steps], label=e)
        ax[7].plot(tok(steps), [r["p"] for r in steps], label=e)
        ax[8].plot(tok(evals), [r["rollback_count"] for r in evals], label=e)
    titles = ["CLM loss", "generalization gap (L_val−L_train)", "acceptability rate (R)",
              "violation count (R)", "Structuring trigger count", "Δp trajectory",
              "λ trajectory", "intermediate channels p", "rollback count"]
    for a, t in zip(ax, titles):
        a.set_title(t); a.set_xlabel("tokens seen (B)"); a.legend(fontsize=7); a.grid(alpha=.3)
    ax[6].set_yscale("symlog", linthresh=1e-9)
    d_star = next(iter(data.values()))[0].get("d_star", 1.573e9)
    for a in ax:
        for f in (0.25, .5, .75, 1.0):
            a.axvline(f * d_star / 1e9, color="grey", lw=.5, ls=":")
    fig.tight_layout(); fig.savefig(OUT / "curves.png", dpi=130)

    # E4 dose table
    lines = ["experiment,dose,tokens_B,L_train,L_val,delta_L_t,gap,acceptability,violations,struct_triggers,dp_plus,dp_minus,rollback,lam,p,plateau"]
    md = ["| exp | dose | tokens (B) | L_val | ΔL_t | gap | acc. | viol | struct | Δp+ | Δp− | rollback | p | plateau |",
          "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for e, (_, _, evals, _) in data.items():
        for r in evals:
            if "dose_checkpoint" not in r:
                continue
            lines.append(",".join(str(x) for x in [e, r["dose_checkpoint"], round(r["tokens_seen"] / 1e9, 4), r["L_train"], r["L_val"],
                                                  r["delta_L_t"], r["generalization_gap"], r["acceptability_rate"], r["violation_count"],
                                                  r["structuring_trigger_count"], r["delta_p_plus"], r["delta_p_minus"],
                                                  r["rollback_count"], r["lam"], r["p"], r["plateau"]]))
            md.append(f"| {e} | {r['dose_checkpoint']:.2f} D* | {r['tokens_seen']/1e9:.3f} | {r['L_val']:.4f} | "
                      f"{(r['delta_L_t'] or 0):.4f} | {r['generalization_gap']:.4f} | {r['acceptability_rate']:.3f} | "
                      f"{r['violation_count']} | {r['structuring_trigger_count']} | {r['delta_p_plus']} | {r['delta_p_minus']} | "
                      f"{r['rollback_count']} | {r['p']} | {r['plateau']} |")
    (OUT / "dose_table.csv").write_text("\n".join(lines) + "\n")
    (OUT / "dose_table.md").write_text("\n".join(md) + "\n")

    # module stats
    mt = ["| exp | struct triggers | repair attempts | repair success | isolate fail | Δp+ | Δp− | rollback | NT rounds | NT Fail | final p | final λ |",
          "|---|---|---|---|---|---|---|---|---|---|---|---|"]
    fe = ["| track | experiment | can run | tokens/s (median) | peak alloc GiB | device used GiB | wall-clock h | stop |",
          "|---|---|---|---|---|---|---|---|"]
    for e, (_, _, evals, end) in data.items():
        last = evals[-1] if evals else {}
        g = lambda k: last.get(k, "")  # noqa: E731
        mt.append(f"| {e} | {g('structuring_trigger_count')} | {g('repair_attempts')} | {g('repair_success')} | {g('isolate_fail')} | "
                  f"{g('delta_p_plus')} | {g('delta_p_minus')} | {g('rollback_count')} | {g('nt_rounds')} | {g('nt_fail_total')} | "
                  f"{g('p')} | {last.get('lam', 0):.2e} |")
        tps = sorted(r["tokens_per_sec"] for r in evals) or [0]
        fe.append(f"| ASUS GX10 | {e} | {'yes' if evals else 'no'} | {tps[len(tps)//2]:.0f} | "
                  f"{max((r['peak_alloc_gib'] for r in evals), default=0)} | {max((r['device_used_gib'] for r in evals), default=0)} | "
                  f"{g('wall_clock_seconds') and g('wall_clock_seconds')/3600:.1f} | {(end or {}).get('stop_reason') or ('running' if not end else 'completed')} |")
    fe.append("| AI-Stack (AMD MI300/ROCm/DeepSpeed) | — | not run on this machine | — | — | — | — | — |")
    (OUT / "module_table.md").write_text("\n".join(mt) + "\n")
    (OUT / "feasibility.md").write_text("\n".join(fe) + "\n")
    print("\n".join(md)); print(); print("\n".join(mt)); print(); print("\n".join(fe))
    print(f"\n→ {OUT}")


if __name__ == "__main__":
    main(sys.argv[1:] or ["E1-full", "E2-full", "E3-full"])
