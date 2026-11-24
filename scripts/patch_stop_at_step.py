"""Add --stop-at-step N to train_lgdsrfpa.py: at step N run the normal eval, then stop via the
existing early-stop path (saves stop_<frac>Dstar.pt + latest.pt, logs run_end). LR schedule untouched."""
import sys
p = sys.argv[1]
s = open(p).read()
if "--stop-at-step" in s:
    print("already patched"); sys.exit(0)
a = '    ap.add_argument("--plateau-delta", type=float, help="test only: plateau ΔL threshold")\n'
assert a in s
s = s.replace(a, a + '    ap.add_argument("--stop-at-step", type=int, default=0, help="stop at this step via the early-stop path (user decision)")\n')
b = '            stop_reason = "divergence"\n'
assert s.count(b) == 1
s = s.replace(b, b + '        if args.stop_at_step and step >= args.stop_at_step:\n            stop_reason = f"user_stop_at_step_{args.stop_at_step}"\n')
open(p, "w").write(s); print("patched", p)
