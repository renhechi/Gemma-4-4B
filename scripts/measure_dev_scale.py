#!/usr/bin/env python3
"""ε scale check: relative deviation of (a) Gated FFN output z (proposal Eq. 8, current E4B impl)
vs (b) whole decoder-layer output h (residual stream; what the 12B E1-full hooked), base vs a checkpoint.
Usage: python3 scripts/measure_dev_scale.py runs/E3-full/latest.pt [n_seqs] [seq_len]"""
import sys
from pathlib import Path

import torch
import torch.nn as nn
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent))
from train_lgdsrfpa import ROOT, DFin, load_model, to_ids  # noqa: E402

ck = sys.argv[1]
n = int(sys.argv[2]) if len(sys.argv) > 2 else 8
T = int(sys.argv[3]) if len(sys.argv) > 3 else 1024
cfg = yaml.safe_load((ROOT / "configs/e4b.yaml").read_text())
dev = torch.device("cuda")
model, smlp, base_mlp, tidx = load_model(cfg, dev)
layer = model.model.language_model.layers[tidx]
data = DFin(Path(cfg["data"]["dfin_dir"]), 4096)
seqs = [data.val_seq(cfg["data"]["n_val_seqs"] + j)[:T] for j in range(n)]


def run():
    zs, hs = [], []
    h1 = layer.mlp.register_forward_hook(lambda m, i, o: zs.append(o.float().mean(1)))
    h2 = layer.register_forward_hook(lambda m, i, o: hs.append((o[0] if isinstance(o, tuple) else o).float().mean(1)))
    with torch.no_grad():
        for s in seqs:
            model.model.language_model(input_ids=to_ids([s], dev))
    h1.remove(); h2.remove()
    return torch.cat(zs), torch.cat(hs)


z0, h0 = run()                                   # base (StructGatedFFN == base MLP at init)
s = torch.load(ck, map_location=dev, weights_only=False)
for k, t in s["mlp"].items():
    setattr(smlp, k, nn.Parameter(t))
z1, h1 = run()
rz = ((z1 - z0).norm(dim=1) / z0.norm(dim=1)).mean().item()
rh = ((h1 - h0).norm(dim=1) / h0.norm(dim=1)).mean().item()
print(f"ckpt step={s['step']}  n={n} T={T}")
print(f"(a) FFN output z      : rel.dev = {rz:.4f}   ‖z‖ ≈ {z0.norm(dim=1).mean():.1f}")
print(f"(b) layer output h    : rel.dev = {rh:.4f}   ‖h‖ ≈ {h0.norm(dim=1).mean():.1f}")
print(f"ratio (a)/(b)         : {rz / max(rh, 1e-12):.1f}×")
