#!/usr/bin/env python3
"""Sanity checks before any real run: loss equivalence, structural-op correctness, throughput."""
import sys
import time
from pathlib import Path

import torch
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent))
from train_lgdsrfpa import ROOT, clm_loss, load_model  # noqa: E402

cfg = yaml.safe_load((ROOT / "configs/e4b.yaml").read_text())
dev = torch.device("cuda")
model, smlp, base_mlp, tidx = load_model(cfg, dev)
print(f"layer={tidx} p={smlp.p} ‖w‖²={float(smlp.sq_norm()):.1f}")
from transformers import AutoTokenizer
tok = AutoTokenizer.from_pretrained(cfg["model"]["local_path"])
txt = ("臺灣證券交易所公布上市公司年報,營業收入較去年同期成長,毛利率亦提升。" * 400)
ids = torch.tensor([tok(txt, add_special_tokens=False)["input_ids"][:4096]], device=dev)

# 1) loss equivalence (HF loss vs chunked loss)
with torch.no_grad():
    l_hf = model(input_ids=ids, labels=ids).loss.float().item()
    l_me = clm_loss(model, ids).item()
print(f"[1] HF loss={l_hf:.5f} chunked={l_me:.5f} diff={abs(l_hf - l_me):.2e}")

# 2) StructGatedFFN ≡ base MLP at init
x = torch.randn(1, 64, smlp.gate_w.shape[1], device=dev, dtype=torch.bfloat16)
with torch.no_grad():
    print(f"[2] max|struct−base| = {(smlp(x) - base_mlp(x)).abs().max().item():.2e}")

# 3) add / prune / restore with optimizer state remap
opt = torch.optim.AdamW(smlp.parameters(), lr=1e-5, fused=True)
loss = clm_loss(model, ids[:, :1024]); loss.backward(); opt.step(); opt.zero_grad()
snap, osnap = smlp.snapshot(), smlp.opt_snapshot(opt)
with torch.no_grad():
    y0 = smlp(x)
d = smlp.gate_w.shape[1]
k = smlp.add_channel(torch.zeros(d), torch.tensor(0.), torch.zeros(d), torch.tensor(1.), torch.zeros(d), opt)
with torch.no_grad():
    print(f"[3a] zero-add: p={smlp.p} new idx={k} Δout={(smlp(x) - y0).abs().max().item():.2e}")
smlp.prune_channels(torch.tensor([0, 1, 2], device=dev), opt)
print(f"[3b] prune 3: p={smlp.p}  opt params={len(opt.param_groups[0]['params'])} "
      f"state shapes ok={all(opt.state[p]['exp_avg'].shape == p.shape for p in smlp.parameters())}")
loss = clm_loss(model, ids[:, :1024]); loss.backward(); opt.step(); opt.zero_grad()
smlp.restore(snap, opt, osnap)
with torch.no_grad():
    print(f"[3c] restore: p={smlp.p} Δout={(smlp(x) - y0).abs().max().item():.2e}")

# 4) throughput / memory, micro-batch sweep
opt = torch.optim.AdamW(smlp.parameters(), lr=1e-5, fused=True)
for mb in (2, 4, 8):
    torch.cuda.reset_peak_memory_stats()
    b = ids.repeat(mb, 1)
    try:
        torch.cuda.synchronize(); t = time.perf_counter()
        for _ in range(3):
            clm_loss(model, b).backward()
        opt.step(); opt.zero_grad()
        torch.cuda.synchronize()
        el = time.perf_counter() - t
        print(f"[4] mb={mb}: {3 * mb * 4096 / el:.0f} tok/s  peak={torch.cuda.max_memory_allocated() / 2**30:.1f} GiB")
    except torch.OutOfMemoryError:
        print(f"[4] mb={mb}: OOM"); break
