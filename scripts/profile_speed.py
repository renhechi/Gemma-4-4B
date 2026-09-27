#!/usr/bin/env python3
"""Where does time go? forward-only (no_grad) vs attention impl vs loss/backward."""
import sys
import time
from pathlib import Path

import torch
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent))
from train_lgdsrfpa import ROOT, clm_loss, load_model  # noqa: E402

impl = sys.argv[1] if len(sys.argv) > 1 else "sdpa"
cfg = yaml.safe_load((ROOT / "configs/e4b.yaml").read_text())
dev = torch.device("cuda")
model, smlp, _, _ = load_model(cfg, dev)
model.config.get_text_config()._attn_implementation = impl
for m in model.modules():
    if hasattr(m, "config") and hasattr(m.config, "_attn_implementation"):
        m.config._attn_implementation = impl
print("attn impl:", impl, "| param devices:", {p.device.type for p in model.parameters()},
      "| dtypes:", {p.dtype for p in model.parameters()})


def t(fn, n=3):
    fn(); torch.cuda.synchronize()
    s = time.perf_counter()
    for _ in range(n):
        fn()
    torch.cuda.synchronize()
    return (time.perf_counter() - s) / n


for T in (1024, 4096):
    ids = torch.randint(10, 200000, (1, T), device=dev)
    with torch.no_grad():
        f = t(lambda: model.model.language_model(input_ids=ids))
        fl = t(lambda: clm_loss(model, ids))
    fb = t(lambda: clm_loss(model, ids).backward(), n=2)
    print(f"T={T}: fwd {T/f:.0f} tok/s | fwd+loss {T/fl:.0f} | fwd+loss+bwd {T/fb:.0f}", flush=True)

with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CUDA, torch.profiler.ProfilerActivity.CPU]) as prof:
    with torch.no_grad():
        model.model.language_model(input_ids=torch.randint(10, 200000, (1, 4096), device=dev))
    torch.cuda.synchronize()
print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=15))
