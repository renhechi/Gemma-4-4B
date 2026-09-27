#!/usr/bin/env python3
"""Train-step throughput at equal tokens per micro-batch: (mb, T) = (4, 4096) vs (16, 1024)."""
import sys, time
from pathlib import Path
import torch, yaml
sys.path.insert(0, str(Path(__file__).resolve().parent))
from train_lgdsrfpa import ROOT, clm_loss, load_model
cfg = yaml.safe_load((ROOT / "configs/e4b.yaml").read_text())
model, smlp, _, _ = load_model(cfg, torch.device("cuda"))
for mb, T in ((4, 4096), (16, 1024)):
    ids = torch.randint(10, 200000, (mb, T), device="cuda")
    clm_loss(model, ids).backward(); torch.cuda.synchronize()
    t = time.perf_counter()
    for _ in range(2):
        clm_loss(model, ids).backward()
    torch.cuda.synchronize()
    print(f"mb={mb} T={T}: {2 * mb * T / (time.perf_counter() - t):.0f} tok/s  peak={torch.cuda.max_memory_allocated()/2**30:.1f}GiB", flush=True)
