#!/usr/bin/env python3
"""LGDSRFPA further-pretraining on Gemma-4-E4B-it (last-layer Gated FFN).

One script, three experiment modes (ED §4.6), identical data order / seed / base:
    --mode E1   Weight-Tuning only (learning goal logged, never acted on)
    --mode E2   Weight-Tuning + Structuring (Selecting → Isolating → Node-Adding)
    --mode E3   Full LGDSRFPA (+ Network-Tuning: λ‖w‖², Node-Pruning, rollback)
E4 (corpus-dose) = the dose checkpoints written by every run (0.25/0.5/0.75/1.0 D*).
E5 (feasibility)  = tokens/sec, peak memory, wall-clock rows in the same metrics file.

Key implementation fact: every layer before the trainable MLP is frozen, so the MLP
input x_c of a sequence never changes during training. Hence z*_c = F_L(x_c; w_0)
is computed exactly with a frozen copy of the base MLP, and the learning goal /
Structuring / pruning trials on the reference set R need only MLP forwards on
cached x (no full-model forward).

Usage:
    python3 scripts/train_lgdsrfpa.py --mode E3 --experiment E3-full
    python3 scripts/train_lgdsrfpa.py --mode E3 --experiment E3-smoke --max-steps 30 \
        --control-interval 5 --eval-interval 10
Resumes automatically from runs/<experiment>/latest.pt.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import yaml

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
from lgdsrfpa4.struct_ffn import StructGatedFFN  # noqa: E402


# ── data ───────────────────────────────────────────────────────────────────────

class DFin:
    """tokens.bin layout: [val docs | train docs(shuffled)]. Train is consumed sequentially,
    so tokens_seen == cumulative corpus dose."""

    def __init__(self, d: Path, seq_len: int):
        self.meta = json.loads((d / "meta.json").read_text())
        self.tok = np.memmap(d / "tokens.bin", dtype=np.uint32, mode="r")
        self.T = seq_len
        self.val_tok = self.meta["val_tokens"]
        self.n_train_seqs = (len(self.tok) - self.val_tok) // seq_len
        self.n_val_seqs = self.val_tok // seq_len

    def train_seq(self, i: int) -> np.ndarray:
        i %= self.n_train_seqs                      # wraps only if D* > unique tokens (logged)
        s = self.val_tok + i * self.T
        return self.tok[s:s + self.T]

    def val_seq(self, i: int) -> np.ndarray:
        return self.tok[i * self.T:(i + 1) * self.T]


def to_ids(arrs, device) -> torch.Tensor:
    return torch.from_numpy(np.stack([a.astype(np.int64) for a in arrs])).to(device)


# ── model ──────────────────────────────────────────────────────────────────────

def load_model(cfg, device):
    from transformers import Gemma4ForConditionalGeneration
    m = Gemma4ForConditionalGeneration.from_pretrained(
        cfg["model"]["local_path"], dtype=torch.bfloat16, attn_implementation="sdpa")
    for attr in ("vision_tower", "audio_tower", "embed_vision", "embed_audio"):
        if hasattr(m.model, attr):
            setattr(m.model, attr, None)
    m.to(device)
    for p in m.parameters():
        p.requires_grad_(False)
    layers = m.model.language_model.layers
    tidx = cfg["model"]["trainable_layer_index"] % len(layers)
    layer = layers[tidx]
    base_mlp = layer.mlp                                 # frozen reference F_L(·; w_0)
    smlp = StructGatedFFN(base_mlp)
    layer.mlp = smlp
    m.config.use_cache = False
    return m, smlp, base_mlp, tidx


def clm_loss(model, ids: torch.Tensor, chunk: int = 1024) -> torch.Tensor:
    """Next-token CE with Gemma final-logit softcapping, computed in checkpointed
    token chunks so the (B·T × 262k) logits tensor is never materialised at once."""
    from torch.utils.checkpoint import checkpoint
    h = model.model.language_model(input_ids=ids).last_hidden_state[:, :-1]
    y = ids[:, 1:]
    cap = model.config.get_text_config().final_logit_softcapping
    W = model.lm_head.weight

    def part(hc, yc):
        lg = (hc @ W.T).float()
        if cap:
            lg = torch.tanh(lg / cap) * cap
        return nn.functional.cross_entropy(lg.reshape(-1, lg.shape[-1]), yc.reshape(-1), reduction="sum")

    tot = h.new_zeros((), dtype=torch.float32)
    for s in range(0, h.shape[1], chunk):
        hc, yc = h[:, s:s + chunk], y[:, s:s + chunk]
        tot = tot + (checkpoint(part, hc, yc, use_reentrant=False) if h.requires_grad else part(hc, yc))
    return tot / y.numel()


class XCapture:
    """Forward pre-hook: grab the MLP input x (B,T,d) of the last micro-batch."""

    def __init__(self, mod: nn.Module):
        self.x = None
        self.on = False
        self.h = mod.register_forward_pre_hook(self._hook)

    def _hook(self, mod, args):
        if self.on:
            self.x = args[0].detach()


# ── LGDSRFPA controller ────────────────────────────────────────────────────────

class Controller:
    def __init__(self, cfg, mode, smlp: StructGatedFFN, base_mlp, x_ref: list[torch.Tensor],
                 log):
        self.cfg, self.mode, self.smlp, self.log = cfg, mode, smlp, log
        self.eps = cfg["learning_goal"]["epsilon"]
        st, nt = cfg["structuring"], cfg["network_tuning"]
        self.T_iso, self.max_adds, self.res_tgt = st["isolating_max_tries"], st["max_adds_per_round"], st["residual_target"]
        self.F_max, self.m_prune, self.max_acc = nt["F_max"], nt["prune_block"], nt["max_accepts_per_round"]
        self.lam = nt["lambda_init"] if mode == "E3" else 0.0
        self.lam_min, self.lam_max = nt["lambda_min"], nt["lambda_max"]
        self.eta, self.gam, self.val_tol = nt["eta_up"], nt["gamma_down"], nt["val_tolerance"]
        self.x_ref = x_ref                                               # list of (T,d) bf16 on GPU
        with torch.no_grad():
            self.xbar = torch.stack([x.float().mean(0) for x in x_ref])  # (N,d)
            self.z_star = torch.stack([base_mlp(x).float().mean(0) for x in x_ref])
        self.z_star_norm = self.z_star.norm(dim=1).clamp_min(1e-8)
        self.c = dict(structuring_trigger_count=0, delta_p_plus=0, delta_p_minus=0,
                      rollback_count=0, repair_attempts=0, repair_success=0,
                      isolate_fail=0, nt_rounds=0, nt_fail_total=0, struct_fail_streak=0)
        self.last_round = {}

    # learning goal (Eq. 10, 13)
    @torch.no_grad()
    def z_ref(self) -> torch.Tensor:
        return torch.stack([self.smlp(x).float().mean(0) for x in self.x_ref])

    @torch.no_grad()
    def deviations(self) -> torch.Tensor:
        return (self.z_ref() - self.z_star).norm(dim=1) / self.z_star_norm

    @torch.no_grad()
    def batch_deviation(self, x: torch.Tensor, base_mlp) -> torch.Tensor:
        z = self.smlp(x).float().mean(1)
        zs = base_mlp(x).float().mean(1)
        return (z - zs).norm(dim=1) / zs.norm(dim=1).clamp_min(1e-8)

    # Algo 1: Selecting — κ = argmin_c (δ[c]² > ε²)
    def select(self, dev: torch.Tensor, skip: set) -> int | None:
        cand = [(d, i) for i, d in enumerate(dev.tolist()) if d > self.eps and i not in skip]
        return min(cand)[1] if cand else None

    # Algo 2: Isolating — γ, ζ s.t. γᵀ(x̄_c − x̄_κ) < −ζ ∀c≠κ  (one-sided form of Eq. 16,
    # required because a single gated channel is monotone along γ)
    @torch.no_grad()
    def isolate(self, k: int):
        others = torch.cat([self.xbar[:k], self.xbar[k + 1:]])
        diff = others - self.xbar[k]
        base = self.xbar[k] - others.mean(0)
        best = (None, -math.inf)
        g = torch.Generator(device=self.xbar.device).manual_seed(1000 + k)
        for t in range(self.T_iso):
            if t == 0:
                gam = base
            elif t == 1:
                gam = self.xbar[k] - others[diff.norm(dim=1).argmin()]
            else:
                gam = base + 0.5 * base.norm() * torch.randn(base.shape, generator=g, device=base.device) / math.sqrt(base.numel())
            gam = gam / gam.norm().clamp_min(1e-12)
            margin = -(diff @ gam).max().item()
            if margin > best[1]:
                best = (gam, margin)
        gam, margin = best
        if gam is None or margin <= 0:
            return None
        return gam, margin / 2.0                                         # (γ, ζ)

    # Algo 3: Node-Adding (Eq. 17–20)
    @torch.no_grad()
    def node_add(self, k: int, gam: torch.Tensor, zeta: float, optimizer):
        beta = 4.0 / zeta
        dt = self.smlp.gate_w.dtype
        gate_row = beta * gam                                            # Eq. 17 (scaled)
        gate_b = torch.tensor(beta * (zeta - float(gam @ self.xbar[k])))  # Eq. 18 (one-sided)
        up_row = torch.zeros_like(gam)
        up_b = torch.tensor(1.0)
        a_k = self.smlp.act_fn((self.x_ref[k].float() @ gate_row + gate_b.to(gam.device)).to(dt)).float().mean()
        if a_k.abs() < 1e-6:
            return None
        dev = self.z_star[k] - self.z_ref()[k]                           # z*_κ − z_κ
        # Eq. 19–20: fix the largest-deviation components until residual ≤ res_tgt·ε·‖z*‖
        target = self.res_tgt * self.eps * self.z_star_norm[k]
        order = dev.abs().argsort(descending=True)
        cum = dev.pow(2).sum() - dev[order].pow(2).cumsum(0)
        n_fix = int((cum.sqrt() > target).sum().item()) + 1
        col = torch.zeros_like(dev)
        sel = order[:n_fix]
        col[sel] = dev[sel] / a_k
        return self.smlp.add_channel(gate_row, gate_b, up_row, up_b, col, optimizer), n_fix

    # Algo 4: Structuring round
    def structuring(self, optimizer) -> dict:
        dev = self.deviations()
        v_before = int((dev > self.eps).sum())
        skip, adds, rb = set(), 0, 0
        v = v_before
        while v > 0 and adds + rb < self.max_adds:
            k = self.select(dev, skip)
            if k is None:
                break
            self.c["repair_attempts"] += 1
            iso = self.isolate(k)
            if iso is None:
                self.c["isolate_fail"] += 1
                skip.add(k)
                continue
            snap, osnap = self.smlp.snapshot(), self.smlp.opt_snapshot(optimizer)
            res = self.node_add(k, *iso, optimizer)
            if res is None:
                skip.add(k)
                continue
            dev2 = self.deviations()
            v2 = int((dev2 > self.eps).sum())
            if dev2[k] <= self.eps and v2 < v:
                self.c["delta_p_plus"] += 1
                self.c["repair_success"] += 1
                adds += 1
                dev, v = dev2, v2
            else:
                self.smlp.restore(snap, optimizer, osnap)
                self.c["rollback_count"] += 1
                rb += 1
                skip.add(k)
        ok = v < v_before
        self.c["struct_fail_streak"] = 0 if ok else self.c["struct_fail_streak"] + 1
        return dict(v_before=v_before, v_after=v, adds=adds, struct_rollbacks=rb)

    # Algo 5/6: Network-Tuning (Node-Pruning with snapshot / rollback, Fail ≤ F_max)
    def network_tuning(self, optimizer) -> dict:
        self.c["nt_rounds"] += 1
        fails = acc = 0
        cursor = 0
        imp = self.smlp.channel_importance(self.x_ref).argsort()
        while fails < self.F_max and acc < self.max_acc and cursor < len(imp):
            idx = imp[cursor:cursor + self.m_prune]
            snap, osnap = self.smlp.snapshot(), self.smlp.opt_snapshot(optimizer)
            self.smlp.prune_channels(idx, optimizer)
            if int((self.deviations() > self.eps).sum()) == 0:           # Eq. 22
                self.c["delta_p_minus"] += len(idx)
                acc += 1
                fails = 0
                cursor = 0
                imp = self.smlp.channel_importance(self.x_ref).argsort()
            else:
                self.smlp.restore(snap, optimizer, osnap)
                self.c["rollback_count"] += 1
                self.c["nt_fail_total"] += 1
                fails += 1
                cursor += self.m_prune
        return dict(prune_accepts=acc, prune_fails=fails)

    # main dispatch after Weight-Tuning (Algo 7 / ED §4.10)
    def control(self, optimizer, val_delta: float | None) -> dict:
        dev = self.deviations()
        v = int((dev > self.eps).sum())
        out = dict(ref_violations=v, dev_mean=float(dev.mean()), dev_max=float(dev.max()))
        if v > 0:
            self.c["structuring_trigger_count"] += 1
            if self.mode in ("E2", "E3"):
                r = self.structuring(optimizer)
                out.update(r)
                if self.mode == "E3":                                    # §4.8 λ rules
                    if r["v_after"] == 0:
                        pass                                             # repaired → hold
                    else:
                        self.lam = max(self.lam_min, self.gam * self.lam)
        else:
            if self.mode == "E3":
                if val_delta is None or val_delta <= self.val_tol:
                    self.lam = min(self.lam_max, self.eta * self.lam)
                out.update(self.network_tuning(optimizer))
        out["p"] = self.smlp.p
        self.last_round = out
        return out


# ── main loop ──────────────────────────────────────────────────────────────────

def lr_at(step, cfg, total):
    wt = cfg["weight_tuning"]
    if step < wt["warmup_steps"]:
        return wt["lr"] * max(1, step) / wt["warmup_steps"]
    prog = min(1.0, (step - wt["warmup_steps"]) / max(1, total - wt["warmup_steps"]))
    return wt["min_lr"] + 0.5 * (wt["lr"] - wt["min_lr"]) * (1 + math.cos(math.pi * prog))


@torch.no_grad()
def eval_val(model, data, n, mb, device):
    model.eval()
    tot = 0.0
    for i in range(0, n, mb):
        ids = to_ids([data.val_seq(j) for j in range(i, min(n, i + mb))], device)
        tot += clm_loss(model, ids).item() * ids.shape[0]
    model.train()
    return tot / n


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=str(ROOT / "configs/e4b.yaml"))
    ap.add_argument("--mode", choices=["E1", "E2", "E3"], required=True)
    ap.add_argument("--experiment", required=True)
    ap.add_argument("--max-steps", type=int, default=0, help="override (smoke); 0 = D* budget")
    ap.add_argument("--control-interval", type=int)
    ap.add_argument("--eval-interval", type=int)
    ap.add_argument("--epsilon", type=float)
    ap.add_argument("--micro-batch", type=int)
    ap.add_argument("--grad-accum", type=int)
    ap.add_argument("--n-val", type=int)
    ap.add_argument("--n-ref", type=int)
    args = ap.parse_args()

    cfg = yaml.safe_load(Path(args.config).read_text())
    ctl, wt = cfg["control"], cfg["weight_tuning"]
    if args.control_interval: ctl["control_interval"] = args.control_interval
    if args.eval_interval: ctl["eval_interval"] = args.eval_interval
    if args.epsilon: cfg["learning_goal"]["epsilon"] = args.epsilon
    if args.micro_batch: wt["micro_batch"] = args.micro_batch
    if args.grad_accum: wt["grad_accum"] = args.grad_accum
    if args.n_val: cfg["data"]["n_val_seqs"] = args.n_val
    if args.n_ref: cfg["data"]["n_ref_seqs"] = args.n_ref

    seed = cfg["run"]["seed"]
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    device = torch.device("cuda")
    out = Path(cfg["run"]["out_root"]) / args.experiment
    out.mkdir(parents=True, exist_ok=True)
    mpath = out / "metrics.jsonl"

    def log(row):
        row = {"experiment": args.experiment, "mode": args.mode, "time": time.time(), **row}
        with open(mpath, "a") as fh:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")

    T, mb, accum = cfg["data"]["seq_len"], wt["micro_batch"], wt["grad_accum"]
    tok_per_step = T * mb * accum
    d_star = cfg["budget"]["d_star_tokens"]
    total_steps = args.max_steps or math.ceil(d_star / tok_per_step)
    dose_base = total_steps if args.max_steps else d_star / tok_per_step     # smoke: fractions of max_steps
    dose_steps = {max(1, round(f * dose_base)): f for f in cfg["budget"]["dose_checkpoints"]}

    data = DFin(Path(cfg["data"]["dfin_dir"]), T)
    n_val, n_ref = cfg["data"]["n_val_seqs"], cfg["data"]["n_ref_seqs"]
    assert data.n_val_seqs >= n_val + n_ref, f"val region too small: {data.n_val_seqs} seqs"
    uniq_train = data.n_train_seqs * T

    print(f"[load] {cfg['model']['local_path']}", flush=True)
    t0 = time.perf_counter()
    model, smlp, base_mlp, tidx = load_model(cfg, device)
    load_s = time.perf_counter() - t0
    cap = XCapture(smlp)

    # reference set R: val sequences [n_val, n_val+n_ref) — cache MLP inputs once
    cap.on = True
    x_ref = []
    with torch.no_grad():
        for j in range(n_ref):
            ids = to_ids([data.val_seq(n_val + j)], device)
            model.model.language_model(input_ids=ids)                     # stop before lm_head
            x_ref.append(cap.x[0].clone())
    cap.on = False
    ctrl = Controller(cfg, args.mode, smlp, base_mlp, x_ref, log)

    optimizer = torch.optim.AdamW(smlp.parameters(), lr=wt["lr"], betas=tuple(wt["betas"]),
                                  weight_decay=0.0, fused=True)
    step = 0
    best_ema = math.inf
    ema = None
    prev_val = None
    val_change = None
    plateau_streak = 0
    wall0 = 0.0
    ck = out / "latest.pt"
    if ck.exists():
        s = torch.load(ck, map_location=device, weights_only=False)
        for n, t in s["mlp"].items():
            setattr(smlp, n, nn.Parameter(t))
        optimizer = torch.optim.AdamW(smlp.parameters(), lr=wt["lr"], betas=tuple(wt["betas"]),
                                      weight_decay=0.0, fused=True)
        optimizer.load_state_dict(s["opt"])
        ctrl.c, ctrl.lam = s["counters"], s["lam"]
        step, prev_val, plateau_streak, best_ema, ema = s["step"], s["prev_val"], s["plateau_streak"], s["best_ema"], s["ema"]
        wall0 = s["wall"]
        torch.set_rng_state(s["rng"])
        print(f"[resume] step={step} p={smlp.p} lam={ctrl.lam:.2e}", flush=True)
    else:
        log(dict(event="run_start", config=cfg, total_steps=total_steps, tok_per_step=tok_per_step,
                 d_star=d_star, unique_train_tokens=int(uniq_train), dfin_meta=data.meta,
                 trainable_layer=tidx, p0=smlp.p0, w_sq_norm=float(smlp.sq_norm().detach()),
                 load_seconds=round(load_s, 1), ref_dev_init=float(ctrl.deviations().max())))

    print(f"[run] {args.experiment} mode={args.mode} steps={total_steps} tok/step={tok_per_step:,} "
          f"D*={d_star/1e9:.3f}B unique_train={uniq_train/1e9:.3f}B p={smlp.p} "
          f"‖w‖²={float(smlp.sq_norm()):.1f}", flush=True)
    if d_star > uniq_train:
        print(f"[warn] D* exceeds unique train tokens → data wraps (epoch>1) after "
              f"{uniq_train/d_star:.2f} D*", flush=True)

    model.train()
    torch.cuda.reset_peak_memory_stats()
    t_int = time.perf_counter()
    tok_int = 0
    wall_start = time.perf_counter() - wall0
    stop_reason = None
    while step < total_steps:
        lr = lr_at(step, cfg, total_steps)
        for g in optimizer.param_groups:
            g["lr"] = lr
        l_sum = 0.0
        for a in range(accum):
            base_i = (step * accum + a) * mb
            ids = to_ids([data.train_seq(base_i + j) for j in range(mb)], device)
            cap.on = a == accum - 1
            loss = clm_loss(model, ids)
            l_sum += loss.item()
            if ctrl.lam > 0:
                loss = loss + ctrl.lam * smlp.sq_norm()                    # Eq. 21 λ‖w‖²
            (loss / accum).backward()
        cap.on = False
        gn = torch.nn.utils.clip_grad_norm_(smlp.parameters(), wt["grad_clip"]).item()
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        step += 1
        tok_int += tok_per_step
        L_train = l_sum / accum
        bdev = ctrl.batch_deviation(cap.x, base_mlp)
        cap.x = None

        # safety (ED §4.7)
        if not math.isfinite(L_train):
            stop_reason = "nan"
        ema = L_train if ema is None else 0.98 * ema + 0.02 * L_train
        best_ema = min(best_ema, ema)
        if step > 50 and ema > cfg["safety"]["divergence_factor"] * best_ema:
            stop_reason = "divergence"

        row = dict(event="step", step=step, tokens_seen=step * tok_per_step, L_train=round(L_train, 5),
                   lr=lr, grad_norm=round(gn, 4), lam=ctrl.lam, p=smlp.p,
                   batch_violations=int((bdev > ctrl.eps).sum()), batch_dev_max=round(float(bdev.max()), 5))

        if step % ctl["control_interval"] == 0 or step in dose_steps:
            r = ctrl.control(optimizer, val_change)
            row.update(ctrl=r, p=smlp.p, lam=ctrl.lam)
            if ctrl.c["struct_fail_streak"] >= cfg["safety"]["struct_fail_streak"]:
                print(f"[warn] {ctrl.c['struct_fail_streak']} consecutive failed structuring rounds", flush=True)

        if step % ctl["eval_interval"] == 0 or step in dose_steps or step == total_steps:
            L_val = eval_val(model, data, n_val, mb, device)
            dL = abs(L_val - prev_val) if prev_val is not None else None
            val_change = (L_val - prev_val) if prev_val is not None else None
            if dL is not None and ctrl.mode == "E3" and L_val - prev_val > ctrl.val_tol:
                ctrl.lam = max(ctrl.lam_min, ctrl.gam * ctrl.lam)          # §4.8: L_val worsened
            plateau_streak = plateau_streak + 1 if (dL is not None and dL < cfg["plateau"]["delta_threshold"]) else 0
            prev_val = L_val
            dev = ctrl.deviations()
            el = time.perf_counter() - t_int
            free, total = torch.cuda.mem_get_info()
            row.update(event="eval", L_val=round(L_val, 5), delta_L_t=dL,
                       generalization_gap=round(L_val - L_train, 5),
                       acceptability_rate=round(1 - float((dev > ctrl.eps).float().mean()), 4),
                       violation_count=int((dev > ctrl.eps).sum()), dev_mean=round(float(dev.mean()), 5),
                       plateau=plateau_streak >= cfg["plateau"]["consecutive_evals"],
                       plateau_streak=plateau_streak,
                       tokens_per_sec=round(tok_int / el, 1),
                       peak_alloc_gib=round(torch.cuda.max_memory_allocated() / 2**30, 2),
                       device_used_gib=round((total - free) / 2**30, 2),
                       wall_clock_seconds=round(time.perf_counter() - wall_start, 1),
                       epoch=round(step * tok_per_step / uniq_train, 4), **ctrl.c)
            t_int, tok_int = time.perf_counter(), 0
            print(f"[{args.experiment}] step {step}/{total_steps} L_tr={L_train:.4f} L_val={L_val:.4f} "
                  f"viol={row['violation_count']}/{n_ref} p={smlp.p} Δp+={ctrl.c['delta_p_plus']} "
                  f"Δp-={ctrl.c['delta_p_minus']} rb={ctrl.c['rollback_count']} lam={ctrl.lam:.2e} "
                  f"tok/s={row['tokens_per_sec']:.0f} mem={row['peak_alloc_gib']}GiB", flush=True)

        if step in dose_steps:
            f = dose_steps[step]
            torch.save({"mlp": {n: p.detach().cpu() for n, p in smlp.named_parameters()},
                        "step": step, "tokens_seen": step * tok_per_step, "dose": f,
                        "counters": dict(ctrl.c), "lam": ctrl.lam, "p": smlp.p,
                        "layer": tidx, "base": cfg["model"]["local_path"]},
                       out / f"dose_{f:.2f}.pt")
            row["dose_checkpoint"] = f
            print(f"[dose] {f:.2f} D* checkpoint saved", flush=True)
        log(row)

        if step % ctl["save_every"] == 0 or step == total_steps or stop_reason:
            tmp = out / "latest.pt.tmp"
            torch.save({"mlp": {n: p.detach() for n, p in smlp.named_parameters()},
                        "opt": optimizer.state_dict(), "counters": ctrl.c, "lam": ctrl.lam,
                        "step": step, "prev_val": prev_val, "plateau_streak": plateau_streak,
                        "best_ema": best_ema, "ema": ema, "rng": torch.get_rng_state(),
                        "wall": time.perf_counter() - wall_start}, tmp)
            os.replace(tmp, ck)
        if stop_reason:
            print(f"[safety] stop: {stop_reason}", flush=True)
            break

    log(dict(event="run_end", step=step, stop_reason=stop_reason, p=smlp.p, lam=ctrl.lam, **ctrl.c,
             wall_clock_seconds=round(time.perf_counter() - wall_start, 1),
             peak_alloc_gib=round(torch.cuda.max_memory_allocated() / 2**30, 2)))
    print(f"[done] {args.experiment} step={step} p={smlp.p} {ctrl.c}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
