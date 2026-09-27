"""Structure-adjustable Gated FFN for the last Gemma-4-E4B decoder layer.

Proposal §3.1–3.3 (Eq. 7–22) mapped onto Gemma4TextMLP:
    z = F_L(x; w) = W_down · ( act(W_gate x + b_gate) ⊙ (W_up x + b_up) )

The HF module is bias-free. Eq. 18 needs a hidden bias w_{p,0}, so this module
adds b_gate / b_up (zero for the original p channels → function identical to
the base model at init). Node-Adding: p → p+1; Node-Pruning: p → p−k.

Optimizer (AdamW) state is remapped along the channel axis on every structural
change, so momentum of surviving channels is preserved (not reset).
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

# param name → channel axis
_CH_AXIS = {"gate_w": 0, "up_w": 0, "gate_b": 0, "up_b": 0, "down_w": 1}


class StructGatedFFN(nn.Module):
    def __init__(self, mlp: nn.Module):
        super().__init__()
        gw = mlp.gate_proj.weight.data
        self.act_fn = mlp.act_fn
        self.gate_w = nn.Parameter(gw.clone())
        self.up_w = nn.Parameter(mlp.up_proj.weight.data.clone())
        self.down_w = nn.Parameter(mlp.down_proj.weight.data.clone())
        self.gate_b = nn.Parameter(torch.zeros(gw.shape[0], dtype=gw.dtype, device=gw.device))
        self.up_b = nn.Parameter(torch.zeros(gw.shape[0], dtype=gw.dtype, device=gw.device))
        self.p0 = gw.shape[0]

    # ── forward ────────────────────────────────────────────────────────────
    @property
    def p(self) -> int:
        return self.gate_w.shape[0]

    def activations(self, x: torch.Tensor) -> torch.Tensor:
        return self.act_fn(F.linear(x, self.gate_w, self.gate_b)) * F.linear(x, self.up_w, self.up_b)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.linear(self.activations(x), self.down_w)

    def sq_norm(self) -> torch.Tensor:
        """‖w‖² of Eq. 21 (all trainable tensors, fp32)."""
        return sum(p.float().pow(2).sum() for p in self.parameters())

    # ── snapshot / rollback (Network-Tuning, Eq. 21 text) ──────────────────
    def snapshot(self) -> dict:
        return {n: p.detach().clone() for n, p in self.named_parameters()}

    def restore(self, snap: dict, optimizer: torch.optim.Optimizer | None = None,
                opt_snap: dict | None = None) -> None:
        old = dict(self.named_parameters())
        for n, t in snap.items():
            setattr(self, n, nn.Parameter(t.clone()))
        if optimizer is not None:
            self._swap_opt_params(optimizer, old, opt_snap)

    def opt_snapshot(self, optimizer) -> dict:
        out = {}
        for n, p in self.named_parameters():
            st = optimizer.state.get(p)
            if st:
                out[n] = {k: (v.clone() if torch.is_tensor(v) else v) for k, v in st.items()}
        return out

    # ── structural ops ─────────────────────────────────────────────────────
    @torch.no_grad()
    def add_channel(self, gate_row, gate_bias, up_row, up_bias, down_col, optimizer=None) -> int:
        """Node-Adding (Eq. 17–20): append one intermediate channel. Returns its index."""
        kw = dict(dtype=self.gate_w.dtype, device=self.gate_w.device)
        new = {
            "gate_w": torch.cat([self.gate_w, gate_row.to(**kw).view(1, -1)], 0),
            "up_w": torch.cat([self.up_w, up_row.to(**kw).view(1, -1)], 0),
            "gate_b": torch.cat([self.gate_b, gate_bias.to(**kw).view(1)], 0),
            "up_b": torch.cat([self.up_b, up_bias.to(**kw).view(1)], 0),
            "down_w": torch.cat([self.down_w, down_col.to(**kw).view(-1, 1)], 1),
        }
        keep = torch.arange(self.p, device=self.gate_w.device)
        self._replace(new, optimizer, keep=keep, n_new=1)
        return self.p - 1

    @torch.no_grad()
    def prune_channels(self, idx: torch.Tensor, optimizer=None) -> None:
        """Node-Pruning (Algo 5): remove channels `idx` (p → p − |idx|)."""
        mask = torch.ones(self.p, dtype=torch.bool, device=self.gate_w.device)
        mask[idx] = False
        keep = mask.nonzero().squeeze(1)
        new = {n: (p.index_select(_CH_AXIS[n], keep)) for n, p in self.named_parameters()}
        self._replace(new, optimizer, keep=keep, n_new=0)

    @torch.no_grad()
    def channel_importance(self, x_list: list[torch.Tensor]) -> torch.Tensor:
        """Saliency s_h = ‖W_down[:,h]‖ · mean_t |a_h(x_t)| over the reference set."""
        acc = torch.zeros(self.p, device=self.gate_w.device, dtype=torch.float32)
        n = 0
        for x in x_list:
            a = self.activations(x.to(self.gate_w.device)).float().abs()
            acc += a.reshape(-1, a.shape[-1]).sum(0)
            n += a.shape[0] * a.shape[1] if a.dim() == 3 else a.shape[0]
        return (acc / max(n, 1)) * self.down_w.float().norm(dim=0)

    # ── internals ──────────────────────────────────────────────────────────
    def _replace(self, new: dict, optimizer, keep: torch.Tensor, n_new: int) -> None:
        old = dict(self.named_parameters())
        for n, t in new.items():
            setattr(self, n, nn.Parameter(t.contiguous()))
        if optimizer is None:
            return
        remapped = {}
        for n, p_old in old.items():
            st = optimizer.state.get(p_old)
            if not st:
                continue
            ax = _CH_AXIS[n]
            s2 = {}
            for k, v in st.items():
                if torch.is_tensor(v) and v.dim() > 0 and v.shape == p_old.shape:
                    v = v.index_select(ax, keep)
                    if n_new:
                        pad_shape = list(v.shape)
                        pad_shape[ax] = n_new
                        v = torch.cat([v, v.new_zeros(pad_shape)], ax)
                s2[k] = v
            remapped[n] = s2
        self._swap_opt_params(optimizer, old, remapped)

    def _swap_opt_params(self, optimizer, old: dict, new_state: dict | None) -> None:
        cur = dict(self.named_parameters())
        old_ids = {id(p): n for n, p in old.items()}
        for g in optimizer.param_groups:
            g["params"] = [cur[old_ids[id(p)]] if id(p) in old_ids else p for p in g["params"]]
        for p in old.values():
            optimizer.state.pop(p, None)
        if new_state:
            for n, st in new_state.items():
                optimizer.state[cur[n]] = st
