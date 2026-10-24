"""Citadel-Rx: continuum-memory test-time-learning receiver (proposal §2.5).

Memory levels. The adaptable parameter set A of the DeepSIC backbone is kept
in K copies θ^(1..K). θ^(1) (the *fast* level) is what detects; it is updated on
every block from the pilots. Level k+1 is updated only every C_{k+1} blocks,
from the *gated* average of level-k increments (Eq. 2.6), and the fast level is
re-anchored to a projection of the slow levels at every consolidation (CMS
Eq. 72), which is the recovery mechanism.

Fast rule (Eq. 2.4, Delta Gradient Descent). Each weight matrix W of the fast
level is updated by  W ← W (I − η δ φ φᵀ) − η ĝ  where φ is the normalised mean
input of that layer over the pilot batch and ĝ is the (Adam-preconditioned or
plain) gradient; the delta term erases the component of W along the current
input direction instead of averaging it in. δ = 0 recovers the reference
online DeepSIC when the rule is 'adam'.

Surprise (Eq. 2.5). S_b = ‖∇_A L_pil(θ^(1); P_b)‖² before adaptation; the
learning rate is η_b = η_max · 2 r_b / (1 + r_b) with r_b = S_b / S̄_b, S̄ an EMA
(a deterministic self-generated rate), and the gate thresholds tighten with r_b.

Gates (G1/G2/G3) are in ``citadel_rx.gates``. The verdict γ_b weights block b's
fast increment in the consolidation sum, so a poisoned block can move only the
fast level and only until the next re-anchoring.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional

import numpy as np
import torch

from ..config import Config
from ..gates import GateController
from .base import BlockOutput
from .deepsic import ParamDict, clone_params, param_dist, param_norm
from .lmmse import ChannelTracker, estimate_noise_var, hard_from_probs, lmmse_soft_symbols, ls_channel_estimate
from .online import DeepSICReceiver
from ..gates.physics import g1_residual


class CitadelRx(DeepSICReceiver):
    name = "citadel"

    def __init__(self, cfg: Config, device: torch.device, single_memory: bool = False):
        super().__init__(cfg, device)
        cc = cfg.citadel
        self.single = single_memory
        self.K = 1 if single_memory else cc.levels
        self.C = [1] if single_memory else list(cc.periods[: self.K])
        if len(self.C) != self.K or self.C[0] != 1:
            raise ValueError("periods must have `levels` entries and start with 1")
        self.A = self.core.adaptable_keys(cc.adaptable)
        self.rule = cc.fast_rule
        self.fast_steps = cc.fast_steps
        self.eta_max = cc.eta_max
        self.use_surprise = cc.surprise_lr
        self.s_ema_coef = cc.surprise_ema
        self.delta = cc.dgd_delta
        self.level_lr = [1.0] + list(cc.level_lr[: self.K - 1]) if self.K > 1 else [1.0]
        self.beta = cc.reanchor_beta
        w = np.array(cc.level_weights[: self.K - 1], dtype=float) if self.K > 1 else np.array([])
        self.level_w = (w / w.sum()) if w.size else w
        gates = [] if single_memory else cc.gates
        self.gate = GateController(gates, cc.gate_mode, cc.g1_tau, cc.g2_tau, cc.g3_quantile,
                                   cc.adaptive_thresholds, dof=self.n_ant * self.n_user)
        self.noise_var = 10 ** (-0.1 * cfg.channel.snr_db)
        self.tracker = ChannelTracker(rho=1.0)
        self.levels: List[ParamDict] = [clone_params(self.params) for _ in range(self.K)]
        self.level_init: List[ParamDict] = [clone_params(self.params) for _ in range(self.K)]
        # increment buffers: buf[k] holds (gamma, delta) of level-k increments since the last
        # consolidation into level k+1
        self.buf: List[List] = [[] for _ in range(self.K)]
        self.S_ema: Optional[float] = None
        self.name = "titans_single" if single_memory else "citadel"

    # ---- lifecycle ----------------------------------------------------------
    def pretrain(self, blocks) -> None:
        super().pretrain(blocks)
        self.levels = [clone_params(self.params) for _ in range(self.K)]
        self.level_init = [clone_params(self.params) for _ in range(self.K)]
        self.tracker = ChannelTracker(rho=1.0)

    def set_channel_rho(self, rho: float) -> None:
        self.tracker.rho = rho

    # ---- surprise -------------------------------------------------------------
    def _surprise(self, yf: torch.Tensor, idx: torch.Tensor) -> float:
        p = {k: (v.detach().clone().requires_grad_(True) if k in self.A else v) for k, v in self.params.items()}
        loss = self.core.detect_loss(p, yf, idx)
        grads = torch.autograd.grad(loss, [p[k] for k in self.A])
        return float(sum((g ** 2).sum() for g in grads).item())

    # ---- fast rule --------------------------------------------------------------
    def _fast_stage(self, p: ParamDict, q: int, X: torch.Tensor, idx: torch.Tensor, eta: float) -> ParamDict:
        keys = [k for k in self.core.keys_for(q) if k in self.A]
        if not keys:
            return p
        p = dict(p)
        leaves = {k: p[k].detach().clone().requires_grad_(True) for k in keys}
        use_adam = self.rule in ("adam", "dgd")
        opt = torch.optim.Adam(list(leaves.values()), lr=eta) if use_adam else None
        use_delta = self.rule in ("dgd", "dgd_plain") and self.delta > 0
        pp = dict(p)
        # normalised mean inputs of each layer over the pilot batch (per user)
        with torch.no_grad():
            phi1 = X.mean(dim=1)                                           # (U, F_in)
            phi1 = phi1 / phi1.norm(dim=-1, keepdim=True).clamp_min(1e-9)
        for _ in range(self.fast_steps):
            pp.update(leaves)
            logits, h = self.core.module_forward(pp, q, X, return_hidden=True)
            tgt = idx.transpose(0, 1)
            ce = torch.nn.functional.cross_entropy(logits.reshape(-1, self.core.order), tgt.reshape(-1),
                                                   reduction="none").reshape(self.core.U, -1).mean(dim=1).sum()
            grads = torch.autograd.grad(ce, list(leaves.values()))
            with torch.no_grad():
                if use_delta:
                    phi2 = h.mean(dim=1)
                    phi2 = phi2 / phi2.norm(dim=-1, keepdim=True).clamp_min(1e-9)
                    for k, t in leaves.items():
                        if k.endswith("fc1.w"):
                            t -= eta * self.delta * torch.einsum("uoi,ui,uj->uoj", t, phi1, phi1)
                        elif k.endswith("fc2.w"):
                            t -= eta * self.delta * torch.einsum("uoi,ui,uj->uoj", t, phi2, phi2)
            if use_adam:
                opt.zero_grad(set_to_none=True)
                for t, g in zip(leaves.values(), grads):
                    t.grad = g
                opt.step()
            else:
                with torch.no_grad():
                    for t, g in zip(leaves.values(), grads):
                        t -= eta * g
        for k, t in leaves.items():
            p[k] = t.detach()
        return p

    def _fast_update(self, yf: torch.Tensor, idx: torch.Tensor, eta: float) -> ParamDict:
        return self.core.sequential_adapt(self.params, yf, idx,
                                          lambda p, q, X, t: self._fast_stage(p, q, X, t, eta))

    # ---- consolidation ------------------------------------------------------------
    def _consolidate(self, b: int) -> Dict[str, Any]:
        events = []
        for k in range(self.K - 1):
            if (b + 1) % self.C[k + 1] != 0 or not self.buf[k]:
                continue
            gs = torch.tensor([g for g, _ in self.buf[k]], dtype=torch.float32)
            denom = float(max(1.0, gs.sum().item()))
            delta_bar = {key: sum(g * d[key] for g, d in self.buf[k]) / denom for key in self.A}
            eta_k = self.level_lr[k + 1]
            for key in self.A:
                self.levels[k + 1][key] = self.levels[k + 1][key] + eta_k * delta_bar[key]
            inc = {key: eta_k * delta_bar[key] for key in self.A}
            self.buf[k + 1].append((float(gs.mean().item()), inc))
            self.buf[k] = []
            events.append({"level": k + 2, "block": b, "mean_gamma": float(gs.mean().item()),
                           "inc_norm": float(torch.sqrt(sum((v ** 2).sum() for v in inc.values())).item())})
        if events:
            # re-anchor the fast level to the projection of the slow levels
            proj = {key: sum(float(w) * self.levels[k + 1][key] for k, w in enumerate(self.level_w)) for key in self.A}
            for key in self.A:
                self.params[key] = (1 - self.beta) * self.params[key] + self.beta * proj[key]
            self.levels[0] = clone_params(self.params)
        return {"consolidations": events}

    # ---- block ---------------------------------------------------------------------
    def process_block(self, b, y_p, idx_p, y_d, aux=None) -> BlockOutput:
        aux = aux or {}
        framer = aux.get("framer")
        yf_p, it_p, yf_d = self.to_feat(y_p), self.to_idx(idx_p), self.to_feat(y_d)
        stats: Dict[str, Any] = {}

        # 1. surprise and self-generated learning rate
        S = self._surprise(yf_p, it_p)
        if self.S_ema is None:
            self.S_ema = S
        r = S / max(self.S_ema, 1e-12)
        eta = self.eta_max * (2 * r / (1 + r)) if self.use_surprise else self.eta_max
        self.S_ema = self.s_ema_coef * self.S_ema + (1 - self.s_ema_coef) * S
        stats.update(surprise=S, surprise_ratio=r, eta=eta)

        # 2. fast update
        before = clone_params(self.params)
        self.params = self._fast_update(yf_p, it_p, eta)
        delta = {k: (self.params[k] - before[k]).detach() for k in self.A}
        stats["fast_step_norm"] = float(torch.sqrt(sum((v ** 2).sum() for v in delta.values())).item())
        with torch.no_grad():
            stats["pilot_loss_post"] = float(self.core.detect_loss(self.params, yf_p, it_p).item())

        # 3. detect data with the fast level
        probs = self.detect(yf_d)
        idx_hat = self.hard(probs)

        # 4. gates
        gamma = 1.0
        if self.gate.gates:
            s_p = self.mod.modulate_indices(idx_p)
            H_ls = ls_channel_estimate(y_p, s_p, self.noise_var)
            nv = estimate_noise_var(y_p, s_p, H_ls)
            fused = self.tracker.fuse(H_ls, nv, y_p.shape[0])
            H = fused["H"]
            g1 = pass_frac = None
            if "g1" in self.gate.gates:
                idx_mm = hard_from_probs(lmmse_soft_symbols(y_d, H, nv, self.mod))
                g1 = g1_residual(y_d, idx_hat, idx_mm, H, self.mod)
            if "g2" in self.gate.gates and framer is not None and framer.framer.enabled:
                pass_frac = float(framer.framer.check(framer.decided_bits_to_subblocks(idx_hat)).mean())
            verdict = self.gate.evaluate(g1, pass_frac, fused["innov_norm"] if "g3" in self.gate.gates else None, r)
            self.gate.update_clean(verdict, pass_frac, g1)
            gamma = verdict.gamma
            stats.update({f"gate_{k}": v for k, v in verdict.stats.items()})
            stats.update({f"verdict_{k}": v for k, v in verdict.verdicts.items()})
            stats["innov_norm"] = fused["innov_norm"]
        stats["gamma"] = gamma

        # 5. store the gated increment and consolidate / re-anchor
        if self.K > 1:
            self.buf[0].append((gamma, delta))
            stats.update(self._consolidate(b))
        self.levels[0] = clone_params(self.params)
        for k in range(self.K):
            stats[f"level{k + 1}_disp"] = param_dist(self.levels[k], self.level_init[k], self.A)
        if self.K > 1:
            stats["fast_to_slow"] = param_dist(self.levels[0], self.levels[1], self.A)
        self.block_count += 1
        return BlockOutput(probs_data=probs, idx_hat=idx_hat, stats=stats)

    def level_snapshot(self, k: int) -> Dict[str, np.ndarray]:
        return {key: self.levels[k][key].detach().cpu().numpy() for key in self.A}

    def attack_surface(self) -> Dict[str, Any]:
        d = super().attack_surface()
        d.update({"keys": self.A, "lr": self.eta_max, "gate": self.gate, "noise_var": self.noise_var})
        return d
