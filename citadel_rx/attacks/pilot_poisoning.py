"""Pilot-poisoning adversaries (proposal §2.4).

All attacks add a perturbation E to the received pilot block Y_p (and optionally
the data block) under an l2 budget ‖E‖_F ≤ ε per block. The budget is set
either explicitly or from a signal-to-interference ratio
    SIR = 10 log10( ‖Y_p^clean‖_F² / ‖E‖_F² ),
i.e. the attacker's energy relative to the legitimate users' energy over the
same symbols. 0 dB means the attacker spends as much energy as all users
together.

A1 — availability poisoning (Wu et al. 2024).  The attacker unrolls the
victim's adaptation map with a differentiable gradient-descent surrogate,
    θ⁺(E) = Adapt(θ; Y_p + E, S_p),
and maximises the detection loss of θ⁺ on a *clean query set* (the clean pilot
observations, which share the block's channel) by projected gradient ascent.
White-box: θ is the victim's current fast level. Transfer: θ is the attacker's
own surrogate receiver, trained on the pilots it observed over the air.

A2 — targeted drift spoofing. Maximise the loss of one user u* and penalise
the loss of the others, so the receiver "forgets" u* only.

A3 — boiling frog. With a small budget on every block, push θ⁺ toward a bad
basin θ_bad (the receiver adapted to a spoofed channel H + ΔH) while keeping
the clean query loss low so decision-based gates (G2) keep passing. Aims at the
slow levels through many consolidations.

adaptive — gate-aware A1: adds a differentiable penalty on the G1 residual
ratio computed from θ⁺'s soft symbols so that the poisoned fast state still
looks physically consistent.

data_jam — not a poisoning attack: Gaussian noise of the same energy on the
data symbols. The fair "is poisoning worse than jamming at equal power" control.
"""
from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

import numpy as np
import torch

from ..config import AttackConfig, Config
from ..modulation import Modulation
from ..receivers.deepsic import DeepSICCore, ParamDict, clone_params
from ..receivers.lmmse import ls_channel_estimate
from ..receivers.online import OnlineDeepSIC


def feat(y: np.ndarray, device) -> torch.Tensor:
    f = np.concatenate([y.real, y.imag], axis=1) if np.iscomplexobj(y) else y
    return torch.as_tensor(np.ascontiguousarray(f), dtype=torch.float32, device=device)


def unfeat(f: torch.Tensor, is_complex: bool) -> np.ndarray:
    a = f.detach().cpu().numpy()
    if is_complex:
        n = a.shape[1] // 2
        return a[:, :n] + 1j * a[:, n:]
    return a


class Attacker:
    def __init__(self, cfg: Config, device: torch.device, rng: np.random.Generator):
        self.cfg = cfg
        self.ac: AttackConfig = cfg.attack
        self.device = device
        self.rng = rng
        self.mod = Modulation(cfg.frame.modulation)
        self.surrogate: Optional[OnlineDeepSIC] = None
        self.h_a: Optional[np.ndarray] = None
        self.theta_bad: Optional[ParamDict] = None
        self._surrogate_cfg = None

    # ---- schedule and budget ------------------------------------------------
    def active(self, b: int) -> bool:
        if self.ac.type == "none":
            return False
        if not (self.ac.start_block <= b < self.ac.end_block):
            return False
        if self.ac.rho_blk < 1.0:
            return bool(self.rng.random() < self.ac.rho_blk)
        return True

    def budget(self, y_clean: np.ndarray, small: bool = False) -> float:
        if self.ac.eps is not None and not small:
            return float(self.ac.eps)
        sir = self.ac.sir_db
        energy = float(np.sum(np.abs(y_clean) ** 2))
        eps = np.sqrt(energy * 10 ** (-sir / 10))
        if small:
            eps = float(self.ac.a3_eps_small) * np.sqrt(energy)
        return float(eps)

    # ---- transfer surrogate -----------------------------------------------------
    def _ensure_surrogate(self) -> OnlineDeepSIC:
        if self.surrogate is None:
            import copy
            scfg = copy.deepcopy(self.cfg)
            scfg.receiver.type = "online_deepsic"
            scfg.channel.seed = self.cfg.channel.seed + 999
            self.surrogate = OnlineDeepSIC(scfg, self.device)
        return self.surrogate

    def observe_clean(self, b: int, y_p: np.ndarray, idx_p: np.ndarray) -> None:
        """The attacker (a transmitter in range) listens to the pilots and trains its surrogate."""
        if self.ac.type == "none" or self.ac.knowledge != "transfer":
            return
        sur = self._ensure_surrogate()
        # the attacker hears the pilots through its own noisy link
        y_obs = y_p + np.sqrt(10 ** (-0.1 * self.cfg.channel.snr_db) / 2) * (
            self.rng.standard_normal(y_p.shape) + (1j * self.rng.standard_normal(y_p.shape) if np.iscomplexobj(y_p) else 0))
        sur.adapt(b, sur.to_feat(y_obs), sur.to_idx(idx_p))

    # ---- rank-1 projection ----------------------------------------------------------
    def _project_structure(self, E: torch.Tensor, is_complex: bool) -> torch.Tensor:
        if self.ac.attacker_channel != "rank1":
            return E
        if self.h_a is None:
            n = E.shape[1] // (2 if is_complex else 1)
            h = self.rng.standard_normal(n) + (1j * self.rng.standard_normal(n) if is_complex else 0)
            self.h_a = h / np.linalg.norm(h)
        e = unfeat(E, is_complex)
        coef = e @ self.h_a.conj()                      # (n,)
        e_proj = np.outer(coef, self.h_a)
        return feat(e_proj, E.device)

    # ---- objectives ---------------------------------------------------------------------
    def _objective(self, core: DeepSICCore, theta_plus: ParamDict, yf_q: torch.Tensor, it_q: torch.Tensor,
                   y_d_feat: Optional[torch.Tensor], H: Optional[np.ndarray]) -> torch.Tensor:
        t = self.ac.type
        probs = core.detect(theta_plus, yf_q)
        logp = torch.log(probs.clamp_min(1e-12))
        nll = -logp.gather(-1, it_q.unsqueeze(-1)).squeeze(-1)          # (n, U)
        per_user = nll.mean(dim=0)
        if t in ("a1", "adaptive"):
            obj = per_user.sum()
            if t == "adaptive" and y_d_feat is not None and H is not None:
                obj = obj - self.ac.gate_aware_weight * self._g1_soft(core, theta_plus, y_d_feat, H)
            return obj
        if t == "a2":
            u = self.ac.target_user
            others = [v for v in range(per_user.shape[0]) if v != u]
            return per_user[u] - self.ac.a2_lambda * per_user[others].sum()
        if t == "a3":
            assert self.theta_bad is not None
            dist = sum(((theta_plus[k] - self.theta_bad[k]) ** 2).sum() for k in self.theta_bad)
            return -dist - self.ac.a2_lambda * per_user.sum()
        raise ValueError(t)

    def _g1_soft(self, core: DeepSICCore, theta: ParamDict, y_d_feat: torch.Tensor, H: np.ndarray) -> torch.Tensor:
        """Differentiable G1 residual with posterior-mean soft symbols (attacker's view)."""
        probs = core.detect(theta, y_d_feat)                            # (n, U, order)
        c = self.mod.constellation_torch(y_d_feat.device)               # (order, 2|1)
        s_soft = probs @ c                                              # (n, U, 2|1)
        Hr = torch.as_tensor(np.real(H), dtype=torch.float32, device=y_d_feat.device)   # (N, U)
        Hi = torch.as_tensor(np.imag(H), dtype=torch.float32, device=y_d_feat.device)
        if self.mod.is_complex:
            sr, si = s_soft[..., 0], s_soft[..., 1]
            yr = sr @ Hr.T - si @ Hi.T
            yi = sr @ Hi.T + si @ Hr.T
            y_hat = torch.cat([yr, yi], dim=1)
        else:
            y_hat = s_soft[..., 0] @ Hr.T
        return ((y_d_feat - y_hat) ** 2).mean()

    # ---- main entry ---------------------------------------------------------------------
    def craft(self, b: int, y_p: np.ndarray, idx_p: np.ndarray, y_d: np.ndarray, receiver,
              noise_var: float) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
        stats: Dict[str, Any] = {"attacked": False}
        if not self.active(b):
            return y_p, y_d, stats
        t = self.ac.type
        is_complex = np.iscomplexobj(y_p)
        stats["attacked"] = True

        if t == "data_jam":
            eps = self.budget(y_d)
            E = self.rng.standard_normal(y_d.shape) + (1j * self.rng.standard_normal(y_d.shape) if is_complex else 0)
            E = E / np.linalg.norm(E) * eps
            stats["eps"] = eps
            return y_p, y_d + E, stats

        # victim or surrogate model
        if self.ac.knowledge == "transfer":
            sur = self._ensure_surrogate()
            core, theta, keys, lr = sur.core, sur.params, list(sur.params.keys()), sur.lr
        else:
            surf = receiver.attack_surface()
            if not surf:
                return y_p, y_d, stats
            core, theta, keys, lr = surf["core"], surf["params"], surf["keys"], surf["lr"]
        theta = clone_params(theta)

        eps = self.budget(y_p, small=(t == "a3"))
        yf_p = feat(y_p, self.device)
        it_p = torch.as_tensor(idx_p, dtype=torch.long, device=self.device)
        y_d_feat = feat(y_d, self.device) if t == "adaptive" else None
        H = ls_channel_estimate(y_p, self.mod.modulate_indices(idx_p), noise_var) if t in ("adaptive", "a3") else None

        if t == "a3" and self.theta_bad is None:
            # spoofed channel: rotate / scale the user gains so the bad basin is well separated
            H_bad = H * (1.0 + 0.5 * self.rng.standard_normal(H.shape))
            s_p = self.mod.modulate_indices(idx_p)
            y_bad = s_p @ H_bad.T
            y_bad = y_bad + np.sqrt(noise_var / 2) * (self.rng.standard_normal(y_bad.shape) + 1j * self.rng.standard_normal(y_bad.shape)) \
                if is_complex else y_bad + np.sqrt(noise_var) * self.rng.standard_normal(y_bad.shape)
            with torch.enable_grad():
                tb = core.functional_adapt(theta, feat(y_bad, self.device), it_p, lr=lr, steps=self.ac.unroll_steps * 5,
                                           keys=keys, create_graph=False)
            self.theta_bad = {k: tb[k].detach() for k in keys}

        E = torch.zeros_like(yf_p, requires_grad=True)
        alpha = self.ac.pgd_alpha * eps
        best_obj, best_E = -np.inf, torch.zeros_like(yf_p)
        for _ in range(self.ac.pgd_steps):
            theta_plus = core.functional_adapt(theta, yf_p + E, it_p, lr=self.ac.unroll_lr, steps=self.ac.unroll_steps,
                                               keys=keys, create_graph=True)
            obj = self._objective(core, theta_plus, yf_p, it_p, y_d_feat, H)
            g, = torch.autograd.grad(obj, E)
            with torch.no_grad():
                if obj.item() > best_obj:
                    best_obj, best_E = obj.item(), E.detach().clone()
                step = g / g.norm().clamp_min(1e-12) * alpha
                E_new = E + step
                E_new = self._project_structure(E_new, is_complex)
                nrm = E_new.norm()
                if nrm > eps:
                    E_new = E_new * (eps / nrm)
                E = E_new.detach().requires_grad_(True)
        # evaluate the last iterate as well and keep the best perturbation seen
        theta_plus = core.functional_adapt(theta, yf_p + E, it_p, lr=self.ac.unroll_lr, steps=self.ac.unroll_steps,
                                           keys=keys, create_graph=False)
        obj = self._objective(core, theta_plus, yf_p, it_p, y_d_feat, H).item()
        if obj > best_obj:
            best_obj, best_E = obj, E.detach().clone()
        E_final = best_E
        E_np = unfeat(E_final, is_complex)
        stats.update(eps=eps, obj=float(best_obj), e_norm=float(np.linalg.norm(E_np)))
        y_p_adv = y_p + E_np
        y_d_adv = y_d
        if self.ac.where in ("pilots_data", "all"):
            eps_d = self.budget(y_d)
            Ed = self.rng.standard_normal(y_d.shape) + (1j * self.rng.standard_normal(y_d.shape) if is_complex else 0)
            y_d_adv = y_d + Ed / np.linalg.norm(Ed) * eps_d
        return y_p_adv, y_d_adv, stats


def make_attacker(cfg: Config, device: torch.device, seed: int) -> Attacker:
    return Attacker(cfg, device, np.random.default_rng(seed + 4242))
