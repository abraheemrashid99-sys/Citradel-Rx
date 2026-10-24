"""Physics-consistency gates (proposal §2.5.2).

Each gate maps block statistics to a verdict γ ∈ [0, 1]. The controller
combines them and maintains the self-generated (adaptive) thresholds: a clean
baseline is tracked by an EMA of each statistic on *accepted* blocks, and the
threshold tightens when the fast level's surprise is high relative to its own
running average (NL's "model generates its own retention gate").

G1 — model-based residual gate.
    r_nn = mean_n || y_d[n] - H_hat s_hat_nn[n] ||^2,  r_mm = same with LMMSE decisions.
    stat = r_nn / r_mm. Decisions of a correctly adapted neural detector are at
    least as consistent with the linear model as LMMSE's (ratio ≈ 1 or below);
    poisoned adaptation yields decisions inconsistent with the model (ratio ≫ 1).
    Using the ratio makes the test insensitive to model mismatch (PA
    nonlinearity) because mismatch inflates both terms.

G2 — CRC gate.
    stat = fraction of CRC-protected sub-blocks that pass with the adapted fast
    level, divided by the clean pass-rate EMA. Proposition 2.3 ties the pass
    probability to the post-adaptation SER, i.e. to the attacker's own
    objective under A1.

G3 — channel-prior likelihood gate.
    stat = normalised innovation energy of the LMMSE channel estimate against
    the AR(1)/Jakes prediction; compared with a chi-square quantile.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np
from scipy.stats import chi2

from ..modulation import Modulation


@dataclass
class GateVerdict:
    gamma: float
    stats: Dict[str, float] = field(default_factory=dict)
    verdicts: Dict[str, float] = field(default_factory=dict)


def _soft_step(stat: float, tau: float, kappa: float, accept_below: bool) -> float:
    """Smooth 0/1 verdict around tau with width kappa*tau."""
    x = (tau - stat) if accept_below else (stat - tau)
    return float(1.0 / (1.0 + np.exp(-x / max(kappa * abs(tau), 1e-9))))


def g1_residual(y_d: np.ndarray, idx_nn: np.ndarray, idx_mm: np.ndarray, H: np.ndarray,
                mod: Modulation) -> float:
    s_nn = mod.modulate_indices(idx_nn)
    s_mm = mod.modulate_indices(idx_mm)
    r_nn = float(np.mean(np.abs(y_d - s_nn @ H.T) ** 2))
    r_mm = float(np.mean(np.abs(y_d - s_mm @ H.T) ** 2))
    return r_nn / max(r_mm, 1e-12)


def g2_crc(pass_frac: float, clean_ema: Optional[float]) -> float:
    if clean_ema is None or clean_ema <= 0:
        return 1.0
    return pass_frac / clean_ema


def g3_prior(innov_norm: float) -> float:
    return innov_norm


class GateController:
    def __init__(self, gates: List[str], mode: str, g1_tau: float, g2_tau: float, g3_quantile: float,
                 adaptive: bool, dof: int, kappa: float = 0.1, ema: float = 0.9):
        self.gates = [g.lower() for g in gates]
        self.mode = mode
        self.g1_tau, self.g2_tau, self.g3_q = g1_tau, g2_tau, g3_quantile
        self.adaptive = adaptive
        self.dof = max(dof, 1)
        self.kappa = kappa
        self.ema = ema
        self.g3_tau = chi2.ppf(g3_quantile, self.dof) / self.dof
        self.clean_pass_ema: Optional[float] = None
        self.clean_g1_ema: Optional[float] = None
        self.n_accepted = 0

    def thresholds(self, surprise_ratio: float) -> Dict[str, float]:
        """Tighten thresholds when surprise is high relative to its EMA (ratio > 1)."""
        t = 1.0
        if self.adaptive:
            t = float(np.clip(1.0 / max(surprise_ratio, 1e-3), 0.5, 1.0)) if surprise_ratio > 1 else 1.0
        return {"g1": self.g1_tau ** t if self.g1_tau >= 1 else self.g1_tau,
                "g2": 1 - (1 - self.g2_tau) * t,
                "g3": self.g3_tau ** t if self.g3_tau >= 1 else self.g3_tau}

    def evaluate(self, g1_stat: Optional[float], pass_frac: Optional[float], innov_norm: Optional[float],
                 surprise_ratio: float = 1.0) -> GateVerdict:
        th = self.thresholds(surprise_ratio)
        stats: Dict[str, float] = {}
        verdicts: Dict[str, float] = {}
        if "g1" in self.gates and g1_stat is not None:
            base = self.clean_g1_ema if self.clean_g1_ema else 1.0
            s = g1_stat / max(base, 1e-9)
            stats["g1"] = s
            verdicts["g1"] = _soft_step(s, th["g1"], self.kappa, accept_below=True)
        if "g2" in self.gates and pass_frac is not None:
            s = g2_crc(pass_frac, self.clean_pass_ema)
            stats["g2"] = s
            verdicts["g2"] = _soft_step(s, th["g2"], self.kappa, accept_below=False)
        if "g3" in self.gates and innov_norm is not None:
            stats["g3"] = innov_norm
            verdicts["g3"] = _soft_step(innov_norm, th["g3"], self.kappa, accept_below=True)
        if not verdicts:
            gamma = 1.0
        elif self.mode == "product":
            gamma = float(np.prod(list(verdicts.values())))
        elif self.mode == "min":
            gamma = float(min(verdicts.values()))
        elif self.mode == "any_hard":
            gamma = float(all(v > 0.5 for v in verdicts.values()))
        else:
            raise ValueError(self.mode)
        return GateVerdict(gamma=gamma, stats=stats, verdicts=verdicts)

    def update_clean(self, verdict: GateVerdict, pass_frac: Optional[float], g1_stat: Optional[float]) -> None:
        """Track clean baselines from blocks the gate accepted (gamma > 0.5). During the first
        few blocks everything is accepted so the EMAs can initialise."""
        accept = verdict.gamma > 0.5 or self.n_accepted < 3
        if not accept:
            return
        self.n_accepted += 1
        if pass_frac is not None:
            self.clean_pass_ema = pass_frac if self.clean_pass_ema is None else \
                self.ema * self.clean_pass_ema + (1 - self.ema) * pass_frac
        if g1_stat is not None:
            self.clean_g1_ema = g1_stat if self.clean_g1_ema is None else \
                self.ema * self.clean_g1_ema + (1 - self.ema) * g1_stat
