"""Physics-consistency gates (proposal §2.5.2).

Each gate maps block statistics to a verdict γ ∈ [0, 1]. Every statistic is
*relative to a model-based reference computed on the same block*, so that a
channel that simply got worse (deep fade, high Doppler) does not look like a
poisoned adaptation: the reference degrades with the channel, the neural
receiver's adaptation does not have to.

G1 — model-based residual gate.
    r_nn = mean_n ‖y_d[n] − Ĥ ŝ_nn[n]‖²,  r_mm = same with LMMSE decisions,
    stat = (r_nn / r_mm) / clean-EMA.  Correct adaptation ⇒ ratio ≈ 1 or below;
    poisoned adaptation ⇒ decisions inconsistent with the linear model ⇒ ≫ 1.
    The ratio is insensitive to model mismatch (PA nonlinearity) because the
    mismatch inflates both terms.

G2 — CRC gate.
    stat = pass_nn / pass_mm : fraction of CRC-protected sub-blocks that pass
    with the adapted fast level, relative to the fraction that pass with the
    LMMSE detector on the same block. Proposition 2.3 ties pass_nn to the
    post-adaptation SER, i.e. to the attacker's own objective under A1.

G3 — channel-prior likelihood gate.
    stat = innovation energy of the LS channel estimate against the AR(1)
    prediction, normalised by its running level on accepted blocks (floored
    by the physical expectation). Catches A2-style physically implausible jumps.

Self-generated thresholds: tightened when the fast level's surprise is high
relative to its running average (NL: the model generates its own retention
gate).
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


def g2_crc(pass_nn: float, pass_mm: float, floor: float = 0.02) -> float:
    return pass_nn / max(pass_mm, floor)


def g3_prior(innov_norm: float) -> float:
    return innov_norm


class GateController:
    def __init__(self, gates: List[str], mode: str, g1_tau: float, g2_tau: float, g3_tau: float,
                 g3_quantile: float, adaptive: bool, dof: int, kappa: float = 0.1, ema: float = 0.9):
        self.gates = [g.lower() for g in gates]
        self.mode = mode
        self.g1_tau, self.g2_tau = g1_tau, g2_tau
        self.adaptive = adaptive
        self.dof = max(dof, 1)
        self.kappa = kappa
        self.ema = ema
        # G3 threshold: ratio to the running innovation level, never below the chi-square floor
        self.g3_tau = max(g3_tau, chi2.ppf(g3_quantile, self.dof) / self.dof)
        self.clean_g1_ema: Optional[float] = None
        self.clean_g2_ema: Optional[float] = None
        self.n_accepted = 0

    def thresholds(self, surprise_ratio: float) -> Dict[str, float]:
        """Tighten thresholds when surprise is high relative to its EMA (ratio > 1)."""
        t = 1.0
        if self.adaptive and surprise_ratio > 1:
            t = float(np.clip(1.0 / surprise_ratio, 0.5, 1.0))
        return {"g1": 1 + (self.g1_tau - 1) * t,            # 1.5 -> 1.25 at the tightest
                "g2": 1 - (1 - self.g2_tau) * t,            # 0.5 -> 0.75 at the tightest
                "g3": 1 + (self.g3_tau - 1) * t}

    def evaluate(self, g1_stat: Optional[float], pass_nn: Optional[float], pass_mm: Optional[float],
                 innov_norm: Optional[float], surprise_ratio: float = 1.0) -> GateVerdict:
        th = self.thresholds(surprise_ratio)
        stats: Dict[str, float] = {}
        verdicts: Dict[str, float] = {}
        if "g1" in self.gates and g1_stat is not None:
            s = g1_stat / max(self.clean_g1_ema or 1.0, 1e-9)
            stats["g1"] = s
            verdicts["g1"] = _soft_step(s, th["g1"], self.kappa, accept_below=True)
        if "g2" in self.gates and pass_nn is not None and pass_mm is not None:
            s = g2_crc(pass_nn, pass_mm) / max(self.clean_g2_ema or 1.0, 1e-9)
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
        elif self.mode == "max":
            gamma = float(max(verdicts.values()))
        elif self.mode == "any_hard":
            gamma = float(all(v > 0.5 for v in verdicts.values()))
        else:
            raise ValueError(self.mode)
        return GateVerdict(gamma=gamma, stats=stats, verdicts=verdicts)

    def update_clean(self, verdict: GateVerdict, g1_stat: Optional[float], pass_nn: Optional[float],
                     pass_mm: Optional[float]) -> bool:
        """Track the clean baselines of the *relative* statistics from accepted blocks (gamma > 0.5);
        the first few blocks are always accepted so the EMAs can initialise. Returns acceptance."""
        accept = verdict.gamma > 0.5 or self.n_accepted < 3
        if not accept:
            return False
        self.n_accepted += 1
        if g1_stat is not None:
            self.clean_g1_ema = g1_stat if self.clean_g1_ema is None else \
                self.ema * self.clean_g1_ema + (1 - self.ema) * g1_stat
        if pass_nn is not None and pass_mm is not None:
            r = g2_crc(pass_nn, pass_mm)
            self.clean_g2_ema = r if self.clean_g2_ema is None else self.ema * self.clean_g2_ema + (1 - self.ema) * r
        return True
