"""Model-based LMMSE channel estimation and detection.

Used (i) as the "adapt nothing" baseline receiver and (ii) inside the gates:
G1 needs a channel estimate and model-based soft decisions, G3 needs the
innovation of the channel estimate against the Jakes/AR(1) prior.

Channel estimate from known pilots S (n_p, U) and observations Y (n_p, N):
    H_ls   = (S^H S + sigma^2 I)^{-1} S^H Y          (regularised LS, rows = users)
    H_hat  = H_ls^T                                   (N, U)
Prior fusion (one-step Kalman predictor with scalar innovation variance):
    H_pred = rho * H_prev ;  H_fused = H_pred + K (H_hat - H_pred).
Detection: per-symbol linear MMSE equaliser then nearest constellation point.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional

import numpy as np
import torch

from ..config import Config
from ..modulation import Modulation
from .base import BlockOutput, Receiver


def ls_channel_estimate(y_p: np.ndarray, s_p: np.ndarray, noise_var: float) -> np.ndarray:
    """Regularised least squares. y_p (n_p, N), s_p (n_p, U) -> H (N, U)."""
    S = s_p
    G = S.conj().T @ S + noise_var * np.eye(S.shape[1])
    H = np.linalg.solve(G, S.conj().T @ y_p)       # (U, N)
    return H.T


def estimate_noise_var(y_p: np.ndarray, s_p: np.ndarray, H: np.ndarray) -> float:
    r = y_p - s_p @ H.T
    return float(np.mean(np.abs(r) ** 2))


def lmmse_soft_symbols(y: np.ndarray, H: np.ndarray, noise_var: float, mod: Modulation) -> np.ndarray:
    """Per-symbol-vector LMMSE equalisation followed by posterior over the constellation.
    Returns probs (n, U, order)."""
    N, U = H.shape
    sigma2 = noise_var
    # symbol energy is 1 for both BPSK and QPSK conventions used here
    W = np.linalg.solve(H.conj().T @ H + sigma2 * np.eye(U), H.conj().T)   # (U, N)
    x_hat = y @ W.T                                                        # (n, U)
    # post-equalisation gain and residual variance per user
    HW = W @ H                                                             # (U, U)
    mu = np.real(np.diag(HW))                                              # (U,)
    nu = np.maximum(mu - mu ** 2, 1e-6)                                    # unbiased residual var approx
    c = mod.constellation()                                                # (order,)
    d2 = np.abs(x_hat[:, :, None] - mu[None, :, None] * c[None, None, :]) ** 2
    logits = -d2 / nu[None, :, None]
    logits -= logits.max(axis=-1, keepdims=True)
    p = np.exp(logits)
    return p / p.sum(axis=-1, keepdims=True)


def hard_from_probs(p: np.ndarray) -> np.ndarray:
    return p.argmax(axis=-1)


@dataclass
class ChannelTracker:
    """Keeps the previous channel estimate and fuses with an AR(1) prior."""
    rho: float
    H_prev: Optional[np.ndarray] = None
    innov_var_ema: Optional[float] = None
    ema: float = 0.9

    def fuse(self, H_hat: np.ndarray, noise_var: float, n_p: int) -> Dict[str, Any]:
        est_var = noise_var / max(n_p, 1)            # LS estimation noise per entry (unit-energy pilots)
        if self.H_prev is None:
            self.H_prev = H_hat
            return {"H": H_hat, "innov": 0.0, "innov_norm": 0.0, "pred": H_hat}
        H_pred = self.rho * self.H_prev + (1 - self.rho) * 0.0
        innov = H_hat - H_pred
        innov_energy = float(np.sum(np.abs(innov) ** 2))
        # expected innovation energy under the prior: process variance + estimation noise
        n_entries = innov.size
        proc_var = (1 - self.rho ** 2) * float(np.mean(np.abs(self.H_prev) ** 2)) if self.rho < 1 else 0.0
        expected = n_entries * (proc_var + est_var) + 1e-9
        if self.innov_var_ema is None:
            self.innov_var_ema = max(innov_energy, expected)
        normalised = innov_energy / max(expected, 0.5 * self.innov_var_ema)
        # Kalman-style gain: trust the estimate in proportion to prior uncertainty
        K = (proc_var + 1e-9) / (proc_var + est_var + 1e-9) if self.rho < 1 else 1.0
        H_fused = H_pred + K * innov
        self.innov_var_ema = self.ema * self.innov_var_ema + (1 - self.ema) * innov_energy
        self.H_prev = H_fused
        return {"H": H_fused, "innov": innov_energy, "innov_norm": normalised, "pred": H_pred,
                "expected": expected}


class LMMSEReceiver(Receiver):
    """Adapt-nothing model-based floor."""

    def __init__(self, cfg: Config, device: torch.device):
        super().__init__(cfg, device)
        self.noise_var = 10 ** (-0.1 * cfg.channel.snr_db)
        self.tracker = ChannelTracker(rho=1.0)

    def process_block(self, b, y_p, idx_p, y_d, aux=None) -> BlockOutput:
        s_p = self.mod.modulate_indices(idx_p)
        H = ls_channel_estimate(y_p, s_p, self.noise_var)
        nv = estimate_noise_var(y_p, s_p, H)
        probs = lmmse_soft_symbols(y_d, H, nv, self.mod)
        pt = torch.as_tensor(probs, dtype=torch.float32, device=self.device)
        return BlockOutput(probs_data=pt, idx_hat=hard_from_probs(probs), stats={"noise_var_hat": nv})
