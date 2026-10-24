"""Block-fading uplink MIMO channel models.

System model (proposal §2.3.1):  y_b[n] = H_b s_b[n] + q_b(.) + n_b[n],
H_b ∈ R^{N_r x U} (real-valued magnitudes as in the reference code; complex
QPSK symbols pass through a real gain matrix, which is exactly the reference
DeepSIC setting), q_b = tanh(0.5 .) when ``linear`` is False.

Fading modes
------------
cosine : the reference deterministic per-user fading, periods (51,39,33,21)
         blocks, depth 0.8..1.0 (exactly `SEDChannel._add_fading`).
jakes  : AR(1) Gauss-Markov block correlation  H_{b+1} = rho H_b + sqrt(1-rho^2) W_b
         around the SED mean, rho = J0(2 pi f_D T_b).
static : no variation.
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Optional

import numpy as np
from scipy.special import j0

from ..config import ChannelConfig

_SED_FADE_PERIODS = np.array([51, 39, 33, 21])
_SED_FADE_DEPTH = 0.8
_COST_SCALING = 0.25
_COST_FRAMES = 25
_COST_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
                         "resources", "cost2100_channel")


@dataclass
class ChannelState:
    H: np.ndarray            # (n_ant, n_user)
    noise_var: float         # per real dimension total (reference convention)
    rho: float               # block-to-block correlation (prior for G3)


class BlockFadingChannel:
    def __init__(self, cfg: ChannelConfig, is_complex: bool, cost_dir: Optional[str] = None):
        self.cfg = cfg
        self.n_user = cfg.n_user
        self.n_ant = cfg.n_ant
        self.is_complex = is_complex
        self.rng = np.random.default_rng(cfg.seed + 1000)
        self.cost_dir = cost_dir or _COST_DIR
        self._H_prev: Optional[np.ndarray] = None
        self.H_mean = self._sed_mean()
        self.rho = 1.0
        if cfg.fading == "jakes":
            self.rho = float(j0(2 * np.pi * cfg.jakes_fd_tb))

    # ---- channel matrices --------------------------------------------
    def _sed_mean(self) -> np.ndarray:
        rows = np.arange(self.n_ant)[:, None]
        cols = np.arange(self.n_user)[None, :]
        return np.exp(-np.abs(rows - cols)).astype(np.float64)

    def _synthetic(self, b: int) -> np.ndarray:
        H = self.H_mean.copy()
        mode = self.cfg.fading
        if mode == "cosine":
            fade = _SED_FADE_DEPTH + (1 - _SED_FADE_DEPTH) * np.cos(2 * np.pi * b / _SED_FADE_PERIODS[: self.n_user])
            H = H * fade[None, :]
        elif mode == "jakes":
            if self._H_prev is None:
                dev = self.rng.standard_normal(H.shape) * 0.1
            else:
                dev = self.rho * (self._H_prev - self.H_mean) + np.sqrt(1 - self.rho ** 2) * 0.1 * self.rng.standard_normal(H.shape)
            H = self.H_mean + dev
        elif mode == "static":
            pass
        else:
            raise ValueError(f"unknown fading mode {mode}")
        self._H_prev = H
        return H

    def _cost2100(self, b: int) -> np.ndarray:
        # Reference convention: y = h @ s with h rows taken from h_i.mat, i.e. row i is the gain
        # vector seen at receive antenna i across users; diagonal pinned to 1, off-diagonal x0.25.
        import scipy.io
        H = np.empty((self.n_ant, self.n_user))
        folder = 1 + (b // _COST_FRAMES)
        folder = ((folder - 1) % 8) + 1  # 8 folders x 25 frames = 200 distinct blocks, then wrap
        for i in range(1, self.n_ant + 1):
            path = os.path.join(self.cost_dir, f"{folder}", f"h_{i}.mat")
            h_row = scipy.io.loadmat(path)["norm_channel"][b % _COST_FRAMES, : self.n_user]
            H[i - 1] = _COST_SCALING * h_row
        d = min(self.n_user, self.n_ant)
        H[np.arange(d), np.arange(d)] = 1.0
        self._H_prev = H
        return H

    def state(self, b: int) -> ChannelState:
        if self.cfg.model == "synthetic":
            H = self._synthetic(b)
        elif self.cfg.model == "cost2100":
            H = self._cost2100(b)
        else:
            raise ValueError(f"unknown channel model {self.cfg.model}")
        noise_var = 10 ** (-0.1 * self.cfg.snr_db)
        return ChannelState(H=H, noise_var=noise_var, rho=self.rho)

    # ---- transmission -------------------------------------------------
    def transmit(self, s: np.ndarray, st: ChannelState, rng: Optional[np.random.Generator] = None) -> np.ndarray:
        """s: (n_sym, n_user) complex or real -> y: (n_sym, n_ant)."""
        rng = rng or self.rng
        conv = s @ st.H.T
        var = st.noise_var
        if self.is_complex:
            w = np.sqrt(var) / 2 * (rng.standard_normal(conv.shape) + 1j * rng.standard_normal(conv.shape))
        else:
            w = np.sqrt(var) * rng.standard_normal(conv.shape)
        y = conv + w
        if not self.cfg.linear:
            y = np.tanh(0.5 * y)
        return y

    def apply_nonlinearity(self, y_lin: np.ndarray) -> np.ndarray:
        return y_lin if self.cfg.linear else np.tanh(0.5 * y_lin)


def make_channel(cfg: ChannelConfig, is_complex: bool) -> BlockFadingChannel:
    return BlockFadingChannel(cfg, is_complex)
