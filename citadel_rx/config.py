"""Experiment configuration.

A single dataclass tree, loadable from YAML, hashable for run records.
Defaults reproduce the reference Meta-DeepSIC setting (TWC 2023 code,
`LINEAR_SYNTH_QPSK.yaml`): 4x4 QPSK, 200 pilot bits, 50 000 data bits per
block, SNR 10 dB, cosine fading on the SED channel.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field, fields, is_dataclass
from typing import Any, Dict, List, Optional

import yaml


@dataclass
class ChannelConfig:
    model: str = "synthetic"          # synthetic | cost2100
    n_user: int = 4
    n_ant: int = 4
    linear: bool = True               # False -> tanh(0.5 y) PA nonlinearity
    fading: str = "cosine"            # cosine (reference) | jakes | static
    jakes_fd_tb: float = 0.05         # normalised Doppler f_D * T_b for 'jakes'
    snr_db: float = 10.0
    seed: int = 0


@dataclass
class FrameConfig:
    modulation: str = "QPSK"          # BPSK | QPSK
    pilot_bits: int = 200             # pilot bits per user per block (reference: 200)
    data_bits: int = 50000            # data bits per user per block (reference: 50 000)
    n_blocks: int = 100
    # CRC framing of the data part (used by gate G2 and the BLER metric)
    crc_bits: int = 8                 # 0 disables CRC framing
    crc_subblock_symbols: int = 32    # symbols per CRC-protected sub-block


@dataclass
class ReceiverConfig:
    type: str = "citadel"             # lmmse | online_deepsic | meta_deepsic | drift_deepsic |
                                      # titans_single | medbn_deepsic | citadel
    iterations: int = 3               # DeepSIC SIC iterations Q
    hidden_base: int = 64
    lr: float = 5e-3
    epochs: int = 300                 # online epochs per block (reference: 300)
    optimizer: str = "adam"
    pretrain: bool = True             # joint pre-training on many channel realisations
    pretrain_blocks: int = 1
    pretrain_bits: int = 5000
    pretrain_snrs: List[float] = field(default_factory=lambda: [2, 4, 6, 8, 10, 12, 14, 16])
    pretrain_epochs: int = 300
    # meta
    meta_lr: float = 0.01
    meta_every: int = 5
    meta_epochs: int = 300
    # drift detection
    drift_threshold: float = 2.0
    # medbn-style robust gradient
    robust_groups: int = 4


@dataclass
class CitadelConfig:
    levels: int = 3
    periods: List[int] = field(default_factory=lambda: [1, 8, 32])
    adaptable: str = "all"            # all | last | mask
    fast_rule: str = "dgd"            # dgd | gd | adam
    fast_steps: int = 300             # fast-level steps per block
    eta_max: float = 5e-3
    surprise_lr: bool = True
    surprise_ema: float = 0.9
    level_lr: List[float] = field(default_factory=lambda: [1.0, 0.5, 0.25])  # eta^(k) for k>=2 (index k-1)
    reanchor_mode: str = "proximal"   # jump (Eq. 72 re-anchor at consolidation) | proximal (per-block pull)
    reanchor_beta: float = 0.25       # jump weight
    prox_lambda: float = 0.5          # proximal pull strength (fraction of eta) toward the slow projection
    level_weights: List[float] = field(default_factory=lambda: [0.5, 0.5])  # Pi weights over slow levels
    gates: List[str] = field(default_factory=lambda: ["g1", "g2", "g3"])
    gate_mode: str = "min"            # min (all gates must pass) | product | max (any) | any_hard
    g1_tau: float = 1.25              # residual ratio threshold (relative to the clean EMA)
    g2_tau: Optional[float] = None    # explicit CRC pass-rate ratio threshold; None -> derived from g2_ser_slack
    g2_ser_slack: float = 0.01        # tolerated SER excess over the reference; tau2 = (1 - slack)^L (Prop. 2.3)
    g3_tau: float = 4.0               # innovation energy relative to its running level
    g3_quantile: float = 0.999        # chi-square floor for the innovation test
    adaptive_thresholds: bool = True
    tighten_floor: float = 0.7        # thresholds shrink at most to this fraction of their slack under surprise
    surprise_clamp: float = 8.0       # surprise ratio clipped to [1/c, c]
    surprise_warmup: int = 5          # blocks used to seed the surprise EMA
    reject_action: str = "revert"     # what happens to the fast level on a rejected block: keep | revert | anchor
    fallback_on_reject: bool = True   # detect the data of a rejected block with the slow projection instead
    dgd_delta: float = 0.1            # scale of the delta-rule decay term


@dataclass
class AttackConfig:
    type: str = "none"                # none | a1 | a2 | a3 | adaptive
    knowledge: str = "whitebox"       # whitebox | transfer
    sir_db: float = 0.0               # perturbation power relative to signal power
    eps: Optional[float] = None       # explicit l2 budget per block (overrides sir_db)
    where: str = "pilots"             # pilots | pilots_data | all
    rho_blk: float = 1.0              # fraction of attacked blocks
    start_block: int = 20
    end_block: int = 60
    pgd_steps: int = 20
    pgd_alpha: float = 0.25           # step as a fraction of eps
    unroll_steps: int = 1             # adaptation steps unrolled by the attacker
    unroll_lr: float = 5e-3
    target_user: int = 0              # A2
    a2_lambda: float = 1.0
    a3_eps_small: float = 0.05
    surrogate_blocks: int = 10        # transfer: blocks of pilots observed by the attacker
    attacker_channel: str = "direct"  # direct | rank1
    gate_aware_weight: float = 1.0    # adaptive attacker: penalty on gate statistics


@dataclass
class RunConfig:
    name: str = "debug"
    out_dir: str = "results"
    device: str = "auto"
    seeds: List[int] = field(default_factory=lambda: [0])
    log_every: int = 10
    save_params: bool = False


@dataclass
class Config:
    channel: ChannelConfig = field(default_factory=ChannelConfig)
    frame: FrameConfig = field(default_factory=FrameConfig)
    receiver: ReceiverConfig = field(default_factory=ReceiverConfig)
    citadel: CitadelConfig = field(default_factory=CitadelConfig)
    attack: AttackConfig = field(default_factory=AttackConfig)
    run: RunConfig = field(default_factory=RunConfig)

    # ---- serialisation -------------------------------------------------
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    def hash(self) -> str:
        return hashlib.sha256(json.dumps(self.to_dict(), sort_keys=True).encode()).hexdigest()[:12]

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "Config":
        return _dc_from_dict(cls, d)

    @classmethod
    def from_yaml(cls, path: str, overrides: Optional[List[str]] = None) -> "Config":
        with open(path) as f:
            d = yaml.safe_load(f) or {}
        cfg = cls.from_dict(d)
        for ov in overrides or []:
            cfg.set(ov)
        return cfg

    def set(self, assignment: str) -> None:
        """Apply a dotted override such as ``attack.sir_db=5``."""
        key, _, raw = assignment.partition("=")
        node: Any = self
        parts = key.strip().split(".")
        for p in parts[:-1]:
            node = getattr(node, p)
        current = getattr(node, parts[-1])
        setattr(node, parts[-1], _coerce(raw.strip(), current))

    @property
    def bits_per_symbol(self) -> int:
        return 1 if self.frame.modulation.upper() == "BPSK" else 2

    @property
    def pilot_symbols(self) -> int:
        return self.frame.pilot_bits // self.bits_per_symbol

    @property
    def data_symbols(self) -> int:
        return self.frame.data_bits // self.bits_per_symbol


def _dc_from_dict(cls, d):
    if not is_dataclass(cls):
        return d
    import typing
    hints = typing.get_type_hints(cls)   # resolves the string annotations of `from __future__ import annotations`
    kwargs = {}
    names = {f.name for f in fields(cls)}
    for k, v in (d or {}).items():
        if k not in names:
            raise KeyError(f"Unknown config key '{k}' for {cls.__name__}")
        t = hints.get(k)
        if isinstance(t, type) and is_dataclass(t):
            kwargs[k] = _dc_from_dict(t, v)
        else:
            kwargs[k] = v
    return cls(**kwargs)


def _coerce(raw: str, current: Any) -> Any:
    if isinstance(current, bool):
        return raw.lower() in ("1", "true", "yes")
    if isinstance(current, int) and not isinstance(current, bool):
        return int(float(raw))
    if isinstance(current, float):
        return float(raw)
    if isinstance(current, list):
        return yaml.safe_load(raw)
    if current is None:
        return yaml.safe_load(raw)
    return raw
