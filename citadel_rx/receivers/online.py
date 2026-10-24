"""Online DeepSIC (reference: `online_deepsic`) and the shared machinery for all
DeepSIC-based receivers: Adam stage training, joint pre-training, detection.
"""
from __future__ import annotations

from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
import torch

from ..config import Config
from .base import BlockOutput, Receiver
from .deepsic import DeepSICCore, ParamDict, clone_params


def adam_train_stage(core: DeepSICCore, p: ParamDict, q: int, X: torch.Tensor, idx: torch.Tensor,
                     lr: float, epochs: int, user_mask: Optional[torch.Tensor] = None,
                     grad_transform: Optional[Callable[[List[torch.Tensor]], List[torch.Tensor]]] = None,
                     keys: Optional[List[str]] = None) -> ParamDict:
    """Train the U stacked modules of stage q with a fresh Adam (reference behaviour:
    a new optimiser per module per block). ``user_mask`` (U,) in {0,1} restricts the
    update to drifted users; ``grad_transform`` lets robust variants replace the
    gradient (median-of-means etc.)."""
    stage_keys = [k for k in core.keys_for(q) if (keys is None or k in keys)]
    p = dict(p)
    leaves = [p[k].detach().clone().requires_grad_(True) for k in stage_keys]
    opt = torch.optim.Adam(leaves, lr=lr)
    pp = dict(p)
    for _ in range(epochs):
        for k, t in zip(stage_keys, leaves):
            pp[k] = t
        ce = core.stage_loss(pp, q, X, idx, reduction="none")         # (U, n)
        per_user = ce.mean(dim=1)
        if user_mask is not None:
            per_user = per_user * user_mask
        loss = per_user.sum()
        if grad_transform is None:
            opt.zero_grad(set_to_none=True)
            loss.backward()
        else:
            grads = grad_transform(_per_group_grads(core, pp, q, X, idx, stage_keys, leaves, user_mask))
            opt.zero_grad(set_to_none=True)
            for t, g in zip(leaves, grads):
                t.grad = g
        opt.step()
    for k, t in zip(stage_keys, leaves):
        p[k] = t.detach()
    return p


def _per_group_grads(core, pp, q, X, idx, stage_keys, leaves, user_mask, groups: int = 4):
    """Gradients computed on ``groups`` disjoint pilot sub-batches (for robust aggregation)."""
    n = X.shape[1]
    perm = torch.randperm(n, device=X.device)
    out = []
    for g in range(groups):
        sel = perm[g::groups]
        ce = core.stage_loss(pp, q, X[:, sel], idx[sel], reduction="none").mean(dim=1)
        if user_mask is not None:
            ce = ce * user_mask
        grads = torch.autograd.grad(ce.sum(), leaves, retain_graph=g < groups - 1)
        out.append(grads)
    return out


class DeepSICReceiver(Receiver):
    """Base for every DeepSIC-backed receiver. Subclasses override ``adapt``."""

    name = "deepsic"

    def __init__(self, cfg: Config, device: torch.device):
        super().__init__(cfg, device)
        rc = cfg.receiver
        self.core = DeepSICCore(self.n_user, self.n_ant, self.mod.order, self.mod.is_complex,
                                rc.iterations, rc.hidden_base, device)
        g = torch.Generator(device="cpu").manual_seed(cfg.channel.seed)
        self.params: ParamDict = self.core.init_params(generator=g if device.type == "cpu" else None)
        self.lr = rc.lr
        self.epochs = rc.epochs
        self.block_count = 0

    # ---- joint pre-training ----------------------------------------------
    def pretrain(self, blocks: List[Tuple[np.ndarray, np.ndarray]]) -> None:
        """Reference `joint_*`: train sequentially on each pre-training block
        (many SNRs / channel realisations), with pretrain_epochs."""
        for y, idx in blocks:
            yf, it = self.to_feat(y), self.to_idx(idx)
            self.params = self.core.sequential_adapt(
                self.params, yf, it,
                lambda p, q, X, t: adam_train_stage(self.core, p, q, X, t, self.lr, self.cfg.receiver.pretrain_epochs))
        self.pretrained_params = clone_params(self.params)

    # ---- per block ------------------------------------------------------
    def adapt(self, b: int, yf: torch.Tensor, idx: torch.Tensor) -> Dict[str, Any]:
        self.params = self.core.sequential_adapt(
            self.params, yf, idx,
            lambda p, q, X, t: adam_train_stage(self.core, p, q, X, t, self.lr, self.epochs))
        return {}

    def detect(self, yf: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            return self.core.detect(self.params, yf)

    def process_block(self, b, y_p, idx_p, y_d, aux=None) -> BlockOutput:
        yf_p, it_p, yf_d = self.to_feat(y_p), self.to_idx(idx_p), self.to_feat(y_d)
        stats = self.adapt(b, yf_p, it_p) or {}
        with torch.no_grad():
            stats["pilot_loss_post"] = float(self.core.detect_loss(self.params, yf_p, it_p).item())
        probs = self.detect(yf_d)
        self.block_count += 1
        return BlockOutput(probs_data=probs, idx_hat=self.hard(probs), stats=stats)

    # ---- attacker view ---------------------------------------------------
    def attack_surface(self) -> Dict[str, Any]:
        return {"core": self.core, "params": self.params, "keys": list(self.params.keys()),
                "lr": self.lr}


class OnlineDeepSIC(DeepSICReceiver):
    name = "online_deepsic"
