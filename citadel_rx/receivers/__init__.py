from .base import Receiver, BlockOutput  # noqa: F401
from .deepsic import DeepSICCore, ParamDict  # noqa: F401


def make_receiver(cfg, device):
    t = cfg.receiver.type
    if t == "lmmse":
        from .lmmse import LMMSEReceiver
        return LMMSEReceiver(cfg, device)
    if t == "online_deepsic":
        from .online import OnlineDeepSIC
        return OnlineDeepSIC(cfg, device)
    if t == "meta_deepsic":
        from .meta import MetaDeepSIC
        return MetaDeepSIC(cfg, device)
    if t == "drift_deepsic":
        from .drift import DriftDeepSIC
        return DriftDeepSIC(cfg, device)
    if t == "medbn_deepsic":
        from .robust import MedBNDeepSIC
        return MedBNDeepSIC(cfg, device)
    if t == "titans_single":
        from .citadel import CitadelRx
        return CitadelRx(cfg, device, single_memory=True)
    if t == "citadel":
        from .citadel import CitadelRx
        return CitadelRx(cfg, device)
    raise ValueError(f"unknown receiver type {t}")
