from __future__ import annotations

from pathlib import Path

import torch

from .config import V2Config
from .model import TwoStageTTS

SCHEMA = "qwen-mimi-tts-v2"


def save_checkpoint(path: str | Path, model: TwoStageTTS, cfg: V2Config, phase: str,
                    epoch: int, optimizer=None, extra: dict | None = None) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    payload = {"schema": SCHEMA, "config": cfg.to_dict(), "phase": phase, "epoch": epoch,
               "model": {k: v.detach().cpu() for k, v in model.state_dict().items()},
               "optimizer": optimizer.state_dict() if optimizer is not None else None,
               "torch_rng": torch.get_rng_state(), "extra": extra or {}}
    temporary = target.with_suffix(target.suffix + ".tmp")
    torch.save(payload, temporary)
    temporary.replace(target)


def load_checkpoint(path: str | Path, device="cpu") -> tuple[TwoStageTTS, V2Config, dict]:
    payload = torch.load(path, map_location="cpu", weights_only=True)
    if payload.get("schema") != SCHEMA:
        raise ValueError("not a v2 checkpoint; legacy weights are architecturally incompatible")
    cfg = V2Config.from_dict(payload["config"])
    model = TwoStageTTS(cfg.model)
    model.load_state_dict(payload["model"], strict=True)
    model.set_phase(payload["phase"])
    model.to(device)
    return model, cfg, payload
