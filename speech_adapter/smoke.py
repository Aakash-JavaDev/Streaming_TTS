from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

from .config import AppConfig, atomic_json


def _peak_rss_mib() -> float | None:
    status = Path("/proc/self/status")
    if not status.is_file():
        return None
    for line in status.read_text(encoding="utf-8").splitlines():
        if line.startswith("VmHWM:"):
            return float(line.split()[1]) / 1024.0
    return None


def _small_token_slice(item: dict[str, Any], maximum_frames: int = 8) -> dict[str, Any]:
    """Keep one real aligned token so a CPU smoke step has a hard memory bound."""
    import torch

    durations = item["durations"].long()
    positive = torch.nonzero(durations > 0, as_tuple=False).flatten()
    if positive.numel() == 0:
        raise RuntimeError(f"record {item['record_id']} has no token with audio frames")
    token = int(positive[0].item())
    frame_start = int(durations[:token].sum().item())
    frame_count = min(int(durations[token].item()), int(maximum_frames))
    return {
        "record_id": item["record_id"],
        "text": item["text"],
        "hidden": item["hidden"][token : token + 1],
        "embeddings": item["embeddings"][token : token + 1],
        "durations": torch.tensor([frame_count], dtype=torch.long),
        "codes": item["codes"][frame_start : frame_start + frame_count],
        "char_offsets": item["char_offsets"][token : token + 1],
        "words": item["words"],
    }


def run_local_smoke(cfg: AppConfig) -> dict[str, Any]:
    """Exercise real prepared data with one bounded CPU optimizer and cache step."""
    import torch

    from .data import SpeechDataset, collate_speech
    from .diagnostics import model_structural_checks
    from .model import adapter_loss
    from .train import build_model

    if cfg.runtime.device != "cpu":
        raise RuntimeError("local-smoke requires runtime.device='cpu'")
    torch.set_num_threads(cfg.runtime.cpu_threads)
    structural = model_structural_checks(cfg)

    dataset = SpeechDataset(cfg, "train", limit=1)
    selected_split = "train"
    if len(dataset) == 0:
        dataset = SpeechDataset(cfg, "val", limit=1)
        selected_split = "val"
    if len(dataset) == 0:
        raise RuntimeError("the limited manifest contains no train or validation record")

    item = _small_token_slice(dataset[0])
    batch = collate_speech([item])
    device = torch.device("cpu")
    model = build_model(cfg, device)
    model.train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.train.learning_rate)
    optimizer.zero_grad(set_to_none=True)
    logits, log_duration = model(batch)
    loss, parts = adapter_loss(
        logits,
        log_duration,
        batch,
        cfg.train.cb0_weight,
        cfg.train.residual_weight,
        cfg.train.duration_weight,
    )
    loss.backward()
    gradients = [parameter.grad for parameter in model.parameters() if parameter.grad is not None]
    gradients_finite = bool(gradients) and all(
        bool(torch.isfinite(gradient).all()) for gradient in gradients
    )
    gradient_norm = math.sqrt(sum(float(gradient.float().square().sum()) for gradient in gradients))
    optimizer.step()

    model.eval()
    rollout_duration = torch.tensor([min(2, int(item["durations"][0]))], dtype=torch.long)
    generated, predicted_duration = model.rollout(
        item["hidden"], item["embeddings"], rollout_duration
    )
    feature_marker = json.loads(
        (cfg.work_dir / "features" / "features.done.json").read_text(encoding="utf-8")
    )
    report = {
        "automatic_pass": bool(
            structural["automatic_pass"]
            and torch.isfinite(loss)
            and gradients_finite
            and generated.shape == (int(rollout_duration[0]), cfg.model.num_codebooks)
        ),
        "record_id": item["record_id"],
        "split": selected_split,
        "bounded_tokens": 1,
        "bounded_frames": int(item["codes"].shape[0]),
        "loss": float(loss.detach()),
        "codec_loss": float(parts["codec"]),
        "duration_loss": float(parts["duration"]),
        "loss_per_codebook": [float(value) for value in parts["per_book"]],
        "cb0_loss": float(parts["per_book"][0]),
        "gradient_norm": gradient_norm,
        "gradients_finite": gradients_finite,
        "rollout_frames": int(generated.shape[0]),
        "rollout_codebooks": int(generated.shape[1]),
        "predicted_duration": int(predicted_duration[0]),
        "peak_process_rss_mib": _peak_rss_mib(),
        "feature_preparation": {
            "device": feature_marker.get("preparation_device"),
            "seconds_rank0": feature_marker.get("preparation_seconds_rank0"),
            "peak_rss_mib_rank0": feature_marker.get("preparation_peak_rss_mib_rank0"),
        },
        "structural": structural,
    }
    atomic_json(cfg.work_dir / "local_smoke_report.json", report)
    return report
