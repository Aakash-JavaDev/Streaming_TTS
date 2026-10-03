from __future__ import annotations

import json
import math
import os
import time
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


def run_smoke(cfg: AppConfig) -> dict[str, Any]:
    """Exercise real prepared data on the configured CPU or CUDA device."""
    import torch

    from .data import SpeechDataset, collate_speech
    from .diagnostics import model_structural_checks
    from .model import adapter_loss
    from .train import build_model, move_batch

    requested_device = cfg.runtime.device
    if requested_device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("smoke requires CUDA, but torch.cuda.is_available() is false")
    use_cuda = requested_device == "cuda" or (
        requested_device == "auto" and torch.cuda.is_available()
    )
    if use_cuda:
        local_rank = int(os.environ.get("LOCAL_RANK", "0"))
        if local_rank >= torch.cuda.device_count():
            raise RuntimeError(
                f"LOCAL_RANK={local_rank} but only {torch.cuda.device_count()} CUDA devices exist"
            )
        device = torch.device(f"cuda:{local_rank}")
        torch.cuda.set_device(device)
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)
    else:
        device = torch.device("cpu")
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
    batch = move_batch(collate_speech([item]), device)
    model = build_model(cfg, device)
    model.train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.train.learning_rate)
    scaler = torch.amp.GradScaler("cuda", enabled=use_cuda)
    optimizer.zero_grad(set_to_none=True)
    started = time.perf_counter()
    with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=use_cuda):
        logits, log_duration = model(batch)
        loss, parts = adapter_loss(
            logits,
            log_duration,
            batch,
            cfg.train.cb0_weight,
            cfg.train.residual_weight,
            cfg.train.duration_weight,
        )
    scaler.scale(loss).backward()
    scaler.unscale_(optimizer)
    gradients = [parameter.grad for parameter in model.parameters() if parameter.grad is not None]
    gradients_finite = bool(gradients) and all(
        bool(torch.isfinite(gradient).all()) for gradient in gradients
    )
    gradient_norm = math.sqrt(
        sum(float(gradient.detach().float().square().sum().item()) for gradient in gradients)
    )
    scaler.step(optimizer)
    scaler.update()

    model.eval()
    rollout_duration = torch.tensor(
        [min(2, int(item["durations"][0]))], dtype=torch.long, device=device
    )
    generated, predicted_duration = model.rollout(
        batch["hidden"][0, :1], batch["embeddings"][0, :1], rollout_duration
    )
    if use_cuda:
        torch.cuda.synchronize(device)
    elapsed = time.perf_counter() - started
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
        "requested_device": requested_device,
        "device": str(device),
        "device_type": device.type,
        "amp_enabled": use_cuda,
        "elapsed_seconds": elapsed,
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
        "cuda": {
            "available": torch.cuda.is_available(),
            "device_count": torch.cuda.device_count(),
            "name": torch.cuda.get_device_name(device) if use_cuda else None,
            "capability": list(torch.cuda.get_device_capability(device)) if use_cuda else None,
            "peak_allocated_mib": (
                torch.cuda.max_memory_allocated(device) / 2**20 if use_cuda else 0.0
            ),
            "peak_reserved_mib": (
                torch.cuda.max_memory_reserved(device) / 2**20 if use_cuda else 0.0
            ),
        },
        "feature_preparation": {
            "device": feature_marker.get("preparation_device"),
            "seconds_rank0": feature_marker.get("preparation_seconds_rank0"),
            "peak_rss_mib_rank0": feature_marker.get("preparation_peak_rss_mib_rank0"),
        },
        "structural": structural,
    }
    atomic_json(cfg.work_dir / "smoke_report.json", report)
    atomic_json(cfg.work_dir / f"{device.type}_smoke_report.json", report)
    if device.type == "cpu":
        atomic_json(cfg.work_dir / "local_smoke_report.json", report)
    return report


def run_local_smoke(cfg: AppConfig) -> dict[str, Any]:
    """Backward-compatible entry point; device selection now follows the config."""
    return run_smoke(cfg)
