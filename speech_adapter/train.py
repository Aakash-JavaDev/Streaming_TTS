from __future__ import annotations

import contextlib
import datetime as dt
import json
import math
import os
import random
import re
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.distributed as dist
from safetensors.torch import load_file, save_file
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader

from .config import AppConfig, atomic_json
from .data import (
    DistributedFrameBatchSampler,
    SpeechDataset,
    collate_speech,
)
from .metrics import (
    CodebookMetrics,
    sequence_accuracy,
    training_health_signals,
    write_epoch_signal_csv,
)
from .model import StreamingSpeechAdapter, adapter_loss


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def distributed(cfg: AppConfig) -> tuple[int, int, int]:
    rank = int(os.environ.get("RANK", "0"))
    world = int(os.environ.get("WORLD_SIZE", "1"))
    local = int(os.environ.get("LOCAL_RANK", rank))
    if world > 1 and not dist.is_initialized():
        use_cuda = cfg.runtime.device == "cuda" or (
            cfg.runtime.device == "auto" and torch.cuda.is_available()
        )
        dist.init_process_group("nccl" if use_cuda else "gloo", timeout=dt.timedelta(hours=2))
    return rank, world, local


def move_batch(batch: dict[str, Any], device: torch.device) -> dict[str, Any]:
    return {
        key: value.to(device, non_blocking=True) if torch.is_tensor(value) else value
        for key, value in batch.items()
    }


def build_model(cfg: AppConfig, device: torch.device) -> StreamingSpeechAdapter:
    marker = json.loads((cfg.work_dir / "features" / "features.done.json").read_text())
    embeddings = np.load(cfg.work_dir / "features" / "embedding_values.npy", mmap_mode="r")
    model = StreamingSpeechAdapter(
        int(marker["hidden_dim"]), int(embeddings.shape[1]), cfg.model
    ).to(device)
    model.assert_budget()
    return model


def _candidate_batch(dataset: SpeechDataset, frame_budget: int, maximum: int) -> dict[str, Any]:
    chosen = []
    total = 0
    for index in sorted(range(len(dataset)), key=dataset.lengths.__getitem__, reverse=True):
        length = dataset.lengths[index]
        if length > frame_budget:
            continue
        if chosen and (total + length > frame_budget or len(chosen) >= maximum):
            continue
        chosen.append(dataset[index])
        total += length
        if total >= int(frame_budget * 0.8) or len(chosen) >= maximum:
            break
    if not chosen:
        longest = max(dataset.lengths)
        raise RuntimeError(f"frame budget {frame_budget} is below every sample; longest={longest}")
    return collate_speech(chosen)


def calibrate_budget(
    model: StreamingSpeechAdapter,
    dataset: SpeechDataset,
    cfg: AppConfig,
    device: torch.device,
) -> int:
    candidate = int(cfg.train.max_frames_per_gpu)
    attempts = 0
    while True:
        attempts += 1
        model.zero_grad(set_to_none=True)
        try:
            batch = move_batch(_candidate_batch(dataset, candidate, cfg.train.max_batch_size), device)
            with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=device.type == "cuda"):
                logits, duration = model(batch)
                loss, _ = adapter_loss(
                    logits,
                    duration,
                    batch,
                    cfg.train.cb0_weight,
                    cfg.train.residual_weight,
                    cfg.train.duration_weight,
                )
            loss.backward()
            model.zero_grad(set_to_none=True)
            del batch, logits, duration, loss
            if device.type == "cuda":
                torch.cuda.empty_cache()
            break
        except torch.cuda.OutOfMemoryError:
            model.zero_grad(set_to_none=True)
            torch.cuda.empty_cache()
            if attempts >= 2:
                raise RuntimeError(
                    f"calibration OOM after retry; set max_frames_per_gpu below {candidate}"
                )
            candidate = max(64, candidate // 2)
            print(f"CUDA OOM during calibration; retrying with {candidate} frames/GPU", flush=True)
    if dist.is_initialized():
        tensor = torch.tensor(candidate, device=device, dtype=torch.long)
        dist.all_reduce(tensor, op=dist.ReduceOp.MIN)
        candidate = int(tensor.item())
    return candidate


def _atomic_weights(model: StreamingSpeechAdapter, path: Path) -> None:
    target = model.module if isinstance(model, DistributedDataParallel) else model
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    save_file({key: value.detach().cpu() for key, value in target.state_dict().items()}, str(temporary))
    temporary.replace(path)


def save_checkpoint(
    model: StreamingSpeechAdapter,
    optimizer: torch.optim.Optimizer,
    scheduler: Any,
    scaler: torch.amp.GradScaler,
    epoch: int,
    best: float,
    cfg: AppConfig,
    name: str,
    run_dir: Path,
) -> None:
    directory = run_dir / "checkpoints"
    directory.mkdir(parents=True, exist_ok=True)
    _atomic_weights(model, directory / f"{name}.safetensors")
    state = {
        "epoch": epoch,
        "best_rollout_cb0_ce": best,
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
        "scaler": scaler.state_dict(),
        "python_rng": random.getstate(),
        "numpy_rng": np.random.get_state(),
        "torch_rng": torch.get_rng_state(),
        "cuda_rng": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
        "config_fingerprint": cfg.fingerprint(),
        "feature_fingerprint": cfg.feature_fingerprint(),
    }
    temporary = directory / f"{name}.state.pt.tmp"
    torch.save(state, temporary)
    temporary.replace(directory / f"{name}.state.pt")


def maybe_resume(
    model: StreamingSpeechAdapter,
    optimizer: torch.optim.Optimizer,
    scheduler: Any,
    scaler: torch.amp.GradScaler,
    cfg: AppConfig,
    device: torch.device,
    run_dir: Path,
) -> tuple[int, float]:
    directory = run_dir / "checkpoints"
    weights = directory / "last.safetensors"
    state_path = directory / "last.state.pt"
    if not cfg.train.resume or not weights.is_file() or not state_path.is_file():
        return 0, math.inf
    state = torch.load(state_path, map_location="cpu", weights_only=False)
    if state.get("config_fingerprint") != cfg.fingerprint():
        raise RuntimeError("checkpoint configuration differs from the current configuration")
    if state.get("feature_fingerprint") != cfg.feature_fingerprint():
        raise RuntimeError("checkpoint feature dataset differs from current sidecars")
    model.load_state_dict(load_file(str(weights), device=str(device)))
    optimizer.load_state_dict(state["optimizer"])
    scheduler.load_state_dict(state["scheduler"])
    scaler.load_state_dict(state["scaler"])
    random.setstate(state["python_rng"])
    np.random.set_state(state["numpy_rng"])
    torch.set_rng_state(state["torch_rng"])
    if torch.cuda.is_available() and state["cuda_rng"]:
        torch.cuda.set_rng_state_all(state["cuda_rng"])
    return int(state["epoch"]), float(state["best_rollout_cb0_ce"])


@torch.no_grad()
def evaluate(
    model: StreamingSpeechAdapter,
    dataset: SpeechDataset,
    device: torch.device,
    limit: int,
    text_mode: str = "correct",
) -> dict[str, Any]:
    model.eval()
    teacher = CodebookMetrics(model.cfg.num_codebooks, model.cfg.codec_vocab)
    rollout = CodebookMetrics(model.cfg.num_codebooks, model.cfg.codec_vocab)
    rollout_sequences = []
    duration_abs = []
    duration_ape = []
    oracle_histogram = np.zeros((model.cfg.num_codebooks, model.cfg.codec_vocab), dtype=np.int64)
    count = min(len(dataset), limit if limit > 0 else len(dataset))
    for index in range(count):
        item = dataset[index]
        if text_mode == "zeroed":
            item = dict(item)
            item["hidden"] = torch.zeros_like(item["hidden"])
            item["embeddings"] = torch.zeros_like(item["embeddings"])
        elif text_mode == "swapped":
            donor = dataset[(index + 1) % len(dataset)]
            if donor["hidden"].shape[0] != item["hidden"].shape[0]:
                slots = torch.arange(item["hidden"].shape[0]) % donor["hidden"].shape[0]
                hidden, embeddings = donor["hidden"][slots], donor["embeddings"][slots]
            else:
                hidden, embeddings = donor["hidden"], donor["embeddings"]
            item = dict(item, hidden=hidden, embeddings=embeddings)
        batch = move_batch(collate_speech([item]), device)
        logits, log_duration = model(batch)
        teacher.update(logits, batch["codes"], batch["frame_mask"])
        generated, predicted_duration, rollout_logits = model.rollout_with_logits(
            batch["hidden"][0, batch["token_mask"][0]],
            batch["embeddings"][0, batch["token_mask"][0]],
            batch["durations"][0, batch["token_mask"][0]],
        )
        target = batch["codes"][0, batch["frame_mask"][0]]
        target_np = target.detach().cpu().numpy()
        for book in range(model.cfg.num_codebooks):
            oracle_histogram[book] += np.bincount(
                target_np[:, book], minlength=model.cfg.codec_vocab
            )
        rollout.update(
            rollout_logits.unsqueeze(0), target[: rollout_logits.shape[0]].unsqueeze(0),
            torch.ones(1, rollout_logits.shape[0], dtype=torch.bool, device=device),
        )
        rollout_sequences.append(
            {"record_id": item["record_id"], **sequence_accuracy(generated, target)}
        )
        true_duration = batch["durations"][0, batch["token_mask"][0]].float()
        difference = (predicted_duration.float() - true_duration).abs()
        duration_abs.extend(difference.cpu().tolist())
        duration_ape.extend((difference / true_duration.clamp(min=1)).cpu().tolist())
    teacher_rows, rollout_rows = teacher.report(), rollout.report()
    oracle_usage = [int(np.count_nonzero(row)) for row in oracle_histogram]
    oracle_entropy = []
    for row in oracle_histogram:
        probabilities = row / max(1, int(row.sum()))
        nonzero = probabilities[probabilities > 0]
        oracle_entropy.append(float(-(nonzero * np.log(nonzero)).sum()))
    return {
        "text_mode": text_mode,
        "samples": count,
        "teacher_forced": teacher_rows,
        "rollout": rollout_rows,
        "rollout_sequences": rollout_sequences,
        "rollout_cb0_ce": rollout_rows[0]["cross_entropy"],
        "duration_mae_frames": float(np.mean(duration_abs)) if duration_abs else None,
        "duration_median_ape": float(np.median(duration_ape)) if duration_ape else None,
        "oracle_codes_used": oracle_usage,
        "oracle_entropy": oracle_entropy,
    }


@torch.no_grad()
def write_epoch_validation_audio(
    model: StreamingSpeechAdapter,
    dataset: SpeechDataset,
    device: torch.device,
    cfg: AppConfig,
    run_dir: Path,
    epoch: int,
) -> dict[str, Any]:
    """Decode fixed validation examples after an epoch using the exact 8-CB codec."""
    from .diagnostics import OnnxMimi, _write_wav

    model.eval()
    decoder = OnnxMimi(
        cfg.paths.mimi_encoder,
        cfg.paths.mimi_decoder,
        num_codebooks=model.cfg.num_codebooks,
    )
    output_dir = run_dir / "epoch_validation" / f"epoch_{epoch:04d}"
    output_dir.mkdir(parents=True, exist_ok=True)
    count = min(len(dataset), cfg.train.eval_audio_samples)
    samples: list[dict[str, Any]] = []
    generation_warnings: list[str] = []
    for index in range(count):
        item = dataset[index]
        batch = move_batch(collate_speech([item]), device)
        logits, _ = model(batch)
        frames = int(batch["frame_mask"][0].sum().item())
        oracle = batch["codes"][0, :frames]
        teacher = logits[0, :frames].argmax(dim=-1)
        hidden = batch["hidden"][0, batch["token_mask"][0]]
        embeddings = batch["embeddings"][0, batch["token_mask"][0]]
        oracle_duration = batch["durations"][0, batch["token_mask"][0]]
        autoregressive, predicted_duration, _ = model.rollout_with_logits(
            hidden, embeddings, oracle_duration
        )
        fully_autoregressive, _, _ = model.rollout_with_logits(hidden, embeddings, None)

        safe_id = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(item["record_id"]))[:80]
        sample_dir = output_dir / f"sample_{index:03d}_{safe_id or 'record'}"
        modes = {
            "oracle": oracle,
            "teacher_forced": teacher,
            "autoregressive_oracle_duration": autoregressive,
            "autoregressive_predicted_duration": fully_autoregressive,
        }
        files: dict[str, str] = {}
        for name, codes in modes.items():
            if codes.ndim != 2 or codes.shape[1] != model.cfg.num_codebooks:
                raise RuntimeError(
                    f"epoch {epoch} {name} produced invalid Mimi shape {tuple(codes.shape)}"
                )
            if codes.shape[0] == 0:
                warning = (
                    f"{item['record_id']} {name}: duration head generated zero audio frames"
                )
                generation_warnings.append(warning)
                files[name] = ""
                continue
            path = sample_dir / f"{name}.wav"
            _write_wav(path, decoder.decode(codes.detach().cpu().numpy()))
            files[name] = str(path.relative_to(run_dir))
        samples.append(
            {
                "record_id": item["record_id"],
                "text": item["text"],
                "target_frames": frames,
                "predicted_duration_frames": int(predicted_duration.sum().item()),
                "teacher_forced_accuracy": sequence_accuracy(teacher, oracle),
                "autoregressive_accuracy": sequence_accuracy(autoregressive, oracle),
                "files": files,
            }
        )
    report = {
        "automatic_pass": len(samples) == count and count > 0 and not generation_warnings,
        "epoch": epoch,
        "samples": samples,
        "num_codebooks": model.cfg.num_codebooks,
        "decoder": cfg.paths.mimi_decoder,
        "generation_warnings": generation_warnings,
        "teacher_forced_definition": "parallel forward pass conditioned on oracle code history",
        "autoregressive_definition": "generated code history; oracle token durations",
        "fully_autoregressive_definition": "generated code history and predicted token durations",
    }
    atomic_json(output_dir / "report.json", report)
    atomic_json(run_dir / "latest_epoch_audio.json", report)
    return report


def require_qualification(cfg: AppConfig) -> None:
    path = cfg.work_dir / "gates" / "gate_report.json"
    if not path.is_file():
        raise RuntimeError("full training requires gates/gate_report.json; run qualification first")
    report = json.loads(path.read_text())
    if report.get("feature_fingerprint") != cfg.feature_fingerprint():
        raise RuntimeError("qualification report belongs to different features")
    for gate in "ABCDEF":
        if report.get("gates", {}).get(gate, {}).get("status") != "passed":
            raise RuntimeError(f"Gate {gate} has not passed")
    context_path = cfg.work_dir / "gates" / "context_selection.json"
    if not context_path.is_file():
        raise RuntimeError("full training requires gates/context_selection.json")
    selection = json.loads(context_path.read_text())
    if int(selection["selected_context_size"]) != cfg.model.text_context_size:
        raise RuntimeError(
            "configured text_context_size does not match the qualified context selection"
        )
    device_path = cfg.work_dir / "gates" / "device_report.json"
    if not device_path.is_file() or not json.loads(device_path.read_text()).get("automatic_pass"):
        raise RuntimeError("full training requires a passing imported Android device report")
    export_path = cfg.work_dir / "gates" / "edge_export" / "export_report.json"
    if not export_path.is_file():
        raise RuntimeError("full training requires the qualified incremental ONNX export")
    export_report = json.loads(export_path.read_text())
    if not export_report.get("pytorch_onnx_pass") or not export_report.get("int8_pass"):
        raise RuntimeError("qualified ONNX or INT8 parity did not pass")


def run_training(
    cfg: AppConfig,
    limit: int = 0,
    epochs: int | None = None,
    experiment: str = "full",
    init_weights: str = "",
) -> dict[str, Any]:
    rank, world, local = distributed(cfg)
    if experiment == "full" and cfg.run_stage == "train_full":
        require_qualification(cfg)
    run_dir = cfg.work_dir if experiment == "full" else cfg.work_dir / "experiments" / experiment
    run_dir.mkdir(parents=True, exist_ok=True)
    seed_everything(cfg.train.seed + rank)
    use_cuda = cfg.runtime.device == "cuda" or (
        cfg.runtime.device == "auto" and torch.cuda.is_available()
    )
    if cfg.runtime.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("runtime.device='cuda' but CUDA is not available")
    device = torch.device(f"cuda:{local}" if use_cuda else "cpu")
    if device.type == "cuda":
        torch.cuda.set_device(device)
        torch.backends.cuda.matmul.allow_tf32 = True
    else:
        torch.set_num_threads(cfg.runtime.cpu_threads)
    train_data = SpeechDataset(cfg, "train", limit=limit)
    val_limit = max(2, min(cfg.train.eval_samples, limit // 10 if limit else cfg.train.eval_samples))
    val_data = SpeechDataset(cfg, "val", limit=val_limit)
    model = build_model(cfg, device)
    if init_weights and not (run_dir / "checkpoints" / "last.safetensors").is_file():
        model.load_state_dict(load_file(init_weights, device=str(device)))
        if rank == 0:
            print(f"initialized {experiment} from {init_weights}", flush=True)
    safe_budget = calibrate_budget(model, train_data, cfg, device)
    sampler = DistributedFrameBatchSampler(
        train_data.lengths,
        safe_budget,
        cfg.train.max_batch_size,
        rank,
        world,
        cfg.train.seed,
        True,
    )
    loader = DataLoader(
        train_data,
        batch_sampler=sampler,
        collate_fn=collate_speech,
        num_workers=cfg.train.num_workers,
        pin_memory=device.type == "cuda",
        persistent_workers=cfg.train.num_workers > 0,
    )
    if world > 1:
        model = DistributedDataParallel(model, device_ids=[local], broadcast_buffers=False)
    base_model = model.module if isinstance(model, DistributedDataParallel) else model
    qualification = experiment != "full"
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=cfg.train.learning_rate,
        weight_decay=0.0 if qualification else cfg.train.weight_decay,
    )
    run_epochs = int(epochs or cfg.train.epochs)
    updates_per_epoch = max(1, math.ceil(len(loader) / cfg.train.grad_accum_steps))
    total_updates = max(1, updates_per_epoch * run_epochs)
    warmup = max(1, int(total_updates * cfg.train.warmup_fraction))

    def lr_factor(step: int) -> float:
        if step < warmup:
            return max(1.0e-3, step / warmup)
        progress = (step - warmup) / max(1, total_updates - warmup)
        cosine = 0.5 * (1.0 + math.cos(math.pi * min(1.0, progress)))
        return cfg.train.min_lr_ratio + (1.0 - cfg.train.min_lr_ratio) * cosine

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_factor)
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")
    start_epoch, best = maybe_resume(
        base_model, optimizer, scheduler, scaler, cfg, device, run_dir
    )
    training_path = run_dir / "training.json"
    if start_epoch and training_path.is_file():
        history = json.loads(training_path.read_text()).get("history", [])
    else:
        history = []
    validation: dict[str, Any] | None = None
    log_path = run_dir / "train_steps.jsonl"
    for epoch in range(start_epoch, run_epochs):
        sampler.set_epoch(epoch)
        model.train()
        optimizer.zero_grad(set_to_none=True)
        started = time.perf_counter()
        seen_frames = 0
        running = 0.0
        running_codebook_loss = np.zeros(cfg.model.num_codebooks, dtype=np.float64)
        running_codebook_frames = 0
        for step, raw_batch in enumerate(loader, 1):
            batch = move_batch(raw_batch, device)
            final_accum = step % cfg.train.grad_accum_steps == 0 or step == len(loader)
            sync = contextlib.nullcontext()
            if isinstance(model, DistributedDataParallel) and not final_accum:
                sync = model.no_sync()
            with sync, torch.autocast(
                device_type="cuda", dtype=torch.float16, enabled=device.type == "cuda"
            ):
                logits, duration = model(batch)
                loss, parts = adapter_loss(
                    logits,
                    duration,
                    batch,
                    cfg.train.cb0_weight,
                    cfg.train.residual_weight,
                    cfg.train.duration_weight,
                )
                loss = loss / cfg.train.grad_accum_steps
            if not torch.isfinite(loss):
                raise FloatingPointError(f"non-finite loss at epoch={epoch + 1} step={step}")
            scaler.scale(loss).backward()
            running += float(loss.item()) * cfg.train.grad_accum_steps
            batch_frames = int(batch["frame_mask"].sum().item())
            seen_frames += batch_frames
            running_codebook_loss += (
                parts["per_book"].detach().float().cpu().numpy() * batch_frames
            )
            running_codebook_frames += batch_frames
            if final_accum:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.train.grad_clip)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
                scheduler.step()
            if rank == 0 and (step == 1 or step % cfg.train.log_every == 0 or step == len(loader)):
                elapsed = max(time.perf_counter() - started, 1e-6)
                message = {
                    "epoch": epoch + 1,
                    "step": step,
                    "steps": len(loader),
                    "loss": running / step,
                    "codec_loss": float(parts["codec"].item()),
                    "batch_loss_per_codebook": [
                        float(value) for value in parts["per_book"].cpu()
                    ],
                    "batch_cb0_loss": float(parts["per_book"][0].item()),
                    "duration_loss": float(parts["duration"].item()),
                    "learning_rate": scheduler.get_last_lr()[0],
                    "frames_per_second_rank0": seen_frames / elapsed,
                    "peak_vram_gib": torch.cuda.max_memory_allocated(device) / 2**30 if device.type == "cuda" else 0,
                }
                print(json.dumps(message), flush=True)
                with log_path.open("a", encoding="utf-8") as handle:
                    handle.write(json.dumps(message) + "\n")
        epoch_codebook_sums = torch.tensor(
            running_codebook_loss, device=device, dtype=torch.float64
        )
        epoch_codebook_frames = torch.tensor(
            running_codebook_frames, device=device, dtype=torch.float64
        )
        if dist.is_initialized():
            dist.all_reduce(epoch_codebook_sums, op=dist.ReduceOp.SUM)
            dist.all_reduce(epoch_codebook_frames, op=dist.ReduceOp.SUM)
            dist.barrier()
        train_codebook_loss = (
            epoch_codebook_sums / epoch_codebook_frames.clamp(min=1)
        ).cpu().tolist()
        validation = None
        if rank == 0:
            score_data = train_data if experiment.startswith(("C_", "D_")) else val_data
            validation = evaluate(base_model, score_data, device, cfg.train.eval_samples)
            audio_report = write_epoch_validation_audio(
                base_model, score_data, device, cfg, run_dir, epoch + 1
            )
            score = float(validation["rollout_cb0_ce"])
            previous_validation = history[-1].get("validation") if history else None
            signals = training_health_signals(
                validation,
                previous_validation=previous_validation,
                vocabulary=cfg.model.codec_vocab,
            )
            atomic_json(
                run_dir / "epoch_validation" / f"epoch_{epoch + 1:04d}" / "metrics.json",
                {
                    "epoch": epoch + 1,
                    "validation": validation,
                    "signals": signals,
                    "audio": audio_report,
                },
            )
            record = {
                "epoch": epoch + 1,
                "train_loss": running / max(1, len(loader)),
                "train_loss_per_codebook": train_codebook_loss,
                "train_cb0_loss": train_codebook_loss[0],
                "validation": validation,
                "validation_audio": {
                    "automatic_pass": audio_report["automatic_pass"],
                    "samples": len(audio_report["samples"]),
                    "report": str(
                        Path("epoch_validation") / f"epoch_{epoch + 1:04d}" / "report.json"
                    ),
                },
                "signals": signals,
                "safe_frames_per_gpu": safe_budget,
            }
            history.append(record)
            compact_books = [
                {
                    "cb": row["codebook"],
                    "signal": row["signal"],
                    "tf_ce": round(row["teacher_ce"], 4),
                    "tf_top1": round(row["teacher_top1"], 4),
                    "ar_ce": round(row["ar_ce"], 4),
                    "ar_top1": round(row["ar_top1"], 4),
                    "gap": round(row["ar_teacher_ce_gap"], 4),
                    "ar_trend": row["ar_trend"],
                    "usage_ratio": round(row["code_usage_ratio"], 3),
                }
                for row in signals["codebooks"]
            ]
            print(
                json.dumps(
                    {
                        "epoch_health": epoch + 1,
                        "signal": signals["overall_signal"],
                        "headline": signals["headline"],
                        "warnings": signals["warnings"],
                        "codebooks": compact_books,
                        "duration_median_ape": signals["duration_median_ape"],
                        "validation_audio_pass": audio_report["automatic_pass"],
                        "validation_audio_warnings": audio_report["generation_warnings"],
                        "validation_audio_report": str(
                            Path("epoch_validation") / f"epoch_{epoch + 1:04d}" / "report.json"
                        ),
                    }
                ),
                flush=True,
            )
            atomic_json(run_dir / "latest_training_signal.json", signals)
            write_epoch_signal_csv(run_dir / "epoch_codebook_signals.csv", history)
            save_checkpoint(
                base_model, optimizer, scheduler, scaler, epoch + 1,
                min(best, score), cfg, "last", run_dir,
            )
            if score < best:
                best = score
                save_checkpoint(
                    base_model, optimizer, scheduler, scaler, epoch + 1,
                    best, cfg, "best", run_dir,
                )
            atomic_json(
                run_dir / "training.json",
                {
                    "complete": epoch + 1 == run_epochs,
                    "config_fingerprint": cfg.fingerprint(),
                    "feature_fingerprint": cfg.feature_fingerprint(),
                    "world_size": world,
                    "parameters": base_model.parameter_count(),
                    "best_rollout_cb0_ce": best,
                    "history": history,
                },
            )
        if dist.is_initialized():
            dist.barrier()
    if rank == 0 and experiment != "full":
        best_path = run_dir / "checkpoints" / "best.safetensors"
        if best_path.is_file():
            base_model.load_state_dict(load_file(str(best_path), device=str(device)))
            validation = None
        if validation is None:
            score_data = train_data if experiment.startswith(("C_", "D_")) else val_data
            validation = evaluate(base_model, score_data, device, cfg.train.eval_samples)
        controls = None
        if experiment.startswith(("D_", "E_")):
            score_data = train_data if experiment.startswith("D_") else val_data
            controls = {
                mode: evaluate(base_model, score_data, device, cfg.train.eval_samples, text_mode=mode)
                for mode in ("zeroed", "swapped")
            }
        if experiment.startswith("smoke_"):
            atomic_json(
                run_dir / "smoke_metrics.json",
                {"automatic_pass": True, "experiment": experiment, "metrics": validation},
            )
        else:
            _write_experiment_gate(cfg, experiment, validation, run_dir, controls)
    if dist.is_initialized():
        dist.destroy_process_group()
    return {"best_rollout_cb0_ce": best, "epochs": run_epochs} if rank == 0 else {}


def _write_experiment_gate(
    cfg: AppConfig,
    experiment: str,
    result: dict[str, Any],
    run_dir: Path,
    controls: dict[str, dict[str, Any]] | None,
) -> None:
    from .diagnostics import update_gate

    gate = experiment[:1].upper()
    automatic = False
    details: dict[str, Any] = {
        "experiment": experiment, "metrics": result, "controls": controls
    }
    if gate == "C":
        teacher_ce = float(result["teacher_forced"][0]["cross_entropy"])
        accuracies = [
            float(row["per_codebook_accuracy"][0])
            for row in result["rollout_sequences"]
        ]
        rollout_accuracy = float(np.mean(accuracies)) if accuracies else 0.0
        details.update({"teacher_cb0_ce": teacher_ce, "rollout_cb0_accuracy": rollout_accuracy})
        automatic = (
            teacher_ce <= cfg.gates.overfit_teacher_forced_ce
            and rollout_accuracy >= cfg.gates.overfit_rollout_cb0_accuracy
        )
    elif gate == "D":
        correct = float(result["rollout_cb0_ce"])
        control_pass = bool(controls) and all(
            correct <= float(controls[mode]["rollout_cb0_ce"]) * (1.0 - cfg.gates.text_control_margin)
            for mode in ("zeroed", "swapped")
        )
        training_record = json.loads((run_dir / "training.json").read_text())
        history = training_record.get("history", [])
        gaps = []
        for item in history:
            val = item["validation"]
            gaps.append(
                float(val["rollout"][0]["cross_entropy"])
                - float(val["teacher_forced"][0]["cross_entropy"])
            )
        shrinking = len(gaps) < 2 or gaps[-1] <= gaps[0]
        automatic = result["samples"] >= 32 and control_pass and shrinking and all(
            row["per_codebook_accuracy"][0] >= 0.95
            for row in result["rollout_sequences"]
        )
    elif gate == "E":
        cb0 = result["rollout"][0]
        correct = float(result["rollout_cb0_ce"])
        control_pass = bool(controls) and all(
            correct <= float(controls[mode]["rollout_cb0_ce"]) * (1.0 - cfg.gates.text_control_margin)
            for mode in ("zeroed", "swapped")
        )
        usage_ratio = cb0["codes_used"] / max(1, result["oracle_codes_used"][0])
        entropy_ratio = cb0["predicted_entropy"] / max(1e-9, result["oracle_entropy"][0])
        automatic = (
            result["samples"] >= min(20, cfg.train.eval_samples)
            and control_pass
            and usage_ratio >= cfg.gates.min_code_usage_ratio
            and entropy_ratio >= cfg.gates.min_entropy_ratio
        )
    elif gate == "F":
        automatic = (
            result.get("duration_median_ape") is not None
            and result["duration_median_ape"] <= cfg.gates.max_duration_median_ape
        )
    atomic_json(run_dir / "gate_metrics.json", {"automatic_pass": automatic, **details})
    update_gate(cfg, gate, automatic, None, note=f"automatic results: {run_dir / 'gate_metrics.json'}")
