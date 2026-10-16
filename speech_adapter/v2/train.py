from __future__ import annotations

import contextlib
import json
import math
import os
import random
import time
from pathlib import Path

import numpy as np
import torch
from torch import distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler

from ..config import atomic_json
from .checkpoint import load_checkpoint, save_checkpoint
from .config import V2Config
from .data import collate, make_dataset, move_batch
from .losses import acoustic_loss, cb0_loss
from .model import TwoStageTTS


def select_device(cfg: V2Config) -> torch.device:
    if cfg.train.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    use_cuda = cfg.train.device == "cuda" or (cfg.train.device == "auto" and torch.cuda.is_available())
    if use_cuda:
        local = int(os.environ.get("LOCAL_RANK", 0))
        torch.cuda.set_device(local)
        return torch.device("cuda", local)
    torch.set_num_threads(cfg.train.cpu_threads)
    if cfg.train.precision != "fp32":
        raise ValueError("CPU training uses fp32; AMP precision is supported on CUDA")
    return torch.device("cpu")


def loss_for(phase: str, outputs, batch: dict, cfg: V2Config):
    if phase == "cb0":
        return cb0_loss(*outputs, batch["codes"], batch["frame_mask"], cfg.train.stop_weight)
    return acoustic_loss(outputs, batch["codes"], batch["frame_mask"])


def amp_context(cfg: V2Config, device: torch.device):
    if device.type != "cuda" or cfg.train.precision == "fp32":
        return contextlib.nullcontext()
    dtype = torch.bfloat16 if cfg.train.precision == "bf16" else torch.float16
    if dtype == torch.bfloat16 and not torch.cuda.is_bf16_supported():
        raise ValueError("bf16 unsupported on this GPU; choose fp16 or fp32")
    return torch.autocast("cuda", dtype=dtype)


def all_ranks_finite(value: torch.Tensor) -> bool:
    flag = torch.isfinite(value).all().to(dtype=torch.int32)
    if dist.is_initialized():
        dist.all_reduce(flag, op=dist.ReduceOp.MIN)
    return bool(flag.item())


def optimizer_step(optimizer, scaler, parameters, max_norm: float) -> tuple[float, bool]:
    """Skip an AMP overflow on EVERY rank before any optimizer can update."""
    scaler.unscale_(optimizer)
    norm = torch.nn.utils.clip_grad_norm_(parameters, max_norm, error_if_nonfinite=False)
    finite = all_ranks_finite(norm)
    if not finite:
        optimizer.zero_grad(set_to_none=True)
        if not scaler.is_enabled():
            raise RuntimeError("non-finite gradient norm without FP16 scaling")
        # Explicit public update makes even a rank with finite local gradients
        # skip and back off. Never let ranks independently update their weights.
        scale = scaler.get_scale() * scaler.get_backoff_factor()
        if not math.isfinite(scale) or scale <= 0:
            raise RuntimeError("FP16 gradient scale became invalid")
        scaler.update(new_scale=scale)
        state = scaler.state_dict()
        state["_growth_tracker"] = 0
        scaler.load_state_dict(state)
        return float(norm), False
    scaler.step(optimizer)
    scaler.update()
    return float(norm), True


def record_context(batch: dict) -> list:
    local = batch["record_ids"]
    if not dist.is_initialized():
        return [local]
    records = [None] * dist.get_world_size()
    dist.all_gather_object(records, local)
    return records


def write_event(path: Path, event: dict) -> None:
    with path.open("a") as out:
        out.write(json.dumps(event, allow_nan=False) + "\n")
    print(json.dumps(event, allow_nan=False), flush=True)


@torch.inference_mode()
def evaluate(model: TwoStageTTS, dataset, cfg: V2Config, device: torch.device, limit: int | None = None) -> dict:
    was_training = model.training
    model.eval()
    totals, count, generated = {}, 0, []
    try:
        for index in range(min(len(dataset), limit or cfg.train.eval_samples)):
            batch = move_batch(collate([dataset[index]]), device)
            outputs = model(batch)
            loss, stats = loss_for(model.phase, outputs, batch, cfg)
            stats = {**stats, "loss": loss}
            frames = int(batch["frame_mask"].sum())
            for key, value in stats.items():
                totals[key] = totals.get(key, 0.0) + float(value) * frames
            count += frames
            # Both phases emit a bounded full inference result. For acoustic phase also
            # report oracle-CB0 accuracy separately from free-running generation.
            if model.phase == "acoustic":
                prediction = outputs.argmax(-1)
                truth = batch["codes"][:, 1:].transpose(1, 2)
                totals["oracle_cb0_acoustic_accuracy"] = totals.get("oracle_cb0_acoustic_accuracy", 0.0) + float((prediction == truth).float().mean()) * frames
            if index == 0:
                result = model.generate(batch["hidden"], batch["embeddings"], batch["token_mask"],
                    cfg.train.max_generate_frames, cfg.train.min_generate_frames, cfg.train.stop_threshold)
                generated.append({"record_id": batch["record_ids"][0], "frames": result.codes.shape[-1],
                                  "termination_reason": result.termination_reason})
        return {**{k: v / count for k, v in totals.items()}, "valid_frames": count,
                "generated": generated, "codec_audio_evaluated": False}
    finally:
        model.train(was_training)


def run_training(cfg: V2Config, phase: str, init: str = "", resume: str = "", limit: int = 0) -> dict:
    cfg.validate()
    if init and resume:
        raise ValueError("choose --init OR --resume")
    if phase == "acoustic" and not (init or resume):
        raise ValueError("Stage B requires --init with a trained Stage A checkpoint")
    device = select_device(cfg)
    rank, world = int(os.environ.get("RANK", 0)), int(os.environ.get("WORLD_SIZE", 1))
    if world > 1 and not dist.is_initialized():
        kwargs = {"device_id": device} if device.type == "cuda" else {}
        dist.init_process_group("nccl" if device.type == "cuda" else "gloo", **kwargs)
    random.seed(cfg.train.seed)
    np.random.seed(cfg.train.seed)
    torch.manual_seed(cfg.train.seed)
    payload = None
    if init or resume:
        model, old_cfg, payload = load_checkpoint(init or resume, device)
        if old_cfg.to_dict()["model"] != cfg.to_dict()["model"]:
            raise ValueError("checkpoint architecture differs from current config")
        if old_cfg.to_dict()["data"] != cfg.to_dict()["data"]:
            raise ValueError("checkpoint feature provenance/data config differs; use the same prepared dataset")
        if init and payload["phase"] != "cb0":
            raise ValueError("--init must identify a Stage A CB0 checkpoint")
        if resume and payload["phase"] != phase:
            raise ValueError("resume phase differs from checkpoint phase")
    else:
        model = TwoStageTTS(cfg.model).to(device)
    model.set_phase(phase)
    model.train()
    parameters = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(parameters,
                                 lr=cfg.train.learning_rate, weight_decay=cfg.train.weight_decay)
    start_epoch = 0
    if resume:
        if payload["optimizer"] is None:
            raise ValueError("resume checkpoint has no optimizer state")
        optimizer.load_state_dict(payload["optimizer"])
        for state in optimizer.state.values():
            for key, value in state.items():
                if torch.is_tensor(value):
                    state[key] = value.to(device)
        start_epoch = int(payload["epoch"]) + 1
        torch.set_rng_state(payload["torch_rng"])
    train_data, val_data = make_dataset(cfg, "train", limit), make_dataset(cfg, "val", limit)
    sampler = DistributedSampler(train_data, world, rank, shuffle=True, seed=cfg.train.seed) if world > 1 else None
    loader = DataLoader(train_data, batch_size=cfg.train.batch_size, shuffle=sampler is None,
                        sampler=sampler, num_workers=cfg.train.num_workers, collate_fn=collate)
    wrapped = DDP(model, device_ids=[device.index] if device.type == "cuda" else None) if world > 1 else model
    scaler = torch.amp.GradScaler("cuda", init_scale=cfg.train.amp_init_scale,
                                 enabled=device.type == "cuda" and cfg.train.precision == "fp16")
    if resume and payload["extra"].get("scaler"):
        scaler.load_state_dict(payload["extra"]["scaler"])
    work = Path(cfg.work_dir)
    history_path = work / f"{phase}_history.jsonl"
    steps_path = work / f"{phase}_steps.jsonl"
    if rank == 0:
        work.mkdir(parents=True, exist_ok=True)
        atomic_json(work / "config.json", cfg.to_dict())
        if not resume:
            history_path.write_text("")
            steps_path.write_text("")
    best = float(payload["extra"].get("best", float("inf"))) if resume else float("inf")
    global_step = int(payload["extra"].get("global_step", start_epoch * len(loader))) if resume else 0
    optimizer_steps = int(payload["extra"].get("optimizer_steps", global_step)) if resume else 0
    skipped_updates = int(payload["extra"].get("skipped_updates", 0)) if resume else 0
    consecutive_overflows = 0
    total_steps = cfg.train.epochs * len(loader)
    warmup = max(1, int(total_steps * cfg.train.warmup_fraction))
    final_report = {}
    try:
        for epoch in range(start_epoch, cfg.train.epochs):
            if sampler is not None:
                sampler.set_epoch(epoch)
            model.train()
            total, samples = 0.0, 0
            window_loss, window_samples, window_stats = 0.0, 0, {}
            window_started = time.perf_counter()
            for epoch_step, batch in enumerate(loader, 1):
                batch = move_batch(batch, device)
                progress = max(0.0, (global_step - warmup) / max(1, total_steps - warmup))
                factor = (global_step + 1) / warmup if global_step < warmup else 0.05 + 0.95 * (1 + math.cos(math.pi * progress)) / 2
                for group in optimizer.param_groups:
                    group["lr"] = cfg.train.learning_rate * factor
                optimizer.zero_grad(set_to_none=True)
                with amp_context(cfg, device):
                    outputs = wrapped(batch)
                    loss, stats = loss_for(phase, outputs, batch, cfg)
                if not all_ranks_finite(loss):
                    raise RuntimeError(f"non-finite {phase} loss; records_by_rank={record_context(batch)}")
                scaler.scale(loss).backward()
                try:
                    norm, updated = optimizer_step(optimizer, scaler, parameters, cfg.train.grad_clip)
                except RuntimeError as exc:
                    raise RuntimeError(f"{exc}; records_by_rank={record_context(batch)}") from exc
                optimizer_steps += int(updated)
                skipped_updates += int(not updated)
                consecutive_overflows = 0 if updated else consecutive_overflows + 1
                if not updated:
                    records = record_context(batch)
                    if rank == 0:
                        write_event(steps_path, {"event": "amp_overflow", "phase": phase,
                            "epoch": epoch, "step": global_step + 1, "epoch_step": epoch_step,
                            "update_skipped": True, "grad_scale": scaler.get_scale(),
                            "skipped_updates": skipped_updates, "records_by_rank": records})
                    if consecutive_overflows >= cfg.train.max_consecutive_overflows:
                        raise RuntimeError(f"persistent FP16 overflow ({consecutive_overflows} consecutive batches); "
                                           f"try precision=fp32; records_by_rank={records}")
                total += float(loss.detach()) * len(batch["record_ids"])
                samples += len(batch["record_ids"])
                global_step += 1
                size = len(batch["record_ids"])
                window_loss += float(loss.detach()) * size
                window_samples += size
                for key, value in stats.items():
                    window_stats[key] = window_stats.get(key, 0.0) + float(value) * size
                if cfg.train.step_size and (epoch_step % cfg.train.step_size == 0 or epoch_step == len(loader)):
                    keys = sorted(window_stats)
                    sums = torch.tensor([window_loss, window_samples] + [window_stats[k] for k in keys],
                                        dtype=torch.float64, device=device)
                    if world > 1:
                        dist.all_reduce(sums)
                    values = sums.tolist()
                    elapsed = time.perf_counter() - window_started
                    if rank == 0:
                        write_event(steps_path, {"event": "train_step", "phase": phase,
                            "epoch": epoch, "step": global_step, "epoch_step": epoch_step,
                            "steps_per_epoch": len(loader), "optimizer_steps": optimizer_steps,
                            "train_loss": values[0] / values[1],
                            "metrics": {k: values[i + 2] / values[1] for i, k in enumerate(keys)},
                            "learning_rate": optimizer.param_groups[0]["lr"],
                            "grad_norm": norm if math.isfinite(norm) else None,
                            "grad_scale": scaler.get_scale(), "update_skipped": not updated,
                            "skipped_updates": skipped_updates, "window_samples": int(values[1]),
                            "samples_per_second": values[1] / max(elapsed, 1e-9)})
                    window_loss, window_samples, window_stats = 0.0, 0, {}
                    window_started = time.perf_counter()
            if world > 1:
                sums = torch.tensor([total, samples], dtype=torch.float64, device=device)
                dist.all_reduce(sums)
                total, samples = sums.tolist()
            # Every rank evaluates the same small deterministic subset. No rank waits
            # inside a DDP forward while rank zero runs a different generation path.
            validation = evaluate(model, val_data, cfg, device)
            final_report = {"epoch": epoch, "phase": phase, "train_loss": total / samples,
                            "validation": validation, "step": global_step,
                            "optimizer_steps": optimizer_steps, "skipped_updates": skipped_updates}
            if rank == 0:
                with history_path.open("a") as out:
                    out.write(json.dumps(final_report) + "\n")
                improved = validation["loss"] < best
                best = min(best, validation["loss"])
                extra = {"best": best, "scaler": scaler.state_dict(), "global_step": global_step,
                         "optimizer_steps": optimizer_steps, "skipped_updates": skipped_updates}
                save_checkpoint(work / "checkpoints" / f"{phase}_last.pt", model, cfg, phase, epoch, optimizer, extra)
                if improved:
                    save_checkpoint(work / "checkpoints" / f"{phase}_best.pt", model, cfg, phase, epoch, optimizer, extra)
                print(json.dumps(final_report), flush=True)
            if world > 1:
                dist.barrier()
        return final_report
    finally:
        if world > 1 and dist.is_initialized():
            dist.destroy_process_group()

