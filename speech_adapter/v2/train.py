from __future__ import annotations

import contextlib
import json
import math
import os
import random
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
        dist.init_process_group("nccl" if device.type == "cuda" else "gloo")
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
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],
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
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda" and cfg.train.precision == "fp16")
    if resume and payload["extra"].get("scaler"):
        scaler.load_state_dict(payload["extra"]["scaler"])
    work = Path(cfg.work_dir)
    history_path = work / f"{phase}_history.jsonl"
    if rank == 0:
        work.mkdir(parents=True, exist_ok=True)
        atomic_json(work / "config.json", cfg.to_dict())
        if not resume:
            history_path.write_text("")
    best = float(payload["extra"].get("best", float("inf"))) if resume else float("inf")
    global_step = start_epoch * len(loader)
    total_steps = cfg.train.epochs * len(loader)
    warmup = max(1, int(total_steps * cfg.train.warmup_fraction))
    final_report = {}
    try:
        for epoch in range(start_epoch, cfg.train.epochs):
            if sampler is not None:
                sampler.set_epoch(epoch)
            model.train()
            total, samples = 0.0, 0
            for batch in loader:
                batch = move_batch(batch, device)
                progress = max(0.0, (global_step - warmup) / max(1, total_steps - warmup))
                factor = (global_step + 1) / warmup if global_step < warmup else 0.05 + 0.95 * (1 + math.cos(math.pi * progress)) / 2
                for group in optimizer.param_groups:
                    group["lr"] = cfg.train.learning_rate * factor
                optimizer.zero_grad(set_to_none=True)
                with amp_context(cfg, device):
                    outputs = wrapped(batch)
                    loss, _ = loss_for(phase, outputs, batch, cfg)
                if not torch.isfinite(loss):
                    raise RuntimeError(f"non-finite {phase} loss; records={batch['record_ids']}")
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                norm = torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], cfg.train.grad_clip)
                if not torch.isfinite(norm):
                    raise RuntimeError("non-finite gradient norm")
                scaler.step(optimizer)
                scaler.update()
                total += float(loss.detach()) * len(batch["record_ids"])
                samples += len(batch["record_ids"])
                global_step += 1
            if world > 1:
                sums = torch.tensor([total, samples], dtype=torch.float64, device=device)
                dist.all_reduce(sums)
                total, samples = sums.tolist()
            # Every rank evaluates the same small deterministic subset. No rank waits
            # inside a DDP forward while rank zero runs a different generation path.
            validation = evaluate(model, val_data, cfg, device)
            final_report = {"epoch": epoch, "phase": phase, "train_loss": total / samples,
                            "validation": validation}
            if rank == 0:
                with history_path.open("a") as out:
                    out.write(json.dumps(final_report) + "\n")
                improved = validation["loss"] < best
                best = min(best, validation["loss"])
                extra = {"best": best, "scaler": scaler.state_dict()}
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
