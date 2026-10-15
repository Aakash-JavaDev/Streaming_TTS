"""Self-contained dummy mechanics/learning check; never claims real speech quality."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch

from .checkpoint import load_checkpoint, save_checkpoint
from .config import Architecture, Training, V2Config
from .data import collate
from .losses import acoustic_loss, cb0_loss
from .model import TwoStageTTS


def tiny_config() -> V2Config:
    return V2Config(model=Architecture(hidden_dim=16, embedding_dim=16, width=32, heads=4,
        ffn_dim=64, cb0_layers=1, acoustic_layers=1, code_embedding_dim=8, codec_vocab=16,
        dropout=0, max_audio_frames=32, max_text_tokens=32),
        train=Training(max_generate_frames=12, batch_size=4))


def dummy_items(count: int = 4, seed: int = 42) -> list[dict]:
    rng = torch.Generator().manual_seed(seed)
    items = []
    for i in range(count):
        cb0 = (torch.arange(6) + 3 * i) % 16
        codes = torch.stack([(cb0 + k + (i if k else 0)) % 16 for k in range(8)])
        items.append({"record_id": f"dummy-{seed}-{i}", "text": "synthetic tensor fixture",
            "hidden": torch.randn(3, 16, generator=rng),
            "embeddings": torch.randn(3, 16, generator=rng), "codes": codes})
    return items


def run_smoke(output_dir: str, steps: int = 160) -> dict:
    if steps < 1:
        raise ValueError("steps must be positive")
    torch.set_num_threads(2)
    torch.manual_seed(42)
    cfg = tiny_config()
    cfg.work_dir = output_dir
    model = TwoStageTTS(cfg.model)
    batch = collate(dummy_items())
    losses = {}
    for phase in ("cb0", "acoustic"):
        model.set_phase(phase)
        model.train()
        opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=0.01)
        first = None
        for _ in range(steps):
            opt.zero_grad(set_to_none=True)
            outputs = model(batch)
            loss, _ = cb0_loss(*outputs, batch["codes"], batch["frame_mask"]) if phase == "cb0" else acoustic_loss(outputs, batch["codes"], batch["frame_mask"])
            first = float(loss.detach()) if first is None else first
            loss.backward()
            torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], 1)
            opt.step()
        model.eval()
        with torch.no_grad():
            outputs = model(batch)
            loss, _ = cb0_loss(*outputs, batch["codes"], batch["frame_mask"]) if phase == "cb0" else acoustic_loss(outputs, batch["codes"], batch["frame_mask"])
        losses[phase] = {"initial": first, "final": float(loss)}
    result = model.generate(batch["hidden"][:1], batch["embeddings"][:1], batch["token_mask"][:1], max_frames=12)
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    checkpoint = out / "dummy_acoustic.pt"
    save_checkpoint(checkpoint, model, cfg, "acoustic", steps)
    restored, _, _ = load_checkpoint(checkpoint)
    replay = restored.generate(batch["hidden"][:1], batch["embeddings"][:1], batch["token_mask"][:1], max_frames=12)
    np.save(out / "dummy_generated_codes.npy", result.codes[0].numpy())
    report = {"passed": all(v["final"] < v["initial"] * 0.25 for v in losses.values()) and torch.equal(result.codes, replay.codes),
              "losses": losses, "generated_codes_shape": list(result.codes.shape),
              "termination_reason": result.termination_reason, "checkpoint_replay_equal": torch.equal(result.codes, replay.codes),
              "dummy_data_only": True, "real_mimi_decoder_tested": False,
              "speech_quality_tested": False, "parameters": sum(p.numel() for p in model.parameters())}
    (out / "smoke_report.json").write_text(json.dumps(report, indent=2))
    return report


def write_dummy_dataset(directory: str) -> V2Config:
    root = Path(directory).resolve()
    root.mkdir(parents=True, exist_ok=True)
    rows = []
    for split, seed in (("train", 42), ("val", 43)):
        for item in dummy_items(seed=seed):
            name = item["record_id"] + ".npz"
            np.savez(root / name, hidden=item["hidden"].numpy(), embeddings=item["embeddings"].numpy(), codes=item["codes"].numpy())
            rows.append({"record_id": item["record_id"], "split": split, "features": name, "text": item["text"]})
    (root / "manifest.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    cfg = tiny_config()
    cfg.train.epochs = 2
    cfg.data.manifest = str(root / "manifest.jsonl")
    cfg.work_dir = str(root / "run")
    (root / "config.json").write_text(json.dumps(cfg.to_dict(), indent=2))
    return cfg
