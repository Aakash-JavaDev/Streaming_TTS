from __future__ import annotations

import csv
import html
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
from safetensors.torch import load_file

from .config import AppConfig, atomic_json
from .data import SpeechDataset, collate_speech
from .diagnostics import OnnxMimi, _write_wav
from .metrics import write_codebook_csv
from .train import build_model, evaluate, move_batch


def load_best(cfg: AppConfig, device: torch.device):
    model = build_model(cfg, device)
    path = cfg.work_dir / "checkpoints" / "best.safetensors"
    if not path.is_file():
        raise FileNotFoundError(f"best checkpoint not found: {path}")
    model.load_state_dict(load_file(str(path), device=str(device)))
    model.eval()
    return model


@torch.no_grad()
def generate_artifacts(cfg: AppConfig, limit: int = 20) -> dict[str, Any]:
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = load_best(cfg, device)
    dataset = SpeechDataset(cfg, "val", limit=limit)
    decoder = OnnxMimi(cfg.paths.mimi_encoder, cfg.paths.mimi_decoder)
    output_dir = cfg.work_dir / "evaluation"
    output_dir.mkdir(parents=True, exist_ok=True)
    metrics = {
        mode: evaluate(model, dataset, device, limit, text_mode=mode)
        for mode in ("correct", "zeroed", "swapped")
    }
    csv_path = output_dir / "per_codebook.csv"
    if csv_path.exists():
        csv_path.unlink()
    for text_mode, result in metrics.items():
        write_codebook_csv(csv_path, result["teacher_forced"], f"{text_mode}_teacher_forced")
        write_codebook_csv(csv_path, result["rollout"], f"{text_mode}_rollout")
    duration_rows = []
    links = []
    cb0_confusion = np.zeros((model.cfg.codec_vocab, model.cfg.codec_vocab), dtype=np.int64)
    predicted_usage = np.zeros((model.cfg.num_codebooks, model.cfg.codec_vocab), dtype=np.int64)
    for index in range(len(dataset)):
        item = dataset[index]
        batch = move_batch(collate_speech([item]), device)
        logits, _ = model(batch)
        frames = int(batch["frame_mask"][0].sum().item())
        teacher = logits[0, :frames].argmax(dim=-1)
        hidden = batch["hidden"][0, batch["token_mask"][0]]
        embeddings = batch["embeddings"][0, batch["token_mask"][0]]
        oracle_duration = batch["durations"][0, batch["token_mask"][0]]
        rollout_oracle, predicted_duration, _ = model.rollout_with_logits(
            hidden, embeddings, oracle_duration
        )
        rollout_predicted, _, _ = model.rollout_with_logits(hidden, embeddings, None)
        oracle = batch["codes"][0, :frames]
        oracle_np = oracle.detach().cpu().numpy()
        rollout_np = rollout_oracle.detach().cpu().numpy()
        np.add.at(cb0_confusion, (oracle_np[:, 0], rollout_np[:, 0]), 1)
        for book in range(model.cfg.num_codebooks):
            predicted_usage[book] += np.bincount(
                rollout_np[:, book], minlength=model.cfg.codec_vocab
            )
        cb0_pred_residual_oracle = oracle.clone()
        cb0_pred_residual_oracle[:, 0] = rollout_oracle[:, 0]
        cb0_oracle_residual_pred = rollout_oracle.clone()
        cb0_oracle_residual_pred[:, 0] = oracle[:, 0]
        modes = {
            "oracle": oracle,
            "teacher_forced": teacher,
            "rollout_oracle_duration": rollout_oracle,
            "rollout_predicted_duration": rollout_predicted,
            "cb0_predicted_residual_oracle": cb0_pred_residual_oracle,
            "cb0_oracle_residual_predicted": cb0_oracle_residual_pred,
        }
        sample_dir = output_dir / "audio" / str(item["record_id"])
        sample_links = []
        for name, codes in modes.items():
            wav = decoder.decode(codes.detach().cpu().numpy())
            path = sample_dir / f"{name}.wav"
            _write_wav(path, wav)
            sample_links.append((name, path.relative_to(output_dir)))
        true = oracle_duration.float()
        error = (predicted_duration.float() - true).abs()
        for token_index in range(true.numel()):
            duration_rows.append(
                {
                    "record_id": item["record_id"],
                    "token_index": token_index,
                    "true_frames": int(true[token_index].item()),
                    "predicted_frames": int(predicted_duration[token_index].item()),
                    "absolute_error": float(error[token_index].item()),
                    "absolute_percentage_error": float(
                        error[token_index].item() / max(1.0, true[token_index].item())
                    ),
                }
            )
        duration_rows.append(
            {
                "record_id": item["record_id"],
                "token_index": "ALL",
                "true_frames": int(true.sum().item()),
                "predicted_frames": int(predicted_duration.sum().item()),
                "absolute_error": float(abs(predicted_duration.sum().item() - true.sum().item())),
                "absolute_percentage_error": float(
                    abs(predicted_duration.sum().item() - true.sum().item())
                    / max(1.0, true.sum().item())
                ),
            }
        )
        links.append({"record_id": item["record_id"], "text": item["text"], "files": sample_links})
    with (output_dir / "duration.csv").open("w", newline="", encoding="utf-8") as handle:
        fields = list(duration_rows[0]) if duration_rows else ["record_id"]
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(duration_rows)
    page = ["<!doctype html><meta charset='utf-8'><h1>Streaming speech listening index</h1>"]
    for sample in links:
        page.append(f"<h2>{html.escape(str(sample['record_id']))}</h2><p>{html.escape(sample['text'])}</p>")
        for mode, relative in sample["files"]:
            source = html.escape(str(relative))
            page.append(f"<p>{html.escape(mode)}<br><audio controls src='{source}'></audio></p>")
    (output_dir / "listening.html").write_text("\n".join(page), encoding="utf-8")
    np.save(output_dir / "cb0_confusion.npy", cb0_confusion)
    np.save(output_dir / "predicted_code_usage.npy", predicted_usage)
    try:
        import matplotlib.pyplot as plt

        top = np.argsort(cb0_confusion.sum(axis=1))[-64:]
        fig, axis = plt.subplots(figsize=(9, 8))
        image = axis.imshow(np.log1p(cb0_confusion[np.ix_(top, top)]), aspect="auto")
        axis.set(title="CB0 confusion (top oracle codes, log counts)", xlabel="predicted", ylabel="oracle")
        fig.colorbar(image, ax=axis)
        fig.tight_layout()
        fig.savefig(output_dir / "cb0_confusion.png", dpi=140)
        plt.close(fig)
        fig, axes = plt.subplots(4, 2, figsize=(14, 12), constrained_layout=True)
        for book, axis in enumerate(axes.flat):
            used = np.flatnonzero(predicted_usage[book])
            axis.bar(used, predicted_usage[book, used], width=1.0)
            axis.set(title=f"CB{book} predicted usage", xlabel="code", ylabel="count")
        fig.savefig(output_dir / "code_usage.png", dpi=140)
        plt.close(fig)
        training_path = cfg.work_dir / "training.json"
        if training_path.is_file():
            history = json.loads(training_path.read_text()).get("history", [])
            if history:
                fig, axes = plt.subplots(1, 2, figsize=(12, 4), constrained_layout=True)
                axes[0].plot([row["epoch"] for row in history], [row["train_loss"] for row in history])
                axes[0].set(title="Training loss", xlabel="epoch")
                axes[1].plot(
                    [row["epoch"] for row in history],
                    [row["validation"]["rollout_cb0_ce"] for row in history],
                )
                axes[1].set(title="Validation rollout CB0 CE", xlabel="epoch")
                fig.savefig(output_dir / "training_curves.png", dpi=140)
                plt.close(fig)
    except Exception as exc:
        (output_dir / "plot_warning.txt").write_text(str(exc), encoding="utf-8")
    atomic_json(output_dir / "metrics.json", metrics)
    conclusion = ["# Experiment conclusion", "", f"Evaluated {len(dataset)} validation samples.", ""]
    for mode, result in metrics.items():
        conclusion.append(f"- {mode}: rollout CB0 CE = {result['rollout_cb0_ce']:.4f}")
    (output_dir / "conclusion.md").write_text("\n".join(conclusion), encoding="utf-8")
    return metrics
