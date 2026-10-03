from __future__ import annotations

import csv
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F


class CodebookMetrics:
    def __init__(self, books: int = 8, vocabulary: int = 2048):
        self.books = books
        self.vocabulary = vocabulary
        self.frames = np.zeros(books, dtype=np.int64)
        self.loss = np.zeros(books, dtype=np.float64)
        self.top1 = np.zeros(books, dtype=np.int64)
        self.top5 = np.zeros(books, dtype=np.int64)
        self.entropy = np.zeros(books, dtype=np.float64)
        self.histogram = np.zeros((books, vocabulary), dtype=np.int64)
        self.repeat_frames = np.zeros(books, dtype=np.int64)
        self.longest_repeat = np.zeros(books, dtype=np.int64)

    @torch.no_grad()
    def update(self, logits: torch.Tensor, labels: torch.Tensor, mask: torch.Tensor) -> None:
        for b in range(logits.shape[0]):
            valid = int(mask[b].sum().item())
            if valid < 1:
                continue
            sample_logits = logits[b, :valid].float()
            sample_labels = labels[b, :valid]
            losses = F.cross_entropy(
                sample_logits.reshape(-1, self.vocabulary), sample_labels.reshape(-1), reduction="none"
            ).view(valid, self.books)
            probabilities = torch.softmax(sample_logits, dim=-1)
            predicted = sample_logits.argmax(dim=-1)
            top5 = sample_logits.topk(5, dim=-1).indices
            entropy = -(probabilities * probabilities.clamp_min(1e-12).log()).sum(dim=-1)
            for book in range(self.books):
                pred = predicted[:, book].cpu().numpy()
                self.frames[book] += valid
                self.loss[book] += float(losses[:, book].sum().item())
                self.top1[book] += int((predicted[:, book] == sample_labels[:, book]).sum().item())
                self.top5[book] += int(
                    (top5[:, book] == sample_labels[:, book, None]).any(dim=-1).sum().item()
                )
                self.entropy[book] += float(entropy[:, book].sum().item())
                self.histogram[book] += np.bincount(pred, minlength=self.vocabulary)
                run = 1
                longest = 1
                repeated = 0
                for left, right in zip(pred, pred[1:]):
                    if left == right:
                        run += 1
                        repeated += 1
                        longest = max(longest, run)
                    else:
                        run = 1
                self.repeat_frames[book] += repeated
                self.longest_repeat[book] = max(self.longest_repeat[book], longest)

    def report(self) -> list[dict[str, Any]]:
        rows = []
        for book in range(self.books):
            count = max(1, int(self.frames[book]))
            ce = float(self.loss[book] / count)
            used = int(np.count_nonzero(self.histogram[book]))
            rows.append(
                {
                    "codebook": book,
                    "frames": int(self.frames[book]),
                    "cross_entropy": ce,
                    "perplexity": float(math.exp(min(ce, 30.0))),
                    "top1_accuracy": float(self.top1[book] / count),
                    "top5_accuracy": float(self.top5[book] / count),
                    "predicted_entropy": float(self.entropy[book] / count),
                    "codes_used": used,
                    "codes_used_fraction": float(used / self.vocabulary),
                    "repeated_fraction": float(self.repeat_frames[book] / max(1, count - 1)),
                    "longest_repeat": int(self.longest_repeat[book]),
                }
            )
        return rows


def sequence_accuracy(predicted: torch.Tensor, target: torch.Tensor) -> dict[str, Any]:
    frames = min(int(predicted.shape[0]), int(target.shape[0]))
    if frames == 0:
        return {"frames": 0, "per_codebook_accuracy": [0.0] * target.shape[-1]}
    matches = predicted[:frames] == target[:frames]
    cb0 = matches[:, 0]
    incorrect = torch.nonzero(~cb0)
    first_error = int(incorrect[0].item()) if incorrect.numel() else frames
    after = cb0[first_error + 1 :] if first_error + 1 < frames else cb0[:0]
    buckets = []
    for start in range(0, frames, 25):
        buckets.append(
            {
                "start": start,
                "end": min(frames, start + 25),
                "cb0_accuracy": float(cb0[start : start + 25].float().mean().item()),
            }
        )
    return {
        "frames": frames,
        "per_codebook_accuracy": matches.float().mean(dim=0).cpu().tolist(),
        "first_cb0_error": first_error if first_error < frames else None,
        "cb0_accuracy_after_first_error": float(after.float().mean().item()) if after.numel() else None,
        "position_buckets": buckets,
    }


def write_codebook_csv(path: str | Path, rows: list[dict[str, Any]], mode: str) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    fields = ["mode"] + list(rows[0].keys()) if rows else ["mode"]
    exists = target.is_file()
    with target.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        if not exists:
            writer.writeheader()
        for row in rows:
            writer.writerow({"mode": mode, **row})


def _loss_trend(current: float, previous: float | None) -> tuple[str, float | None]:
    """Classify CE movement while ignoring normal sub-percent validation noise."""
    if previous is None or not math.isfinite(previous):
        return "baseline", None
    delta = current - previous
    tolerance = max(0.01, abs(previous) * 0.005)
    if delta < -tolerance:
        return "improving", delta
    if delta > tolerance:
        return "worsening", delta
    return "flat", delta


def training_health_signals(
    validation: dict[str, Any],
    previous_validation: dict[str, Any] | None = None,
    vocabulary: int = 2048,
) -> dict[str, Any]:
    """Turn teacher-forced and autoregressive validation metrics into clear signals.

    Autoregressive metrics use the model's own previous Mimi codes while retaining
    oracle token durations. Duration quality is reported independently.
    """
    teacher = validation.get("teacher_forced", [])
    rollout = validation.get("rollout", [])
    if not teacher or len(teacher) != len(rollout):
        raise ValueError("validation must contain matching teacher-forced and rollout codebooks")
    previous_teacher = (previous_validation or {}).get("teacher_forced", [])
    previous_rollout = (previous_validation or {}).get("rollout", [])
    oracle_usage = validation.get("oracle_codes_used", [])
    oracle_entropy = validation.get("oracle_entropy", [])
    random_ce = math.log(vocabulary)
    books: list[dict[str, Any]] = []
    warnings: list[str] = []
    improving = worsening = collapsed = 0

    for index, (tf, ar) in enumerate(zip(teacher, rollout)):
        tf_ce = float(tf["cross_entropy"])
        ar_ce = float(ar["cross_entropy"])
        prior_tf = (
            float(previous_teacher[index]["cross_entropy"])
            if index < len(previous_teacher) else None
        )
        prior_ar = (
            float(previous_rollout[index]["cross_entropy"])
            if index < len(previous_rollout) else None
        )
        tf_trend, tf_delta = _loss_trend(tf_ce, prior_tf)
        ar_trend, ar_delta = _loss_trend(ar_ce, prior_ar)
        used_reference = int(oracle_usage[index]) if index < len(oracle_usage) else 0
        entropy_reference = float(oracle_entropy[index]) if index < len(oracle_entropy) else 0.0
        usage_ratio = float(ar["codes_used"]) / max(1, used_reference)
        entropy_ratio = float(ar["predicted_entropy"]) / max(1e-9, entropy_reference)
        collapse = bool(usage_ratio < 0.25 or entropy_ratio < 0.50)
        gap = ar_ce - tf_ce

        if ar_trend == "improving":
            improving += 1
        elif ar_trend == "worsening":
            worsening += 1
        if collapse:
            collapsed += 1

        if ar_trend == "worsening" or collapse or gap > 1.0:
            signal = "red"
        elif ar_trend in {"baseline", "flat"} or gap > 0.35:
            signal = "yellow"
        else:
            signal = "green"
        books.append(
            {
                "codebook": int(tf.get("codebook", index)),
                "signal": signal,
                "teacher_ce": tf_ce,
                "teacher_ce_vs_random": tf_ce - random_ce,
                "teacher_trend": tf_trend,
                "teacher_delta": tf_delta,
                "teacher_top1": float(tf["top1_accuracy"]),
                "ar_ce": ar_ce,
                "ar_ce_vs_random": ar_ce - random_ce,
                "ar_trend": ar_trend,
                "ar_delta": ar_delta,
                "ar_top1": float(ar["top1_accuracy"]),
                "ar_teacher_ce_gap": gap,
                "codes_used": int(ar["codes_used"]),
                "oracle_codes_used": used_reference,
                "code_usage_ratio": usage_ratio,
                "entropy_ratio": entropy_ratio,
                "repeated_fraction": float(ar["repeated_fraction"]),
                "longest_repeat": int(ar["longest_repeat"]),
                "collapse_warning": collapse,
            }
        )

    cb0 = books[0]
    if cb0["ar_trend"] == "worsening":
        warnings.append("CB0 autoregressive loss worsened; intelligibility may be regressing")
    if cb0["collapse_warning"]:
        warnings.append("CB0 code usage/entropy is collapsing")
    if cb0["ar_teacher_ce_gap"] > 1.0:
        warnings.append("CB0 teacher-to-AR gap is severe; exposure error is dominating")
    elif cb0["ar_teacher_ce_gap"] > 0.35:
        warnings.append("CB0 teacher-to-AR gap is growing")
    if collapsed:
        warnings.append(f"{collapsed}/{len(books)} codebooks show possible code collapse")
    duration_ape = validation.get("duration_median_ape")
    if duration_ape is not None and float(duration_ape) > 0.50:
        warnings.append("median duration error exceeds 50%")

    has_previous = bool(previous_teacher and previous_rollout)
    if not all(math.isfinite(float(row["teacher_ce"])) and math.isfinite(float(row["ar_ce"])) for row in books):
        overall = "red"
        warnings.insert(0, "non-finite validation loss detected")
    elif cb0["signal"] == "red" or worsening >= max(2, len(books) // 2) or collapsed >= max(2, len(books) // 2):
        overall = "red"
    elif has_previous and cb0["ar_trend"] == "improving" and improving >= max(1, len(books) // 2) and not warnings:
        overall = "green"
    else:
        overall = "yellow"

    if overall == "green":
        headline = "teacher-forced and autoregressive validation are improving together"
    elif overall == "red":
        headline = warnings[0] if warnings else "validation shows a material regression"
    elif not has_previous:
        headline = "baseline epoch recorded; trend signals begin next epoch"
    else:
        headline = "training needs observation; improvement is not yet consistent"
    return {
        "overall_signal": overall,
        "headline": headline,
        "evaluation_protocol": "teacher-forced plus autoregressive Mimi codes with oracle token durations",
        "random_guess_cross_entropy": random_ce,
        "samples": int(validation.get("samples", 0)),
        "improving_codebooks": improving,
        "flat_codebooks": len(books) - improving - worsening,
        "worsening_codebooks": worsening,
        "collapsed_codebooks": collapsed,
        "cb0": cb0,
        "duration_mae_frames": validation.get("duration_mae_frames"),
        "duration_median_ape": duration_ape,
        "warnings": warnings,
        "codebooks": books,
    }


def write_epoch_signal_csv(path: str | Path, history: list[dict[str, Any]]) -> None:
    """Rewrite a resume-safe epoch/codebook table from canonical training history."""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "epoch", "overall_signal", "codebook", "signal", "teacher_ce",
        "teacher_trend", "teacher_delta", "teacher_top1", "ar_ce", "ar_trend",
        "ar_delta", "ar_top1", "ar_teacher_ce_gap", "code_usage_ratio",
        "entropy_ratio", "repeated_fraction", "longest_repeat", "collapse_warning",
    ]
    temporary = target.with_suffix(target.suffix + ".tmp")
    with temporary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for epoch_row in history:
            signal = epoch_row.get("signals", {})
            for book in signal.get("codebooks", []):
                writer.writerow(
                    {
                        "epoch": epoch_row["epoch"],
                        "overall_signal": signal.get("overall_signal"),
                        **{field: book.get(field) for field in fields if field not in {"epoch", "overall_signal"}},
                    }
                )
    temporary.replace(target)


def atomic_metrics(path: str | Path, metrics: dict[str, Any]) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_suffix(target.suffix + ".tmp")
    temporary.write_text(json.dumps(metrics, indent=2, sort_keys=True), encoding="utf-8")
    temporary.replace(target)
