from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class PathsConfig:
    data_dir: str
    source_jsonl: str
    qwen_model: str
    kokoro_model: str = ""
    kokoro_voices: str = ""
    mimi_encoder: str = ""
    mimi_decoder: str = ""
    mimi_weights: str = ""
    work_dir: str = "/kaggle/working/streaming_speech"


@dataclass
class ModelConfig:
    model_dim: int = 256
    temporal_layers: int = 2
    attention_heads: int = 4
    ffn_dim: int = 512
    text_context_size: int = 8
    audio_cache_frames: int = 64
    num_codebooks: int = 8
    codec_vocab: int = 2048
    dropout: float = 0.0


@dataclass
class TrainConfig:
    seed: int = 42
    epochs: int = 20
    learning_rate: float = 1.0e-4
    weight_decay: float = 0.01
    warmup_fraction: float = 0.05
    min_lr_ratio: float = 0.05
    grad_clip: float = 1.0
    duration_weight: float = 0.1
    cb0_weight: float = 4.0
    residual_weight: float = 1.0
    max_frames_per_gpu: int = 1200
    max_batch_size: int = 8
    grad_accum_steps: int = 1
    num_workers: int = 2
    val_fraction: float = 0.1
    eval_samples: int = 64
    eval_audio_samples: int = 2
    save_every: int = 1
    log_every: int = 25
    resume: bool = True


@dataclass
class GateConfig:
    codec_samples: int = 20
    alignment_samples: int = 50
    alignment_pass_fraction: float = 0.95
    overfit_teacher_forced_ce: float = 0.05
    overfit_rollout_cb0_accuracy: float = 0.99
    text_control_margin: float = 0.05
    min_code_usage_ratio: float = 0.50
    min_entropy_ratio: float = 0.75
    max_duration_median_ape: float = 0.20
    max_device_p50_ms: float = 20.0
    max_device_p95_ms: float = 40.0
    max_device_worst_ms: float = 80.0
    max_first_audio_ms: float = 300.0
    max_int8_weight_bytes: int = 12 * 1024 * 1024
    max_working_memory_bytes: int = 32 * 1024 * 1024
    max_cache_bytes: int = 1024 * 1024


@dataclass
class RuntimeConfig:
    """Execution limits that do not change the adapter architecture."""

    profile: str = "kaggle"
    device: str = "auto"
    require_two_gpus: bool = True
    manifest_record_limit: int = 0
    cpu_threads: int = 2


@dataclass
class AppConfig:
    paths: PathsConfig
    model: ModelConfig = field(default_factory=ModelConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    gates: GateConfig = field(default_factory=GateConfig)
    runtime: RuntimeConfig = field(default_factory=RuntimeConfig)
    feature_mode: str = "legacy_answer"
    run_stage: str = "preflight"
    confirm_full_training: str = ""

    @classmethod
    def load(cls, path: str | Path) -> "AppConfig":
        source = Path(path)
        raw = json.loads(source.read_text(encoding="utf-8"))
        cfg = cls(
            paths=PathsConfig(**raw["paths"]),
            model=ModelConfig(**raw.get("model", {})),
            train=TrainConfig(**raw.get("train", {})),
            gates=GateConfig(**raw.get("gates", {})),
            runtime=RuntimeConfig(**raw.get("runtime", {})),
            feature_mode=str(raw.get("feature_mode", "legacy_answer")),
            run_stage=str(raw.get("run_stage", "preflight")),
            confirm_full_training=str(raw.get("confirm_full_training", "")),
        )
        cfg.validate()
        return cfg

    def validate(self) -> None:
        if self.run_stage not in {"preflight", "prepare", "qualify", "train_full", "evaluate", "export"}:
            raise ValueError(f"unsupported run_stage={self.run_stage!r}")
        if self.feature_mode not in {
            "legacy_answer", "answer_only_recompute", "deployment_context"
        }:
            raise ValueError(
                "feature_mode must be 'legacy_answer', 'answer_only_recompute', "
                "or 'deployment_context'"
            )
        if self.runtime.profile not in {"local_cpu", "kaggle"}:
            raise ValueError("runtime.profile must be 'local_cpu' or 'kaggle'")
        if self.runtime.device not in {"auto", "cpu", "cuda"}:
            raise ValueError("runtime.device must be 'auto', 'cpu', or 'cuda'")
        if self.runtime.manifest_record_limit < 0:
            raise ValueError("runtime.manifest_record_limit cannot be negative")
        if 0 < self.runtime.manifest_record_limit < 4:
            raise ValueError("runtime.manifest_record_limit must be zero or at least four")
        if self.runtime.cpu_threads < 1:
            raise ValueError("runtime.cpu_threads must be positive")
        if self.model.model_dim % self.model.attention_heads:
            raise ValueError("model_dim must be divisible by attention_heads")
        if self.model.num_codebooks != 8 or self.model.codec_vocab != 2048:
            raise ValueError("this dataset contract requires 8 codebooks with vocabulary 2048")
        if self.model.audio_cache_frames < 1 or self.model.text_context_size < 1:
            raise ValueError("cache sizes must be positive")
        if not 0.0 < self.train.val_fraction < 1.0:
            raise ValueError("val_fraction must be between 0 and 1")
        if self.train.max_frames_per_gpu < 1 or self.train.max_batch_size < 1:
            raise ValueError("batch limits must be positive")
        if self.train.eval_samples < 1 or self.train.eval_audio_samples < 1:
            raise ValueError("validation sample counts must be positive")
        if self.run_stage == "train_full" and self.confirm_full_training != "TRAIN_FULL_DATASET":
            raise ValueError(
                "full training is locked; set confirm_full_training to TRAIN_FULL_DATASET"
            )

    @property
    def work_dir(self) -> Path:
        return Path(self.paths.work_dir)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def fingerprint(self) -> str:
        payload = json.dumps(self.to_dict(), sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()

    def feature_fingerprint(self) -> str:
        payload = {
            "feature_schema": 2,
            "data_dir": self.paths.data_dir,
            "source_jsonl": self.paths.source_jsonl,
            "qwen_model": self.paths.qwen_model,
            "feature_mode": self.feature_mode,
            "manifest_record_limit": self.runtime.manifest_record_limit,
            "seed": self.train.seed,
            "val_fraction": self.train.val_fraction,
        }
        dataset_report = self.work_dir / "dataset_report.json"
        if dataset_report.is_file():
            try:
                payload["dataset_signature"] = json.loads(
                    dataset_report.read_text(encoding="utf-8")
                ).get("dataset_signature")
            except (OSError, json.JSONDecodeError):
                payload["dataset_signature"] = "unreadable"
        raw = json.dumps(payload, sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def atomic_json(path: str | Path, value: Any) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_suffix(target.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True), encoding="utf-8")
    temporary.replace(target)


def sha256_file(path: str | Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while True:
            block = handle.read(chunk_size)
            if not block:
                break
            digest.update(block)
    return digest.hexdigest()
