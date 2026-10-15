from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path


@dataclass
class Architecture:
    hidden_dim: int = 1536
    embedding_dim: int = 1536
    width: int = 512
    heads: int = 8
    ffn_dim: int = 1536
    cb0_layers: int = 4
    acoustic_layers: int = 2
    code_embedding_dim: int = 128
    num_codebooks: int = 8
    codec_vocab: int = 2048
    dropout: float = 0.1
    gate_bias: float = -3.0
    max_audio_frames: int = 1500
    max_text_tokens: int = 2048

    def validate(self) -> None:
        for key, value in asdict(self).items():
            if key not in {"dropout", "gate_bias"} and (not isinstance(value, int) or value < 1):
                raise ValueError(f"model.{key} must be positive")
        if self.width % self.heads or (self.width // self.heads) % 2:
            raise ValueError("width must be divisible by heads, with an even RoPE head width")
        if self.num_codebooks != 8:
            raise ValueError("v2 requires exactly eight codebooks")
        if not 0 <= self.dropout < 1:
            raise ValueError("dropout must be in [0,1)")


@dataclass
class Data:
    manifest: str = ""
    legacy_config: str = ""
    codes_layout: str = "K_T"
    embedding_model_dir: str = ""
    embedding_cache: str = ""
    model_id: str = "Qwen/Qwen2.5-1.5B-Instruct"
    model_revision: str = ""
    hidden_layer: str = "final_normalized"
    hidden_alignment: str = "after_consuming_token"


@dataclass
class Training:
    seed: int = 42
    epochs: int = 20
    batch_size: int = 8
    learning_rate: float = 3e-4
    weight_decay: float = 0.01
    stop_weight: float = 0.1
    grad_clip: float = 1.0
    warmup_fraction: float = 0.05
    num_workers: int = 0
    cpu_threads: int = 2
    device: str = "auto"
    precision: str = "fp32"
    eval_samples: int = 8
    max_generate_frames: int = 300
    min_generate_frames: int = 1
    stop_threshold: float = 0.5


@dataclass
class V2Config:
    model: Architecture = field(default_factory=Architecture)
    data: Data = field(default_factory=Data)
    train: Training = field(default_factory=Training)
    work_dir: str = "runs/v2"

    @classmethod
    def from_dict(cls, raw: dict) -> "V2Config":
        unknown = set(raw) - {"model", "data", "train", "work_dir"}
        if unknown:
            raise ValueError(f"unknown v2 config fields: {sorted(unknown)}")
        cfg = cls(Architecture(**raw.get("model", {})), Data(**raw.get("data", {})),
                  Training(**raw.get("train", {})), raw.get("work_dir", "runs/v2"))
        cfg.validate()
        return cfg

    @classmethod
    def load(cls, path: str | Path) -> "V2Config":
        path = Path(path)
        text = path.read_text()
        if path.suffix.lower() in {".yaml", ".yml"}:
            import yaml
            raw = yaml.safe_load(text)
        else:
            raw = json.loads(text)
        return cls.from_dict(raw)

    def validate(self) -> None:
        self.model.validate()
        if self.data.manifest and self.data.legacy_config:
            raise ValueError("choose data.manifest OR data.legacy_config")
        if self.data.codes_layout not in {"K_T", "T_K"}:
            raise ValueError("codes_layout must be explicit: K_T or T_K")
        if self.data.hidden_alignment not in {"predicting_token", "after_consuming_token"}:
            raise ValueError("record the hidden-state/token alignment convention")
        if self.train.device not in {"auto", "cpu", "cuda"}:
            raise ValueError("device must be auto, cpu, or cuda")
        if self.train.precision not in {"fp32", "bf16", "fp16"}:
            raise ValueError("precision must be fp32, bf16, or fp16")
        for name in ("epochs", "batch_size", "cpu_threads", "eval_samples", "max_generate_frames"):
            if getattr(self.train, name) < 1:
                raise ValueError(f"train.{name} must be positive")
        if self.train.num_workers < 0 or self.train.learning_rate <= 0 or self.train.grad_clip <= 0:
            raise ValueError("invalid worker count, learning rate, or gradient clipping")
        if self.train.weight_decay < 0 or self.train.stop_weight < 0:
            raise ValueError("loss weight/weight decay must be nonnegative")
        if not 0 <= self.train.warmup_fraction < 1:
            raise ValueError("warmup_fraction must be in [0,1)")
        if not 0 <= self.train.stop_threshold <= 1:
            raise ValueError("stop_threshold must be in [0,1]")
        if not 1 <= self.train.min_generate_frames <= self.train.max_generate_frames <= self.model.max_audio_frames:
            raise ValueError("generation limits must satisfy 1 <= min <= max <= max_audio_frames")

    def to_dict(self) -> dict:
        return asdict(self)
