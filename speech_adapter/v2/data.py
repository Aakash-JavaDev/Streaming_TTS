from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

from .config import V2Config
from .embeddings import EmbeddingCache


def _feature_arrays(arrays, cache: EmbeddingCache | None = None) -> dict:
    hidden = np.asarray(arrays["hidden"], dtype=np.float32).copy()
    if "embeddings" in arrays:
        embeddings = np.asarray(arrays["embeddings"], dtype=np.float32).copy()
    elif cache is not None and "token_ids" in arrays:
        embeddings = cache.lookup(np.asarray(arrays["token_ids"], dtype=np.int64))
    else:
        raise ValueError("provide frozen embeddings, or token_ids and an embedding cache")
    result = {"hidden": torch.from_numpy(hidden), "embeddings": torch.from_numpy(embeddings)}
    if "token_ids" in arrays:
        ids = np.asarray(arrays["token_ids"])
        if ids.ndim != 1 or not np.issubdtype(ids.dtype, np.integer) or np.any(ids < 0):
            raise ValueError("token_ids must be nonnegative integer [L]")
        result["token_ids"] = torch.from_numpy(ids.astype(np.int64).copy())
    return result


def validate_item(item: dict, cfg: V2Config, require_codes: bool = True) -> None:
    h, e = item["hidden"], item["embeddings"]
    if h.ndim != 2 or e.ndim != 2 or h.shape[0] != e.shape[0] or h.shape[0] < 1:
        raise ValueError("hidden/embedding sequences must be nonempty, aligned [L,D]")
    if h.shape[1] != cfg.model.hidden_dim or e.shape[1] != cfg.model.embedding_dim:
        raise ValueError("dataset feature widths differ from configured Qwen widths")
    if h.shape[0] > cfg.model.max_text_tokens or not torch.isfinite(h).all() or not torch.isfinite(e).all():
        raise ValueError("invalid/non-finite/overlong text features")
    if "token_ids" in item and len(item["token_ids"]) != h.shape[0]:
        raise ValueError("token IDs and hidden states are not aligned")
    if require_codes:
        codes = item["codes"]
        if codes.ndim != 2 or codes.shape[0] != 8 or not 1 <= codes.shape[1] <= cfg.model.max_audio_frames:
            raise ValueError("codes must be nonempty [8,T] within max_audio_frames")
        if codes.dtype != torch.long or codes.min() < 0 or codes.max() >= cfg.model.codec_vocab:
            raise ValueError("invalid Mimi code dtype/range")


class NPZDataset(Dataset):
    """JSONL {record_id, split, features: sample.npz}; paths relative to manifest."""
    def __init__(self, cfg: V2Config, split: str, limit: int = 0):
        self.cfg = cfg
        self.root = Path(cfg.data.manifest).resolve().parent
        self.rows = [json.loads(line) for line in Path(cfg.data.manifest).read_text().splitlines() if line.strip()]
        ids = [str(r["record_id"]) for r in self.rows]
        if len(ids) != len(set(ids)):
            raise ValueError("record IDs must be unique across all splits")
        self.rows = [r for r in self.rows if r["split"] == split]
        if limit:
            self.rows = self.rows[:limit]
        if not self.rows:
            raise ValueError(f"no {split} records in {cfg.data.manifest}")
        cache_path = getattr(cfg.data, "embedding_cache", "")
        self.cache = EmbeddingCache(cache_path) if cache_path else None

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index: int) -> dict:
        row = self.rows[index]
        path = self.root / row["features"]
        with np.load(path, allow_pickle=False) as arrays:
            item = _feature_arrays(arrays, self.cache)
            codes = np.asarray(arrays["codes"])
            if not np.issubdtype(codes.dtype, np.integer):
                raise ValueError("Mimi codes must be stored as integers, not rounded floats")
            if self.cfg.data.codes_layout == "T_K":
                codes = codes.T
            item["codes"] = torch.from_numpy(codes.astype(np.int64).copy())
        item["record_id"], item["text"] = str(row["record_id"]), str(row.get("text", ""))
        item["timestamps"] = row.get("timestamps")
        validate_item(item, self.cfg)
        return item


class PreparedLegacyDataset(Dataset):
    """Reuse existing mmap Qwen feature sidecars; no duration/owner conditioning."""
    def __init__(self, cfg: V2Config, split: str, limit: int = 0):
        from ..config import AppConfig
        from ..data import SpeechDataset
        legacy = AppConfig.load(cfg.data.legacy_config)
        self.dataset = SpeechDataset(legacy, split, limit=limit)
        self.cfg = cfg

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        old = self.dataset[index]
        item = {"record_id": str(old["record_id"]), "text": old["text"],
                "hidden": old["hidden"].clone(), "embeddings": old["embeddings"],
                "codes": old["codes"].T.contiguous(), "timestamps": old["words"]}
        validate_item(item, self.cfg)
        return item


def make_dataset(cfg: V2Config, split: str, limit: int = 0) -> Dataset:
    if cfg.data.legacy_config:
        return PreparedLegacyDataset(cfg, split, limit)
    if cfg.data.manifest:
        return NPZDataset(cfg, split, limit)
    raise ValueError("set data.manifest or data.legacy_config")


def collate(items: list[dict]) -> dict:
    if not items:
        raise ValueError("empty batch")
    b = len(items)
    length = max(i["hidden"].shape[0] for i in items)
    frames = max(i["codes"].shape[1] for i in items)
    batch = {
        "hidden": torch.zeros(b, length, items[0]["hidden"].shape[1]),
        "embeddings": torch.zeros(b, length, items[0]["embeddings"].shape[1]),
        "codes": torch.zeros(b, 8, frames, dtype=torch.long),
        "token_mask": torch.zeros(b, length, dtype=torch.bool),
        "frame_mask": torch.zeros(b, frames, dtype=torch.bool),
        "record_ids": [i["record_id"] for i in items],
    }
    for j, item in enumerate(items):
        n, t = item["hidden"].shape[0], item["codes"].shape[1]
        batch["hidden"][j, :n] = item["hidden"]
        batch["embeddings"][j, :n] = item["embeddings"]
        batch["codes"][j, :, :t] = item["codes"]
        batch["token_mask"][j, :n] = True
        batch["frame_mask"][j, :t] = True
    return batch


def move_batch(batch: dict, device: torch.device) -> dict:
    return {k: v.to(device) if torch.is_tensor(v) else v for k, v in batch.items()}


def load_inference_features(path: str, cfg: V2Config, embedding_cache: str = "") -> dict:
    cache_path = embedding_cache or cfg.data.embedding_cache
    cache = EmbeddingCache(cache_path) if cache_path else None
    with np.load(path, allow_pickle=False) as arrays:
        # Deliberately do not read codes/lengths/timestamps during synthesis.
        item = _feature_arrays(arrays, cache)
    validate_item(item, cfg, require_codes=False)
    return {"hidden": item["hidden"][None], "embeddings": item["embeddings"][None],
            "token_mask": torch.ones(1, len(item["hidden"]), dtype=torch.bool)}
