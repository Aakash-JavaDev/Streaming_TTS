"""Frozen Qwen input-table extraction; never construct or run the language model."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
from safetensors import safe_open

INPUT_KEYS = ("model.embed_tokens.weight", "model.model.embed_tokens.weight", "transformer.wte.weight")


def embedding_rows(model_dir: str | Path, token_ids: np.ndarray) -> np.ndarray:
    root = Path(model_dir)
    if not root.is_dir():
        raise FileNotFoundError(f"expected local safetensors checkpoint directory: {root}")
    cfg_path = root / "config.json"
    cfg = json.loads(cfg_path.read_text()) if cfg_path.is_file() else {}
    # lm_head is not generally the input table. Only use a documented tied alias.
    keys = INPUT_KEYS + (("lm_head.weight",) if cfg.get("tie_word_embeddings") is True else ())
    found: tuple[Path, str] | None = None
    indices = sorted(root.glob("*.safetensors.index.json"))
    if indices:
        mapping = json.loads(indices[0].read_text()).get("weight_map", {})
        for key in keys:
            if key in mapping:
                found = (root / mapping[key], key)
                break
    if found is None:
        files = sorted(root.glob("*.safetensors"))
        for key in keys:
            for path in files:
                with safe_open(path, framework="pt", device="cpu") as handle:
                    if key in handle.keys():
                        found = (path, key)
                        break
            if found:
                break
    if found is None:
        raise FileNotFoundError("input embedding table not found (untied lm_head is not an input table)")
    with safe_open(found[0], framework="pt", device="cpu") as handle:
        table = handle.get_tensor(found[1])
    if table.ndim != 2 or not table.is_floating_point():
        raise ValueError("input table must be a floating-point matrix; quantized packed tables are unsupported")
    if cfg.get("hidden_size") is not None and table.shape[1] != int(cfg["hidden_size"]):
        raise ValueError("embedding width differs from checkpoint hidden_size")
    ids = np.asarray(token_ids)
    if ids.ndim != 1 or not np.issubdtype(ids.dtype, np.integer):
        raise ValueError("token IDs must be a one-dimensional integer array")
    if ids.size and (ids.min() < 0 or ids.max() >= table.shape[0]):
        raise ValueError("token ID outside the checkpoint input vocabulary")
    result = table.index_select(0, torch.from_numpy(ids.astype(np.int64))).float().numpy()
    if not np.isfinite(result).all():
        raise ValueError("non-finite token embeddings")
    return result.astype(np.float16)


def prepare_embedding_cache(model_dir: str, ids_file: str, output_dir: str) -> dict:
    ids = np.unique(np.load(ids_file, allow_pickle=False))
    values = embedding_rows(model_dir, ids)
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    np.save(out / "embedding_ids.npy", ids.astype(np.int64))
    np.save(out / "embedding_values.npy", values)
    meta = {"model_dir": str(Path(model_dir).resolve()), "rows": len(ids), "width": values.shape[1]}
    (out / "embedding_provenance.json").write_text(json.dumps(meta, indent=2))
    return meta


class EmbeddingCache:
    def __init__(self, directory: str | Path):
        root = Path(directory)
        self.ids = np.load(root / "embedding_ids.npy", mmap_mode="r", allow_pickle=False)
        self.values = np.load(root / "embedding_values.npy", mmap_mode="r", allow_pickle=False)
        if self.ids.ndim != 1 or self.values.ndim != 2 or len(self.ids) != len(self.values):
            raise ValueError("malformed embedding cache")
        if len(self.ids) < 1 or np.any(np.diff(self.ids) <= 0):
            raise ValueError("embedding_ids must be nonempty, sorted, unique")

    def lookup(self, ids: np.ndarray) -> np.ndarray:
        loc = np.searchsorted(self.ids, ids)
        if np.any(loc >= len(self.ids)) or not np.array_equal(self.ids[np.minimum(loc, len(self.ids)-1)], ids):
            raise ValueError("token missing from frozen embedding cache")
        return np.asarray(self.values[loc], dtype=np.float32).copy()
