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
    
    ids = np.asarray(token_ids)
    if ids.ndim != 1 or not np.issubdtype(ids.dtype, np.integer):
        raise ValueError("token IDs must be a one-dimensional integer array")

    with safe_open(found[0], framework="pt", device="cpu") as handle:
        key = found[1]
        table = handle.get_tensor(key)

        if table.ndim != 2:
            raise ValueError("input embedding table must be a matrix")

        vocab_size = table.shape[0]
        if ids.size and (ids.min() < 0 or ids.max() >= vocab_size):
            raise ValueError("token ID outside the checkpoint input vocabulary")
        ids = ids.astype(np.int64, copy=False)

        if table.is_floating_point():
            # Existing FP32 / FP16 / BF16 checkpoint path.
            width = table.shape[1]
            result = (
                table.index_select(0, torch.from_numpy(ids.copy()))
                .float()
                .numpy()
                .astype(np.float16)
            )

        elif table.dtype == torch.uint32:
            # Targeted support for MLX affine 4-bit, group_size=64.
            prefix = key.rsplit(".", 1)[0]
            scale_key = prefix + ".scales"
            bias_key = prefix + ".biases"

            if scale_key not in handle.keys() or bias_key not in handle.keys():
                raise ValueError("packed MLX embeddings require scales and biases")

            scales = handle.get_tensor(scale_key)
            biases = handle.get_tensor(bias_key)

            bits = 4
            values_per_word = 32 // bits
            group_size = 64
            width = table.shape[1] * values_per_word

            expected_shape = (vocab_size, width // group_size)
            if width % group_size:
                raise ValueError("embedding width is not divisible by group_size")
            if tuple(scales.shape) != expected_shape:
                raise ValueError("scales shape does not match MLX 4-bit/64 layout")
            if tuple(biases.shape) != expected_shape:
                raise ValueError("biases shape does not match MLX 4-bit/64 layout")
            if not scales.is_floating_point() or not biases.is_floating_point():
                raise ValueError("MLX affine scales and biases must be floating point")

            if cfg.get("hidden_size") is not None:
                if width != int(cfg["hidden_size"]):
                    raise ValueError(
                        "packed embedding width does not match hidden_size; "
                        "this decoder supports MLX affine 4-bit/64 only"
                    )

            # Keep the vocabulary table packed. Decode requested rows in batches.
            packed_table = table.numpy()
            shifts = np.arange(0, 32, bits, dtype=np.uint32)
            result = np.empty((len(ids), width), dtype=np.float16)

            for start in range(0, len(ids), 512):
                batch_ids = ids[start:start + 512]
                n = len(batch_ids)
                row_indices = torch.from_numpy(batch_ids.copy())

                # NumPy handles uint32 indexing and bit operations on CPU.
                packed = packed_table[batch_ids]
                codes = (
                    (packed[..., None] >> shifts) & np.uint32(15)
                ).reshape(n, width).astype(np.float32)

                s = scales.index_select(0, row_indices).float().numpy()
                b = biases.index_select(0, row_indices).float().numpy()

                values = (
                    codes.reshape(n, width // group_size, group_size)
                    * s[..., None]
                    + b[..., None]
                )

                result[start:start + n] = (
                    values.reshape(n, width).astype(np.float16)
                )

        else:
            raise ValueError(f"unsupported embedding dtype: {table.dtype}")

    if cfg.get("hidden_size") is not None:
        if width != int(cfg["hidden_size"]):
            raise ValueError("embedding width differs from checkpoint hidden_size")

    if not np.isfinite(result).all():
        raise ValueError("non-finite or FP16-overflowed token embeddings")

    return result

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
