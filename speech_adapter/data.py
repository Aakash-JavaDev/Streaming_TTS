from __future__ import annotations

import hashlib
import json
import os
import random
import shutil
import time
from collections import OrderedDict, defaultdict
from pathlib import Path
from typing import Any, Iterator, Sequence

import numpy as np

from .config import AppConfig, atomic_json

REQUIRED_SHARD_FILES = (
    "codes.npy",
    "offsets.npy",
    "hidden.npy",
    "hidden_offsets.npy",
    "token_ids.npy",
    "token_char_offsets.npy",
    "ids.jsonl",
    "alignment.jsonl",
    "pieces.jsonl",
)


def jsonl(path: str | Path) -> Iterator[Any]:
    with Path(path).open(encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"invalid JSON at {path}:{line_no}: {exc}") from exc


def discover_shards(data_dir: str | Path) -> list[Path]:
    root = Path(data_dir)
    if not root.is_dir():
        raise FileNotFoundError(f"encoded shard directory does not exist: {root}")
    shards = sorted(path for path in root.glob("shard_*") if path.is_dir())
    complete = [s for s in shards if all((s / name).is_file() for name in REQUIRED_SHARD_FILES)]
    if not complete:
        missing = {
            str(s): [name for name in REQUIRED_SHARD_FILES if not (s / name).is_file()]
            for s in shards[:10]
        }
        raise FileNotFoundError(f"no complete shards under {root}; missing={missing}")
    return complete


def _source_files(path: str | Path) -> list[Path]:
    source = Path(path)
    if source.is_file():
        return [source]
    if source.is_dir():
        files = sorted(source.glob("*.jsonl"))
        if files:
            return files
    raise FileNotFoundError(f"source JSONL file/directory does not exist or is empty: {source}")


def source_rows(
    path: str | Path, wanted_ids: set[str] | None = None
) -> dict[str, dict[str, str]]:
    rows: dict[str, dict[str, str]] = {}
    fallback_id = 0
    for source in _source_files(path):
        for item in jsonl(source):
            record_id = str(item.get("id", fallback_id))
            fallback_id += 1
            if wanted_ids is not None and record_id not in wanted_ids:
                continue
            question = str(item.get("question", "")).strip()
            answer = str(item.get("answer") or item.get("text") or item.get("response") or "").strip()
            if question and answer:
                rows[record_id] = {"question": question, "answer": answer}
            if wanted_ids is not None and len(rows) == len(wanted_ids):
                break
        if wanted_ids is not None and len(rows) == len(wanted_ids):
            break
    if not rows:
        raise RuntimeError(f"no usable question/answer rows in {path}")
    return rows


def _split(record_id: str, seed: int, val_fraction: float) -> str:
    token = hashlib.sha256(f"{seed}:{record_id}".encode()).digest()
    value = int.from_bytes(token[:8], "big") / float(2**64)
    return "val" if value < val_fraction else "train"


def _file_signature(path: Path) -> dict[str, Any]:
    stat = path.stat()
    return {"path": str(path), "bytes": stat.st_size, "mtime_ns": stat.st_mtime_ns}


def dataset_signature(shards: Sequence[Path], source_jsonl: str | Path) -> str:
    payload = [_file_signature(path) for path in _source_files(source_jsonl)]
    for shard in shards:
        payload.extend(_file_signature(shard / name) for name in REQUIRED_SHARD_FILES)
    raw = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(raw.encode()).hexdigest()


def build_manifest(cfg: AppConfig, force: bool = False) -> dict[str, Any]:
    """Validate shards and write a compact, restart-safe record manifest."""
    work = cfg.work_dir
    work.mkdir(parents=True, exist_ok=True)
    manifest_path = work / "manifest.jsonl"
    report_path = work / "dataset_report.json"
    quarantine_path = work / "quarantine.jsonl"
    shards = discover_shards(cfg.paths.data_dir)
    signature = dataset_signature(shards, cfg.paths.source_jsonl)
    record_limit = int(cfg.runtime.manifest_record_limit)
    if not force and manifest_path.is_file() and report_path.is_file():
        report = json.loads(report_path.read_text())
        if (
            report.get("dataset_signature") == signature
            and int(report.get("record_limit", 0)) == record_limit
            and report.get("complete")
        ):
            print(f"manifest already valid: {manifest_path}")
            return report

    wanted_ids: set[str] | None = None
    if record_limit:
        wanted_ids = set()
        candidate_budget = max(1024, record_limit * 20)
        for shard in shards:
            for raw_id in jsonl(shard / "ids.jsonl"):
                wanted_ids.add(str(raw_id))
                if len(wanted_ids) >= candidate_budget:
                    break
            if len(wanted_ids) >= candidate_budget:
                break
    sources = source_rows(cfg.paths.source_jsonl, wanted_ids=wanted_ids)
    manifest_tmp = manifest_path.with_suffix(".jsonl.tmp")
    quarantine_tmp = quarantine_path.with_suffix(".jsonl.tmp")
    good = bad = frames = tokens = 0
    split_counts = {"train": 0, "val": 0}
    hidden_dim = None
    validated_shards = 0
    with manifest_tmp.open("w", encoding="utf-8") as output, quarantine_tmp.open(
        "w", encoding="utf-8"
    ) as rejected:
        for shard_no, shard in enumerate(shards, 1):
            validated_shards += 1
            offsets = np.load(shard / "offsets.npy", mmap_mode="r")
            hidden_offsets = np.load(shard / "hidden_offsets.npy", mmap_mode="r")
            codes = np.load(shard / "codes.npy", mmap_mode="r")
            hidden = np.load(shard / "hidden.npy", mmap_mode="r")
            token_ids = np.load(shard / "token_ids.npy", mmap_mode="r")
            char_offsets = np.load(shard / "token_char_offsets.npy", mmap_mode="r")
            ids = list(jsonl(shard / "ids.jsonl"))
            alignments = list(jsonl(shard / "alignment.jsonl"))
            pieces = list(jsonl(shard / "pieces.jsonl"))
            n = len(ids)
            errors = []
            if offsets.ndim != 1 or len(offsets) != n + 1:
                errors.append(f"offsets shape {offsets.shape} for {n} rows")
            if hidden_offsets.ndim != 1 or len(hidden_offsets) != n + 1:
                errors.append(f"hidden_offsets shape {hidden_offsets.shape} for {n} rows")
            if len(alignments) != n or len(pieces) != n:
                errors.append("JSONL row counts disagree")
            if codes.ndim != 2 or codes.shape[1] != cfg.model.num_codebooks:
                errors.append(f"codes must be [frames,8], got {codes.shape}")
            if hidden.ndim != 2:
                errors.append(f"hidden must be rank 2, got {hidden.shape}")
            if token_ids.ndim != 1 or char_offsets.ndim != 2 or char_offsets.shape[1] != 2:
                errors.append("token arrays have invalid ranks")
            if errors:
                raise RuntimeError(f"invalid shard {shard}: {'; '.join(errors)}")
            if int(offsets[-1]) != codes.shape[0] or int(hidden_offsets[-1]) != hidden.shape[0]:
                raise RuntimeError(f"terminal offsets disagree with arrays in {shard}")
            if token_ids.shape[0] != hidden.shape[0] or char_offsets.shape[0] != hidden.shape[0]:
                raise RuntimeError(f"hidden/token arrays disagree in {shard}")
            if np.any(np.diff(offsets) < 0) or np.any(np.diff(hidden_offsets) < 0):
                raise RuntimeError(f"non-monotonic offsets in {shard}")
            if codes.size and (int(codes.min()) < 0 or int(codes.max()) >= cfg.model.codec_vocab):
                raise RuntimeError(f"code ID outside [0,{cfg.model.codec_vocab - 1}] in {shard}")
            hidden_dim = hidden_dim or int(hidden.shape[1])
            if int(hidden.shape[1]) != hidden_dim:
                raise RuntimeError(f"hidden dimension changed in {shard}")

            for row, raw_id in enumerate(ids):
                record_id = str(raw_id)
                try:
                    source = sources[record_id]
                    lo, hi = int(offsets[row]), int(offsets[row + 1])
                    hlo, hhi = int(hidden_offsets[row]), int(hidden_offsets[row + 1])
                    alignment = alignments[row]
                    words = alignment.get("words", [])
                    if hi <= lo or hhi <= hlo or not words:
                        raise ValueError("empty codes, hidden states, or word alignment")
                    if int(words[0]["start_frame"]) != 0 or int(words[-1]["end_frame"]) != hi - lo:
                        raise ValueError("word alignment does not cover all Mimi frames")
                    previous = 0
                    for word in words:
                        start, end = int(word["start_frame"]), int(word["end_frame"])
                        if start != previous or end <= start:
                            raise ValueError("word alignment is not gap-free and monotonic")
                        previous = end
                    split = _split(record_id, cfg.train.seed, cfg.train.val_fraction)
                    entry = {
                        "record_id": record_id,
                        "shard": str(shard),
                        "row": row,
                        "code_lo": lo,
                        "code_hi": hi,
                        "frames": hi - lo,
                        "legacy_tokens": hhi - hlo,
                        "split": split,
                        "question": source["question"],
                        "answer": source["answer"],
                        "words": words,
                    }
                    output.write(json.dumps(entry, ensure_ascii=False) + "\n")
                    good += 1
                    split_counts[split] += 1
                    frames += hi - lo
                    tokens += hhi - hlo
                    if record_limit and good >= record_limit:
                        break
                except Exception as exc:
                    bad += 1
                    rejected.write(
                        json.dumps({"record_id": record_id, "shard": str(shard), "row": row, "reason": str(exc)})
                        + "\n"
                    )
            print(f"validated shard {shard_no}/{len(shards)}: {shard.name} ({n} rows)", flush=True)
            if record_limit and good >= record_limit:
                print(f"local manifest limit reached: {good} records", flush=True)
                break
    if good < 4:
        raise RuntimeError(f"only {good} valid records found")
    manifest_tmp.replace(manifest_path)
    quarantine_tmp.replace(quarantine_path)
    report = {
        "complete": True,
        "dataset_signature": signature,
        "record_limit": record_limit,
        "records": good,
        "quarantined": bad,
        "frames": frames,
        "audio_hours": frames / 12.5 / 3600.0,
        "legacy_tokens": tokens,
        "hidden_dim": hidden_dim,
        "shards": validated_shards,
        "available_shards": len(shards),
        "split_counts": split_counts,
        "manifest": str(manifest_path),
        "quarantine": str(quarantine_path),
    }
    estimated_feature_bytes = tokens * (int(hidden_dim or 0) * 2 + 12)
    report["estimated_feature_gib"] = estimated_feature_bytes / 2**30
    report["recommended_working_gib"] = estimated_feature_bytes * 1.25 / 2**30 + 2.0
    report["free_working_gib"] = shutil.disk_usage(work).free / 2**30
    atomic_json(report_path, report)
    return report


def allocate_token_frames(
    text: str, token_offsets: np.ndarray, words: Sequence[dict[str, Any]], total_frames: int
) -> np.ndarray:
    """Allocate a gap-free frame span to every assistant token, including punctuation."""
    n = int(token_offsets.shape[0])
    if n < 1 or total_frames < 1 or not words:
        raise ValueError("alignment allocation received empty input")
    word_char_centers: list[float] = []
    word_frame_centers: list[float] = []
    cursor = 0
    lower = text.lower()
    for word in words:
        raw = str(word["word"])
        found = text.find(raw, cursor)
        if found < 0:
            found = lower.find(raw.lower(), cursor)
        if found < 0:
            raise ValueError(f"aligned word {raw!r} is absent from answer")
        word_char_centers.append(found + len(raw) / 2.0)
        word_frame_centers.append((float(word["start_frame"]) + float(word["end_frame"])) / 2.0)
        cursor = found + len(raw)
    token_centers = []
    for start, end in np.asarray(token_offsets, dtype=np.int64):
        if end > start:
            token_centers.append((float(start) + float(end)) / 2.0)
        else:
            token_centers.append(token_centers[-1] if token_centers else 0.0)
    centers = np.interp(token_centers, word_char_centers, word_frame_centers)
    boundaries = np.zeros(n + 1, dtype=np.int32)
    boundaries[-1] = int(total_frames)
    if n > 1:
        boundaries[1:-1] = np.rint((centers[:-1] + centers[1:]) / 2.0).astype(np.int32)
    boundaries = np.maximum.accumulate(np.clip(boundaries, 0, total_frames))
    boundaries[-1] = total_frames
    durations = np.diff(boundaries).astype(np.int32)
    if durations.sum() != total_frames or np.any(durations < 0):
        raise RuntimeError("token allocation is not gap-free")
    return durations


class MMapLRU:
    def __init__(self, capacity: int = 8):
        self.capacity = max(1, int(capacity))
        self._items: OrderedDict[str, np.ndarray] = OrderedDict()

    def get(self, path: str | Path, *, dtype: Any = None, shape: tuple[int, ...] | None = None) -> np.ndarray:
        key = f"{path}|{dtype}|{shape}"
        if key in self._items:
            value = self._items.pop(key)
            self._items[key] = value
            return value
        if dtype is None:
            value = np.load(path, mmap_mode="r")
        else:
            if shape is None:
                raise ValueError("raw memmap requires shape")
            value = np.memmap(path, mode="r", dtype=dtype, shape=shape)
        self._items[key] = value
        while len(self._items) > self.capacity:
            self._items.popitem(last=False)
        return value


def load_manifest(path: str | Path, split: str | None = None, limit: int = 0) -> list[dict[str, Any]]:
    rows = []
    for item in jsonl(path):
        if split is None or item["split"] == split:
            rows.append(item)
            if limit > 0 and len(rows) >= limit:
                break
    return rows


def load_feature_index(path: str | Path) -> dict[str, dict[str, Any]]:
    return {str(row["record_id"]): row for row in jsonl(path)}


def _qwen_embedding_rows(model_dir: str | Path, token_ids: np.ndarray) -> np.ndarray:
    """Read Qwen's embedding tensor without constructing the full language model."""
    from safetensors import safe_open

    root = Path(model_dir)
    keys = (
        "model.embed_tokens.weight",
        "model.model.embed_tokens.weight",
        "transformer.wte.weight",
        "lm_head.weight",
    )
    tensor_path: Path | None = None
    tensor_key: str | None = None
    indices = sorted(root.glob("*.safetensors.index.json"))
    if indices:
        weight_map = json.loads(indices[0].read_text(encoding="utf-8")).get("weight_map", {})
        for key in keys:
            filename = weight_map.get(key)
            if filename:
                tensor_path, tensor_key = root / filename, key
                break
    if tensor_path is None:
        for path in sorted(root.glob("*.safetensors")):
            with safe_open(path, framework="pt", device="cpu") as handle:
                available = set(handle.keys())
            tensor_key = next((key for key in keys if key in available), None)
            if tensor_key is not None:
                tensor_path = path
                break
    if tensor_path is None or tensor_key is None:
        raise FileNotFoundError(
            f"could not find Qwen input embeddings in safetensors under {root}"
        )
    print(f"loading Qwen embedding only: {tensor_path.name}:{tensor_key}", flush=True)
    with safe_open(tensor_path, framework="pt", device="cpu") as handle:
        table = handle.get_tensor(tensor_key)
    ids = np.asarray(token_ids, dtype=np.int64)
    if ids.size and (int(ids.min()) < 0 or int(ids.max()) >= int(table.shape[0])):
        raise RuntimeError(
            f"token ID range [{ids.min()},{ids.max()}] exceeds {table.shape[0]} embedding rows"
        )
    import torch

    selected = table.index_select(0, torch.from_numpy(ids)).float().numpy().astype(np.float16)
    del table
    return selected


def _feature_worker(cfg: AppConfig) -> tuple[Any, Any, int, int, int, Any]:
    import torch
    import torch.distributed as dist

    rank = int(os.environ.get("RANK", "0"))
    world = int(os.environ.get("WORLD_SIZE", "1"))
    local_rank = int(os.environ.get("LOCAL_RANK", rank))
    use_cuda = cfg.runtime.device == "cuda" or (
        cfg.runtime.device == "auto" and torch.cuda.is_available()
    )
    if cfg.runtime.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("runtime.device='cuda' but CUDA is not available")
    if use_cuda:
        torch.cuda.set_device(local_rank)
    if world > 1:
        dist.init_process_group("nccl" if use_cuda else "gloo")
    device = torch.device(f"cuda:{local_rank}" if use_cuda else "cpu")
    if device.type == "cpu":
        torch.set_num_threads(cfg.runtime.cpu_threads)
    return torch, dist, rank, world, local_rank, device


def _ensure_manifest(cfg: AppConfig, rank: int, world: int, dist: Any) -> None:
    """Build the prerequisite once, then release all feature workers."""
    manifest = cfg.work_dir / "manifest.jsonl"
    report = cfg.work_dir / "dataset_report.json"
    if rank == 0:
        if not manifest.is_file() or not report.is_file():
            print("manifest is missing; building it before feature preparation", flush=True)
        build_manifest(cfg)
    if world > 1:
        dist.barrier()
    if not manifest.is_file() or not report.is_file():
        raise RuntimeError("manifest preparation did not produce its required outputs")


def _feature_complete(
    cfg: AppConfig, done_path: Path, selected: Sequence[dict[str, Any]], torch: Any,
    dist: Any, world: int, device: Any,
) -> tuple[bool, bool]:
    local = False
    final_path = done_path.parent / "features.done.json"
    final_valid = False
    if final_path.is_file():
        final_marker = json.loads(final_path.read_text())
        final_valid = final_marker.get("feature_fingerprint") == cfg.feature_fingerprint()
    if done_path.is_file():
        marker = json.loads(done_path.read_text())
        local = (
            final_valid
            and marker.get("feature_mode") == cfg.feature_mode
            and marker.get("feature_fingerprint") == cfg.feature_fingerprint()
            and marker.get("records") == len(selected)
        )
    complete = local
    if world > 1:
        flag = torch.tensor(int(local), device=device, dtype=torch.int32)
        dist.all_reduce(flag, op=dist.ReduceOp.MIN)
        complete = bool(flag.item())
    return local, complete


def _merge_feature_indices(
    cfg: AppConfig, feature_dir: Path, world: int, hidden_dim: int,
    embedding_values: np.ndarray, runtime_stats: dict[str, Any] | None = None,
) -> None:
    indices: list[dict[str, Any]] = []
    all_ids = []
    for worker in range(world):
        indices.extend(jsonl(feature_dir / f"index_rank{worker:02d}.jsonl"))
        all_ids.append(np.load(feature_dir / f"used_ids_rank{worker:02d}.npy"))
    indices.sort(key=lambda item: str(item["record_id"]))
    temporary = feature_dir / "index.jsonl.tmp"
    with temporary.open("w", encoding="utf-8") as handle:
        for item in indices:
            handle.write(json.dumps(item) + "\n")
    temporary.replace(feature_dir / "index.jsonl")
    unique = np.unique(np.concatenate(all_ids)).astype(np.int32)
    if int(embedding_values.shape[0]) != int(unique.size):
        raise RuntimeError("embedding rows do not match the used token ID set")
    np.save(feature_dir / "embedding_ids.npy", unique)
    np.save(feature_dir / "embedding_values.npy", embedding_values)
    marker = {
        "feature_fingerprint": cfg.feature_fingerprint(),
        "feature_mode": cfg.feature_mode,
        "records": len(indices),
        "vocabulary_rows": int(unique.size),
        "hidden_dim": hidden_dim,
        "embedding_dim": int(embedding_values.shape[1]),
    }
    marker.update(runtime_stats or {})
    atomic_json(feature_dir / "features.done.json", marker)


def _peak_rss_mib() -> float | None:
    status = Path("/proc/self/status")
    if not status.is_file():
        return None
    for line in status.read_text(encoding="utf-8").splitlines():
        if line.startswith("VmHWM:"):
            return float(line.split()[1]) / 1024.0
    return None


def _prepare_legacy_answer_features(cfg: AppConfig) -> None:
    """Index answer-only shard features; do not run the Qwen language model."""
    torch, dist, rank, world, local_rank, device = _feature_worker(cfg)
    _ensure_manifest(cfg, rank, world, dist)
    print(
        f"rank {rank}/{world}: reusing answer-only shard hidden/token features; "
        f"local_rank={local_rank}",
        flush=True,
    )
    rows = load_manifest(cfg.work_dir / "manifest.jsonl")
    selected = [row for index, row in enumerate(rows) if index % world == rank]
    feature_dir = cfg.work_dir / "features"
    feature_dir.mkdir(parents=True, exist_ok=True)
    index_path = feature_dir / f"index_rank{rank:02d}.jsonl"
    done_path = feature_dir / f"rank{rank:02d}.done.json"
    local_complete, all_complete = _feature_complete(
        cfg, done_path, selected, torch, dist, world, device
    )
    if all_complete:
        print(f"rank {rank}: answer-only features already complete", flush=True)
        if world > 1:
            dist.barrier()
            dist.destroy_process_group()
        return
    if local_complete:
        print(f"rank {rank}: rebuilding index because another rank is incomplete", flush=True)
    dataset_report = json.loads((cfg.work_dir / "dataset_report.json").read_text())
    hidden_dim = int(dataset_report["hidden_dim"])
    cache = MMapLRU(12)
    token_offset = 0
    used_ids: set[int] = set()
    temporary = Path(str(index_path) + ".tmp")
    with temporary.open("w", encoding="utf-8") as output:
        for position, row in enumerate(selected, 1):
            shard = Path(row["shard"])
            shard_row = int(row["row"])
            hidden_offsets = cache.get(shard / "hidden_offsets.npy")
            hidden_lo = int(hidden_offsets[shard_row])
            hidden_hi = int(hidden_offsets[shard_row + 1])
            token_ids = np.asarray(
                cache.get(shard / "token_ids.npy")[hidden_lo:hidden_hi], dtype=np.int32
            )
            char_offsets = np.asarray(
                cache.get(shard / "token_char_offsets.npy")[hidden_lo:hidden_hi],
                dtype=np.int32,
            )
            if token_ids.size < 1 or char_offsets.shape != (token_ids.size, 2):
                raise RuntimeError(f"invalid legacy answer features for {row['record_id']}")
            durations = allocate_token_frames(
                row["answer"], char_offsets, row["words"], int(row["frames"])
            )
            output.write(
                json.dumps(
                    {
                        "record_id": row["record_id"],
                        "source": "legacy_answer",
                        "rank": rank,
                        "token_offset": token_offset,
                        "tokens": int(token_ids.size),
                        "hidden_dim": hidden_dim,
                        "shard": str(shard),
                        "hidden_lo": hidden_lo,
                        "hidden_hi": hidden_hi,
                        "durations": durations.tolist(),
                    }
                )
                + "\n"
            )
            token_offset += int(token_ids.size)
            used_ids.update(int(value) for value in token_ids)
            if position == 1 or position % 500 == 0 or position == len(selected):
                print(f"rank {rank}: indexed {position}/{len(selected)}", flush=True)
    temporary.replace(index_path)
    np.save(
        feature_dir / f"used_ids_rank{rank:02d}.npy",
        np.asarray(sorted(used_ids), dtype=np.int32),
    )
    atomic_json(
        done_path,
        {
            "feature_fingerprint": cfg.feature_fingerprint(),
            "feature_mode": cfg.feature_mode,
            "records": len(selected),
            "tokens": token_offset,
            "hidden_dim": hidden_dim,
        },
    )
    if world > 1:
        dist.barrier()
    if rank == 0:
        all_ids = [
            np.load(feature_dir / f"used_ids_rank{worker:02d}.npy")
            for worker in range(world)
        ]
        unique = np.unique(np.concatenate(all_ids)).astype(np.int32)
        values = _qwen_embedding_rows(cfg.paths.qwen_model, unique)
        _merge_feature_indices(cfg, feature_dir, world, hidden_dim, values)
        print(
            f"answer-only feature preparation complete: {len(rows)} records; "
            f"no Qwen forward passes; {unique.size} embedding rows",
            flush=True,
        )
    if world > 1:
        dist.barrier()
        dist.destroy_process_group()


def _prepare_recomputed_features(cfg: AppConfig, include_chat_context: bool) -> None:
    """Recompute answer states, optionally after the deployment chat prefix."""
    from transformers import AutoModelForCausalLM, AutoTokenizer

    torch, dist, rank, world, local_rank, device = _feature_worker(cfg)
    started = time.perf_counter()
    _ensure_manifest(cfg, rank, world, dist)
    source_mode = "deployment_context" if include_chat_context else "answer_only_recompute"
    print(
        f"rank {rank}/{world}: {source_mode} feature worker; "
        f"local_rank={local_rank}; Qwen={cfg.paths.qwen_model}",
        flush=True,
    )
    dataset_report = json.loads((cfg.work_dir / "dataset_report.json").read_text())
    feature_dir = cfg.work_dir / "features"
    existing_bytes = sum(
        path.stat().st_size for path in feature_dir.glob("**/*") if path.is_file()
    ) if feature_dir.exists() else 0
    available_bytes = shutil.disk_usage(cfg.work_dir).free + existing_bytes
    required_bytes = int(dataset_report["estimated_feature_gib"] * 2**30 * 1.25 + 2 * 2**30)
    if available_bytes < required_bytes:
        raise RuntimeError(
            f"deployment-context sidecars need about {required_bytes / 2**30:.1f} GiB including reserve; "
            f"only {available_bytes / 2**30:.1f} GiB is available"
        )
    rows = load_manifest(cfg.work_dir / "manifest.jsonl")
    selected = [row for index, row in enumerate(rows) if index % world == rank]
    feature_dir.mkdir(parents=True, exist_ok=True)
    hidden_path = feature_dir / f"hidden_rank{rank:02d}.f16"
    ids_path = feature_dir / f"token_ids_rank{rank:02d}.i32"
    chars_path = feature_dir / f"char_offsets_rank{rank:02d}.i32"
    index_path = feature_dir / f"index_rank{rank:02d}.jsonl"
    done_path = feature_dir / f"rank{rank:02d}.done.json"
    local_complete, all_complete = _feature_complete(
        cfg, done_path, selected, torch, dist, world, device
    )
    if all_complete:
        print(f"rank {rank}: {source_mode} features already complete", flush=True)
        if world > 1:
            dist.barrier()
            dist.destroy_process_group()
        return
    if local_complete:
        print(f"rank {rank}: rebuilding sidecar because another rank is incomplete", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(cfg.paths.qwen_model, use_fast=True)
    dtype = torch.float16 if device.type == "cuda" else torch.float32
    print(f"rank {rank}: loading full Qwen onto {device}", flush=True)
    model = AutoModelForCausalLM.from_pretrained(
        cfg.paths.qwen_model, dtype=dtype, low_cpu_mem_usage=True
    ).to(device)
    print(f"rank {rank}: Qwen loaded; beginning {len(selected)} records", flush=True)
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    hidden_dim = int(model.config.hidden_size)
    token_offset = 0
    used_ids: set[int] = set()
    suffix = ".tmp"
    with open(str(hidden_path) + suffix, "wb") as hidden_out, open(
        str(ids_path) + suffix, "wb"
    ) as ids_out, open(str(chars_path) + suffix, "wb") as chars_out, open(
        str(index_path) + suffix, "w", encoding="utf-8"
    ) as index_out:
        for position, row in enumerate(selected, 1):
            answer = tokenizer(
                row["answer"], add_special_tokens=False, return_offsets_mapping=True
            )
            answer_ids = [int(value) for value in answer["input_ids"]]
            offsets = np.asarray(answer["offset_mapping"], dtype=np.int32)
            if not answer_ids or offsets.shape != (len(answer_ids), 2):
                raise RuntimeError(f"invalid assistant tokenization for {row['record_id']}")
            prefix_ids = []
            if include_chat_context:
                prefix_ids = tokenizer.apply_chat_template(
                    [{"role": "user", "content": row["question"]}],
                    tokenize=True,
                    add_generation_prompt=True,
                )
            full = torch.tensor([list(prefix_ids) + answer_ids], device=device, dtype=torch.long)
            with torch.inference_mode(), torch.autocast(
                device_type=device.type, enabled=device.type == "cuda"
            ):
                result = model(full, output_hidden_states=True, use_cache=False)
                states = (
                    result.hidden_states[-1][0, -len(answer_ids) :]
                    .float().cpu().numpy().astype(np.float16)
                )
            states.tofile(hidden_out)
            np.asarray(answer_ids, dtype=np.int32).tofile(ids_out)
            offsets.tofile(chars_out)
            durations = allocate_token_frames(
                row["answer"], offsets, row["words"], int(row["frames"])
            )
            index_out.write(
                json.dumps(
                    {
                        "record_id": row["record_id"],
                        "source": source_mode,
                        "rank": rank,
                        "token_offset": token_offset,
                        "tokens": len(answer_ids),
                        "hidden_dim": hidden_dim,
                        "durations": durations.tolist(),
                    }
                )
                + "\n"
            )
            token_offset += len(answer_ids)
            used_ids.update(answer_ids)
            if position == 1 or position % 100 == 0 or position == len(selected):
                print(f"rank {rank}: features {position}/{len(selected)}", flush=True)
    Path(str(hidden_path) + suffix).replace(hidden_path)
    Path(str(ids_path) + suffix).replace(ids_path)
    Path(str(chars_path) + suffix).replace(chars_path)
    Path(str(index_path) + suffix).replace(index_path)
    np.save(
        feature_dir / f"used_ids_rank{rank:02d}.npy",
        np.asarray(sorted(used_ids), dtype=np.int32),
    )
    atomic_json(
        done_path,
        {
            "feature_fingerprint": cfg.feature_fingerprint(),
            "feature_mode": cfg.feature_mode,
            "records": len(selected),
            "tokens": token_offset,
            "hidden_dim": hidden_dim,
        },
    )
    if world > 1:
        dist.barrier()
    if rank == 0:
        all_ids = [
            np.load(feature_dir / f"used_ids_rank{worker:02d}.npy")
            for worker in range(world)
        ]
        unique = np.unique(np.concatenate(all_ids)).astype(np.int32)
        embedding = model.get_input_embeddings().weight.detach()
        values = (
            embedding[torch.from_numpy(unique.astype(np.int64)).to(device)]
            .float().cpu().numpy().astype(np.float16)
        )
        _merge_feature_indices(
            cfg,
            feature_dir,
            world,
            hidden_dim,
            values,
            {
                "preparation_device": str(device),
                "preparation_seconds_rank0": time.perf_counter() - started,
                "preparation_peak_rss_mib_rank0": _peak_rss_mib(),
            },
        )
    if world > 1:
        dist.barrier()
        dist.destroy_process_group()


def prepare_qwen_features(cfg: AppConfig) -> None:
    """Prepare the selected answer-only or deployment-context feature contract."""
    if cfg.feature_mode == "legacy_answer":
        _prepare_legacy_answer_features(cfg)
    elif cfg.feature_mode == "answer_only_recompute":
        _prepare_recomputed_features(cfg, include_chat_context=False)
    else:
        _prepare_recomputed_features(cfg, include_chat_context=True)


class SpeechDataset:
    def __init__(self, cfg: AppConfig, split: str, limit: int = 0):
        import torch

        self.torch = torch
        self.cfg = cfg
        self.rows = load_manifest(cfg.work_dir / "manifest.jsonl", split=split, limit=limit)
        self.features = load_feature_index(cfg.work_dir / "features" / "index.jsonl")
        feature_marker = json.loads(
            (cfg.work_dir / "features" / "features.done.json").read_text()
        )
        if feature_marker.get("feature_fingerprint") != cfg.feature_fingerprint():
            raise RuntimeError("feature sidecars do not match the configured dataset/Qwen context")
        ids = np.load(cfg.work_dir / "features" / "embedding_ids.npy", mmap_mode="r")
        values = np.load(cfg.work_dir / "features" / "embedding_values.npy", mmap_mode="r")
        self.embedding_by_id = {int(token): index for index, token in enumerate(ids)}
        self.embedding_values = values
        self.cache = MMapLRU(12)
        self.lengths = [int(row["frames"]) for row in self.rows]
        missing = [row["record_id"] for row in self.rows if row["record_id"] not in self.features]
        if missing:
            raise RuntimeError(f"{len(missing)} records lack prepared features; first={missing[:3]}")

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> dict[str, Any]:
        torch = self.torch
        row = self.rows[index]
        ref = self.features[row["record_id"]]
        source = str(ref.get("source", "deployment_context"))
        rank = int(ref["rank"])
        offset, count, hidden_dim = int(ref["token_offset"]), int(ref["tokens"]), int(ref["hidden_dim"])
        feature_dir = self.cfg.work_dir / "features"
        if source == "legacy_answer":
            shard = Path(ref["shard"])
            hidden_lo, hidden_hi = int(ref["hidden_lo"]), int(ref["hidden_hi"])
            hidden = self.cache.get(shard / "hidden.npy")[hidden_lo:hidden_hi]
            token_ids = self.cache.get(shard / "token_ids.npy")[hidden_lo:hidden_hi]
            char_offsets = self.cache.get(shard / "token_char_offsets.npy")[hidden_lo:hidden_hi]
        elif source in {"answer_only_recompute", "deployment_context"}:
            marker = json.loads((feature_dir / f"rank{rank:02d}.done.json").read_text())
            total_tokens = int(marker["tokens"])
            hidden = self.cache.get(
                feature_dir / f"hidden_rank{rank:02d}.f16",
                dtype=np.float16,
                shape=(total_tokens, hidden_dim),
            )[offset : offset + count]
            token_ids = self.cache.get(
                feature_dir / f"token_ids_rank{rank:02d}.i32",
                dtype=np.int32,
                shape=(total_tokens,),
            )[offset : offset + count]
            char_offsets = self.cache.get(
                feature_dir / f"char_offsets_rank{rank:02d}.i32",
                dtype=np.int32,
                shape=(total_tokens, 2),
            )[offset : offset + count]
        else:
            raise RuntimeError(f"unknown feature source {source!r} for {row['record_id']}")
        if int(hidden.shape[0]) != count or int(hidden.shape[1]) != hidden_dim:
            raise RuntimeError(f"hidden shape mismatch for {row['record_id']}: {hidden.shape}")
        embedding_rows = [self.embedding_by_id[int(token)] for token in token_ids]
        embeddings = np.asarray(self.embedding_values[embedding_rows], dtype=np.float32)
        codes_all = self.cache.get(Path(row["shard"]) / "codes.npy")
        codes = np.asarray(codes_all[int(row["code_lo"]) : int(row["code_hi"])], dtype=np.int64)
        durations = np.asarray(ref["durations"], dtype=np.int64)
        if durations.shape[0] != count or int(durations.sum()) != codes.shape[0]:
            raise RuntimeError(f"duration/code mismatch for {row['record_id']}")
        return {
            "record_id": row["record_id"],
            "text": row["answer"],
            "hidden": torch.from_numpy(np.asarray(hidden, dtype=np.float32)),
            "embeddings": torch.from_numpy(embeddings),
            "durations": torch.from_numpy(durations),
            "codes": torch.from_numpy(codes.copy()),
            "char_offsets": np.asarray(char_offsets, dtype=np.int32).copy(),
            "words": row["words"],
        }


def collate_speech(items: Sequence[dict[str, Any]]) -> dict[str, Any]:
    import torch

    batch = len(items)
    max_tokens = max(int(item["hidden"].shape[0]) for item in items)
    max_frames = max(int(item["codes"].shape[0]) for item in items)
    hidden_dim = int(items[0]["hidden"].shape[1])
    embedding_dim = int(items[0]["embeddings"].shape[1])
    hidden = torch.zeros(batch, max_tokens, hidden_dim)
    embeddings = torch.zeros(batch, max_tokens, embedding_dim)
    durations = torch.zeros(batch, max_tokens, dtype=torch.long)
    token_mask = torch.zeros(batch, max_tokens, dtype=torch.bool)
    codes = torch.zeros(batch, max_frames, 8, dtype=torch.long)
    frame_mask = torch.zeros(batch, max_frames, dtype=torch.bool)
    owners = torch.zeros(batch, max_frames, dtype=torch.long)
    for b, item in enumerate(items):
        n, frames = item["hidden"].shape[0], item["codes"].shape[0]
        hidden[b, :n] = item["hidden"]
        embeddings[b, :n] = item["embeddings"]
        durations[b, :n] = item["durations"]
        token_mask[b, :n] = True
        codes[b, :frames] = item["codes"]
        frame_mask[b, :frames] = True
        owner = torch.repeat_interleave(torch.arange(n), item["durations"])
        if owner.numel() != frames:
            raise RuntimeError("frame-owner construction failed")
        owners[b, :frames] = owner
    return {
        "record_ids": [item["record_id"] for item in items],
        "texts": [item["text"] for item in items],
        "hidden": hidden,
        "embeddings": embeddings,
        "durations": durations,
        "token_mask": token_mask,
        "codes": codes,
        "frame_mask": frame_mask,
        "owners": owners,
    }


class DistributedFrameBatchSampler:
    """Deterministic variable-size batches with an equal step count on every rank."""

    def __init__(
        self,
        lengths: Sequence[int],
        frame_budget: int,
        max_batch_size: int,
        rank: int,
        world_size: int,
        seed: int,
        shuffle: bool,
    ):
        self.lengths = list(map(int, lengths))
        self.frame_budget = int(frame_budget)
        self.max_batch_size = int(max_batch_size)
        self.rank = int(rank)
        self.world_size = int(world_size)
        self.seed = int(seed)
        self.shuffle = bool(shuffle)
        self.epoch = 0

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    def _global_batches(self) -> list[list[int]]:
        indices = list(range(len(self.lengths)))
        rng = random.Random(self.seed + self.epoch)
        if self.shuffle:
            rng.shuffle(indices)
        bucket = max(32, self.max_batch_size * 16)
        ordered: list[int] = []
        for start in range(0, len(indices), bucket):
            part = indices[start : start + bucket]
            part.sort(key=self.lengths.__getitem__)
            if self.shuffle and rng.random() < 0.5:
                part.reverse()
            ordered.extend(part)
        batches: list[list[int]] = []
        current: list[int] = []
        current_frames = 0
        for index in ordered:
            length = self.lengths[index]
            if length > self.frame_budget:
                raise RuntimeError(
                    f"sample {index} has {length} frames, above per-GPU budget {self.frame_budget}"
                )
            if current and (current_frames + length > self.frame_budget or len(current) >= self.max_batch_size):
                batches.append(current)
                current, current_frames = [], 0
            current.append(index)
            current_frames += length
        if current:
            batches.append(current)
        if self.shuffle:
            rng.shuffle(batches)
        if not batches:
            return []
        original = list(batches)
        cursor = 0
        while len(batches) % self.world_size:
            batches.append(list(original[cursor % len(original)]))
            cursor += 1
        return batches

    def __iter__(self) -> Iterator[list[int]]:
        yield from self._global_batches()[self.rank :: self.world_size]

    def __len__(self) -> int:
        return len(self._global_batches()) // self.world_size
