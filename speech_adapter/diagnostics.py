from __future__ import annotations

import html
import json
import math
import time
from functools import lru_cache
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from .config import AppConfig, atomic_json, sha256_file
from .data import SpeechDataset, discover_shards, load_manifest

SAMPLE_RATE = 24_000
FRAME_RATE = 12.5


def _providers() -> list[str]:
    import onnxruntime as ort

    available = ort.get_available_providers()
    preferred = ["CUDAExecutionProvider", "CPUExecutionProvider"]
    return [provider for provider in preferred if provider in available]


class OnnxMimi:
    def __init__(self, encoder: str, decoder: str, num_codebooks: int = 8):
        import onnxruntime as ort

        self.encoder = ort.InferenceSession(encoder, providers=_providers())
        self.decoder = ort.InferenceSession(decoder, providers=_providers())
        self.num_codebooks = int(num_codebooks)
        decoder_shape = self.decoder.get_inputs()[0].shape
        decoder_books = decoder_shape[1] if len(decoder_shape) > 1 else None
        if decoder_books != self.num_codebooks:
            raise RuntimeError(
                f"Mimi decoder expects {decoder_books!r} codebooks, but the adapter produces "
                f"{self.num_codebooks}. Do not zero-pad missing residual codebooks: use a true "
                f"{self.num_codebooks}-codebook decoder ONNX."
            )

    def encode(self, waveform: np.ndarray) -> np.ndarray:
        model_input = self.encoder.get_inputs()[0]
        audio = np.asarray(waveform, dtype=np.float32).reshape(1, 1, -1)
        raw = np.asarray(self.encoder.run(None, {model_input.name: audio})[0])
        if raw.ndim != 3 or raw.shape[0] != 1:
            raise RuntimeError(f"unexpected Mimi encoder output {raw.shape}")
        raw = raw[0]
        if raw.shape[0] in (self.num_codebooks, 32):
            codes = raw[: self.num_codebooks].T
        elif raw.shape[1] in (self.num_codebooks, 32):
            codes = raw[:, : self.num_codebooks]
        else:
            raise RuntimeError(f"cannot locate codebook axis in Mimi output {raw.shape}")
        codes = np.asarray(codes, dtype=np.int64)
        if codes.ndim != 2 or codes.shape[1] != self.num_codebooks or codes.size == 0:
            raise RuntimeError(
                f"normalized Mimi codes must be [frames,{self.num_codebooks}], got {codes.shape}"
            )
        if int(codes.min()) < 0 or int(codes.max()) >= 2048:
            raise RuntimeError("Mimi encoder produced an invalid code ID")
        return codes

    def decode(self, codes: np.ndarray) -> np.ndarray:
        model_input = self.decoder.get_inputs()[0]
        values = np.asarray(codes, dtype=np.int64)
        if values.ndim != 2 or values.shape[1] != self.num_codebooks:
            raise ValueError(
                f"decode expects [frames,{self.num_codebooks}], got {values.shape}"
            )
        transposed = values.T[None]
        output = self.decoder.run(None, {model_input.name: transposed})[0]
        waveform = np.asarray(output, dtype=np.float32).reshape(-1)
        if not waveform.size or not np.isfinite(waveform).all():
            raise RuntimeError("Mimi decoder returned empty or non-finite audio")
        return waveform


def inspect_mimi_contract(cfg: AppConfig) -> dict[str, Any]:
    """Verify that codec targets and decoder agree on the eight-book contract."""
    import onnxruntime as ort

    expected = int(cfg.model.num_codebooks)
    required = {
        "encoder": Path(cfg.paths.mimi_encoder),
        "decoder": Path(cfg.paths.mimi_decoder),
    }
    missing = [f"{name}={path}" for name, path in required.items() if not path.is_file()]
    if missing:
        raise FileNotFoundError("missing Mimi ONNX assets: " + ", ".join(missing))
    encoder = ort.InferenceSession(str(required["encoder"]), providers=_providers())
    decoder = ort.InferenceSession(str(required["decoder"]), providers=_providers())
    encoder_shape = encoder.get_outputs()[0].shape
    decoder_shape = decoder.get_inputs()[0].shape
    encoder_books = encoder_shape[1] if len(encoder_shape) > 1 else None
    decoder_books = decoder_shape[1] if len(decoder_shape) > 1 else None
    encoder_ok = isinstance(encoder_books, int) and encoder_books >= expected
    decoder_ok = decoder_books == expected
    decoder_smoke = False
    decoder_samples = 0
    if decoder_ok:
        test_codes = np.zeros((1, expected, 2), dtype=np.int64)
        test_audio = np.asarray(
            decoder.run(None, {decoder.get_inputs()[0].name: test_codes})[0]
        )
        decoder_samples = int(test_audio.size)
        decoder_smoke = bool(test_audio.size and np.isfinite(test_audio).all())
    report = {
        "automatic_pass": bool(encoder_ok and decoder_ok and decoder_smoke),
        "expected_codebooks": expected,
        "encoder_output_shape": encoder_shape,
        "encoder_output_codebooks": encoder_books,
        "encoder_contract": (
            f"first {expected} sequential residual codebooks are used"
            if encoder_ok else "incompatible"
        ),
        "decoder_input_shape": decoder_shape,
        "decoder_input_codebooks": decoder_books,
        "decoder_contract": "exact" if decoder_ok else "incompatible",
        "decoder_smoke_pass": decoder_smoke,
        "decoder_smoke_samples": decoder_samples,
        "zero_padding_allowed": False,
        "warning": None if decoder_ok else (
            f"The adapter produces {expected} books but this decoder consumes {decoder_books}. "
            "Padding the remaining books with token 0 adds real quantizer embeddings and corrupts audio."
        ),
        "encoder": str(required["encoder"]),
        "decoder": str(required["decoder"]),
    }
    atomic_json(cfg.work_dir / "gates" / "mimi_contract.json", report)
    return report


class StatefulMimi:
    def __init__(self, weights: str, device: str = "cuda:0"):
        import torch
        from moshi.models import loaders

        self.torch = torch
        self.device = torch.device(device)
        self.model = loaders.get_mimi(weights, device=self.device)
        self.model.set_num_codebooks(8)

    def decode(self, codes: np.ndarray) -> np.ndarray:
        chunks = []
        tensor = self.torch.from_numpy(np.asarray(codes, dtype=np.int64).T[None]).to(self.device)
        with self.torch.no_grad(), self.model.streaming(batch_size=1):
            for frame in range(tensor.shape[-1]):
                chunk = self.model.decode(tensor[:, :, frame : frame + 1])
                chunks.append(chunk.detach().float().cpu())
        if not chunks:
            raise RuntimeError("stateful Mimi produced no chunks")
        return self.torch.cat(chunks, dim=-1)[0, 0].numpy()


@lru_cache(maxsize=2)
def _kokoro_resources(model_path: str, voices_path: str):
    import onnxruntime as ort
    from kokoro_onnx.tokenizer import Tokenizer

    session = ort.InferenceSession(model_path, providers=_providers())
    tokenizer = Tokenizer()
    voices = np.fromfile(voices_path, dtype=np.float32).reshape(-1, 1, 256)
    return session, tokenizer, voices


def synthesize_kokoro(text: str, model_path: str, voices_path: str) -> np.ndarray:
    session, tokenizer, voices = _kokoro_resources(model_path, voices_path)
    re = __import__("re")
    sentences = [part.strip() for part in re.split(r"(?<=[.!?])\s+", text.strip()) if part.strip()]
    chunks: list[str] = []
    current = ""

    def flush() -> None:
        nonlocal current
        if current:
            chunks.append(current)
            current = ""

    def hard_split(phonemes: str) -> None:
        for offset in range(0, len(phonemes), 500):
            piece = phonemes[offset : offset + 500].strip()
            if piece:
                chunks.append(piece)

    for sentence in sentences or [text.strip()]:
        sentence_phonemes = tokenizer.phonemize(sentence, lang="en-us")
        if not sentence_phonemes:
            continue
        if len(sentence_phonemes) <= 500:
            combined = f"{current} {sentence_phonemes}".strip() if current else sentence_phonemes
            if len(combined) <= 500:
                current = combined
            else:
                flush()
                current = sentence_phonemes
            continue
        flush()
        buffered_text = ""
        buffered_phonemes = ""
        for word in sentence.split():
            trial_text = f"{buffered_text} {word}".strip()
            trial_phonemes = tokenizer.phonemize(trial_text, lang="en-us")
            if len(trial_phonemes) <= 500:
                buffered_text, buffered_phonemes = trial_text, trial_phonemes
                continue
            if buffered_phonemes:
                chunks.append(buffered_phonemes)
            word_phonemes = tokenizer.phonemize(word, lang="en-us")
            if len(word_phonemes) <= 500:
                buffered_text, buffered_phonemes = word, word_phonemes
            else:
                hard_split(word_phonemes)
                buffered_text = buffered_phonemes = ""
        if buffered_phonemes:
            chunks.append(buffered_phonemes)
    flush()
    audio_parts = []
    for piece in chunks:
        ids = tokenizer.tokenize(piece)
        if not ids:
            continue
        style = voices[min(len(ids), voices.shape[0] - 1)]
        tokens = np.asarray([[0, *ids, 0]], dtype=np.int64)
        output = session.run(
            None,
            {"input_ids": tokens, "style": style, "speed": np.ones(1, np.float32)},
        )[0]
        audio_parts.append(np.asarray(output, dtype=np.float32).reshape(-1))
    if not audio_parts:
        raise RuntimeError("Kokoro produced no audio")
    audio = np.concatenate(audio_parts)
    if not np.isfinite(audio).all():
        raise RuntimeError("Kokoro produced non-finite audio")
    return audio


def _audio_metrics(reference: np.ndarray, candidate: np.ndarray) -> dict[str, float]:
    length = min(reference.size, candidate.size)
    ref = np.asarray(reference[:length], dtype=np.float64)
    cand = np.asarray(candidate[:length], dtype=np.float64)
    error = ref - cand
    alpha = float(np.dot(cand, ref) / max(np.dot(ref, ref), 1e-12))
    target = alpha * ref
    noise = cand - target
    si_sdr = 10.0 * math.log10(max(np.dot(target, target), 1e-12) / max(np.dot(noise, noise), 1e-12))
    return {
        "samples": length,
        "duration_seconds": candidate.size / SAMPLE_RATE,
        "mse": float(np.mean(error**2)),
        "si_sdr_db": si_sdr,
        "rms_dbfs": float(20 * math.log10(max(np.sqrt(np.mean(cand**2)), 1e-12))),
        "peak": float(np.max(np.abs(cand))),
        "clipped_fraction": float(np.mean(np.abs(cand) >= 0.999)),
    }


def _write_wav(path: Path, waveform: np.ndarray) -> None:
    import soundfile as sf

    path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(path, np.asarray(waveform, dtype=np.float32), SAMPLE_RATE)


def run_gate_a(cfg: AppConfig) -> dict[str, Any]:
    for label, path in {
        "Kokoro model": cfg.paths.kokoro_model,
        "Kokoro voices": cfg.paths.kokoro_voices,
        "Mimi encoder": cfg.paths.mimi_encoder,
        "Mimi decoder": cfg.paths.mimi_decoder,
        "stateful Mimi weights": cfg.paths.mimi_weights,
    }.items():
        if not path or not Path(path).is_file():
            raise FileNotFoundError(f"Gate A requires {label}: {path!r}")
    rows = load_manifest(cfg.work_dir / "manifest.jsonl")
    rows.sort(key=lambda item: item["record_id"])
    rows = rows[: cfg.gates.codec_samples]
    offline = OnnxMimi(cfg.paths.mimi_encoder, cfg.paths.mimi_decoder)
    streaming = StatefulMimi(cfg.paths.mimi_weights)
    output_dir = cfg.work_dir / "gates" / "A_codec"
    results = []
    all_exact = True
    for position, row in enumerate(rows, 1):
        source = synthesize_kokoro(row["answer"], cfg.paths.kokoro_model, cfg.paths.kokoro_voices)
        fresh_codes = offline.encode(source)
        codes = np.load(Path(row["shard"]) / "codes.npy", mmap_mode="r")[
            int(row["code_lo"]) : int(row["code_hi"])
        ].astype(np.int64)
        exact = fresh_codes.shape == codes.shape and np.array_equal(fresh_codes, codes)
        all_exact = all_exact and exact
        offline_audio = offline.decode(codes)
        streaming_audio = streaming.decode(codes)
        sample_dir = output_dir / str(row["record_id"])
        _write_wav(sample_dir / "source.wav", source)
        _write_wav(sample_dir / "stored_offline.wav", offline_audio)
        _write_wav(sample_dir / "stored_streaming.wav", streaming_audio)
        result = {
            "record_id": row["record_id"],
            "codes_exact": exact,
            "frames": int(codes.shape[0]),
            "expected_seconds": codes.shape[0] / FRAME_RATE,
            "offline": _audio_metrics(source, offline_audio),
            "streaming": _audio_metrics(source, streaming_audio),
        }
        results.append(result)
        print(f"Gate A {position}/{len(rows)} id={row['record_id']} exact={exact}", flush=True)
    automatic = all_exact and all(
        item["offline"]["clipped_fraction"] < 0.01
        and item["streaming"]["clipped_fraction"] < 0.01
        and item["offline"]["samples"] > 0
        and item["streaming"]["samples"] > 0
        for item in results
    )
    report = {
        "automatic_pass": automatic,
        "manual_required": True,
        "samples": results,
        "asset_hashes": {
            "kokoro": sha256_file(cfg.paths.kokoro_model),
            "voices": sha256_file(cfg.paths.kokoro_voices),
            "mimi_encoder": sha256_file(cfg.paths.mimi_encoder),
            "mimi_decoder": sha256_file(cfg.paths.mimi_decoder),
            "mimi_weights": sha256_file(cfg.paths.mimi_weights),
        },
    }
    atomic_json(output_dir / "report.json", report)
    return report


def run_gate_b(cfg: AppConfig) -> dict[str, Any]:
    import matplotlib.pyplot as plt

    dataset = SpeechDataset(cfg, "val", limit=cfg.gates.alignment_samples)
    decoder = OnnxMimi(cfg.paths.mimi_encoder, cfg.paths.mimi_decoder)
    output_dir = cfg.work_dir / "gates" / "B_alignment"
    output_dir.mkdir(parents=True, exist_ok=True)
    links = []
    for index in range(len(dataset)):
        item = dataset[index]
        waveform = decoder.decode(item["codes"].numpy())
        fig, axes = plt.subplots(2, 1, figsize=(14, 7), constrained_layout=True)
        times = np.arange(waveform.size) / SAMPLE_RATE
        axes[0].plot(times, waveform, linewidth=0.5)
        axes[0].set(title=f"{item['record_id']} waveform", xlabel="seconds")
        axes[0].grid(alpha=0.2)
        axes[1].specgram(waveform, NFFT=1024, Fs=SAMPLE_RATE, noverlap=768)
        axes[1].set(title=item["text"], xlabel="seconds")
        frame_cursor = 0
        for token, duration in enumerate(item["durations"].tolist()):
            if duration:
                axes[0].axvline(frame_cursor / FRAME_RATE, color="tab:blue", alpha=0.15)
            frame_cursor += duration
        for word in item["words"]:
            boundary = int(word["start_frame"]) / FRAME_RATE
            axes[0].axvline(boundary, color="tab:red", alpha=0.35)
            axes[0].text(boundary, 0.95, str(word["word"]), rotation=90, transform=axes[0].get_xaxis_transform())
        name = f"{index:03d}_{item['record_id']}.png"
        fig.savefig(output_dir / name, dpi=120)
        plt.close(fig)
        links.append({"record_id": item["record_id"], "image": name})
    rows = "\n".join(
        f'<article><h2>{html.escape(str(item["record_id"]))}</h2><img src="{html.escape(item["image"])}" style="max-width:100%"></article>'
        for item in links
    )
    (output_dir / "index.html").write_text(
        "<!doctype html><meta charset='utf-8'><title>Alignment review</title>"
        "<h1>Gate B alignment review</h1><p>Review every panel, then use approve-gate.</p>" + rows,
        encoding="utf-8",
    )
    report = {"automatic_pass": len(links) >= cfg.gates.alignment_samples, "manual_required": True, "panels": links}
    atomic_json(output_dir / "report.json", report)
    return report


def model_structural_checks(cfg: AppConfig) -> dict[str, Any]:
    import torch

    from .model import StreamingSpeechAdapter, adapter_loss

    torch.manual_seed(cfg.train.seed)
    local_cfg = cfg.model
    feature_marker_path = cfg.work_dir / "features" / "features.done.json"
    if feature_marker_path.is_file():
        feature_marker = json.loads(feature_marker_path.read_text())
        text_dim = int(feature_marker["hidden_dim"])
        embedding_dim = int(
            np.load(cfg.work_dir / "features" / "embedding_values.npy", mmap_mode="r").shape[1]
        )
    else:
        text_dim = embedding_dim = 1536
    model = StreamingSpeechAdapter(text_dim, embedding_dim, local_cfg)
    model.assert_budget()
    model.eval()
    batch, tokens, frames = 1, 6, 12
    durations = torch.tensor([[2, 2, 2, 2, 2, 2]])
    owners = torch.repeat_interleave(torch.arange(tokens), durations[0]).unsqueeze(0)
    sample = {
        "hidden": torch.randn(batch, tokens, text_dim),
        "embeddings": torch.randn(batch, tokens, embedding_dim),
        "durations": durations,
        "token_mask": torch.ones(batch, tokens, dtype=torch.bool),
        "codes": torch.randint(0, 2048, (batch, frames, 8)),
        "frame_mask": torch.ones(batch, frames, dtype=torch.bool),
        "owners": owners,
    }
    logits, log_duration = model(sample)
    changed = {key: value.clone() if hasattr(value, "clone") else value for key, value in sample.items()}
    changed["codes"][:, 8:] = torch.randint(0, 2048, changed["codes"][:, 8:].shape)
    logits_changed, _ = model(changed)
    causal = torch.allclose(logits[:, :8], logits_changed[:, :8], atol=1e-5, rtol=1e-4)
    sequence_steps = local_cfg.audio_cache_frames + 5
    temporal_input = torch.randn(1, sequence_steps, local_cfg.model_dim)
    valid_mask = torch.ones(1, sequence_steps, dtype=torch.bool)
    full = temporal_input
    for block in model.temporal:
        full = block(full, valid_mask)
    head_dim = local_cfg.model_dim // local_cfg.attention_heads
    key_caches = [
        torch.zeros(1, local_cfg.attention_heads, local_cfg.audio_cache_frames, head_dim)
        for _ in model.temporal
    ]
    value_caches = [torch.zeros_like(cache) for cache in key_caches]
    incremental = []
    cache_valid = 0
    for step in range(sequence_steps):
        value = temporal_input[:, step : step + 1]
        next_valid = cache_valid
        for layer, block in enumerate(model.temporal):
            value, key_caches[layer], value_caches[layer], next_valid = block.step(
                value, key_caches[layer], value_caches[layer], cache_valid, step
            )
        cache_valid = next_valid
        incremental.append(value)
    incremental_tensor = torch.cat(incremental, dim=1)
    cache_equivalent = torch.allclose(full, incremental_tensor, atol=2e-5, rtol=2e-4)
    model.train()
    loss, _ = adapter_loss(logits, log_duration, sample, 4.0, 0.0, 0.0)
    loss.backward()
    cb0_gradient = model.hidden_projection.weight.grad is not None and float(
        model.hidden_projection.weight.grad.abs().sum()
    ) > 0
    report = {
        "parameter_count": model.parameter_count(),
        "fp16_weight_mib": model.parameter_count() * 2 / 2**20,
        "int8_weight_mib": model.parameter_count() / 2**20,
        "kv_cache_kib": (
            local_cfg.temporal_layers * 2 * local_cfg.audio_cache_frames
            * local_cfg.model_dim * 2 / 1024
        ),
        "causality_pass": bool(causal),
        "cache_equivalence_pass": bool(cache_equivalent),
        "cb0_shared_gradient_pass": bool(cb0_gradient),
        "automatic_pass": bool(
            causal and cache_equivalent and cb0_gradient
            and model.parameter_count() <= 12_000_000
        ),
    }
    atomic_json(cfg.work_dir / "gates" / "structural.json", report)
    return report


def update_gate(
    cfg: AppConfig,
    gate: str,
    automatic_pass: bool,
    manual_pass: bool | None,
    reviewed: int = 0,
    passed: int = 0,
    note: str = "",
) -> dict[str, Any]:
    gate = gate.upper()
    if gate not in "ABCDEF":
        raise ValueError("gate must be A, B, C, D, E, or F")
    path = cfg.work_dir / "gates" / "gate_report.json"
    report = json.loads(path.read_text()) if path.is_file() else {
        "feature_fingerprint": cfg.feature_fingerprint(), "gates": {}
    }
    if report.get("feature_fingerprint") != cfg.feature_fingerprint():
        raise RuntimeError("existing gate report belongs to different features")
    if gate == "B" and manual_pass:
        if reviewed < cfg.gates.alignment_samples or passed / max(1, reviewed) < cfg.gates.alignment_pass_fraction:
            raise ValueError("Gate B approval does not meet the configured review threshold")
    status = "passed" if automatic_pass and manual_pass is True else "pending"
    if manual_pass is False or not automatic_pass:
        status = "failed"
    report["gates"][gate] = {
        "status": status,
        "automatic_pass": bool(automatic_pass),
        "manual_pass": manual_pass,
        "reviewed": int(reviewed),
        "passed": int(passed),
        "note": note,
        "updated_unix": time.time(),
    }
    atomic_json(path, report)
    return report


def preflight(cfg: AppConfig, require_two_gpus: bool = True) -> dict[str, Any]:
    import os
    import shutil
    import torch

    required = {"data_dir": cfg.paths.data_dir, "source_jsonl": cfg.paths.source_jsonl, "qwen_model": cfg.paths.qwen_model}
    missing = [f"{name}={path}" for name, path in required.items() if not Path(path).exists()]
    if missing:
        raise FileNotFoundError("missing required inputs: " + ", ".join(missing))
    shards = discover_shards(cfg.paths.data_dir)
    gpu_count = torch.cuda.device_count()
    if require_two_gpus and gpu_count != 2:
        raise RuntimeError(f"select Kaggle GPU T4 x2; detected {gpu_count} CUDA devices")
    gpus = []
    for index in range(gpu_count):
        props = torch.cuda.get_device_properties(index)
        gpus.append({"index": index, "name": props.name, "vram_gib": props.total_memory / 2**30})
    disk_probe = cfg.work_dir
    while not disk_probe.exists() and disk_probe != disk_probe.parent:
        disk_probe = disk_probe.parent
    free = shutil.disk_usage(disk_probe)
    page_size = os.sysconf("SC_PAGE_SIZE")
    physical_pages = os.sysconf("SC_PHYS_PAGES")
    report = {
        "profile": cfg.runtime.profile,
        "selected_device": cfg.runtime.device,
        "python": __import__("sys").version,
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "gpus": gpus,
        "shards": len(shards),
        "manifest_record_limit": cfg.runtime.manifest_record_limit,
        "cpu_threads_configured": cfg.runtime.cpu_threads,
        "logical_cpu_count": os.cpu_count(),
        "physical_ram_gib": page_size * physical_pages / 2**30,
        "free_working_gib": free.free / 2**30,
        "config_fingerprint": cfg.fingerprint(),
    }
    print(json.dumps(report, indent=2))
    return report
