#!/usr/bin/env python3
"""Create listenable correct-8cb versus invalid zero-padded-32 Mimi reconstructions."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import onnxruntime as ort
import soundfile as sf


def session(path: Path) -> ort.InferenceSession:
    return ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--audio", type=Path, required=True)
    parser.add_argument("--encoder-8cb", type=Path, required=True)
    parser.add_argument("--decoder-8cb", type=Path, required=True)
    parser.add_argument("--decoder-32cb", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seconds", type=float, default=4.0)
    args = parser.parse_args()

    waveform, sample_rate = sf.read(args.audio, dtype="float32", always_2d=False)
    if waveform.ndim == 2:
        waveform = waveform.mean(axis=1)
    if sample_rate != 24_000:
        raise ValueError(f"Mimi requires 24 kHz input; received {sample_rate}")
    waveform = np.asarray(waveform[: int(sample_rate * args.seconds)], dtype=np.float32)
    encoder = session(args.encoder_8cb)
    encoded = np.asarray(
        encoder.run(
            None, {encoder.get_inputs()[0].name: waveform.reshape(1, 1, -1)}
        )[0],
        dtype=np.int64,
    )
    if encoded.ndim != 3 or encoded.shape[1] != 8:
        raise RuntimeError(f"expected encoder codes [batch,8,frames], got {encoded.shape}")

    decoder8 = session(args.decoder_8cb)
    correct = np.asarray(
        decoder8.run(None, {decoder8.get_inputs()[0].name: encoded})[0], dtype=np.float32
    ).reshape(-1)
    decoder32 = session(args.decoder_32cb)
    padded = np.pad(encoded, ((0, 0), (0, 24), (0, 0)))
    wrong = np.asarray(
        decoder32.run(None, {decoder32.get_inputs()[0].name: padded})[0], dtype=np.float32
    ).reshape(-1)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    sf.write(args.output_dir / "source.wav", waveform, sample_rate)
    sf.write(args.output_dir / "oracle_8cb_correct.wav", correct, sample_rate)
    sf.write(args.output_dir / "oracle_8cb_wrong_zero_padded.wav", wrong, sample_rate)
    length = min(correct.size, wrong.size)
    difference = correct[:length].astype(np.float64) - wrong[:length].astype(np.float64)
    report = {
        "source_seconds": waveform.size / sample_rate,
        "code_shape": list(encoded.shape),
        "correct_samples": int(correct.size),
        "wrong_samples": int(wrong.size),
        "correct_rms": float(np.sqrt(np.mean(correct.astype(np.float64) ** 2))),
        "wrong_rms": float(np.sqrt(np.mean(wrong.astype(np.float64) ** 2))),
        "difference_rms": float(np.sqrt(np.mean(difference**2))),
        "zero_padding_equivalent": bool(
            np.allclose(correct[:length], wrong[:length], atol=1e-5, rtol=1e-4)
        ),
    }
    (args.output_dir / "report.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
