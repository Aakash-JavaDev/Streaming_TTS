"""Optional real codec backends; core training and tests need no Mimi download."""
from __future__ import annotations

from pathlib import Path
import wave

import numpy as np
import torch


def validate_codes(codes: torch.Tensor) -> torch.Tensor:
    if codes.ndim != 3 or codes.shape[0] != 1 or codes.shape[1] != 8 or codes.shape[2] < 1:
        raise ValueError("Mimi decoder requires [1,8,T]")
    if codes.dtype != torch.long or codes.min() < 0 or codes.max() >= 2048:
        raise ValueError("Mimi decoder requires integer code IDs in [0,2047]")
    return codes.detach().cpu()


class OnnxDecoder:
    def __init__(self, path: str):
        import onnxruntime as ort
        self.session = ort.InferenceSession(path, providers=["CPUExecutionProvider"])
        inputs = self.session.get_inputs()
        if len(inputs) != 1 or len(inputs[0].shape) != 3 or inputs[0].type != "tensor(int64)":
            raise ValueError("expected stateless ONNX with one int64 [B,8,T] input")
        books = inputs[0].shape[1]
        if isinstance(books, int) and books != 8:
            raise ValueError("ONNX decoder must be exported for eight codebooks")
        self.name = inputs[0].name

    def decode(self, codes: torch.Tensor) -> np.ndarray:
        values = validate_codes(codes).numpy()
        audio = np.asarray(self.session.run(None, {self.name: values})[0], dtype=np.float32)
        return checked_audio(audio)


class MoshiDecoder:
    def __init__(self, weights: str, device: str = "cpu"):
        from moshi.models import loaders
        self.device = torch.device(device)
        self.model = loaders.get_mimi(weights, device=self.device)
        self.model.set_num_codebooks(8)
        self.model.eval()

    @torch.inference_mode()
    def decode(self, codes: torch.Tensor) -> np.ndarray:
        codes = validate_codes(codes).to(self.device)
        chunks = []
        with self.model.streaming(batch_size=1):
            for t in range(codes.shape[-1]):
                chunk = self.model.decode(codes[:, :, t:t+1])
                chunks.append(chunk.detach().float().cpu())
        return checked_audio(torch.cat(chunks, dim=-1).numpy())


def checked_audio(audio) -> np.ndarray:
    values = np.asarray(audio, dtype=np.float32).reshape(-1)
    if not values.size or not np.isfinite(values).all():
        raise ValueError("codec returned empty/non-finite waveform")
    return values


def write_wav(path: str | Path, audio: np.ndarray) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    pcm = (np.clip(checked_audio(audio), -1, 1) * 32767).astype("<i2")
    with wave.open(str(path), "wb") as out:
        out.setnchannels(1)
        out.setsampwidth(2)
        out.setframerate(24000)
        out.writeframes(pcm.tobytes())
