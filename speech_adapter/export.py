from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
from safetensors.torch import load_file

from .config import AppConfig, atomic_json
from .train import build_model


class TextStepExport(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(
        self,
        hidden: torch.Tensor,
        embedding: torch.Tensor,
        cache: torch.Tensor,
        valid_mask: torch.Tensor,
    ):
        fused = self.model.fuse_text(hidden, embedding).unsqueeze(1)
        next_cache = torch.cat((cache[:, 1:], fused), dim=1)
        next_mask = torch.cat(
            (valid_mask[:, 1:], torch.ones_like(valid_mask[:, :1])), dim=1
        )
        all_context = self.model.text_context(next_cache, next_mask)
        context = all_context[:, -1]
        duration = self.model.duration_head(context).squeeze(-1)
        return context, duration, next_cache, next_mask


class AudioStepExport(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(
        self,
        text_context: torch.Tensor,
        previous_codes: torch.Tensor,
        first_frame: torch.Tensor,
        key_caches: torch.Tensor,
        value_caches: torch.Tensor,
        valid_mask: torch.Tensor,
        position: torch.Tensor,
    ):
        previous = self.model.embed_frame(previous_codes[:, None, :])[:, 0]
        bos = self.model.audio_bos.view(1, -1).expand_as(previous)
        previous = torch.where(first_frame.reshape(-1, 1), bos, previous)
        value = self.model.audio_input(torch.cat((text_context, previous), dim=-1)).unsqueeze(1)
        next_keys = []
        next_values = []
        next_mask = torch.cat((valid_mask[:, 1:], torch.ones_like(valid_mask[:, :1])), dim=1)
        for layer, block in enumerate(self.model.temporal):
            normalized = block.attn_norm(value)
            q, k, v = block.attn.qkv(normalized).chunk(3, dim=-1)
            q, k, v = block.attn._split(q), block.attn._split(k), block.attn._split(v)
            angles = position.float().reshape(1, 1) * block.attn.rope.inv_freq.float().reshape(1, -1)
            cos = torch.cat((angles.cos(), angles.cos()), dim=-1)[None].to(q.dtype)
            sin = torch.cat((angles.sin(), angles.sin()), dim=-1)[None].to(q.dtype)
            q = q * cos + torch.cat((-q[..., q.shape[-1] // 2 :], q[..., : q.shape[-1] // 2]), -1) * sin
            k = k * cos + torch.cat((-k[..., k.shape[-1] // 2 :], k[..., : k.shape[-1] // 2]), -1) * sin
            layer_k = torch.cat((key_caches[layer, :, :, 1:], k), dim=2)
            layer_v = torch.cat((value_caches[layer, :, :, 1:], v), dim=2)
            scores = torch.matmul(q, layer_k.transpose(-1, -2)) / math.sqrt(block.attn.head_dim)
            scores = scores.masked_fill(
                ~next_mask[:, None, None], torch.finfo(scores.dtype).min
            )
            weights = torch.softmax(scores.float(), dim=-1).to(scores.dtype)
            update = torch.matmul(weights, layer_v).transpose(1, 2).reshape(value.shape)
            value = value + block.attn.out(update)
            value = value + block.ffn(block.ffn_norm(value))
            next_keys.append(layer_k)
            next_values.append(layer_v)
        hidden = self.model.temporal_norm(value[:, 0])
        prefix_sum = torch.zeros_like(hidden)
        codes = []
        logits = []
        for book in range(self.model.cfg.num_codebooks):
            prefix = prefix_sum / math.sqrt(max(book, 1))
            conditioned = self.model.depth_norm(hidden + self.model.depth_mix(prefix))
            book_logits = conditioned @ self.model.codec_weight[book] + self.model.codec_bias[book]
            code = book_logits.argmax(dim=-1)
            logits.append(book_logits)
            codes.append(code)
            prefix_sum = prefix_sum + self.model.code_embedding(
                code + book * self.model.cfg.codec_vocab
            )
        return (
            torch.stack(codes, dim=-1),
            torch.stack(logits, dim=1),
            torch.stack(next_keys),
            torch.stack(next_values),
            next_mask,
            position + 1,
        )


def _ort_session(path: Path):
    import onnxruntime as ort

    return ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])


def export_adapter(
    cfg: AppConfig,
    parity_steps: int = 100,
    checkpoint_path: str = "",
    output_dir_path: str = "",
) -> dict[str, Any]:
    device = torch.device("cpu")
    model = build_model(cfg, device)
    checkpoint = (
        Path(checkpoint_path) if checkpoint_path
        else cfg.work_dir / "checkpoints" / "best.safetensors"
    )
    model.load_state_dict(load_file(str(checkpoint), device="cpu"))
    model.eval()
    output_dir = Path(output_dir_path) if output_dir_path else cfg.work_dir / "export"
    output_dir.mkdir(parents=True, exist_ok=True)
    text_model, audio_model = TextStepExport(model).eval(), AudioStepExport(model).eval()
    text_dim = model.hidden_projection.in_features
    embedding_dim = model.embedding_projection.in_features
    dim, context = model.cfg.model_dim, model.cfg.text_context_size
    heads = model.cfg.attention_heads
    head_dim = dim // heads
    layers, window = model.cfg.temporal_layers, model.cfg.audio_cache_frames
    text_inputs = (
        torch.randn(1, text_dim),
        torch.randn(1, embedding_dim),
        torch.zeros(1, context, dim),
        torch.zeros(1, context, dtype=torch.bool),
    )
    audio_inputs = (
        torch.randn(1, dim),
        torch.zeros(1, 8, dtype=torch.long),
        torch.ones(1, dtype=torch.bool),
        torch.zeros(layers, 1, heads, window, head_dim),
        torch.zeros(layers, 1, heads, window, head_dim),
        torch.zeros(1, window, dtype=torch.bool),
        torch.zeros(1, dtype=torch.long),
    )
    text_path, audio_path = output_dir / "text_step.onnx", output_dir / "audio_step.onnx"
    torch.onnx.export(
        text_model,
        text_inputs,
        text_path,
        input_names=["hidden", "embedding", "text_cache", "text_valid"],
        output_names=["context", "log_duration", "next_text_cache", "next_text_valid"],
        opset_version=18,
        dynamo=False,
    )
    torch.onnx.export(
        audio_model,
        audio_inputs,
        audio_path,
        input_names=[
            "text_context", "previous_codes", "first_frame", "key_caches",
            "value_caches", "audio_valid", "position",
        ],
        output_names=[
            "codes", "logits", "next_key_caches", "next_value_caches",
            "next_audio_valid", "next_position",
        ],
        opset_version=18,
        dynamo=False,
    )
    text_session, audio_session = _ort_session(text_path), _ort_session(audio_path)
    rng = np.random.default_rng(cfg.train.seed)
    text_cache = np.zeros((1, context, dim), np.float32)
    text_valid = np.zeros((1, context), bool)
    key_cache = np.zeros((layers, 1, heads, window, head_dim), np.float32)
    value_cache = np.zeros_like(key_cache)
    audio_valid = np.zeros((1, window), bool)
    position = np.zeros(1, np.int64)
    previous = np.zeros((1, 8), np.int64)
    max_text_error = max_audio_error = 0.0
    cb0_agreement = []
    context_value = None
    for step in range(parity_steps):
        hidden = rng.standard_normal((1, text_dim)).astype(np.float32)
        embedding = rng.standard_normal((1, embedding_dim)).astype(np.float32)
        torch_text = text_model(
            torch.from_numpy(hidden), torch.from_numpy(embedding),
            torch.from_numpy(text_cache), torch.from_numpy(text_valid),
        )
        ort_text = text_session.run(None, {
            "hidden": hidden, "embedding": embedding,
            "text_cache": text_cache, "text_valid": text_valid,
        })
        max_text_error = max(max_text_error, float(np.max(np.abs(torch_text[0].detach().numpy() - ort_text[0]))))
        context_value, text_cache, text_valid = ort_text[0], ort_text[2], ort_text[3]
        first = np.asarray([step == 0], dtype=bool)
        torch_audio = audio_model(
            torch.from_numpy(context_value), torch.from_numpy(previous), torch.from_numpy(first),
            torch.from_numpy(key_cache), torch.from_numpy(value_cache),
            torch.from_numpy(audio_valid), torch.from_numpy(position),
        )
        ort_audio = audio_session.run(None, {
            "text_context": context_value, "previous_codes": previous,
            "first_frame": first, "key_caches": key_cache,
            "value_caches": value_cache, "audio_valid": audio_valid,
            "position": position,
        })
        torch_logits = torch_audio[1].detach().numpy()
        max_audio_error = max(max_audio_error, float(np.max(np.abs(torch_logits - ort_audio[1]))))
        cb0_agreement.append(bool(torch_logits[:, 0].argmax(-1)[0] == ort_audio[1][:, 0].argmax(-1)[0]))
        previous, key_cache, value_cache, audio_valid, position = (
            ort_audio[0], ort_audio[2], ort_audio[3], ort_audio[4], ort_audio[5]
        )
    quantized = output_dir / "audio_step.int8.onnx"
    quantization_error = None
    int8_cb0_agreement = None
    int8_max_logit_error = None
    try:
        from onnxruntime.quantization import QuantType, quantize_dynamic

        quantize_dynamic(str(audio_path), str(quantized), weight_type=QuantType.QInt8)
        int8_session = _ort_session(quantized)
        rng = np.random.default_rng(cfg.train.seed + 1)
        fp_keys = np.zeros((layers, 1, heads, window, head_dim), np.float32)
        fp_values = np.zeros_like(fp_keys)
        q_keys = fp_keys.copy()
        q_values = fp_values.copy()
        fp_mask = np.zeros((1, window), bool)
        q_mask = fp_mask.copy()
        fp_pos = np.zeros(1, np.int64)
        q_pos = fp_pos.copy()
        fp_previous = np.zeros((1, 8), np.int64)
        q_previous = fp_previous.copy()
        agreements = []
        errors = []
        for step in range(parity_steps):
            context_input = rng.standard_normal((1, dim)).astype(np.float32)
            first = np.asarray([step == 0], dtype=bool)
            fp = audio_session.run(None, {
                "text_context": context_input, "previous_codes": fp_previous,
                "first_frame": first, "key_caches": fp_keys,
                "value_caches": fp_values, "audio_valid": fp_mask,
                "position": fp_pos,
            })
            quant = int8_session.run(None, {
                "text_context": context_input, "previous_codes": q_previous,
                "first_frame": first, "key_caches": q_keys,
                "value_caches": q_values, "audio_valid": q_mask,
                "position": q_pos,
            })
            agreements.append(bool(fp[1][:, 0].argmax(-1)[0] == quant[1][:, 0].argmax(-1)[0]))
            errors.append(float(np.max(np.abs(fp[1] - quant[1]))))
            fp_previous, fp_keys, fp_values, fp_mask, fp_pos = fp[0], fp[2], fp[3], fp[4], fp[5]
            q_previous, q_keys, q_values, q_mask, q_pos = (
                quant[0], quant[2], quant[3], quant[4], quant[5]
            )
        int8_cb0_agreement = float(np.mean(agreements))
        int8_max_logit_error = float(max(errors))
    except Exception as exc:
        quantization_error = f"quantization failed: {exc}"
    report = {
        "steps": parity_steps,
        "max_text_absolute_error": max_text_error,
        "max_audio_logit_absolute_error": max_audio_error,
        "cb0_argmax_agreement": float(np.mean(cb0_agreement)),
        "pytorch_onnx_pass": max_text_error < 1e-3 and max_audio_error < 1e-3 and all(cb0_agreement),
        "text_onnx_bytes": text_path.stat().st_size,
        "audio_onnx_bytes": audio_path.stat().st_size,
        "int8_audio_onnx_bytes": quantized.stat().st_size if quantized.is_file() else None,
        "int8_cb0_argmax_agreement": int8_cb0_agreement,
        "int8_max_logit_error": int8_max_logit_error,
        "int8_pass": bool(int8_cb0_agreement is not None and int8_cb0_agreement >= 0.99),
        "quantization_note": quantization_error,
    }
    atomic_json(output_dir / "export_report.json", report)
    return report
