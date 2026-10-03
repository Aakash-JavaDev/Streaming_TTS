from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import ModelConfig


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1.0e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        scale = value.float().pow(2).mean(dim=-1, keepdim=True).add(self.eps).rsqrt()
        return (value * scale.to(value.dtype)) * self.weight


def _rotate_half(value: torch.Tensor) -> torch.Tensor:
    left, right = value.chunk(2, dim=-1)
    return torch.cat((-right, left), dim=-1)


class RotaryEmbedding(nn.Module):
    def __init__(self, head_dim: int, base: float = 10_000.0):
        super().__init__()
        if head_dim % 2:
            raise ValueError("rotary head dimension must be even")
        inv = 1.0 / (base ** (torch.arange(0, head_dim, 2).float() / head_dim))
        self.register_buffer("inv_freq", inv, persistent=False)

    def apply(
        self, query: torch.Tensor, key: torch.Tensor, positions: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        angles = torch.outer(positions.float(), self.inv_freq.float())
        cos = torch.cat((angles.cos(), angles.cos()), dim=-1)[None, None].to(query.dtype)
        sin = torch.cat((angles.sin(), angles.sin()), dim=-1)[None, None].to(query.dtype)
        return query * cos + _rotate_half(query) * sin, key * cos + _rotate_half(key) * sin


class CausalSelfAttention(nn.Module):
    def __init__(self, dim: int, heads: int, window: int):
        super().__init__()
        if dim % heads:
            raise ValueError("attention dimension must be divisible by heads")
        self.dim = dim
        self.heads = heads
        self.head_dim = dim // heads
        self.window = int(window)
        self.qkv = nn.Linear(dim, dim * 3, bias=False)
        self.out = nn.Linear(dim, dim, bias=False)
        self.rope = RotaryEmbedding(self.head_dim)

    def _split(self, value: torch.Tensor) -> torch.Tensor:
        batch, steps, _ = value.shape
        return value.view(batch, steps, self.heads, self.head_dim).transpose(1, 2)

    def forward(self, value: torch.Tensor, valid_mask: torch.Tensor) -> torch.Tensor:
        batch, steps, _ = value.shape
        q, k, v = self.qkv(value).chunk(3, dim=-1)
        q, k, v = self._split(q), self._split(k), self._split(v)
        positions = torch.arange(steps, device=value.device)
        q, k = self.rope.apply(q, k, positions)
        scores = torch.matmul(q, k.transpose(-1, -2)) / math.sqrt(self.head_dim)
        row = positions[:, None]
        col = positions[None, :]
        causal = (col <= row) & (col >= row - self.window + 1)
        allowed = causal[None, None] & valid_mask[:, None, None, :]
        scores = scores.masked_fill(~allowed, torch.finfo(scores.dtype).min)
        weights = torch.softmax(scores.float(), dim=-1).to(scores.dtype)
        weights = torch.nan_to_num(weights)
        output = torch.matmul(weights, v).transpose(1, 2).reshape(batch, steps, self.dim)
        return self.out(output) * valid_mask.unsqueeze(-1)

    def step(
        self,
        value: torch.Tensor,
        key_cache: torch.Tensor,
        value_cache: torch.Tensor,
        valid: int,
        position: int,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, int]:
        """One fixed-cache step. Caches are chronological with newest at the end."""
        if value.shape[1] != 1:
            raise ValueError("incremental attention expects exactly one step")
        q, k, v = self.qkv(value).chunk(3, dim=-1)
        q, k, v = self._split(q), self._split(k), self._split(v)
        pos = torch.tensor([position], device=value.device)
        q, k = self.rope.apply(q, k, pos)
        next_k = torch.roll(key_cache, shifts=-1, dims=2)
        next_v = torch.roll(value_cache, shifts=-1, dims=2)
        next_k[:, :, -1:, :] = k
        next_v[:, :, -1:, :] = v
        next_valid = min(self.window, int(valid) + 1)
        scores = torch.matmul(q, next_k.transpose(-1, -2)) / math.sqrt(self.head_dim)
        slots = torch.arange(self.window, device=value.device)
        allowed = slots >= self.window - next_valid
        scores = scores.masked_fill(~allowed[None, None, None], torch.finfo(scores.dtype).min)
        weights = torch.softmax(scores.float(), dim=-1).to(scores.dtype)
        output = torch.matmul(weights, next_v).transpose(1, 2).reshape(value.shape[0], 1, self.dim)
        return self.out(output), next_k, next_v, next_valid


class CausalBlock(nn.Module):
    def __init__(self, dim: int, heads: int, ffn_dim: int, window: int, dropout: float):
        super().__init__()
        self.attn_norm = RMSNorm(dim)
        self.attn = CausalSelfAttention(dim, heads, window)
        self.ffn_norm = RMSNorm(dim)
        self.ffn = nn.Sequential(
            nn.Linear(dim, ffn_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(ffn_dim, dim),
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, value: torch.Tensor, valid_mask: torch.Tensor) -> torch.Tensor:
        value = value + self.dropout(self.attn(self.attn_norm(value), valid_mask))
        value = value + self.dropout(self.ffn(self.ffn_norm(value))) * valid_mask.unsqueeze(-1)
        return value

    def step(
        self,
        value: torch.Tensor,
        key_cache: torch.Tensor,
        value_cache: torch.Tensor,
        valid: int,
        position: int,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, int]:
        update, next_k, next_v, next_valid = self.attn.step(
            self.attn_norm(value), key_cache, value_cache, valid, position
        )
        value = value + update
        value = value + self.ffn(self.ffn_norm(value))
        return value, next_k, next_v, next_valid


class TextContextAttention(nn.Module):
    def __init__(self, dim: int, heads: int, window: int):
        super().__init__()
        self.window = int(window)
        self.attn = CausalSelfAttention(dim, heads, window)
        self.norm = RMSNorm(dim)

    def forward(self, value: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        return self.norm(value + self.attn(value, mask))

    def step(
        self, value: torch.Tensor, cache: torch.Tensor, valid: int
    ) -> tuple[torch.Tensor, torch.Tensor, int]:
        next_cache = torch.roll(cache, shifts=-1, dims=1)
        next_cache[:, -1:, :] = value
        next_valid = min(self.window, int(valid) + 1)
        q = self.attn._split(self.attn.qkv(value).chunk(3, dim=-1)[0])
        projected = self.attn.qkv(next_cache)
        _, keys, values = projected.chunk(3, dim=-1)
        keys, values = self.attn._split(keys), self.attn._split(values)
        positions = torch.arange(self.window, device=value.device)
        query_pos = torch.tensor([self.window - 1], device=value.device)
        q, _ = self.attn.rope.apply(q, q, query_pos)
        dummy = keys
        _, keys = self.attn.rope.apply(dummy, keys, positions)
        scores = torch.matmul(q, keys.transpose(-1, -2)) / math.sqrt(self.attn.head_dim)
        allowed = positions >= self.window - next_valid
        scores = scores.masked_fill(~allowed[None, None, None], torch.finfo(scores.dtype).min)
        weights = torch.softmax(scores.float(), dim=-1).to(scores.dtype)
        context = torch.matmul(weights, values).transpose(1, 2).reshape(value.shape)
        context = self.attn.out(context)
        return self.norm(value + context), next_cache, next_valid


@dataclass
class StreamingState:
    text_cache: torch.Tensor
    text_valid: int
    key_caches: list[torch.Tensor]
    value_caches: list[torch.Tensor]
    audio_valid: int
    audio_position: int
    previous_codes: torch.Tensor | None


class StreamingSpeechAdapter(nn.Module):
    def __init__(self, text_dim: int, embedding_dim: int, cfg: ModelConfig):
        super().__init__()
        self.cfg = cfg
        dim = cfg.model_dim
        self.hidden_norm = nn.LayerNorm(text_dim)
        self.embedding_norm = nn.LayerNorm(embedding_dim)
        self.hidden_projection = nn.Linear(text_dim, dim)
        self.embedding_projection = nn.Linear(embedding_dim, dim)
        self.fusion_norm = RMSNorm(dim)
        self.text_context = TextContextAttention(dim, cfg.attention_heads, cfg.text_context_size)
        self.duration_head = nn.Sequential(nn.Linear(dim, dim // 2), nn.SiLU(), nn.Linear(dim // 2, 1))
        self.code_embedding = nn.Embedding(cfg.num_codebooks * cfg.codec_vocab, dim)
        self.audio_bos = nn.Parameter(torch.randn(dim) * (dim ** -0.5))
        self.audio_input = nn.Linear(dim * 2, dim)
        self.temporal = nn.ModuleList(
            CausalBlock(dim, cfg.attention_heads, cfg.ffn_dim, cfg.audio_cache_frames, cfg.dropout)
            for _ in range(cfg.temporal_layers)
        )
        self.temporal_norm = RMSNorm(dim)
        self.depth_mix = nn.Linear(dim, dim, bias=False)
        self.depth_norm = RMSNorm(dim)
        self.codec_weight = nn.Parameter(
            torch.randn(cfg.num_codebooks, dim, cfg.codec_vocab) * (dim ** -0.5)
        )
        self.codec_bias = nn.Parameter(torch.zeros(cfg.num_codebooks, cfg.codec_vocab))

    def parameter_count(self) -> int:
        return sum(parameter.numel() for parameter in self.parameters())

    def assert_budget(self, maximum: int = 12_000_000) -> None:
        count = self.parameter_count()
        if count > maximum:
            raise RuntimeError(f"adapter has {count:,} parameters, above budget {maximum:,}")

    def fuse_text(self, hidden: torch.Tensor, embeddings: torch.Tensor) -> torch.Tensor:
        fused = self.hidden_projection(self.hidden_norm(hidden))
        fused = fused + self.embedding_projection(self.embedding_norm(embeddings))
        return self.fusion_norm(fused)

    def encode_text(
        self, hidden: torch.Tensor, embeddings: torch.Tensor, token_mask: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        fused = self.fuse_text(hidden, embeddings)
        context = self.text_context(fused, token_mask)
        duration = self.duration_head(context).squeeze(-1)
        return context * token_mask.unsqueeze(-1), duration

    def embed_frame(self, codes: torch.Tensor) -> torch.Tensor:
        books = torch.arange(self.cfg.num_codebooks, device=codes.device).view(1, 1, -1)
        values = self.code_embedding(codes.long() + books * self.cfg.codec_vocab)
        return values.sum(dim=-2) / math.sqrt(self.cfg.num_codebooks)

    def _previous_frame_embeddings(self, codes: torch.Tensor) -> torch.Tensor:
        batch, frames, _ = codes.shape
        embedded = self.embed_frame(codes)
        bos = self.audio_bos.view(1, 1, -1).expand(batch, 1, -1)
        return torch.cat((bos, embedded[:, :-1]), dim=1) if frames else embedded

    def frame_logits(self, frame_hidden: torch.Tensor, teacher_codes: torch.Tensor) -> torch.Tensor:
        books = torch.arange(self.cfg.num_codebooks, device=teacher_codes.device).view(1, 1, -1)
        values = self.code_embedding(teacher_codes.long() + books * self.cfg.codec_vocab)
        inclusive = values.cumsum(dim=2)
        prefix = torch.cat((torch.zeros_like(inclusive[:, :, :1]), inclusive[:, :, :-1]), dim=2)
        counts = torch.arange(self.cfg.num_codebooks, device=teacher_codes.device).float().sqrt().clamp(min=1)
        prefix = prefix / counts.view(1, 1, -1, 1)
        conditioned = self.depth_norm(frame_hidden[:, :, None, :] + self.depth_mix(prefix))
        return torch.einsum("btkd,kdv->btkv", conditioned, self.codec_weight) + self.codec_bias

    def forward(self, batch: dict[str, torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
        text_context, log_duration = self.encode_text(
            batch["hidden"], batch["embeddings"], batch["token_mask"]
        )
        gather = batch["owners"].unsqueeze(-1).expand(-1, -1, self.cfg.model_dim)
        frame_text = torch.gather(text_context, 1, gather)
        previous = self._previous_frame_embeddings(batch["codes"])
        frame_hidden = self.audio_input(torch.cat((frame_text, previous), dim=-1))
        for block in self.temporal:
            frame_hidden = block(frame_hidden, batch["frame_mask"])
        frame_hidden = self.temporal_norm(frame_hidden)
        return self.frame_logits(frame_hidden, batch["codes"]), log_duration

    def make_streaming_state(self, batch_size: int, device: torch.device, dtype: torch.dtype) -> StreamingState:
        head_dim = self.cfg.model_dim // self.cfg.attention_heads
        shape = (batch_size, self.cfg.attention_heads, self.cfg.audio_cache_frames, head_dim)
        return StreamingState(
            text_cache=torch.zeros(batch_size, self.cfg.text_context_size, self.cfg.model_dim, device=device, dtype=dtype),
            text_valid=0,
            key_caches=[torch.zeros(shape, device=device, dtype=dtype) for _ in self.temporal],
            value_caches=[torch.zeros(shape, device=device, dtype=dtype) for _ in self.temporal],
            audio_valid=0,
            audio_position=0,
            previous_codes=None,
        )

    def text_step(
        self, hidden: torch.Tensor, embedding: torch.Tensor, state: StreamingState
    ) -> tuple[torch.Tensor, torch.Tensor]:
        fused = self.fuse_text(hidden, embedding).unsqueeze(1)
        context, state.text_cache, state.text_valid = self.text_context.step(
            fused, state.text_cache, state.text_valid
        )
        duration = self.duration_head(context[:, 0]).squeeze(-1)
        return context[:, 0], duration

    def audio_step(
        self, text_context: torch.Tensor, state: StreamingState
    ) -> tuple[torch.Tensor, list[torch.Tensor]]:
        batch = text_context.shape[0]
        if state.previous_codes is None:
            previous = self.audio_bos.view(1, -1).expand(batch, -1)
        else:
            previous = self.embed_frame(state.previous_codes[:, None, :])[:, 0]
        value = self.audio_input(torch.cat((text_context, previous), dim=-1)).unsqueeze(1)
        next_keys: list[torch.Tensor] = []
        next_values: list[torch.Tensor] = []
        next_valid = state.audio_valid
        for index, block in enumerate(self.temporal):
            value, key, val, layer_valid = block.step(
                value,
                state.key_caches[index],
                state.value_caches[index],
                state.audio_valid,
                state.audio_position,
            )
            next_keys.append(key)
            next_values.append(val)
            next_valid = layer_valid
        state.key_caches, state.value_caches = next_keys, next_values
        state.audio_valid = next_valid
        state.audio_position += 1
        hidden = self.temporal_norm(value[:, 0])
        generated: list[torch.Tensor] = []
        prefix_sum = torch.zeros(batch, self.cfg.model_dim, device=hidden.device, dtype=hidden.dtype)
        logits_by_book: list[torch.Tensor] = []
        for book in range(self.cfg.num_codebooks):
            prefix = prefix_sum / math.sqrt(max(book, 1))
            conditioned = self.depth_norm(hidden + self.depth_mix(prefix))
            logits = conditioned @ self.codec_weight[book] + self.codec_bias[book]
            code = logits.argmax(dim=-1)
            logits_by_book.append(logits)
            generated.append(code)
            embedded = self.code_embedding(code + book * self.cfg.codec_vocab)
            prefix_sum = prefix_sum + embedded
        codes = torch.stack(generated, dim=-1)
        state.previous_codes = codes
        return codes, logits_by_book

    @torch.no_grad()
    def rollout(
        self,
        hidden: torch.Tensor,
        embeddings: torch.Tensor,
        durations: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        self.eval()
        device, dtype = hidden.device, next(self.parameters()).dtype
        state = self.make_streaming_state(1, device, dtype)
        contexts = []
        predicted_durations = []
        for index in range(hidden.shape[0]):
            context, log_duration = self.text_step(hidden[index : index + 1], embeddings[index : index + 1], state)
            contexts.append(context)
            predicted_durations.append(torch.expm1(log_duration.float()).clamp(0, 250).round().long())
        predicted = torch.cat(predicted_durations)
        selected = durations.long() if durations is not None else predicted
        frames = []
        for context, count in zip(contexts, selected.tolist()):
            for _ in range(int(count)):
                codes, _ = self.audio_step(context, state)
                frames.append(codes[0])
        output = torch.stack(frames) if frames else torch.empty(0, self.cfg.num_codebooks, dtype=torch.long, device=device)
        return output, predicted

    @torch.no_grad()
    def rollout_with_logits(
        self,
        hidden: torch.Tensor,
        embeddings: torch.Tensor,
        durations: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        self.eval()
        device, dtype = hidden.device, next(self.parameters()).dtype
        state = self.make_streaming_state(1, device, dtype)
        contexts: list[torch.Tensor] = []
        predicted_durations: list[torch.Tensor] = []
        for index in range(hidden.shape[0]):
            context, log_duration = self.text_step(
                hidden[index : index + 1], embeddings[index : index + 1], state
            )
            contexts.append(context)
            predicted_durations.append(
                torch.expm1(log_duration.float()).clamp(0, 250).round().long()
            )
        predicted = torch.cat(predicted_durations)
        selected = durations.long() if durations is not None else predicted
        frames: list[torch.Tensor] = []
        frame_logits: list[torch.Tensor] = []
        for context, count in zip(contexts, selected.tolist()):
            for _ in range(int(count)):
                codes, logits = self.audio_step(context, state)
                frames.append(codes[0])
                frame_logits.append(torch.stack([book[0] for book in logits]))
        codes_out = (
            torch.stack(frames)
            if frames
            else torch.empty(0, self.cfg.num_codebooks, dtype=torch.long, device=device)
        )
        logits_out = (
            torch.stack(frame_logits)
            if frame_logits
            else torch.empty(
                0, self.cfg.num_codebooks, self.cfg.codec_vocab, device=device, dtype=dtype
            )
        )
        return codes_out, predicted, logits_out


def adapter_loss(
    logits: torch.Tensor,
    log_duration: torch.Tensor,
    batch: dict[str, torch.Tensor],
    cb0_weight: float,
    residual_weight: float,
    duration_weight: float,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    vocabulary = logits.shape[-1]
    raw = F.cross_entropy(
        logits.float().reshape(-1, vocabulary), batch["codes"].reshape(-1), reduction="none"
    ).view_as(batch["codes"])
    weights = torch.tensor(
        [cb0_weight] + [residual_weight] * (batch["codes"].shape[-1] - 1),
        device=logits.device,
    )
    valid = batch["frame_mask"].unsqueeze(-1)
    codec = (raw * weights.view(1, 1, -1) * valid).sum() / (
        valid.sum().clamp(min=1) * weights.sum()
    )
    target = torch.log1p(batch["durations"].float())
    duration = ((log_duration - target).square() * batch["token_mask"]).sum() / batch[
        "token_mask"
    ].sum().clamp(min=1)
    total = codec + duration_weight * duration
    per_book = (raw * valid).sum(dim=(0, 1)) / valid.sum().clamp(min=1)
    return total, {"codec": codec.detach(), "duration": duration.detach(), "per_book": per_book.detach()}
