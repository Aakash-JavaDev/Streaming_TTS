from __future__ import annotations

import math
from dataclasses import dataclass

import torch
from torch import nn
from torch.nn import functional as F

from ..model import RMSNorm, RotaryEmbedding
from .config import Architecture


def positions(length: int, width: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    t = torch.arange(length, device=device, dtype=torch.float32)[:, None]
    rates = torch.exp(torch.arange(0, width, 2, device=device).float() * (-math.log(10000) / width))
    result = torch.zeros(length, width, device=device)
    result[:, 0::2] = torch.sin(t * rates)
    result[:, 1::2] = torch.cos(t * rates[:width // 2])
    return result.to(dtype)[None]


def check_mask(mask: torch.Tensor, batch: int, length: int, name: str) -> None:
    if mask.dtype != torch.bool or mask.shape != (batch, length):
        raise ValueError(f"{name} must be bool [{batch},{length}]")
    if not mask.any(dim=1).all():
        raise ValueError(f"{name} cannot contain empty sequences")
    if length > 1 and (mask[:, 1:] & ~mask[:, :-1]).any():
        raise ValueError(f"{name} must be right padded")


class Conditioner(nn.Module):
    def __init__(self, cfg: Architecture):
        super().__init__()
        self.cfg = cfg
        self.hidden_norm = RMSNorm(cfg.hidden_dim)
        self.hidden_projection = nn.Linear(cfg.hidden_dim, cfg.width, bias=False)
        self.embedding_projection = nn.Linear(cfg.embedding_dim, cfg.width, bias=False)
        self.gate = nn.Linear(2 * cfg.width, cfg.width)
        nn.init.zeros_(self.gate.weight)
        nn.init.constant_(self.gate.bias, cfg.gate_bias)

    def forward(self, hidden: torch.Tensor, embeddings: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        if hidden.ndim != 3 or embeddings.ndim != 3:
            raise ValueError("hidden and embeddings must be [B,L,D]")
        b, length, d = hidden.shape
        if d != self.cfg.hidden_dim or embeddings.shape != (b, length, self.cfg.embedding_dim):
            raise ValueError("cached hidden/embedding dimensions do not match model config")
        if length > self.cfg.max_text_tokens:
            raise ValueError("text exceeds max_text_tokens; do not silently truncate aligned data")
        check_mask(mask, b, length, "token_mask")
        # Zero before normalization/projection: padded NaNs must not contaminate gradients.
        hidden = hidden.masked_fill(~mask[..., None], 0)
        embeddings = embeddings.masked_fill(~mask[..., None], 0)
        p = self.embedding_projection(embeddings)
        q = self.hidden_projection(self.hidden_norm(hidden))
        gate = torch.sigmoid(self.gate(torch.cat((p, q), -1)))
        z = p + gate * q + positions(length, self.cfg.width, p.device, p.dtype)
        return z.masked_fill(~mask[..., None], 0)


class SelfAttention(nn.Module):
    def __init__(self, cfg: Architecture, causal: bool):
        super().__init__()
        self.width, self.heads = cfg.width, cfg.heads
        self.head_dim = cfg.width // cfg.heads
        self.causal, self.dropout = causal, cfg.dropout
        self.qkv = nn.Linear(cfg.width, 3 * cfg.width, bias=False)
        self.out = nn.Linear(cfg.width, cfg.width, bias=False)
        self.rope = RotaryEmbedding(self.head_dim) if causal else None

    def split(self, value: torch.Tensor) -> torch.Tensor:
        b, n, _ = value.shape
        return value.reshape(b, n, self.heads, self.head_dim).transpose(1, 2)

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        b, n, _ = x.shape
        q, k, v = (self.split(t) for t in self.qkv(x).chunk(3, -1))
        if self.rope is not None:
            q, k = self.rope.apply(q, k, torch.arange(n, device=x.device))
        allowed = mask[:, None, None, :].expand(b, 1, n, n)
        if self.causal:
            allowed = allowed & torch.ones(n, n, device=x.device, dtype=torch.bool).tril()[None, None]
        y = F.scaled_dot_product_attention(q, k, v, attn_mask=allowed,
                                           dropout_p=self.dropout if self.training else 0.0)
        y = self.out(y.transpose(1, 2).reshape(b, n, self.width))
        return y.masked_fill(~mask[..., None], 0)

    def step(self, x: torch.Tensor, cache: tuple | None, position: int) -> tuple[torch.Tensor, tuple]:
        if not self.causal or x.shape[1] != 1 or self.training:
            raise ValueError("cached step requires causal attention, one position, eval mode")
        q, k, v = (self.split(t) for t in self.qkv(x).chunk(3, -1))
        q, k = self.rope.apply(q, k, torch.tensor([position], device=x.device))
        if cache is not None:
            k, v = torch.cat((cache[0], k), 2), torch.cat((cache[1], v), 2)
        y = F.scaled_dot_product_attention(q, k, v)
        return self.out(y.transpose(1, 2).reshape(x.shape[0], 1, self.width)), (k, v)


class CrossAttention(nn.Module):
    def __init__(self, cfg: Architecture):
        super().__init__()
        self.width, self.heads, self.head_dim = cfg.width, cfg.heads, cfg.width // cfg.heads
        self.q = nn.Linear(cfg.width, cfg.width, bias=False)
        self.kv = nn.Linear(cfg.width, 2 * cfg.width, bias=False)
        self.out = nn.Linear(cfg.width, cfg.width, bias=False)
        self.dropout = cfg.dropout

    def split(self, x: torch.Tensor) -> torch.Tensor:
        return x.reshape(x.shape[0], x.shape[1], self.heads, self.head_dim).transpose(1, 2)

    def prepare(self, text: torch.Tensor) -> tuple:
        return tuple(self.split(t) for t in self.kv(text).chunk(2, -1))

    def forward(self, x: torch.Tensor, text: torch.Tensor, text_mask: torch.Tensor,
                projected: tuple | None = None) -> torch.Tensor:
        k, v = self.prepare(text) if projected is None else projected
        y = F.scaled_dot_product_attention(self.split(self.q(x)), k, v,
                attn_mask=text_mask[:, None, None, :],
                dropout_p=self.dropout if self.training else 0.0)
        return self.out(y.transpose(1, 2).reshape(x.shape))


class Block(nn.Module):
    def __init__(self, cfg: Architecture, causal: bool):
        super().__init__()
        self.self_norm, self.cross_norm, self.ffn_norm = (RMSNorm(cfg.width) for _ in range(3))
        self.self_attention = SelfAttention(cfg, causal)
        self.cross_attention = CrossAttention(cfg)
        self.ffn_in = nn.Linear(cfg.width, 2 * cfg.ffn_dim, bias=False)
        self.ffn_out = nn.Linear(cfg.ffn_dim, cfg.width, bias=False)
        self.dropout = nn.Dropout(cfg.dropout)

    def ffn(self, x: torch.Tensor) -> torch.Tensor:
        a, gate = self.ffn_in(self.ffn_norm(x)).chunk(2, -1)
        return self.ffn_out(a * F.silu(gate))

    def forward(self, x: torch.Tensor, z: torch.Tensor, audio_mask: torch.Tensor,
                text_mask: torch.Tensor) -> torch.Tensor:
        x = x + self.dropout(self.self_attention(self.self_norm(x), audio_mask))
        x = x + self.dropout(self.cross_attention(self.cross_norm(x), z, text_mask))
        x = x + self.dropout(self.ffn(x))
        return x.masked_fill(~audio_mask[..., None], 0)

    def step(self, x: torch.Tensor, z: torch.Tensor, text_mask: torch.Tensor,
             cache: tuple | None, position: int, text_kv: tuple) -> tuple:
        update, cache = self.self_attention.step(self.self_norm(x), cache, position)
        x = x + update
        x = x + self.cross_attention(self.cross_norm(x), z, text_mask, text_kv)
        return x + self.ffn(x), cache


class CB0Generator(nn.Module):
    def __init__(self, cfg: Architecture):
        super().__init__()
        self.cfg = cfg
        self.embedding = nn.Embedding(cfg.codec_vocab, cfg.code_embedding_dim)
        self.projection = nn.Linear(cfg.code_embedding_dim, cfg.width, bias=False)
        self.bos = nn.Parameter(torch.randn(cfg.width) * cfg.width**-0.5)
        self.blocks = nn.ModuleList(Block(cfg, causal=True) for _ in range(cfg.cb0_layers))
        self.norm = RMSNorm(cfg.width)
        self.code_head = nn.Linear(cfg.width, cfg.codec_vocab)
        self.stop_head = nn.Linear(cfg.width, 1)

    def shifted_inputs(self, cb0: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        cb0 = cb0.masked_fill(~mask, 0)
        previous = self.projection(self.embedding(cb0[:, :-1]))
        return torch.cat((self.bos[None, None].expand(cb0.shape[0], 1, -1), previous), 1)

    def forward(self, cb0: torch.Tensor, z: torch.Tensor, mask: torch.Tensor,
                text_mask: torch.Tensor) -> tuple:
        if cb0.ndim != 2 or cb0.shape[1] > self.cfg.max_audio_frames:
            raise ValueError("CB0 must be [B,T] within max_audio_frames")
        check_mask(mask, *cb0.shape, "frame_mask")
        x = self.shifted_inputs(cb0, mask)
        for block in self.blocks:
            x = block(x, z, mask, text_mask)
        x = self.norm(x)
        return self.code_head(x), self.stop_head(x).squeeze(-1)

    def step(self, previous: torch.Tensor | None, z: torch.Tensor, text_mask: torch.Tensor,
             caches: list, position: int, text_kvs: list) -> tuple:
        x = self.bos[None, None].expand(z.shape[0], 1, -1) if previous is None else self.projection(self.embedding(previous[:, None]))
        next_caches = []
        for block, cache, text_kv in zip(self.blocks, caches, text_kvs):
            x, next_cache = block.step(x, z, text_mask, cache, position, text_kv)
            next_caches.append(next_cache)
        x = self.norm(x)
        return self.code_head(x)[:, 0], self.stop_head(x)[:, 0, 0], next_caches


class AcousticCompletion(nn.Module):
    def __init__(self, cfg: Architecture):
        super().__init__()
        self.cfg = cfg
        self.embedding = nn.Embedding(cfg.codec_vocab, cfg.code_embedding_dim)
        self.projection = nn.Linear(cfg.code_embedding_dim, cfg.width, bias=False)
        self.blocks = nn.ModuleList(Block(cfg, causal=False) for _ in range(cfg.acoustic_layers))
        self.norm = RMSNorm(cfg.width)
        self.heads = nn.ModuleList(nn.Linear(cfg.width, cfg.codec_vocab) for _ in range(7))

    def forward(self, cb0: torch.Tensor, z: torch.Tensor, mask: torch.Tensor,
                text_mask: torch.Tensor) -> torch.Tensor:
        if cb0.ndim != 2 or cb0.shape[1] > self.cfg.max_audio_frames:
            raise ValueError("CB0 must be [B,T] within max_audio_frames")
        check_mask(mask, *cb0.shape, "frame_mask")
        x = self.projection(self.embedding(cb0.masked_fill(~mask, 0)))
        x = x + positions(cb0.shape[1], self.cfg.width, x.device, x.dtype)
        x = x.masked_fill(~mask[..., None], 0)
        for block in self.blocks:
            x = block(x, z, mask, text_mask)
        x = self.norm(x)
        return torch.stack([head(x) for head in self.heads], dim=2)  # B,T,7,V


@dataclass
class Generation:
    codes: torch.Tensor  # B,8,T (initial inference contract B=1)
    termination_reason: str


class TwoStageTTS(nn.Module):
    def __init__(self, cfg: Architecture):
        super().__init__()
        cfg.validate()
        self.cfg = cfg
        self.conditioner = Conditioner(cfg)
        self.cb0 = CB0Generator(cfg)
        self.acoustic = AcousticCompletion(cfg)
        self.phase = "cb0"
        self.set_phase("cb0")

    def set_phase(self, phase: str) -> None:
        if phase not in {"cb0", "acoustic"}:
            raise ValueError("phase must be cb0 or acoustic")
        self.phase = phase
        self.requires_grad_(False)
        if phase == "cb0":
            self.conditioner.requires_grad_(True)
            self.cb0.requires_grad_(True)
        else:
            self.acoustic.requires_grad_(True)
        self.zero_grad(set_to_none=True)
        self.train(self.training)

    def train(self, mode: bool = True):
        super().train(mode)
        if self.phase == "acoustic":
            self.conditioner.eval()
            self.cb0.eval()
        else:
            self.acoustic.eval()
        return self

    def condition_text(self, batch: dict) -> torch.Tensor:
        return self.conditioner(batch["hidden"], batch["embeddings"], batch["token_mask"])

    def forward(self, batch: dict, phase: str | None = None):
        phase = self.phase if phase is None else phase
        if phase != self.phase:
            raise ValueError("call set_phase before changing the training objective")
        if phase == "cb0":
            z = self.condition_text(batch)
            return self.cb0(batch["codes"][:, 0], z, batch["frame_mask"], batch["token_mask"])
        with torch.no_grad():
            z = self.condition_text(batch)
        return self.acoustic(batch["codes"][:, 0], z, batch["frame_mask"], batch["token_mask"])

    @torch.inference_mode()
    def generate(self, hidden: torch.Tensor, embeddings: torch.Tensor, token_mask: torch.Tensor,
                 max_frames: int = 300, min_frames: int = 1, stop_threshold: float = 0.5) -> Generation:
        if hidden.shape[0] != 1:
            raise ValueError("v2 free-running inference currently requires B=1")
        if not 1 <= min_frames <= max_frames <= self.cfg.max_audio_frames:
            raise ValueError("invalid generation frame limits")
        if not 0 <= stop_threshold <= 1:
            raise ValueError("stop_threshold must be in [0,1]")
        was_training = self.training
        self.eval()
        try:
            z = self.conditioner(hidden, embeddings, token_mask)
            text_kvs = [block.cross_attention.prepare(z) for block in self.cb0.blocks]
            caches = [None] * len(self.cb0.blocks)
            previous, frames = None, []
            reason = "max_frames"
            for position in range(max_frames):
                logits, stop, caches = self.cb0.step(previous, z, token_mask, caches, position, text_kvs)
                previous = logits.argmax(-1)
                frames.append(previous)
                if len(frames) >= min_frames and torch.sigmoid(stop).item() >= stop_threshold:
                    reason = "stop"
                    break
            cb0 = torch.stack(frames, dim=1)
            mask = torch.ones_like(cb0, dtype=torch.bool)
            residuals = self.acoustic(cb0, z, mask, token_mask).argmax(-1).transpose(1, 2)
            return Generation(torch.cat((cb0[:, None], residuals), dim=1), reason)
        finally:
            self.train(was_training)
