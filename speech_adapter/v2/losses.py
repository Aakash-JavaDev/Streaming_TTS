from __future__ import annotations

import torch
from torch.nn import functional as F


def cb0_loss(logits: torch.Tensor, stop_logits: torch.Tensor, codes: torch.Tensor,
             mask: torch.Tensor, stop_weight: float = 0.1) -> tuple[torch.Tensor, dict]:
    if mask.dtype != torch.bool or mask.shape != stop_logits.shape or not mask.any():
        raise ValueError("invalid/nonempty valid-frame mask")
    ce = F.cross_entropy(logits[mask].float(), codes[:, 0][mask])
    lengths = mask.sum(1)
    if (lengths == 0).any():
        raise ValueError("empty audio sequences")
    target = torch.zeros_like(stop_logits)
    target[torch.arange(len(lengths), device=mask.device), lengths - 1] = 1
    positive = target[mask].sum()
    negative = mask.sum() - positive
    pos_weight = (negative / positive).clamp(min=1.0)
    stop = F.binary_cross_entropy_with_logits(stop_logits[mask].float(), target[mask].float(), pos_weight=pos_weight)
    return ce + stop_weight * stop, {"cb0_ce": ce.detach(), "stop_bce": stop.detach()}


def acoustic_loss(logits: torch.Tensor, codes: torch.Tensor, mask: torch.Tensor) -> tuple[torch.Tensor, dict]:
    if mask.dtype != torch.bool or not mask.any():
        raise ValueError("invalid/nonempty valid-frame mask")
    targets = codes[:, 1:].transpose(1, 2)
    values = torch.stack([F.cross_entropy(logits[:, :, k][mask].float(), targets[:, :, k][mask]) for k in range(7)])
    return values.mean(), {f"cb{k+1}_ce": values[k].detach() for k in range(7)}
