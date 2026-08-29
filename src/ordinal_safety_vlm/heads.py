from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from .configuration import OrdinalConfig


SUPPORTED_HEAD_ARCHS = {
    "linear_ln_linear",
    "linear_gelu_linear",
    "linear_gelu_ln_linear",
}


def _projection(hidden_size: int, output_size: int, architecture: str) -> nn.Sequential:
    architecture = str(architecture).strip().lower()
    if architecture not in SUPPORTED_HEAD_ARCHS:
        raise ValueError(
            f"Unsupported head architecture {architecture!r}; "
            f"expected one of {sorted(SUPPORTED_HEAD_ARCHS)}"
        )
    layers: list[nn.Module] = [nn.Linear(hidden_size, hidden_size)]
    if architecture == "linear_ln_linear":
        layers.extend([nn.LayerNorm(hidden_size), nn.Linear(hidden_size, output_size)])
    elif architecture == "linear_gelu_linear":
        layers.extend([nn.GELU(), nn.Linear(hidden_size, output_size)])
    else:
        layers.extend(
            [nn.GELU(), nn.LayerNorm(hidden_size), nn.Linear(hidden_size, output_size)]
        )
    return nn.Sequential(*layers)


class OrdinalHead(nn.Module):
    _THRESHOLD_EPS = 1e-4

    def __init__(
        self,
        hidden_size: int,
        num_classes: int,
        architecture: str,
        threshold_init_gap: float = 1.25,
    ) -> None:
        super().__init__()
        self.num_classes = int(num_classes)
        self.z_proj = _projection(hidden_size, 1, architecture)
        gap = float(threshold_init_gap)
        if gap <= self._THRESHOLD_EPS:
            raise ValueError("threshold_init_gap must be greater than 1e-4")
        inverse_softplus = math.log(math.expm1(gap - self._THRESHOLD_EPS))
        self.threshold_delta = nn.Parameter(
            torch.full((self.num_classes - 1,), inverse_softplus)
        )

    def thresholds(self) -> torch.Tensor:
        return torch.cumsum(F.softplus(self.threshold_delta) + self._THRESHOLD_EPS, dim=0)

    def forward(self, pooled_hidden: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        z = self.z_proj(pooled_hidden).squeeze(-1)
        logits = z.unsqueeze(-1) - self.thresholds().view(1, -1)
        return logits, z


class ClassificationHead(nn.Module):
    def __init__(self, hidden_size: int, num_labels: int, architecture: str) -> None:
        super().__init__()
        self.proj = _projection(hidden_size, num_labels, architecture)

    def forward(self, pooled_hidden: torch.Tensor) -> torch.Tensor:
        return self.proj(pooled_hidden)


class SafetyHeads(nn.Module):
    """Heads whose state-dict names match the portable export exactly."""

    def __init__(self, config: OrdinalConfig) -> None:
        super().__init__()
        self.ordinal_head = OrdinalHead(
            config.hidden_size,
            config.num_classes,
            config.head_arch,
            config.threshold_init_gap,
        )
        self.category_head: ClassificationHead | None = None
        self.teacher_heads = nn.ModuleDict()
        if config.auxiliary.enabled:
            if not config.auxiliary.category_labels:
                raise ValueError("Auxiliary heads are enabled but category_labels is empty")
            self.category_head = ClassificationHead(
                config.hidden_size,
                len(config.auxiliary.category_labels),
                config.auxiliary.head_arch,
            )
            self.teacher_heads = nn.ModuleDict(
                {
                    name: ClassificationHead(
                        config.hidden_size,
                        len(labels),
                        config.auxiliary.head_arch,
                    )
                    for name, labels in config.auxiliary.teacher_labels.items()
                }
            )


def cumulative_to_class_probs(ordinal_probs: torch.Tensor) -> torch.Tensor:
    batch_size, thresholds = ordinal_probs.shape
    num_classes = thresholds + 1
    output = ordinal_probs.new_zeros((batch_size, num_classes))
    output[:, 0] = 1.0 - ordinal_probs[:, 0]
    for index in range(1, num_classes - 1):
        output[:, index] = ordinal_probs[:, index - 1] - ordinal_probs[:, index]
    output[:, -1] = ordinal_probs[:, -1]
    output = output.clamp_min(0.0)
    return output / output.sum(dim=-1, keepdim=True).clamp_min(1e-8)


def class_probs_to_score(
    class_probs: torch.Tensor,
    *,
    minimum: float,
    maximum: float,
) -> torch.Tensor:
    num_classes = int(class_probs.size(-1))
    ids = torch.arange(
        1,
        num_classes + 1,
        dtype=class_probs.dtype,
        device=class_probs.device,
    )
    expected_id = (class_probs * ids.view(1, -1)).sum(dim=-1)
    scale = (expected_id - 1.0) / float(num_classes - 1)
    return float(minimum) + scale * (float(maximum) - float(minimum))


def last_non_padding_pool(
    hidden: torch.Tensor,
    attention_mask: torch.Tensor | None,
) -> torch.Tensor:
    if attention_mask is None or attention_mask.ndim != 2:
        return hidden[:, -1, :]
    positions = torch.arange(hidden.size(1), device=hidden.device).unsqueeze(0)
    positions = positions.expand(attention_mask.size(0), -1)
    indices = positions.masked_fill(attention_mask.long() <= 0, -1).max(dim=-1).values
    indices = indices.clamp_min(0)
    rows = torch.arange(hidden.size(0), device=hidden.device)
    return hidden[rows, indices, :]
