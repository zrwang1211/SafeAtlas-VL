from __future__ import annotations

import torch
import torch.nn.functional as F


def ordinal_targets(
    label_ids: torch.Tensor,
    *,
    num_classes: int = 5,
    mode: str = "hard",
    gaussian_sigma: float = 0.75,
) -> torch.Tensor:
    """Build cumulative P(y > k) targets for labels encoded as 1..K."""

    labels = label_ids.long()
    thresholds = torch.arange(
        1,
        int(num_classes),
        dtype=torch.long,
        device=labels.device,
    ).view(1, -1)
    normalized_mode = str(mode).strip().lower()
    if normalized_mode == "hard":
        return (labels.view(-1, 1) > thresholds).to(torch.float32)
    if normalized_mode != "gaussian_soft":
        raise ValueError("mode must be hard or gaussian_soft")
    if gaussian_sigma <= 0:
        raise ValueError("gaussian_sigma must be positive")

    classes = torch.arange(
        1,
        int(num_classes) + 1,
        dtype=torch.float32,
        device=labels.device,
    ).view(1, -1)
    distance = classes - labels.float().view(-1, 1)
    class_mass = torch.exp(-0.5 * (distance / float(gaussian_sigma)).square())
    class_mass = class_mass / class_mass.sum(dim=-1, keepdim=True).clamp_min(1e-12)
    cumulative = []
    for threshold in range(1, int(num_classes)):
        cumulative.append(class_mass[:, threshold:].sum(dim=-1))
    return torch.stack(cumulative, dim=-1)


def ordinal_bce_loss(
    ordinal_logits: torch.Tensor,
    label_ids: torch.Tensor,
    *,
    num_classes: int = 5,
    mode: str = "hard",
    gaussian_sigma: float = 0.75,
) -> torch.Tensor:
    targets = ordinal_targets(
        label_ids,
        num_classes=num_classes,
        mode=mode,
        gaussian_sigma=gaussian_sigma,
    ).to(dtype=ordinal_logits.dtype)
    return F.binary_cross_entropy_with_logits(ordinal_logits, targets, reduction="mean")
