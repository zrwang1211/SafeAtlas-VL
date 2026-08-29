from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Mapping, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F
from safetensors.torch import save_file

from .configuration import OrdinalConfig
from .constants import normalize_target
from .heads import SafetyHeads, last_non_padding_pool
from .losses import ordinal_bce_loss
from .model import _extract_hidden
from .prompts import PromptBundle


IGNORE_INDEX = -100
CATEGORY_LABELS = (
    "none",
    "violation of personal property",
    "persuasion and manipulation",
    "illegal activities",
    "influence operations",
    "fraud or deceptive action",
    "defamation",
    "security threats",
    "privacy",
    "dangerous information",
    "false beliefs",
    "erosion of trust in public information",
    "unfair",
    "toxic",
    "trade and compliance",
    "risky financial practices",
)
CATEGORY_TO_ID = {label: index for index, label in enumerate(CATEGORY_LABELS)}


class SafeAtlasHeadCollator:
    """Build processor inputs and Stage-2 labels from flattened public rows."""

    def __init__(self, processor: Any, prompts: PromptBundle | None = None) -> None:
        self.processor = processor
        self.prompts = prompts or PromptBundle.from_packaged()

    def __call__(self, records: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
        messages = []
        ordinal_labels = []
        category_labels = []
        teacher_labels = {"judge1": [], "judge2": [], "judge3": []}
        for record in records:
            target = normalize_target(str(record.get("target_name") or record.get("target") or ""))
            messages.append(
                self.prompts.build_messages(
                    target_name=target,
                    image=record.get("image"),
                    request=str(record.get("request") or ""),
                    response=str(record.get("response") or ""),
                )
            )
            ordinal_labels.append(int(record["label_id"]))
            category = str(record.get("category") or "").strip().lower().replace("_", " ")
            category_labels.append(CATEGORY_TO_ID.get(category, IGNORE_INDEX))
            normalized_teacher = _teacher_label_ids(record.get("teacher_head"), target=target)
            for name in teacher_labels:
                teacher_labels[name].append(normalized_teacher[name])

        inputs = _apply_chat_template(self.processor, messages)
        inputs["ordinal_labels"] = torch.tensor(ordinal_labels, dtype=torch.long)
        inputs["category_labels"] = torch.tensor(category_labels, dtype=torch.long)
        for name, values in teacher_labels.items():
            inputs[f"teacher_{name}_labels"] = torch.tensor(values, dtype=torch.long)
        return inputs


class FrozenBackboneSafetyModel(nn.Module):
    """Frozen multimodal backbone with trainable SafeAtlas prediction heads."""

    def __init__(
        self,
        backbone: nn.Module,
        config: OrdinalConfig,
        *,
        category_loss_weight: float = 0.2,
        teacher_loss_weight: float = 0.2,
        gaussian_sigma: float = 0.75,
    ) -> None:
        super().__init__()
        self.backbone = backbone
        self.config = config
        self.heads = SafetyHeads(config)
        self.category_loss_weight = float(category_loss_weight)
        self.teacher_loss_weight = float(teacher_loss_weight)
        self.gaussian_sigma = float(gaussian_sigma)
        self.backbone.requires_grad_(False)
        self.backbone.eval()

    def train(self, mode: bool = True) -> "FrozenBackboneSafetyModel":
        self.training = mode
        self.backbone.eval()
        self.heads.train(mode)
        return self

    def forward(
        self,
        ordinal_labels: torch.Tensor,
        category_labels: torch.Tensor,
        teacher_judge1_labels: torch.Tensor,
        teacher_judge2_labels: torch.Tensor,
        teacher_judge3_labels: torch.Tensor,
        **inputs: Any,
    ) -> Dict[str, Any]:
        inputs.pop("labels", None)
        attention_mask = inputs.get("attention_mask")
        with torch.no_grad():
            hidden = self._forward_hidden(inputs)
            pooled = last_non_padding_pool(hidden, attention_mask)

        head_parameter = next(self.heads.ordinal_head.parameters())
        pooled = pooled.to(device=head_parameter.device, dtype=head_parameter.dtype)
        ordinal_labels = ordinal_labels.to(head_parameter.device)
        ordinal_logits, z = self.heads.ordinal_head(pooled)
        loss = ordinal_bce_loss(
            ordinal_logits,
            ordinal_labels,
            num_classes=self.config.num_classes,
            mode="gaussian_soft",
            gaussian_sigma=self.gaussian_sigma,
        )

        category_logits = None
        if self.heads.category_head is not None:
            category_logits = self.heads.category_head(pooled)
            category_loss = _masked_cross_entropy(
                category_logits,
                category_labels.to(category_logits.device),
            )
            if category_loss is not None:
                loss = loss + self.category_loss_weight * category_loss

        teacher_inputs = {
            "judge1": teacher_judge1_labels,
            "judge2": teacher_judge2_labels,
            "judge3": teacher_judge3_labels,
        }
        active_teacher_losses = []
        for name, head in self.heads.teacher_heads.items():
            logits = head(pooled)
            head_loss = _masked_cross_entropy(
                logits,
                teacher_inputs[name].to(logits.device),
            )
            if head_loss is not None:
                active_teacher_losses.append(head_loss)
        if active_teacher_losses:
            loss = loss + self.teacher_loss_weight * torch.stack(active_teacher_losses).mean()

        return {"loss": loss, "logits": ordinal_logits, "z": z, "category_logits": category_logits}

    def _forward_hidden(self, inputs: Mapping[str, Any]) -> torch.Tensor:
        candidates = []
        base_model = getattr(self.backbone, "base_model", None)
        if base_model is not None:
            candidates.extend([getattr(base_model, "model", None), base_model])
        candidates.append(getattr(self.backbone, "model", None))
        for module in candidates:
            if not isinstance(module, nn.Module):
                continue
            kwargs = dict(inputs, use_cache=False, return_dict=True)
            try:
                output = module(**kwargs)
            except TypeError:
                kwargs.pop("use_cache", None)
                try:
                    output = module(**kwargs)
                except TypeError:
                    continue
            hidden = _extract_hidden(output)
            if hidden is not None:
                return hidden
        output = self.backbone(
            **dict(inputs, use_cache=False, output_hidden_states=True, return_dict=True)
        )
        hidden = _extract_hidden(output)
        if hidden is None:
            raise RuntimeError("Backbone returned no final hidden states")
        return hidden


def save_head_bundle(model: FrozenBackboneSafetyModel, output_dir: str | Path) -> None:
    """Save only the trained heads, metadata, and canonical prompts."""

    root = Path(output_dir).expanduser().resolve()
    root.mkdir(parents=True, exist_ok=True)
    state = {
        key: value.detach().cpu().contiguous()
        for key, value in model.heads.state_dict().items()
    }
    save_file(
        state,
        str(root / model.config.heads_file),
        metadata={"format": "pt", "safety_ds_format": "ordinal_heads_v1"},
    )
    raw = dict(model.config.raw)
    raw["learned_thresholds"] = [
        float(value)
        for value in model.heads.ordinal_head.thresholds().detach().float().cpu().tolist()
    ]
    (root / "ordinal_config.json").write_text(
        json.dumps(raw, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    prompt_dir = root / "prompts"
    prompt_dir.mkdir(exist_ok=True)
    prompts = PromptBundle.from_packaged()
    for target in ("image", "request", "response"):
        prompt = prompts.for_target(target)
        (prompt_dir / f"{target}_system.txt").write_text(prompt.system, encoding="utf-8")
        (prompt_dir / f"{target}_template.txt").write_text(prompt.user_template, encoding="utf-8")


def _apply_chat_template(processor: Any, messages: Sequence[Any]) -> Dict[str, Any]:
    base = {
        "tokenize": True,
        "add_generation_prompt": True,
        "return_tensors": "pt",
        "return_dict": True,
    }
    attempts = [
        {**base, "padding": True, "return_attention_mask": True},
        {**base, "processor_kwargs": {"padding": True, "return_attention_mask": True}},
    ]
    last_error: Exception | None = None
    for kwargs in attempts:
        try:
            output = processor.apply_chat_template(messages, **kwargs)
            return dict(output)
        except (TypeError, ValueError) as exc:
            last_error = exc
    raise RuntimeError("Processor rejected the SafeAtlas training batch") from last_error


def _teacher_label_ids(value: Any, *, target: str) -> Dict[str, int]:
    ignored = {"judge1": IGNORE_INDEX, "judge2": IGNORE_INDEX, "judge3": IGNORE_INDEX}
    if target == "image" or not isinstance(value, Mapping):
        return ignored
    qwen = str(value.get("qwen3guard") or "").strip().upper()
    guardreasoner = str(value.get("guardreasoner_vl") or "").strip()
    llamaguard = str(value.get("llamaguard4") or "").strip()
    return {
        "judge1": {"S": 0, "C": 1, "U": 2}.get(qwen, IGNORE_INDEX),
        "judge2": {"0": 0, "1": 1}.get(guardreasoner, IGNORE_INDEX),
        "judge3": {"0": 0, "1": 1}.get(llamaguard, IGNORE_INDEX),
    }


def _masked_cross_entropy(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor | None:
    active = labels != IGNORE_INDEX
    if not bool(active.any()):
        return None
    return F.cross_entropy(logits[active], labels[active])
