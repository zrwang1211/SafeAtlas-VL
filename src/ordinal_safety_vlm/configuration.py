from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Mapping, Sequence

from .constants import LABELS, TEACHER_LABELS
from .hub import RepositoryFiles


@dataclass(frozen=True)
class AuxiliaryConfig:
    enabled: bool
    head_arch: str
    category_labels: tuple[str, ...]
    teacher_labels: Dict[str, tuple[str, ...]]


@dataclass(frozen=True)
class OrdinalConfig:
    raw: Dict[str, Any]
    heads_file: str
    num_classes: int
    labels: tuple[str, ...]
    head_arch: str
    hidden_size: int
    head_dtype: str
    threshold_init_gap: float
    score_range_min: float
    score_range_max: float
    chat_template_kwargs: Dict[str, Any]
    max_pixels: int | None
    min_pixels: int | None
    prompts: Dict[str, Dict[str, str | None]]
    auxiliary: AuxiliaryConfig

    @classmethod
    def from_repository(cls, files: RepositoryFiles) -> "OrdinalConfig":
        return cls.from_dict(files.read_json("ordinal_config.json"))

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "OrdinalConfig":
        raw = dict(payload)
        if raw.get("format") != "safety_ds_standalone_ordinal":
            raise ValueError(
                "Unsupported model format. Expected safety_ds_standalone_ordinal."
            )
        if raw.get("weights_format") != "heads_only":
            raise ValueError("Published runtime requires a heads-only portable export")

        labels = tuple(str(x) for x in raw.get("labels") or LABELS)
        num_classes = int(raw.get("num_classes", len(labels)))
        if num_classes != len(labels) or num_classes < 2:
            raise ValueError(
                f"Invalid ordinal labels: num_classes={num_classes}, labels={labels}"
            )

        aux_raw = raw.get("auxiliary_heads")
        if not isinstance(aux_raw, Mapping):
            aux_raw = {}
        teacher_raw = aux_raw.get("teacher_labels")
        teacher_labels: Dict[str, tuple[str, ...]] = {}
        for name, defaults in TEACHER_LABELS.items():
            values = teacher_raw.get(name) if isinstance(teacher_raw, Mapping) else None
            teacher_labels[name] = tuple(str(x) for x in (values or defaults))

        prompt_raw = raw.get("prompts")
        prompts: Dict[str, Dict[str, str | None]] = {}
        for target in ("request", "response", "image"):
            row = prompt_raw.get(target) if isinstance(prompt_raw, Mapping) else None
            row = row if isinstance(row, Mapping) else {}
            prompts[target] = {
                "system_prompt_file": _optional_string(row.get("system_prompt_file")),
                "user_template_file": _optional_string(row.get("user_template_file")),
            }

        return cls(
            raw=dict(raw),
            heads_file=str(raw.get("heads_file") or "ordinal_heads.safetensors"),
            num_classes=num_classes,
            labels=labels,
            head_arch=str(raw.get("head_arch") or "linear_ln_linear"),
            hidden_size=int(raw["hidden_size"]),
            head_dtype=_head_dtype(raw.get("head_dtype", "float32")),
            threshold_init_gap=float(raw.get("threshold_init_gap", 1.25)),
            score_range_min=float(raw.get("score_range_min", 0.0)),
            score_range_max=float(raw.get("score_range_max", 100.0)),
            chat_template_kwargs=dict(raw.get("chat_template_kwargs") or {}),
            max_pixels=_optional_int(raw.get("max_pixels")),
            min_pixels=_optional_int(raw.get("min_pixels")),
            prompts=prompts,
            auxiliary=AuxiliaryConfig(
                enabled=bool(aux_raw.get("enabled", False)),
                head_arch=str(aux_raw.get("head_arch") or raw.get("head_arch") or "linear_ln_linear"),
                category_labels=tuple(str(x) for x in aux_raw.get("category_labels") or ()),
                teacher_labels=teacher_labels,
            ),
        )


def _optional_string(value: Any) -> str | None:
    text = str(value or "").strip()
    return text or None


def _optional_int(value: Any) -> int | None:
    return None if value in (None, "") else int(value)


def _head_dtype(value: Any) -> str:
    dtype = str(value or "float32").strip().lower()
    if dtype != "float32":
        raise ValueError("SafeAtlas prediction heads must use float32 parameters")
    return dtype
