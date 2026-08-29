from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import torch
from PIL import Image

from .constants import normalize_target
from .model import PortableOrdinalModel
from .prompts import PromptBundle


@dataclass(frozen=True)
class Prediction:
    target_name: str
    safety_label: str
    risk_score: float
    z: float
    thresholds: list[float]
    ordinal_probs: list[float]
    class_probs: dict[str, float]
    category: str | None = None
    category_probs: dict[str, float] = field(default_factory=dict)
    teacher_predictions: dict[str, str] = field(default_factory=dict)
    teacher_probs: dict[str, dict[str, float]] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class SafetyPredictor:
    """High-level image/request/response inference API."""

    def __init__(
        self,
        model_name_or_path: str,
        *,
        revision: str | None = None,
        device_map: str = "auto",
        dtype: str = "bfloat16",
        trust_remote_code: bool = False,
    ) -> None:
        self.model = PortableOrdinalModel(
            model_name_or_path=model_name_or_path,
            revision=revision,
            device_map=device_map,
            dtype=dtype,
            trust_remote_code=trust_remote_code,
        )
        self.prompts = PromptBundle.from_model_repository(
            self.model.repository,
            self.model.config,
        )

    def predict(
        self,
        *,
        image: str | Path | Image.Image,
        target_name: str,
        request: str = "",
        response: str = "",
    ) -> Prediction:
        target = normalize_target(target_name)
        messages = self.prompts.build_messages(
            target_name=target,
            image=_load_rgb_image(image),
            request=request,
            response=response,
        )
        inputs = self._apply_chat_template(messages)
        tensors = self.model.predict_tensors(dict(inputs))
        return self._prediction(target=target, tensors=tensors)

    def _apply_chat_template(self, messages: list[dict[str, Any]]) -> Any:
        processor = self.model.processor
        optional = dict(self.model.config.chat_template_kwargs)
        chat_template = str(getattr(processor, "chat_template", "") or "")
        if "enable_thinking" not in chat_template:
            optional.pop("enable_thinking", None)
        processor_kwargs = dict(optional.pop("processor_kwargs", {}) or {})
        processor_kwargs.update({"padding": True, "return_attention_mask": True})
        base = {
            "tokenize": True,
            "return_tensors": "pt",
            "return_dict": True,
            "add_generation_prompt": False,
        }
        base.update(optional)

        attempts = [
            {**base, **processor_kwargs},
            {**base, "processor_kwargs": processor_kwargs},
            {
                key: value
                for key, value in {**base, **processor_kwargs}.items()
                if key not in {"enable_thinking", "add_generation_prompt"}
            },
        ]
        last_error: Exception | None = None
        for kwargs in attempts:
            try:
                return processor.apply_chat_template([messages], **kwargs)
            except (TypeError, ValueError) as exc:
                last_error = exc
        raise RuntimeError("Processor rejected all supported chat-template forms") from last_error

    def _prediction(
        self,
        *,
        target: str,
        tensors: dict[str, Any],
    ) -> Prediction:
        config = self.model.config
        class_values = _row(tensors["class_probs"])
        class_probs = {
            label: float(class_values[label_index])
            for label_index, label in enumerate(config.labels)
        }
        class_index = int(tensors["class_index"][0].detach().cpu().item())

        category = None
        category_probs: dict[str, float] = {}
        if "category_probs" in tensors:
            values = _row(tensors["category_probs"])
            category_probs = {
                label: float(values[label_index])
                for label_index, label in enumerate(config.auxiliary.category_labels)
            }
            category = config.auxiliary.category_labels[max(range(len(values)), key=values.__getitem__)]

        teacher_predictions: dict[str, str] = {}
        teacher_probs: dict[str, dict[str, float]] = {}
        if target != "image":
            for name, probability_tensor in tensors.get("teacher_probs", {}).items():
                values = _row(probability_tensor)
                labels = config.auxiliary.teacher_labels[name]
                teacher_probs[name] = {
                    label: float(values[label_index])
                    for label_index, label in enumerate(labels)
                }
                teacher_predictions[name] = labels[max(range(len(values)), key=values.__getitem__)]

        return Prediction(
            target_name=target,
            safety_label=config.labels[class_index],
            risk_score=float(tensors["risk_score"][0].detach().float().cpu().item()),
            z=float(tensors["z"][0].detach().float().cpu().item()),
            thresholds=_values(tensors["thresholds"]),
            ordinal_probs=_row(tensors["ordinal_probs"]),
            class_probs=class_probs,
            category=category,
            category_probs=category_probs,
            teacher_predictions=teacher_predictions,
            teacher_probs=teacher_probs,
        )

def _load_rgb_image(value: str | Path | Image.Image) -> Image.Image:
    if isinstance(value, Image.Image):
        image = value
    else:
        path = Path(value).expanduser()
        if not path.is_file():
            raise FileNotFoundError(path)
        with Image.open(path) as opened:
            opened.seek(0)
            image = opened.convert("RGB").copy()
    return image.convert("RGB") if image.mode != "RGB" else image.copy()


def _row(tensor: torch.Tensor) -> list[float]:
    return _values(tensor[0])


def _values(tensor: torch.Tensor) -> list[float]:
    return [float(value) for value in tensor.detach().float().cpu().tolist()]
