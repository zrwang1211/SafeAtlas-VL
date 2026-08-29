from __future__ import annotations

from typing import Any, Dict, Iterable, Mapping

import torch
import torch.nn as nn

from .configuration import OrdinalConfig
from .heads import (
    SafetyHeads,
    class_probs_to_score,
    cumulative_to_class_probs,
    last_non_padding_pool,
)
from .hub import RepositoryFiles


MODEL_CLASS_NAMES = (
    "AutoModelForImageTextToText",
    "AutoModelForMultimodalLM",
    "AutoModelForVision2Seq",
)


def resolve_dtype(name: str) -> torch.dtype | str:
    normalized = str(name or "auto").strip().lower()
    if normalized == "auto":
        return "auto"
    mapping = {
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
        "float32": torch.float32,
    }
    if normalized not in mapping:
        raise ValueError(f"Unsupported dtype {name!r}; expected auto/bfloat16/float16/float32")
    return mapping[normalized]


class PortableOrdinalModel:
    def __init__(
        self,
        *,
        model_name_or_path: str,
        revision: str | None = None,
        device_map: str = "auto",
        dtype: str = "bfloat16",
        trust_remote_code: bool = False,
    ) -> None:
        self.model_name_or_path = str(model_name_or_path)
        self.repository = RepositoryFiles(self.model_name_or_path, revision=revision)
        self.config = OrdinalConfig.from_repository(self.repository)
        self.processor = self._load_processor(trust_remote_code=trust_remote_code)
        self.backbone = self._load_backbone(
            device_map=device_map,
            dtype=dtype,
            trust_remote_code=trust_remote_code,
        )
        self._configure_processor()
        self.heads = SafetyHeads(self.config)
        self._load_heads()
        self.backbone.eval()
        self.heads.eval()
        self.heads.to(self.runtime_device())

    def _load_processor(self, *, trust_remote_code: bool) -> Any:
        from transformers import AutoProcessor

        return AutoProcessor.from_pretrained(
            self.model_name_or_path,
            revision=self.repository.revision,
            trust_remote_code=trust_remote_code,
        )

    def _load_backbone(
        self,
        *,
        device_map: str,
        dtype: str,
        trust_remote_code: bool,
    ) -> nn.Module:
        import transformers

        kwargs: Dict[str, Any] = {
            "revision": self.repository.revision,
            "trust_remote_code": trust_remote_code,
            "dtype": resolve_dtype(dtype),
        }
        if str(device_map).lower() not in {"", "none"}:
            kwargs["device_map"] = device_map
        last_error: Exception | None = None
        for class_name in MODEL_CLASS_NAMES:
            model_class = getattr(transformers, class_name, None)
            if model_class is None:
                continue
            try:
                return model_class.from_pretrained(self.model_name_or_path, **kwargs)
            except Exception as exc:
                last_error = exc
        raise RuntimeError(
            f"No supported Transformers multimodal class could load {self.model_name_or_path!r}"
        ) from last_error

    def _configure_processor(self) -> None:
        tokenizer = getattr(self.processor, "tokenizer", None)
        if tokenizer is not None:
            tokenizer.padding_side = "right"
            tokenizer.truncation_side = "right"
        image_processor = getattr(self.processor, "image_processor", None)
        if image_processor is not None:
            if self.config.max_pixels is not None and hasattr(image_processor, "max_pixels"):
                image_processor.max_pixels = self.config.max_pixels
            if self.config.min_pixels is not None and hasattr(image_processor, "min_pixels"):
                image_processor.min_pixels = self.config.min_pixels

    def _load_heads(self) -> None:
        try:
            from safetensors.torch import load_file
        except ImportError as exc:
            raise ImportError("Loading ordinal heads requires safetensors") from exc
        state = load_file(str(self.repository.resolve(self.config.heads_file)), device="cpu")
        non_fp32 = [name for name, value in state.items() if value.dtype != torch.float32]
        if non_fp32:
            raise RuntimeError(
                "SafeAtlas head weights must be float32; "
                f"found incompatible tensors: {non_fp32[:5]}"
            )
        incompatible = self.heads.load_state_dict(state, strict=True)
        if incompatible.missing_keys or incompatible.unexpected_keys:
            raise RuntimeError(
                "Ordinal head state mismatch: "
                f"missing={incompatible.missing_keys}, unexpected={incompatible.unexpected_keys}"
            )

    def runtime_device(self) -> torch.device:
        device = getattr(self.backbone, "device", None)
        if isinstance(device, torch.device) and device.type != "meta":
            return device
        device_map = getattr(self.backbone, "hf_device_map", None)
        if isinstance(device_map, dict):
            for value in device_map.values():
                if isinstance(value, int):
                    return torch.device(f"cuda:{value}")
                if isinstance(value, str) and value.startswith("cuda"):
                    return torch.device(value)
                if value in {"cpu", "mps", "xpu"}:
                    return torch.device(value)
        try:
            parameter_device = next(self.backbone.parameters()).device
            if parameter_device.type != "meta":
                return parameter_device
        except StopIteration:
            pass
        return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    def backbone_dtype(self) -> torch.dtype | None:
        try:
            return next(self.backbone.parameters()).dtype
        except StopIteration:
            return None

    def move_inputs(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        device = self.runtime_device()
        dtype = self.backbone_dtype()
        moved: Dict[str, Any] = {}
        for key, value in inputs.items():
            if not hasattr(value, "to"):
                moved[key] = value
            elif getattr(getattr(value, "dtype", None), "is_floating_point", False) and dtype:
                moved[key] = value.to(device=device, dtype=dtype)
            else:
                moved[key] = value.to(device=device)
        return moved

    @torch.inference_mode()
    def predict_tensors(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        inputs = self.move_inputs(inputs)
        attention_mask = inputs.get("attention_mask")
        hidden = self._forward_hidden(**inputs)
        pooled = last_non_padding_pool(hidden, attention_mask)
        head_parameter = next(self.heads.ordinal_head.parameters())
        pooled = pooled.to(device=head_parameter.device, dtype=head_parameter.dtype)

        ordinal_logits, z = self.heads.ordinal_head(pooled)
        ordinal_probs = torch.sigmoid(ordinal_logits)
        class_probs = cumulative_to_class_probs(ordinal_probs)
        risk_score = class_probs_to_score(
            class_probs,
            minimum=self.config.score_range_min,
            maximum=self.config.score_range_max,
        )
        output: Dict[str, Any] = {
            "z": z,
            "thresholds": self.heads.ordinal_head.thresholds(),
            "ordinal_probs": ordinal_probs,
            "class_probs": class_probs,
            "risk_score": risk_score,
            "class_index": class_probs.argmax(dim=-1),
        }
        if self.heads.category_head is not None:
            category_logits = self.heads.category_head(pooled)
            output["category_probs"] = torch.softmax(category_logits, dim=-1)
        teacher_probs: Dict[str, torch.Tensor] = {}
        for name, head in self.heads.teacher_heads.items():
            teacher_probs[name] = torch.softmax(head(pooled), dim=-1)
        output["teacher_probs"] = teacher_probs
        return output

    def _forward_hidden(self, **inputs: Any) -> torch.Tensor:
        for module in self._hidden_candidates():
            output = self._call_hidden_module(module, inputs)
            hidden = _extract_hidden(output)
            if hidden is not None:
                return hidden

        fallback = dict(inputs)
        fallback.update(use_cache=False, output_hidden_states=True, return_dict=True)
        output = self.backbone(**fallback)
        hidden = _extract_hidden(output)
        if hidden is None:
            raise RuntimeError("Backbone returned no final token hidden states")
        return hidden

    def _hidden_candidates(self) -> Iterable[nn.Module]:
        candidates: list[nn.Module] = []
        seen: set[int] = set()

        def add(module: Any) -> None:
            if isinstance(module, nn.Module) and id(module) not in seen:
                seen.add(id(module))
                candidates.append(module)

        get_base = getattr(self.backbone, "get_base_model", None)
        if callable(get_base):
            try:
                add(get_base())
            except Exception:
                pass
        base_model = getattr(self.backbone, "base_model", None)
        if base_model is not None:
            add(getattr(base_model, "model", None))
            add(base_model)
        add(getattr(self.backbone, "model", None))
        return candidates

    @staticmethod
    def _call_hidden_module(module: nn.Module, inputs: Dict[str, Any]) -> Any:
        kwargs = dict(inputs)
        kwargs.update(use_cache=False, return_dict=True)
        try:
            return module(**kwargs)
        except TypeError:
            kwargs.pop("use_cache", None)
            try:
                return module(**kwargs)
            except TypeError:
                return None


def _extract_hidden(output: Any) -> torch.Tensor | None:
    if output is None:
        return None
    if isinstance(output, Mapping):
        last_hidden = output.get("last_hidden_state")
        if isinstance(last_hidden, torch.Tensor) and last_hidden.ndim == 3:
            return last_hidden
        hidden_states = output.get("hidden_states")
        if hidden_states and isinstance(hidden_states[-1], torch.Tensor):
            return hidden_states[-1]
    last_hidden = getattr(output, "last_hidden_state", None)
    if isinstance(last_hidden, torch.Tensor) and last_hidden.ndim == 3:
        return last_hidden
    hidden_states = getattr(output, "hidden_states", None)
    if hidden_states and isinstance(hidden_states[-1], torch.Tensor):
        return hidden_states[-1]
    if isinstance(output, (tuple, list)):
        for item in output:
            if isinstance(item, torch.Tensor) and item.ndim == 3:
                return item
    return None
