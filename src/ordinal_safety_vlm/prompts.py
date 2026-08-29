from __future__ import annotations

import re
from dataclasses import dataclass
from importlib.resources import files as package_files
from pathlib import Path
from typing import Any, Dict

from .configuration import OrdinalConfig
from .constants import TARGETS, normalize_target
from .hub import RepositoryFiles


_IMAGE_TOKEN = re.compile(r"<\s*image\s*>", re.IGNORECASE)


@dataclass(frozen=True)
class TargetPrompt:
    system: str
    user_template: str

    def render(self, *, request: str = "", response: str = "") -> str:
        try:
            rendered = self.user_template.format(request=request, response=response)
        except KeyError as exc:
            raise ValueError(f"Unsupported prompt placeholder: {exc}") from exc
        if len(_IMAGE_TOKEN.findall(rendered)) != 1:
            raise ValueError(
                "Every target template must contain exactly one <image> placeholder"
            )
        return rendered


class PromptBundle:
    def __init__(self, prompts: Dict[str, TargetPrompt]) -> None:
        missing = [target for target in TARGETS if target not in prompts]
        if missing:
            raise ValueError(f"Missing target prompts: {missing}")
        self._prompts = dict(prompts)

    @classmethod
    def from_packaged(cls) -> "PromptBundle":
        """Load the canonical explicit prompts used to construct training rows."""

        return cls(
            {
                target: TargetPrompt(
                    system=_read_packaged_prompt(f"{target}_system.txt"),
                    user_template=_read_packaged_prompt(f"{target}_template.txt"),
                )
                for target in TARGETS
            }
        )

    @classmethod
    def from_model_repository(
        cls,
        repository: RepositoryFiles,
        config: OrdinalConfig,
    ) -> "PromptBundle":
        prompts: Dict[str, TargetPrompt] = {}
        for target in TARGETS:
            row = config.prompts[target]
            system = _read_model_or_packaged_prompt(
                repository,
                model_relative=row.get("system_prompt_file"),
                packaged_name=f"{target}_system.txt",
            )
            template = _read_model_or_packaged_prompt(
                repository,
                model_relative=row.get("user_template_file"),
                packaged_name=f"{target}_template.txt",
            )
            prompts[target] = TargetPrompt(system=system, user_template=template)
        return cls(prompts)

    def for_target(self, target_name: str) -> TargetPrompt:
        return self._prompts[normalize_target(target_name)]

    def build_messages(
        self,
        *,
        target_name: str,
        image: Any,
        request: str = "",
        response: str = "",
    ) -> list[dict[str, Any]]:
        target = normalize_target(target_name)
        if image is None:
            raise ValueError("All three targets require an image")
        if target in {"request", "response"} and not str(request).strip():
            raise ValueError(f"{target} target requires request text")
        if target == "response" and not str(response).strip():
            raise ValueError("response target requires response text")
        prompt = self._prompts[target]
        user_text = prompt.render(request=str(request), response=str(response))
        user_content = _insert_image_at_placeholder(user_text, image)
        return [
            {"role": "system", "content": [{"type": "text", "text": prompt.system}]},
            {"role": "user", "content": user_content},
        ]


def _packaged_prompt_path(name: str) -> Path:
    resource = package_files("ordinal_safety_vlm").joinpath("resources", name)
    return Path(str(resource))


def _read_packaged_prompt(name: str) -> str:
    path = _packaged_prompt_path(name)
    if not path.is_file():
        raise FileNotFoundError(f"Missing packaged prompt: {name}")
    return path.read_text(encoding="utf-8")


def _read_model_or_packaged_prompt(
    repository: RepositoryFiles,
    *,
    model_relative: str | None,
    packaged_name: str,
) -> str:
    if model_relative:
        return repository.resolve(model_relative).read_text(encoding="utf-8")
    return _read_packaged_prompt(packaged_name)


def _insert_image_at_placeholder(text: str, image: Any) -> list[dict[str, Any]]:
    match = _IMAGE_TOKEN.search(text)
    if match is None:
        raise ValueError("Rendered template has no <image> placeholder")
    before = _IMAGE_TOKEN.sub("", text[: match.start()])
    after = _IMAGE_TOKEN.sub("", text[match.end() :])
    blocks: list[dict[str, Any]] = []
    if before:
        blocks.append({"type": "text", "text": before})
    blocks.append({"type": "image", "image": image})
    if after:
        blocks.append({"type": "text", "text": after})
    return blocks
