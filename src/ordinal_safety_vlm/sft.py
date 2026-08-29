from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Mapping

from PIL import Image

from .constants import PUBLIC_TO_DISPLAY_LABEL
from .data import DEFAULT_DATASET_ID, SafeAtlasRecord, iter_safeatlas_records
from .prompts import PromptBundle


_QWEN3GUARD = {"S": "safe", "C": "controversial", "U": "unsafe"}
_BINARY_GUARD = {"0": "safe", "1": "unsafe"}


def format_assistant_target(
    record: SafeAtlasRecord,
    *,
    include_teacher_heads: bool = True,
) -> str:
    """Render the structured Stage-1 assistant target used by SafeAtlas Guard."""

    lines = [
        f"Safety: {PUBLIC_TO_DISPLAY_LABEL[record.safety_label]}",
        f"Categories: {record.category}",
    ]
    if include_teacher_heads and record.target_name != "image" and record.teacher_head:
        qwen = _QWEN3GUARD.get(str(record.teacher_head.get("qwen3guard", "")).upper())
        guardreasoner = _BINARY_GUARD.get(str(record.teacher_head.get("guardreasoner_vl", "")))
        llamaguard = _BINARY_GUARD.get(str(record.teacher_head.get("llamaguard4", "")))
        if None not in (qwen, guardreasoner, llamaguard):
            lines.extend(
                [
                    "AuxiliaryJudgments:",
                    f"Judge1: {qwen}",
                    f"Judge2: {guardreasoner}",
                    f"Judge3: {llamaguard}",
                ]
            )
    return "\n".join(lines)


def to_sharegpt_record(
    record: SafeAtlasRecord,
    *,
    image_path: str,
    prompts: PromptBundle | None = None,
    include_teacher_heads: bool = True,
) -> Dict[str, Any]:
    """Convert a public record into LLaMA-Factory's multimodal messages format."""

    bundle = prompts or PromptBundle.from_packaged()
    prompt = bundle.for_target(record.target_name)
    user_text = prompt.render(request=record.request, response=record.response)
    return {
        "messages": [
            {"role": "system", "content": prompt.system},
            {"role": "user", "content": user_text},
            {
                "role": "assistant",
                "content": format_assistant_target(
                    record,
                    include_teacher_heads=include_teacher_heads,
                ),
            },
        ],
        "images": [image_path],
    }


def export_sft_dataset(
    output_dir: str | Path,
    *,
    dataset_name_or_path: str = DEFAULT_DATASET_ID,
    split: str = "train",
    max_samples: int | None = None,
    include_teacher_heads: bool = True,
    image_format: str = "jpg",
) -> Dict[str, Any]:
    """Materialize a LLaMA-Factory JSONL view without duplicating images per annotation."""

    root = Path(output_dir).expanduser().resolve()
    image_dir = root / "images"
    image_dir.mkdir(parents=True, exist_ok=True)
    output_jsonl = root / f"safeatlas_{split}.jsonl"
    prompts = PromptBundle.from_packaged()
    saved_images: set[str] = set()
    rows = 0

    normalized_format = str(image_format).strip().lower().lstrip(".")
    if normalized_format not in {"jpg", "png", "webp"}:
        raise ValueError("image_format must be jpg, png, or webp")

    with output_jsonl.open("w", encoding="utf-8") as handle:
        for record in iter_safeatlas_records(
            dataset_name_or_path,
            split=split,
            streaming=True,
            max_samples=max_samples,
        ):
            relative_image = Path("images") / f"{record.image_id}.{normalized_format}"
            absolute_image = root / relative_image
            if record.image_id not in saved_images:
                _save_image(record.image, absolute_image, normalized_format)
                saved_images.add(record.image_id)
            payload = to_sharegpt_record(
                record,
                image_path=relative_image.as_posix(),
                prompts=prompts,
                include_teacher_heads=include_teacher_heads,
            )
            handle.write(json.dumps(payload, ensure_ascii=False) + "\n")
            rows += 1

    dataset_key = f"safeatlas_{split}"
    dataset_info = {
        dataset_key: {
            "file_name": output_jsonl.name,
            "formatting": "sharegpt",
            "columns": {"messages": "messages", "images": "images"},
        }
    }
    dataset_info_path = root / "dataset_info.json"
    dataset_info_path.write_text(
        json.dumps(dataset_info, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    summary = {
        "dataset": dataset_name_or_path,
        "split": split,
        "rows": rows,
        "images": len(saved_images),
        "jsonl": str(output_jsonl),
        "dataset_info": str(dataset_info_path),
    }
    (root / "export_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return summary


def _save_image(image: Any, path: Path, image_format: str) -> None:
    if not isinstance(image, Image.Image):
        if isinstance(image, Mapping) and image.get("path"):
            with Image.open(str(image["path"])) as opened:
                image = opened.convert("RGB").copy()
        else:
            raise TypeError(f"Expected a PIL image, got {type(image)!r}")
    path.parent.mkdir(parents=True, exist_ok=True)
    if image_format == "jpg":
        image.convert("RGB").save(path, format="JPEG", quality=95, subsampling=0)
    elif image_format == "png":
        image.save(path, format="PNG", optimize=True)
    else:
        image.convert("RGB").save(path, format="WEBP", quality=95, method=4)
