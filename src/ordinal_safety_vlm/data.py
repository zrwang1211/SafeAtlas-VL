from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterator, Mapping

from .constants import PUBLIC_LABEL_TO_ID, normalize_safety_label, normalize_target


DEFAULT_DATASET_ID = "zrwang1211/SafeAtlas-VL"


@dataclass(frozen=True)
class SafeAtlasRecord:
    """One flattened annotation from the public SafeAtlas-VL images view."""

    annotation_id: str
    image_id: str
    image: Any
    target_name: str
    request: str
    response: str
    safety_label: str
    label_id: int
    category: str
    teacher_head: Dict[str, str]

    def as_training_dict(self) -> Dict[str, Any]:
        return {
            "sample_id": self.annotation_id,
            "image_id": self.image_id,
            "image": self.image,
            "target_name": self.target_name,
            "request": self.request,
            "response": self.response,
            "label_name": self.safety_label,
            "label_id": self.label_id,
            "category": self.category,
            "teacher_head": dict(self.teacher_head),
        }


def record_from_public_row(
    image_row: Mapping[str, Any],
    annotation: Mapping[str, Any],
) -> SafeAtlasRecord:
    """Validate and flatten one nested public annotation."""

    image_id = str(image_row.get("image_id") or "").strip()
    annotation_id = str(annotation.get("annotation_id") or "").strip()
    if not image_id or not annotation_id:
        raise ValueError("SafeAtlas rows require non-empty image_id and annotation_id")

    target_name = normalize_target(str(annotation.get("target") or ""))
    safety_label = normalize_safety_label(str(annotation.get("safety_label") or ""))
    category = str(annotation.get("category") or "").strip().lower().replace("_", " ")
    if not category:
        raise ValueError(f"Missing category for annotation {annotation_id}")
    if safety_label == "safe_core" and category != "none":
        raise ValueError(
            f"safe_core annotation {annotation_id} must use category='none', got {category!r}"
        )

    request = str(annotation.get("request") or "")
    response = str(annotation.get("response") or "")
    if target_name in {"request", "response"} and not request.strip():
        raise ValueError(f"{target_name} annotation {annotation_id} has no request text")
    if target_name == "response" and not response.strip():
        raise ValueError(f"response annotation {annotation_id} has no response text")

    raw_teacher = annotation.get("teacher_head")
    teacher_head = {
        str(key): str(value)
        for key, value in (raw_teacher.items() if isinstance(raw_teacher, Mapping) else ())
        if value is not None and str(value).strip()
    }
    if target_name == "image":
        teacher_head = {}

    image = image_row.get("image")
    if image is None:
        raise ValueError(f"Missing image payload for image_id={image_id}")

    return SafeAtlasRecord(
        annotation_id=annotation_id,
        image_id=image_id,
        image=image,
        target_name=target_name,
        request=request,
        response=response,
        safety_label=safety_label,
        label_id=PUBLIC_LABEL_TO_ID[safety_label],
        category=category,
        teacher_head=teacher_head,
    )


def iter_safeatlas_records(
    dataset_name_or_path: str = DEFAULT_DATASET_ID,
    *,
    split: str = "train",
    streaming: bool = True,
    max_samples: int | None = None,
) -> Iterator[SafeAtlasRecord]:
    """Stream flattened annotations from the Hub or a local dataset repository.

    A local repository path is expected to contain the released
    ``data/shards/<split>/*/images.parquet`` layout.
    """

    try:
        from datasets import load_dataset
    except ImportError as exc:
        raise ImportError(
            "Loading SafeAtlas-VL requires the project dependencies. "
            "Install with `pip install -e .`."
        ) from exc

    source = Path(dataset_name_or_path).expanduser()
    if source.is_dir():
        pattern = str(source / "data" / "shards" / split / "*" / "images.parquet")
        dataset = load_dataset(
            "parquet",
            data_files={split: pattern},
            split=split,
            streaming=streaming,
        )
    else:
        dataset = load_dataset(
            dataset_name_or_path,
            "images",
            split=split,
            streaming=streaming,
        )

    emitted = 0
    for image_row in dataset:
        annotations = image_row.get("annotations") or []
        if not isinstance(annotations, (list, tuple)):
            raise ValueError("Expected annotations to be a list in the images configuration")
        for annotation in annotations:
            if not isinstance(annotation, Mapping):
                raise ValueError("Expected every nested annotation to be a mapping")
            yield record_from_public_row(image_row, annotation)
            emitted += 1
            if max_samples is not None and emitted >= int(max_samples):
                return
