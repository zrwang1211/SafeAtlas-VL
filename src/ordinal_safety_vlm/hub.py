from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict


class RepositoryFiles:
    """Resolve files from either a local model directory or the HF Hub."""

    def __init__(self, model_name_or_path: str, *, revision: str | None = None) -> None:
        self.model_name_or_path = str(model_name_or_path)
        self.revision = revision
        local = Path(self.model_name_or_path).expanduser()
        self.local_root = local.resolve() if local.is_dir() else None

    def resolve(self, relative_path: str) -> Path:
        relative_path = str(relative_path).replace("\\", "/").lstrip("/")
        if not relative_path or ".." in Path(relative_path).parts:
            raise ValueError(f"Unsafe repository-relative path: {relative_path!r}")
        if self.local_root is not None:
            path = (self.local_root / relative_path).resolve()
            if self.local_root not in path.parents and path != self.local_root:
                raise ValueError(f"Path escapes model directory: {relative_path!r}")
            if not path.is_file():
                raise FileNotFoundError(path)
            return path

        try:
            from huggingface_hub import hf_hub_download
        except ImportError as exc:
            raise ImportError("Remote model loading requires huggingface-hub") from exc
        return Path(
            hf_hub_download(
                repo_id=self.model_name_or_path,
                filename=relative_path,
                revision=self.revision,
            )
        )

    def read_json(self, relative_path: str) -> Dict[str, Any]:
        payload = json.loads(self.resolve(relative_path).read_text(encoding="utf-8"))
        if not isinstance(payload, dict):
            raise ValueError(f"Expected a JSON object in {relative_path}")
        return payload
