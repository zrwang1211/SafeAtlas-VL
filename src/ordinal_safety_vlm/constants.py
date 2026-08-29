from __future__ import annotations

LABELS = (
    "safe core",
    "safe leaning disputed",
    "boundary uncertain",
    "unsafe leaning disputed",
    "unsafe core",
)

PUBLIC_LABELS = (
    "safe_core",
    "safe_leaning_disputed",
    "boundary_uncertain",
    "unsafe_leaning_disputed",
    "unsafe_core",
)

PUBLIC_LABEL_TO_ID = {label: index + 1 for index, label in enumerate(PUBLIC_LABELS)}
PUBLIC_TO_DISPLAY_LABEL = dict(zip(PUBLIC_LABELS, LABELS))
DISPLAY_TO_PUBLIC_LABEL = dict(zip(LABELS, PUBLIC_LABELS))

TARGETS = ("image", "request", "response")
TEACHER_LABELS = {
    "judge1": ("safe", "controversial", "unsafe"),
    "judge2": ("safe", "unsafe"),
    "judge3": ("safe", "unsafe"),
}


def normalize_target(value: str) -> str:
    target = str(value or "").strip().lower().replace("_", "-")
    aliases = {
        "prompt": "request",
        "user": "request",
        "image-only": "image",
        "visual": "image",
        "vision": "image",
    }
    target = aliases.get(target, target)
    if target not in TARGETS:
        raise ValueError(f"target must be one of {TARGETS}, got {value!r}")
    return target


def normalize_safety_label(value: str) -> str:
    """Return the canonical public dataset label with underscores."""

    normalized = str(value or "").strip().lower().replace("-", "_").replace(" ", "_")
    if normalized not in PUBLIC_LABEL_TO_ID:
        raise ValueError(f"Unknown safety label: {value!r}")
    return normalized
