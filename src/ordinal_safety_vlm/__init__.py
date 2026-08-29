"""Public runtime and data adapters for SafeAtlas Guard."""

from .data import SafeAtlasRecord, iter_safeatlas_records
from .predictor import Prediction, SafetyPredictor
from .prompts import PromptBundle
from .sft import format_assistant_target, to_sharegpt_record

__all__ = [
    "Prediction",
    "PromptBundle",
    "SafeAtlasRecord",
    "SafetyPredictor",
    "format_assistant_target",
    "iter_safeatlas_records",
    "to_sharegpt_record",
]
__version__ = "1.0.0"
