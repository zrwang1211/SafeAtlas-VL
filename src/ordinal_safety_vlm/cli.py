from __future__ import annotations

import argparse
import json
import sys
from typing import Iterable

from .data import DEFAULT_DATASET_ID, iter_safeatlas_records
from .predictor import SafetyPredictor
from .sft import export_sft_dataset


def _model_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--model", required=True, help="Local model directory or HF repository id")
    parser.add_argument("--revision", default=None)
    parser.add_argument("--device-map", default="auto")
    parser.add_argument(
        "--dtype",
        choices=("auto", "bfloat16", "float16", "float32"),
        default="bfloat16",
    )
    parser.add_argument(
        "--trust-remote-code",
        action=argparse.BooleanOptionalAction,
        default=False,
    )


def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(prog="safeatlas")
    subparsers = parser.add_subparsers(dest="command", required=True)

    single = subparsers.add_parser("predict", help="Predict one sample")
    _model_arguments(single)
    single.add_argument("--image", required=True)
    single.add_argument("--target", required=True, choices=("image", "request", "response"))
    single.add_argument("--request", default="")
    single.add_argument("--response", default="")

    inspect_data = subparsers.add_parser(
        "inspect-data",
        help="Validate and print a few records from the public dataset",
    )
    inspect_data.add_argument("--dataset", default=DEFAULT_DATASET_ID)
    inspect_data.add_argument("--split", default="train")
    inspect_data.add_argument("--limit", type=int, default=3)

    export_sft = subparsers.add_parser(
        "export-sft",
        help="Export the public dataset to multimodal ShareGPT JSONL",
    )
    export_sft.add_argument("--dataset", default=DEFAULT_DATASET_ID)
    export_sft.add_argument("--split", default="train")
    export_sft.add_argument("--output-dir", required=True)
    export_sft.add_argument("--max-samples", type=int, default=None)
    export_sft.add_argument("--image-format", choices=("jpg", "png", "webp"), default="jpg")
    export_sft.add_argument(
        "--teacher-heads",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    return parser.parse_args(argv)


def _load_predictor(args: argparse.Namespace) -> SafetyPredictor:
    return SafetyPredictor(
        args.model,
        revision=args.revision,
        device_map=args.device_map,
        dtype=args.dtype,
        trust_remote_code=bool(args.trust_remote_code),
    )


def _run_single(args: argparse.Namespace) -> int:
    prediction = _load_predictor(args).predict(
        image=args.image,
        target_name=args.target,
        request=args.request,
        response=args.response,
    )
    print(json.dumps(prediction.to_dict(), ensure_ascii=False, indent=2))
    return 0


def _run_inspect_data(args: argparse.Namespace) -> int:
    if args.limit <= 0:
        raise ValueError("limit must be positive")
    for record in iter_safeatlas_records(
        args.dataset,
        split=args.split,
        streaming=True,
        max_samples=args.limit,
    ):
        payload = record.as_training_dict()
        payload["image"] = {
            "type": type(record.image).__name__,
            "size": list(getattr(record.image, "size", ())),
        }
        print(json.dumps(payload, ensure_ascii=False))
    return 0


def _run_export_sft(args: argparse.Namespace) -> int:
    summary = export_sft_dataset(
        args.output_dir,
        dataset_name_or_path=args.dataset,
        split=args.split,
        max_samples=args.max_samples,
        include_teacher_heads=bool(args.teacher_heads),
        image_format=args.image_format,
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


def main(argv: Iterable[str] | None = None) -> int:
    try:
        args = parse_args(argv)
        handlers = {
            "predict": _run_single,
            "inspect-data": _run_inspect_data,
            "export-sft": _run_export_sft,
        }
        return handlers[args.command](args)
    except Exception as exc:
        print(f"[FATAL] {type(exc).__name__}: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
