from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any, Mapping

import torch
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader, IterableDataset

from ordinal_safety_vlm.configuration import OrdinalConfig
from ordinal_safety_vlm.constants import LABELS, TEACHER_LABELS
from ordinal_safety_vlm.data import iter_safeatlas_records
from ordinal_safety_vlm.training import (
    CATEGORY_LABELS,
    FrozenBackboneSafetyModel,
    SafeAtlasHeadCollator,
    save_head_bundle,
)


class PublicRecordDataset(IterableDataset):
    def __init__(self, dataset: str, split: str, max_samples: int | None) -> None:
        super().__init__()
        self.dataset = dataset
        self.split = split
        self.max_samples = max_samples

    def __iter__(self):
        for record in iter_safeatlas_records(
            self.dataset,
            split=self.split,
            streaming=True,
            max_samples=self.max_samples,
        ):
            yield record.as_training_dict()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the frozen-backbone SafeAtlas heads")
    parser.add_argument("--config", required=True, help="Stage-2 JSON configuration")
    parser.add_argument("--model", help="Override the Stage-1 checkpoint")
    parser.add_argument("--dataset", help="Override the dataset repository or directory")
    parser.add_argument("--output-dir", help="Override the output directory")
    parser.add_argument("--max-steps", type=int, default=0, help="Stop after this many updates")
    parser.add_argument("--max-samples", type=int, help="Read at most this many rows")
    return parser.parse_args()


def load_run_config(path: str | Path) -> dict[str, Any]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("Stage-2 configuration must be a JSON object")
    return payload


def build_head_config(run: Mapping[str, Any]) -> OrdinalConfig:
    category_to_id = {label: index for index, label in enumerate(CATEGORY_LABELS)}
    payload = {
        "format": "safety_ds_standalone_ordinal",
        "format_version": 1,
        "weights_format": "heads_only",
        "heads_file": "ordinal_heads.safetensors",
        "backbone_location": ".",
        "processor_location": ".",
        "num_classes": len(LABELS),
        "labels": list(LABELS),
        "label_to_class_index": {label: index for index, label in enumerate(LABELS)},
        "label_to_ordinal_id": {label: index + 1 for index, label in enumerate(LABELS)},
        "head_arch": str(run["head_arch"]),
        "hidden_size": int(run["hidden_size"]),
        "head_dtype": str(run["head_dtype"]),
        "threshold_init_gap": float(run["threshold_init_gap"]),
        "pooling": "last_non_padding_token",
        "score_range_min": 0.0,
        "score_range_max": 100.0,
        "max_pixels": None,
        "min_pixels": None,
        "chat_template_kwargs": {
            "enable_thinking": False,
            "add_generation_prompt": True,
        },
        "auxiliary_heads": {
            "enabled": True,
            "head_arch": str(run["head_arch"]),
            "category_labels": list(CATEGORY_LABELS),
            "category_to_id": category_to_id,
            "teacher_labels": {name: list(labels) for name, labels in TEACHER_LABELS.items()},
        },
        "prompts": {
            target: {
                "system_prompt_file": f"prompts/{target}_system.txt",
                "user_template_file": f"prompts/{target}_template.txt",
            }
            for target in ("request", "response", "image")
        },
    }
    return OrdinalConfig.from_dict(payload)


def positive_int(run: Mapping[str, Any], key: str) -> int:
    value = int(run[key])
    if value <= 0:
        raise ValueError(f"{key} must be positive")
    return value


def cosine_multiplier(step: int, *, warmup_steps: int, total_steps: int, floor: float) -> float:
    if warmup_steps > 0 and step < warmup_steps:
        return float(step + 1) / float(warmup_steps)
    progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
    progress = min(max(progress, 0.0), 1.0)
    return floor + (1.0 - floor) * 0.5 * (1.0 + math.cos(math.pi * progress))


def main() -> None:
    args = parse_args()
    run = load_run_config(args.config)
    if args.model:
        run["model"] = args.model
    if args.dataset:
        run["dataset"] = args.dataset
    if args.output_dir:
        run["output_dir"] = args.output_dir

    batch_size = positive_int(run, "batch_size")
    accumulation = positive_int(run, "gradient_accumulation_steps")
    epochs = positive_int(run, "epochs")
    expected_processes = positive_int(run, "expected_num_processes")
    configured_samples = positive_int(run, "num_train_samples")
    max_samples = args.max_samples if args.max_samples is not None else None
    if max_samples is not None and max_samples <= 0:
        raise ValueError("max_samples must be positive")

    try:
        from accelerate import Accelerator
        from transformers import AutoModelForImageTextToText, AutoProcessor
    except ImportError as exc:
        raise ImportError("Install training dependencies with `pip install -e '.[train]'`") from exc

    dtype_name = str(run["dtype"])
    dtype = {
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
        "float32": torch.float32,
    }.get(dtype_name)
    if dtype is None:
        raise ValueError("dtype must be bfloat16, float16, or float32")

    torch.manual_seed(int(run["seed"]))
    if torch.cuda.is_available() and bool(run["tf32"]):
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    accelerator = Accelerator(
        gradient_accumulation_steps=accumulation,
        mixed_precision=(
            "no" if dtype_name == "float32" else ("bf16" if dtype_name == "bfloat16" else "fp16")
        ),
    )
    if accelerator.num_processes != expected_processes:
        accelerator.print(
            f"Warning: configuration expects {expected_processes} processes, "
            f"but accelerate started {accelerator.num_processes}."
        )

    model_path = str(run["model"])
    processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=False)
    backbone = AutoModelForImageTextToText.from_pretrained(
        model_path,
        dtype=dtype,
        trust_remote_code=False,
    )
    head_config = build_head_config(run)
    if head_config.head_dtype != "float32":
        raise ValueError("Stage-2 prediction heads must use head_dtype=float32")
    hidden_size = int(getattr(backbone.config, "hidden_size", head_config.hidden_size))
    if hidden_size != head_config.hidden_size:
        raise ValueError(
            f"Configured hidden_size={head_config.hidden_size} does not match "
            f"backbone hidden_size={hidden_size}"
        )

    model = FrozenBackboneSafetyModel(
        backbone,
        head_config,
        category_loss_weight=float(run["category_loss_weight"]),
        teacher_loss_weight=float(run["teacher_loss_weight"]),
        gaussian_sigma=float(run["gaussian_sigma"]),
    )
    model.heads.float()
    if any(parameter.dtype != torch.float32 for parameter in model.heads.parameters()):
        raise RuntimeError("Failed to initialize all Stage-2 head parameters in float32")
    threshold_parameters = [model.heads.ordinal_head.threshold_delta]
    threshold_ids = {id(parameter) for parameter in threshold_parameters}
    head_parameters = [
        parameter for parameter in model.heads.parameters() if id(parameter) not in threshold_ids
    ]
    head_lr = float(run["learning_rate_head"])
    threshold_lr = float(run["learning_rate_threshold"])
    optimizer = torch.optim.AdamW(
        [
            {"params": head_parameters, "lr": head_lr},
            {"params": threshold_parameters, "lr": threshold_lr},
        ],
        weight_decay=float(run["weight_decay"]),
        fused=bool(run["use_fused_adamw"]) and torch.cuda.is_available(),
    )

    samples_for_schedule = min(configured_samples, max_samples or configured_samples)
    global_batch_size = batch_size * accelerator.num_processes * accumulation
    total_steps = math.ceil(samples_for_schedule / global_batch_size) * epochs
    if args.max_steps > 0:
        total_steps = min(total_steps, args.max_steps)
    if total_steps <= 0:
        raise ValueError("The computed number of training steps is zero")
    warmup_steps = int(total_steps * float(run["warmup_ratio"]))
    threshold_floor = float(run["learning_rate_threshold_floor"]) / threshold_lr
    scheduler = LambdaLR(
        optimizer,
        lr_lambda=[
            lambda step: cosine_multiplier(
                step,
                warmup_steps=warmup_steps,
                total_steps=total_steps,
                floor=0.0,
            ),
            lambda step: cosine_multiplier(
                step,
                warmup_steps=warmup_steps,
                total_steps=total_steps,
                floor=threshold_floor,
            ),
        ],
    )

    dataset = PublicRecordDataset(str(run["dataset"]), str(run["split"]), max_samples)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        collate_fn=SafeAtlasHeadCollator(processor),
        num_workers=int(run["num_workers"]),
        pin_memory=bool(run["dataloader_pin_memory"]),
    )
    model, optimizer, loader, scheduler = accelerator.prepare(
        model,
        optimizer,
        loader,
        scheduler,
    )
    trainable_parameters = [
        parameter for parameter in model.parameters() if parameter.requires_grad
    ]

    update_step = 0
    model.train()
    for epoch in range(epochs):
        for inputs in loader:
            with accelerator.accumulate(model):
                output = model(**inputs)
                loss = output["loss"]
                accelerator.backward(loss)
                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(trainable_parameters, 1.0)
                optimizer.step()
                if accelerator.sync_gradients:
                    scheduler.step()
                optimizer.zero_grad(set_to_none=True)
            if accelerator.sync_gradients:
                update_step += 1
                if update_step % int(run["log_steps"]) == 0:
                    accelerator.print(
                        f"epoch={epoch + 1} step={update_step}/{total_steps} "
                        f"loss={float(loss.detach()):.6f}"
                    )
                if update_step >= total_steps:
                    break
        if update_step >= total_steps:
            break

    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        unwrapped = accelerator.unwrap_model(model)
        output_dir = Path(str(run["output_dir"]))
        save_head_bundle(unwrapped, output_dir)
        summary = {
            "model": model_path,
            "dataset": str(run["dataset"]),
            "split": str(run["split"]),
            "processes": accelerator.num_processes,
            "global_batch_size": global_batch_size,
            "update_steps": update_step,
            "epochs": epochs,
        }
        (output_dir / "training_summary.json").write_text(
            json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        accelerator.print(f"Saved SafeAtlas heads to {output_dir.resolve()}")


if __name__ == "__main__":
    main()
