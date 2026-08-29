# Training SafeAtlas Guard

SafeAtlas Guard uses two distinct training stages. The supplied configurations
cover the 2B, 4B, and 8B variants and assume one node with eight GPUs.

## Stage 1: multimodal instruction tuning

Stage 1 performs full-parameter supervised fine-tuning of a Qwen3-VL Instruct
backbone. For each image, request, or response target, the model learns to emit
a structured judgment containing the five-level safety label and harm category.
Request and response targets also include three teacher judgments.

Install the training dependencies and export the public dataset:

```bash
pip install -e ".[train]"

safeatlas export-sft \
  --dataset zrwang1211/SafeAtlas-VL \
  --split train \
  --output-dir outputs/safeatlas_sft
```

The export writes `safeatlas_train.jsonl`, an `images` directory, and the
`dataset_info.json` consumed by LLaMA-Factory. Then launch the configuration for
the desired model size:

```bash
llamafactory-cli train training/stage1/safeatlas_guard_2b.yaml
llamafactory-cli train training/stage1/safeatlas_guard_4b.yaml
llamafactory-cli train training/stage1/safeatlas_guard_8b.yaml
```

These are alternative runs; execute only the model size you need. Each uses a
per-device batch size of 4 and gradient accumulation of 4, giving a global
batch size of 128 on eight GPUs. LLaMA-Factory automatically launches one
worker per visible GPU when multiple GPUs are available.

## Stage 2: frozen-backbone prediction heads

Stage 2 loads the completed Stage-1 checkpoint and freezes the entire
multimodal backbone. It trains:

- a five-level cumulative ordinal head with Gaussian-smoothed targets;
- a 16-way harm-category head, where `none` is the category for safe-core
  examples;
- three teacher-simulation heads for request and response targets.

Launch the matching Stage-2 configuration:

```bash
accelerate launch --num_processes 8 training/train_heads.py --config training/stage2/safeatlas_guard_2b.json
accelerate launch --num_processes 8 training/train_heads.py --config training/stage2/safeatlas_guard_4b.json
accelerate launch --num_processes 8 training/train_heads.py --config training/stage2/safeatlas_guard_8b.json
```

Again, execute only one model-size command. The Stage-2 configurations use a
per-device batch size of 8 and gradient accumulation of 4, giving a global
batch size of 256 on eight GPUs. They preserve the learning rates, loss
weights, Gaussian smoothing, and ordinal-threshold initialization used by the
original training runs.

The frozen backbone is loaded in BF16. The prediction heads are comparatively
small, and the four learned scalar thresholds directly define the boundaries
between the five ordinal classes. Small threshold updates can be lost at BF16
precision, which can shift those boundaries. All head parameters, thresholds,
and optimizer states therefore remain in FP32 and are saved in FP32. BF16
autocast may accelerate the dense projections, while threshold construction
and ordinal logits remain in FP32.

The Stage-2 output contains `ordinal_heads.safetensors`,
`ordinal_config.json`, and the prompt files. A standalone release combines
these files with the corresponding Stage-1 Transformers checkpoint.

## Configuration summary

| Variant | Backbone | Hidden size | GPUs | Stage-1 batch / accumulation | Stage-2 batch / accumulation |
| --- | --- | ---: | ---: | ---: | ---: |
| 2B | `Qwen/Qwen3-VL-2B-Instruct` | 2048 | 8 | 4 / 4 | 8 / 4 |
| 4B | `Qwen/Qwen3-VL-4B-Instruct` | 2560 | 8 | 4 / 4 | 8 / 4 |
| 8B | `Qwen/Qwen3-VL-8B-Instruct` | 4096 | 8 | 4 / 4 | 8 / 4 |

Model, dataset, and output paths are portable defaults in the configuration
files. They can be overridden for a local run:

```bash
accelerate launch --num_processes 8 training/train_heads.py \
  --config training/stage2/safeatlas_guard_2b.json \
  --model /path/to/stage1_checkpoint \
  --dataset zrwang1211/SafeAtlas-VL \
  --output-dir /path/to/output
```

Before a full run, add the following options for a one-update smoke test:

```bash
--max-samples 64 --max-steps 1
```
