# FineForge

**Local LoRA fine-tuning pipeline for consumer GPUs.**

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-green.svg)](https://www.python.org/downloads/)
[![PyPI](https://img.shields.io/badge/pypi-fineforge-orange.svg)](https://pypi.org/project/fineforge/)

FineForge is an end-to-end CLI tool that takes you from raw chat data to a fine-tuned model running in Ollama. It curates your dataset (validate, score, deduplicate, split), trains QLoRA adapters with 4-bit quantization on consumer GPUs, evaluates before/after quality, and exports the result to GGUF for local inference.

## Pipeline

```
  JSONL Data ──> Curate ──> Train (QLoRA) ──> Evaluate ──> Export (GGUF) ──> Ollama
     │             │            │                │              │              │
     │        validate       4-bit quant     base vs tuned    merge +       register
     │        score          LoRA adapters   side-by-side     quantize      & run
     │        dedup          SFTTrainer      score compare    Modelfile
     │        split
```

## Hardware Requirements

| Component | Minimum | Recommended |
|-----------|---------|-------------|
| GPU VRAM  | 8 GB    | 16-24 GB    |
| RAM       | 16 GB   | 32+ GB      |
| Disk      | 20 GB   | 50+ GB      |

Tested on NVIDIA RTX 3090 (24 GB VRAM) with 7B parameter models. QLoRA 4-bit quantization makes fine-tuning 7B models possible on 8 GB GPUs.

## Quick Start

### Install

```bash
# Core (dataset tools, CLI) -- no GPU required
pip install fineforge

# With training support (requires CUDA GPU)
pip install fineforge[train]

# Everything (training + GGUF export + dev tools)
pip install fineforge[all]
```

Or install from source:

```bash
git clone https://github.com/dbhavery/fineforge.git
cd fineforge
pip install -e ".[dev]"
```

### Full Workflow

```bash
# 1. Prepare your dataset
fineforge prepare my_chats.jsonl --output-dir ./data --min-quality 0.4

# 2. Create a training config
cat > config.yaml << 'EOF'
base_model: unsloth/Qwen2.5-7B
dataset_path: ./data/train.jsonl
output_dir: ./output
lora_r: 16
lora_alpha: 32
num_epochs: 3
learning_rate: 2e-4
batch_size: 4
max_seq_length: 2048
EOF

# 3. Train
fineforge train config.yaml

# 4. Evaluate
fineforge eval ./output/adapter --prompts test_prompts.yaml --base-model unsloth/Qwen2.5-7B

# 5. Export to GGUF and register in Ollama
fineforge export ./output/adapter \
  --base-model unsloth/Qwen2.5-7B \
  --quantization q4_k_m \
  --ollama-name my-tuned-model
```

## Dataset Format

FineForge uses the OpenAI chat format. Each line in your JSONL file is a conversation:

```json
{
  "messages": [
    {"role": "system", "content": "You are a helpful coding assistant."},
    {"role": "user", "content": "How do I reverse a list in Python?"},
    {"role": "assistant", "content": "Use the built-in `reversed()` function or slice notation: `my_list[::-1]`."}
  ]
}
```

Requirements:
- Each sample must have at least one `user` and one `assistant` message.
- `system` message is optional but recommended.
- Multi-turn conversations (multiple user/assistant pairs) are supported.

See `examples/chat_format.jsonl` for more examples.

## Training Config Reference

| Parameter | Default | Description |
|-----------|---------|-------------|
| `base_model` | `unsloth/Qwen2.5-7B` | HuggingFace model ID |
| `dataset_path` | `./data/train.jsonl` | Path to training JSONL |
| `output_dir` | `./output` | Output directory |
| `lora_r` | `16` | LoRA rank |
| `lora_alpha` | `32` | LoRA scaling factor |
| `lora_dropout` | `0.05` | LoRA dropout |
| `lora_target_modules` | `[q_proj, k_proj, v_proj, o_proj]` | Target linear layers |
| `learning_rate` | `2e-4` | Peak learning rate |
| `num_epochs` | `3` | Training epochs |
| `batch_size` | `4` | Per-device batch size |
| `gradient_accumulation_steps` | `4` | Steps before optimizer update |
| `max_seq_length` | `2048` | Max token sequence length |
| `warmup_steps` | `10` | LR warmup steps |
| `fp16` | `true` | Mixed-precision FP16 |
| `bf16` | `false` | Mixed-precision BF16 (Ampere+) |
| `seed` | `42` | Random seed |

## CLI Reference

### `fineforge prepare`

Curate and validate a chat dataset.

```
fineforge prepare <input.jsonl> [OPTIONS]

Options:
  -o, --output-dir    Output directory for train/eval JSONL  [default: ./data]
  -q, --min-quality   Minimum quality score (0.0-1.0)        [default: 0.3]
  -e, --eval-ratio    Fraction held out for evaluation       [default: 0.1]
  -s, --seed          Random seed for split                  [default: 42]
```

### `fineforge train`

Run QLoRA fine-tuning from a YAML config.

```
fineforge train <config.yaml>
```

### `fineforge eval`

Evaluate a fine-tuned model against its base model.

```
fineforge eval <adapter_path> [OPTIONS]

Options:
  -p, --prompts       Path to test prompts YAML              [required]
  -b, --base-model    Base model to compare against          [default: unsloth/Qwen2.5-7B]
  -o, --output        Save results to JSON file
```

### `fineforge export`

Export adapter to GGUF and optionally register in Ollama.

```
fineforge export <adapter_path> [OPTIONS]

Options:
  -b, --base-model       Base model ID                       [default: unsloth/Qwen2.5-7B]
  -f, --format           Export format                       [default: gguf]
  -q, --quantization     GGUF quantization type              [default: q4_k_m]
  --ollama-name          Register in Ollama with this name
  --system-prompt        Default system prompt for Modelfile
  -o, --output-dir       Output directory                    [default: ./output]
  --llama-cpp-path       Path to llama.cpp directory
```

## Architecture

```
fineforge/
  __init__.py        # Package version
  cli.py             # Click CLI entry point
  config.py          # TrainConfig dataclass + YAML I/O
  dataset.py         # Load, validate, score, filter, split JSONL datasets
  trainer.py         # QLoRA training wrapper (lazy torch/transformers imports)
  evaluator.py       # Before/after model comparison
  exporter.py        # GGUF export + Ollama registration
```

Key design decisions:
- **Lazy imports**: `torch`, `transformers`, `peft`, `trl` are only imported inside the functions that need them. The CLI, dataset tools, and tests all work without GPU dependencies installed.
- **Modular pipeline**: Each stage (prepare, train, eval, export) is independent. You can use just the dataset curator without ever training.
- **Config-driven**: Training is fully configured via YAML. No code changes needed to adjust hyperparameters.

## License

MIT License. See [LICENSE](LICENSE) for details.
