# FineForge

QLoRA fine-tuning CLI for consumer GPUs -- takes you from raw chat data to a fine-tuned model running in Ollama, entirely on local hardware.

## Why I Built This

Cloud fine-tuning costs $2-5/hr on GPU instances and requires uploading your data to third-party servers. With QLoRA 4-bit quantization, a 7B parameter model fits in 8GB VRAM and trains on a single consumer GPU. I wanted a single CLI that handles the full pipeline: curate dataset, train adapter, evaluate against base model, export to GGUF, register in Ollama. No notebooks, no cloud, no manual steps between stages.

## What It Does

- **Full pipeline in 5 commands** -- `prepare` (validate/score/dedup/split), `train` (QLoRA), `eval` (base vs. tuned comparison), `export` (GGUF + Ollama registration)
- **Trains 7B models on 8GB VRAM** -- QLoRA 4-bit quantization reduces memory footprint by 75% vs. full fine-tuning; $0 cloud compute cost
- **YAML training configs** -- all hyperparameters in one file, no code changes between training runs
- **Dataset curation** -- validates chat format, scores quality, deduplicates, splits train/eval automatically
- **Before/after evaluation** -- side-by-side comparison of base model vs. fine-tuned model on test prompts

## Key Technical Decisions

- **QLoRA over full fine-tuning** -- makes 7B-13B models trainable on consumer GPUs (RTX 3060 and up). Full fine-tuning of a 7B model requires 56+ GB VRAM; QLoRA brings it under 8GB.
- **Click CLI over notebook workflow** -- reproducible training runs, scriptable in CI/CD. Notebooks encourage one-off experiments; CLI enforces repeatable configuration.
- **Lazy imports for GPU dependencies** -- `torch`, `transformers`, `peft`, `trl` are only imported inside functions that need them. The CLI, dataset tools, and tests all run without a GPU installed. `pip install fineforge` works on any machine; `pip install fineforge[train]` adds GPU dependencies.
- **OpenAI chat format** -- standard JSONL with `messages` array. Compatible with existing chat export tools, no custom format to learn.

## Quick Start

```bash
# Core (dataset tools, no GPU required)
pip install fineforge

# With training support (requires CUDA GPU)
pip install fineforge[train]

# Full pipeline
fineforge prepare my_chats.jsonl --output-dir ./data --min-quality 0.4
fineforge train config.yaml
fineforge eval ./output/adapter --prompts test_prompts.yaml
fineforge export ./output/adapter --quantization q4_k_m --ollama-name my-model
```

## Lessons Learned

**VRAM management is the critical constraint, and one size does not fit all.** Batch size and gradient accumulation need careful tuning per model architecture. A config that trains Qwen2.5-7B smoothly will OOM on Llama-3-8B because of different attention head counts and hidden dimensions. The default config (`batch_size: 4, gradient_accumulation: 4`) is conservative enough for most 7B models on 24GB, but 8GB cards need `batch_size: 1` with higher gradient accumulation to compensate. I learned this by bricking training runs, not from documentation.

## Tests

```bash
pip install -e ".[dev]"
pytest tests/ -v
```

---

MIT License. See [LICENSE](LICENSE).
