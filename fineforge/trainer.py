"""QLoRA training wrapper for FineForge.

All heavy ML imports (torch, transformers, peft, trl, bitsandbytes)
are lazy-loaded inside functions so this module can be imported without
GPU dependencies installed.
"""

import json
import time
from pathlib import Path
from typing import Any, Optional

from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn

from fineforge.config import TrainConfig

console = Console()


class TrainingError(Exception):
    """Raised when training fails."""


class Trainer:
    """QLoRA fine-tuning trainer.

    Wraps the HuggingFace ecosystem (transformers + peft + trl) to
    provide a simple interface for LoRA fine-tuning on consumer GPUs.

    All heavy dependencies are imported lazily at training time, not
    at import time. This allows the rest of the project (CLI, dataset
    tools, tests) to work without torch installed.

    Args:
        config: Training configuration.
    """

    def __init__(self, config: TrainConfig) -> None:
        self.config = config
        self._model: Any = None
        self._tokenizer: Any = None

    def _import_dependencies(self) -> dict[str, Any]:
        """Lazily import all heavy ML dependencies.

        Returns:
            Dict mapping module names to imported modules.

        Raises:
            ImportError: If required packages are not installed.
        """
        missing: list[str] = []
        modules: dict[str, Any] = {}

        try:
            import torch
            modules["torch"] = torch
        except ImportError:
            missing.append("torch")

        try:
            import transformers
            modules["transformers"] = transformers
        except ImportError:
            missing.append("transformers")

        try:
            import peft
            modules["peft"] = peft
        except ImportError:
            missing.append("peft")

        try:
            import trl
            modules["trl"] = trl
        except ImportError:
            missing.append("trl")

        try:
            import datasets
            modules["datasets"] = datasets
        except ImportError:
            missing.append("datasets")

        if missing:
            raise ImportError(
                f"Missing required packages for training: {', '.join(missing)}. "
                f"Install them with: pip install fineforge[train]"
            )

        return modules

    def _check_gpu(self) -> dict[str, Any]:
        """Check GPU availability and return device info.

        Returns:
            Dict with gpu_available, device_name, vram_gb, cuda_version.
        """
        try:
            import torch
        except ImportError:
            return {
                "gpu_available": False,
                "device_name": "N/A (torch not installed)",
                "vram_gb": 0,
                "cuda_version": "N/A",
            }

        if torch.cuda.is_available():
            device = torch.cuda.get_device_properties(0)
            return {
                "gpu_available": True,
                "device_name": device.name,
                "vram_gb": round(device.total_mem / (1024**3), 1),
                "cuda_version": torch.version.cuda or "N/A",
            }
        return {
            "gpu_available": False,
            "device_name": "CPU only",
            "vram_gb": 0,
            "cuda_version": "N/A",
        }

    def train(self) -> Path:
        """Run the full QLoRA fine-tuning pipeline.

        Steps:
        1. Validate config
        2. Check GPU
        3. Load base model with 4-bit quantization
        4. Apply LoRA adapters
        5. Load and tokenize dataset
        6. Train with SFTTrainer
        7. Save the final adapter

        Returns:
            Path to the saved adapter directory.

        Raises:
            TrainingError: If training fails at any step.
            ImportError: If required packages are missing.
        """
        # Validate config
        errors = self.config.validate()
        if errors:
            raise TrainingError(
                f"Invalid config: {'; '.join(errors)}"
            )

        # Import heavy deps
        deps = self._import_dependencies()
        torch = deps["torch"]
        transformers = deps["transformers"]
        peft = deps["peft"]
        trl = deps["trl"]
        datasets_lib = deps["datasets"]

        # Check GPU
        gpu_info = self._check_gpu()
        console.print(f"[bold]GPU:[/bold] {gpu_info['device_name']}")
        if gpu_info["gpu_available"]:
            console.print(f"[bold]VRAM:[/bold] {gpu_info['vram_gb']} GB")
            console.print(f"[bold]CUDA:[/bold] {gpu_info['cuda_version']}")
        else:
            console.print(
                "[yellow]Warning: No GPU detected. Training on CPU will be "
                "extremely slow.[/yellow]"
            )

        output_dir = Path(self.config.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        console.print(f"\n[bold cyan]Loading model:[/bold cyan] {self.config.base_model}")

        try:
            # 4-bit quantization config
            bnb_config = transformers.BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.bfloat16 if self.config.bf16 else torch.float16,
                bnb_4bit_use_double_quant=True,
            )

            # Load base model
            model = transformers.AutoModelForCausalLM.from_pretrained(
                self.config.base_model,
                quantization_config=bnb_config,
                device_map="auto",
                trust_remote_code=self.config.trust_remote_code,
            )

            tokenizer = transformers.AutoTokenizer.from_pretrained(
                self.config.base_model,
                trust_remote_code=self.config.trust_remote_code,
            )
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token

            console.print("[green]Model loaded successfully.[/green]")

            # Apply LoRA
            console.print("[bold cyan]Applying LoRA adapters...[/bold cyan]")
            lora_config = peft.LoraConfig(
                r=self.config.lora_r,
                lora_alpha=self.config.lora_alpha,
                lora_dropout=self.config.lora_dropout,
                target_modules=self.config.lora_target_modules,
                bias="none",
                task_type="CAUSAL_LM",
            )
            model = peft.get_peft_model(model, lora_config)

            trainable, total = model.get_nb_trainable_parameters()
            pct = round(100 * trainable / total, 2) if total > 0 else 0
            console.print(
                f"[green]LoRA applied:[/green] {trainable:,} / {total:,} "
                f"parameters trainable ({pct}%)"
            )

            # Load dataset
            console.print(
                f"[bold cyan]Loading dataset:[/bold cyan] {self.config.dataset_path}"
            )
            dataset_path = Path(self.config.dataset_path)
            if not dataset_path.exists():
                raise TrainingError(
                    f"Dataset not found: {self.config.dataset_path}"
                )

            dataset = datasets_lib.load_dataset(
                "json", data_files=str(dataset_path), split="train"
            )
            console.print(f"[green]Dataset loaded:[/green] {len(dataset)} samples")

            # Format for SFTTrainer
            def format_chat(example: dict) -> dict:
                """Apply chat template to messages."""
                text = tokenizer.apply_chat_template(
                    example["messages"],
                    tokenize=False,
                    add_generation_prompt=False,
                )
                return {"text": text}

            dataset = dataset.map(format_chat)

            # Training arguments
            training_args = transformers.TrainingArguments(
                output_dir=str(output_dir / "checkpoints"),
                num_train_epochs=self.config.num_epochs,
                per_device_train_batch_size=self.config.batch_size,
                gradient_accumulation_steps=self.config.gradient_accumulation_steps,
                learning_rate=self.config.learning_rate,
                warmup_steps=self.config.warmup_steps,
                fp16=self.config.fp16,
                bf16=self.config.bf16,
                logging_steps=self.config.logging_steps,
                save_steps=self.config.save_steps,
                save_total_limit=3,
                seed=self.config.seed,
                report_to="none",
                optim="paged_adamw_8bit",
                max_grad_norm=0.3,
                lr_scheduler_type="cosine",
            )

            # Create SFTTrainer
            trainer = trl.SFTTrainer(
                model=model,
                tokenizer=tokenizer,
                args=training_args,
                train_dataset=dataset,
                max_seq_length=self.config.max_seq_length,
                dataset_text_field="text",
                packing=False,
            )

            # Train
            console.print("\n[bold green]Starting training...[/bold green]")
            start_time = time.time()
            train_result = trainer.train()
            elapsed = time.time() - start_time

            console.print(
                f"\n[bold green]Training complete![/bold green] "
                f"({elapsed:.1f}s)"
            )
            console.print(
                f"  Final loss: {train_result.training_loss:.4f}"
            )

            # Save adapter
            adapter_dir = output_dir / "adapter"
            model.save_pretrained(str(adapter_dir))
            tokenizer.save_pretrained(str(adapter_dir))

            # Save training metadata
            metadata = {
                "base_model": self.config.base_model,
                "training_loss": train_result.training_loss,
                "training_time_seconds": round(elapsed, 1),
                "num_samples": len(dataset),
                "config": self.config.to_dict(),
            }
            with open(output_dir / "training_metadata.json", "w") as f:
                json.dump(metadata, f, indent=2)

            console.print(f"[bold green]Adapter saved to:[/bold green] {adapter_dir}")
            return adapter_dir

        except ImportError as e:
            raise TrainingError(f"Missing dependency: {e}") from e
        except Exception as e:
            raise TrainingError(f"Training failed: {e}") from e
