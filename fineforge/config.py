"""Training configuration for FineForge.

Defines the TrainConfig dataclass and YAML loading utilities.
"""

from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Optional

import yaml


@dataclass
class TrainConfig:
    """Configuration for a QLoRA fine-tuning run.

    Attributes:
        base_model: HuggingFace model ID or local path.
        dataset_path: Path to the training JSONL file.
        output_dir: Directory to save checkpoints and final adapter.
        lora_r: LoRA rank (dimensionality of the low-rank matrices).
        lora_alpha: LoRA scaling factor.
        lora_dropout: Dropout probability for LoRA layers.
        lora_target_modules: Which linear layers to apply LoRA to.
        learning_rate: Peak learning rate for the optimizer.
        num_epochs: Number of training epochs.
        batch_size: Per-device training batch size.
        gradient_accumulation_steps: Gradient accumulation before optimizer step.
        max_seq_length: Maximum sequence length for tokenization.
        warmup_steps: Linear warmup steps for the learning rate scheduler.
        fp16: Use mixed-precision (FP16) training.
        bf16: Use BF16 mixed-precision (preferred on Ampere+ GPUs).
        logging_steps: Log metrics every N steps.
        save_steps: Save a checkpoint every N steps.
        eval_steps: Run evaluation every N steps (0 = end of epoch only).
        seed: Random seed for reproducibility.
        chat_template: Chat template name (e.g. "chatml", "llama3").
        eval_dataset_path: Optional path to a separate eval JSONL file.
    """

    base_model: str = "unsloth/Qwen2.5-7B"
    dataset_path: str = "./data/train.jsonl"
    output_dir: str = "./output"

    # LoRA hyperparameters
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    lora_target_modules: list[str] = field(
        default_factory=lambda: ["q_proj", "k_proj", "v_proj", "o_proj"]
    )

    # Training hyperparameters
    learning_rate: float = 2e-4
    num_epochs: int = 3
    batch_size: int = 4
    gradient_accumulation_steps: int = 4
    max_seq_length: int = 2048
    warmup_steps: int = 10
    fp16: bool = True
    bf16: bool = False
    logging_steps: int = 10
    save_steps: int = 100
    eval_steps: int = 0
    seed: int = 42

    # Data handling
    chat_template: str = "chatml"
    eval_dataset_path: Optional[str] = None

    def validate(self) -> list[str]:
        """Validate configuration values.

        Returns:
            List of validation error messages. Empty if valid.
        """
        errors: list[str] = []

        if self.lora_r < 1:
            errors.append(f"lora_r must be >= 1, got {self.lora_r}")
        if self.lora_alpha < 1:
            errors.append(f"lora_alpha must be >= 1, got {self.lora_alpha}")
        if not 0.0 <= self.lora_dropout < 1.0:
            errors.append(
                f"lora_dropout must be in [0.0, 1.0), got {self.lora_dropout}"
            )
        if self.learning_rate <= 0:
            errors.append(
                f"learning_rate must be > 0, got {self.learning_rate}"
            )
        if self.num_epochs < 1:
            errors.append(f"num_epochs must be >= 1, got {self.num_epochs}")
        if self.batch_size < 1:
            errors.append(f"batch_size must be >= 1, got {self.batch_size}")
        if self.gradient_accumulation_steps < 1:
            errors.append(
                f"gradient_accumulation_steps must be >= 1, "
                f"got {self.gradient_accumulation_steps}"
            )
        if self.max_seq_length < 64:
            errors.append(
                f"max_seq_length must be >= 64, got {self.max_seq_length}"
            )
        if self.fp16 and self.bf16:
            errors.append("fp16 and bf16 cannot both be True")
        if not self.base_model:
            errors.append("base_model must not be empty")
        if not self.lora_target_modules:
            errors.append("lora_target_modules must not be empty")

        return errors

    def to_dict(self) -> dict:
        """Serialize config to a plain dictionary."""
        return asdict(self)

    def save_yaml(self, path: str | Path) -> None:
        """Save configuration to a YAML file.

        Args:
            path: Destination file path.
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            yaml.dump(self.to_dict(), f, default_flow_style=False, sort_keys=False)

    @classmethod
    def from_yaml(cls, path: str | Path) -> "TrainConfig":
        """Load configuration from a YAML file.

        Args:
            path: Path to the YAML configuration file.

        Returns:
            A TrainConfig instance populated from the YAML.

        Raises:
            FileNotFoundError: If the YAML file does not exist.
            yaml.YAMLError: If the file is not valid YAML.
            TypeError: If the YAML contains unknown fields.
        """
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Config file not found: {path}")

        with open(path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)

        if data is None:
            return cls()

        if not isinstance(data, dict):
            raise TypeError(f"Expected a YAML mapping, got {type(data).__name__}")

        # Filter to only known fields
        known_fields = {f.name for f in cls.__dataclass_fields__.values()}
        unknown = set(data.keys()) - known_fields
        if unknown:
            raise TypeError(
                f"Unknown config fields: {', '.join(sorted(unknown))}"
            )

        return cls(**data)
