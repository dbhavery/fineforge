"""Tests for fineforge.config module."""

import tempfile
from pathlib import Path

import pytest
import yaml

from fineforge.config import TrainConfig


class TestTrainConfigDefaults:
    """Tests for default values."""

    def test_default_values(self) -> None:
        """Default config should have sensible values."""
        config = TrainConfig()
        assert config.base_model == "unsloth/Qwen2.5-7B"
        assert config.lora_r == 16
        assert config.lora_alpha == 32
        assert config.num_epochs == 3
        assert config.fp16 is True
        assert config.bf16 is False

    def test_default_validates(self) -> None:
        """Default config should pass validation."""
        config = TrainConfig()
        errors = config.validate()
        assert errors == []


class TestTrainConfigValidation:
    """Tests for config validation."""

    def test_invalid_lora_r(self) -> None:
        """lora_r < 1 should fail validation."""
        config = TrainConfig(lora_r=0)
        errors = config.validate()
        assert any("lora_r" in e for e in errors)

    def test_invalid_learning_rate(self) -> None:
        """learning_rate <= 0 should fail validation."""
        config = TrainConfig(learning_rate=-0.001)
        errors = config.validate()
        assert any("learning_rate" in e for e in errors)

    def test_fp16_bf16_conflict(self) -> None:
        """Both fp16 and bf16 True should fail."""
        config = TrainConfig(fp16=True, bf16=True)
        errors = config.validate()
        assert any("fp16" in e and "bf16" in e for e in errors)

    def test_empty_model_name(self) -> None:
        """Empty base_model should fail validation."""
        config = TrainConfig(base_model="")
        errors = config.validate()
        assert any("base_model" in e for e in errors)

    def test_invalid_dropout(self) -> None:
        """Dropout outside [0, 1) should fail."""
        config = TrainConfig(lora_dropout=1.5)
        errors = config.validate()
        assert any("lora_dropout" in e for e in errors)

    def test_max_seq_length_too_small(self) -> None:
        """max_seq_length < 64 should fail."""
        config = TrainConfig(max_seq_length=32)
        errors = config.validate()
        assert any("max_seq_length" in e for e in errors)


class TestTrainConfigYaml:
    """Tests for YAML serialization/deserialization."""

    def test_save_and_load(self, tmp_path: Path) -> None:
        """Config should round-trip through YAML."""
        config = TrainConfig(
            base_model="test/model",
            lora_r=32,
            num_epochs=5,
            learning_rate=1e-4,
        )
        yaml_path = tmp_path / "config.yaml"
        config.save_yaml(yaml_path)

        loaded = TrainConfig.from_yaml(yaml_path)
        assert loaded.base_model == "test/model"
        assert loaded.lora_r == 32
        assert loaded.num_epochs == 5
        assert loaded.learning_rate == 1e-4

    def test_load_nonexistent_file(self) -> None:
        """Loading a missing file should raise FileNotFoundError."""
        with pytest.raises(FileNotFoundError):
            TrainConfig.from_yaml("/nonexistent/config.yaml")

    def test_load_unknown_fields(self, tmp_path: Path) -> None:
        """Unknown fields in YAML should raise TypeError."""
        yaml_path = tmp_path / "bad.yaml"
        with open(yaml_path, "w") as f:
            yaml.dump({"base_model": "test", "unknown_field": True}, f)

        with pytest.raises(TypeError, match="Unknown config fields"):
            TrainConfig.from_yaml(yaml_path)

    def test_load_empty_yaml(self, tmp_path: Path) -> None:
        """An empty YAML file should return default config."""
        yaml_path = tmp_path / "empty.yaml"
        yaml_path.write_text("")

        config = TrainConfig.from_yaml(yaml_path)
        assert config.base_model == "unsloth/Qwen2.5-7B"

    def test_to_dict(self) -> None:
        """to_dict should return a plain dictionary."""
        config = TrainConfig()
        d = config.to_dict()
        assert isinstance(d, dict)
        assert d["base_model"] == "unsloth/Qwen2.5-7B"
        assert d["lora_r"] == 16
