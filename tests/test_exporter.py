"""Tests for fineforge.exporter module."""

import pytest

from fineforge.exporter import generate_modelfile


class TestGenerateModelfile:
    """Tests for Ollama Modelfile generation."""

    def test_basic_modelfile(self) -> None:
        """Generate a basic Modelfile with just a model path."""
        content = generate_modelfile("/path/to/model.gguf")
        assert "FROM /path/to/model.gguf" in content
        assert "PARAMETER temperature 0.7" in content
        assert "PARAMETER top_p 0.9" in content

    def test_modelfile_with_system_prompt(self) -> None:
        """Modelfile should include the system prompt."""
        content = generate_modelfile(
            "/model.gguf",
            system_prompt="You are a coding assistant.",
        )
        assert "SYSTEM" in content
        assert "You are a coding assistant." in content

    def test_modelfile_without_system_prompt(self) -> None:
        """Modelfile without system prompt should not have SYSTEM directive."""
        content = generate_modelfile("/model.gguf", system_prompt="")
        assert "SYSTEM" not in content

    def test_modelfile_custom_temperature(self) -> None:
        """Custom temperature should be reflected in the Modelfile."""
        content = generate_modelfile("/model.gguf", temperature=0.3)
        assert "PARAMETER temperature 0.3" in content

    def test_modelfile_with_stop_tokens(self) -> None:
        """Stop tokens should be included as PARAMETER directives."""
        content = generate_modelfile(
            "/model.gguf",
            stop_tokens=["<|end|>", "<|user|>"],
        )
        assert 'PARAMETER stop "<|end|>"' in content
        assert 'PARAMETER stop "<|user|>"' in content

    def test_modelfile_no_stop_tokens(self) -> None:
        """Without stop tokens, no stop parameters should appear."""
        content = generate_modelfile("/model.gguf")
        assert "PARAMETER stop" not in content
