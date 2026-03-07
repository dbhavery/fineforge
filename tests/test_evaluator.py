"""Tests for fineforge.evaluator module."""

from pathlib import Path

import pytest
import yaml

from fineforge.evaluator import (
    EvalPrompt,
    load_prompts,
    score_response,
)


class TestLoadPrompts:
    """Tests for loading prompts from YAML."""

    def test_load_valid_prompts(self, tmp_path: Path) -> None:
        """Load a valid prompts YAML file."""
        prompts_data = {
            "prompts": [
                {
                    "name": "greeting",
                    "system": "You are a helpful assistant.",
                    "user": "Hello!",
                    "expected_keywords": ["hello", "help"],
                    "max_tokens": 64,
                },
                {
                    "name": "code",
                    "user": "Write a Python hello world.",
                    "expected_keywords": ["print", "hello"],
                },
            ]
        }
        yaml_path = tmp_path / "prompts.yaml"
        with open(yaml_path, "w") as f:
            yaml.dump(prompts_data, f)

        prompts = load_prompts(yaml_path)
        assert len(prompts) == 2
        assert prompts[0].name == "greeting"
        assert prompts[0].system == "You are a helpful assistant."
        assert prompts[0].max_tokens == 64
        assert prompts[1].system == ""
        assert prompts[1].max_tokens == 256  # default

    def test_load_nonexistent_file(self) -> None:
        """Raise FileNotFoundError for missing file."""
        with pytest.raises(FileNotFoundError):
            load_prompts("/nonexistent/prompts.yaml")

    def test_load_invalid_format(self, tmp_path: Path) -> None:
        """Raise ValueError for invalid YAML structure."""
        yaml_path = tmp_path / "bad.yaml"
        with open(yaml_path, "w") as f:
            yaml.dump({"not_prompts": []}, f)

        with pytest.raises(ValueError, match="prompts"):
            load_prompts(yaml_path)

    def test_load_missing_required_fields(self, tmp_path: Path) -> None:
        """Raise ValueError when prompt missing name or user."""
        yaml_path = tmp_path / "incomplete.yaml"
        with open(yaml_path, "w") as f:
            yaml.dump({"prompts": [{"name": "test"}]}, f)

        with pytest.raises(ValueError, match="'user'"):
            load_prompts(yaml_path)


class TestScoreResponse:
    """Tests for score_response."""

    def test_empty_response(self) -> None:
        """Empty response should score 0."""
        assert score_response("", []) == 0.0
        assert score_response("  ", []) == 0.0

    def test_good_response_with_keywords(self) -> None:
        """Response containing expected keywords should score well."""
        response = (
            "Python is a programming language that is widely used for "
            "web development, data science, and automation. It has a "
            "clean syntax and a large standard library."
        )
        score = score_response(response, ["python", "programming", "syntax"])
        assert score > 0.5

    def test_response_missing_keywords(self) -> None:
        """Response missing all keywords should score lower."""
        response = "The weather is sunny today."
        score_with = score_response(response, ["python", "code", "function"])
        score_without = score_response(response, [])
        # With missing keywords, score should be lower
        assert score_with < score_without or score_with <= 0.5

    def test_short_response_scores_lower(self) -> None:
        """Very short response should score lower than a detailed one."""
        short_score = score_response("Yes.", [])
        long_score = score_response(
            "Yes, that is correct. Here is a detailed explanation of why "
            "this approach works and how you can implement it effectively "
            "in your project with proper error handling.",
            [],
        )
        assert long_score > short_score

    def test_no_keywords_neutral(self) -> None:
        """With no expected keywords, keyword score should be neutral (0.5)."""
        response = "This is a reasonable response with decent length."
        score = score_response(response, [])
        assert score > 0.0
