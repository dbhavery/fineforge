"""Tests for fineforge.dataset module."""

import json
import tempfile
from pathlib import Path

import pytest

from fineforge.dataset import (
    filter_dataset,
    load_jsonl,
    save_jsonl,
    score_sample,
    split_dataset,
    validate_sample,
)


def _make_sample(
    system: str = "You are a helpful assistant.",
    user: str = "What is Python?",
    assistant: str = "Python is a high-level programming language known for its readability and versatility.",
) -> dict:
    """Create a valid chat sample."""
    messages = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": user})
    messages.append({"role": "assistant", "content": assistant})
    return {"messages": messages}


def _write_jsonl(path: Path, samples: list[dict]) -> None:
    """Write samples to a JSONL file."""
    with open(path, "w", encoding="utf-8") as f:
        for s in samples:
            f.write(json.dumps(s) + "\n")


class TestLoadJsonl:
    """Tests for load_jsonl."""

    def test_load_valid_file(self, tmp_path: Path) -> None:
        """Load a valid JSONL file with multiple samples."""
        samples = [_make_sample(), _make_sample(user="How are you?")]
        jsonl_path = tmp_path / "data.jsonl"
        _write_jsonl(jsonl_path, samples)

        loaded = load_jsonl(jsonl_path)
        assert len(loaded) == 2
        assert loaded[0]["messages"][1]["content"] == "What is Python?"

    def test_load_file_not_found(self) -> None:
        """Raise FileNotFoundError for missing files."""
        with pytest.raises(FileNotFoundError):
            load_jsonl("/nonexistent/path/data.jsonl")

    def test_load_empty_file(self, tmp_path: Path) -> None:
        """Raise ValueError for an empty file."""
        jsonl_path = tmp_path / "empty.jsonl"
        jsonl_path.write_text("")

        with pytest.raises(ValueError, match="No valid samples"):
            load_jsonl(jsonl_path)

    def test_load_with_blank_lines(self, tmp_path: Path) -> None:
        """Blank lines should be skipped."""
        jsonl_path = tmp_path / "blanks.jsonl"
        with open(jsonl_path, "w") as f:
            f.write(json.dumps(_make_sample()) + "\n")
            f.write("\n")
            f.write("\n")
            f.write(json.dumps(_make_sample(user="Second")) + "\n")

        loaded = load_jsonl(jsonl_path)
        assert len(loaded) == 2


class TestValidateSample:
    """Tests for validate_sample."""

    def test_valid_sample(self) -> None:
        """A proper sample should have no errors."""
        errors = validate_sample(_make_sample())
        assert errors == []

    def test_missing_messages_key(self) -> None:
        """Sample without 'messages' key should fail."""
        errors = validate_sample({"text": "hello"})
        assert any("Missing 'messages'" in e for e in errors)

    def test_not_a_dict(self) -> None:
        """Non-dict sample should fail."""
        errors = validate_sample("just a string")
        assert any("must be a dict" in e for e in errors)

    def test_missing_user_message(self) -> None:
        """Sample without user message should fail."""
        sample = {
            "messages": [
                {"role": "system", "content": "sys"},
                {"role": "assistant", "content": "response"},
            ]
        }
        errors = validate_sample(sample)
        assert any("user" in e for e in errors)

    def test_missing_assistant_message(self) -> None:
        """Sample without assistant message should fail."""
        sample = {
            "messages": [
                {"role": "user", "content": "question"},
            ]
        }
        errors = validate_sample(sample)
        assert any("assistant" in e for e in errors)

    def test_invalid_role(self) -> None:
        """Messages with unknown roles should fail."""
        sample = {
            "messages": [
                {"role": "user", "content": "hi"},
                {"role": "bot", "content": "hello"},
            ]
        }
        errors = validate_sample(sample)
        assert any("invalid role" in e for e in errors)

    def test_missing_content(self) -> None:
        """Message without 'content' should fail."""
        sample = {
            "messages": [
                {"role": "user"},
                {"role": "assistant", "content": "ok"},
            ]
        }
        errors = validate_sample(sample)
        assert any("missing 'content'" in e for e in errors)


class TestScoreSample:
    """Tests for score_sample."""

    def test_good_sample_scores_high(self) -> None:
        """A well-formed sample with good responses should score above 0.5."""
        sample = _make_sample(
            user="Explain the difference between lists and tuples in Python.",
            assistant=(
                "Lists and tuples are both sequence types in Python. "
                "Lists are mutable, meaning you can add, remove, or change elements. "
                "Tuples are immutable and cannot be modified after creation. "
                "Lists use square brackets, tuples use parentheses. "
                "Tuples are slightly faster and can be used as dictionary keys."
            ),
        )
        score = score_sample(sample)
        assert score > 0.5

    def test_empty_sample_scores_zero(self) -> None:
        """A sample with no messages should score 0."""
        score = score_sample({"messages": []})
        assert score == 0.0

    def test_short_response_scores_low(self) -> None:
        """A very short assistant response should score low."""
        sample = _make_sample(assistant="Yes")
        score = score_sample(sample)
        assert score < 0.5


class TestFilterDataset:
    """Tests for filter_dataset."""

    def test_removes_duplicates(self) -> None:
        """Exact duplicate samples should be removed."""
        sample = _make_sample()
        filtered, stats = filter_dataset([sample, sample, sample])
        assert stats.duplicates_removed == 2
        assert len(filtered) <= 1

    def test_removes_invalid(self) -> None:
        """Invalid samples should be excluded."""
        valid = _make_sample()
        invalid = {"not": "a valid sample"}
        filtered, stats = filter_dataset([valid, invalid])
        assert stats.total_raw == 2


class TestSplitDataset:
    """Tests for split_dataset."""

    def test_split_ratio(self) -> None:
        """Train/eval split should respect the ratio."""
        samples = [_make_sample(user=f"Question {i}") for i in range(20)]
        train, eval_set = split_dataset(samples, eval_ratio=0.2, seed=42)
        assert len(train) + len(eval_set) == 20
        assert len(eval_set) >= 3  # ~20% of 20

    def test_split_deterministic(self) -> None:
        """Same seed should produce the same split."""
        samples = [_make_sample(user=f"Q{i}") for i in range(10)]
        train1, eval1 = split_dataset(samples, eval_ratio=0.2, seed=99)
        train2, eval2 = split_dataset(samples, eval_ratio=0.2, seed=99)
        assert train1 == train2
        assert eval1 == eval2

    def test_zero_eval_ratio(self) -> None:
        """Eval ratio of 0 should put everything in train."""
        samples = [_make_sample()]
        train, eval_set = split_dataset(samples, eval_ratio=0.0)
        assert len(train) == 1
        assert len(eval_set) == 0


class TestSaveJsonl:
    """Tests for save_jsonl."""

    def test_round_trip(self, tmp_path: Path) -> None:
        """Save and reload should produce identical data."""
        original = [_make_sample(), _make_sample(user="Second question")]
        path = tmp_path / "output.jsonl"
        save_jsonl(original, path)
        reloaded = load_jsonl(path)
        assert len(reloaded) == len(original)
        assert reloaded[0]["messages"] == original[0]["messages"]
