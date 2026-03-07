"""Dataset curator for FineForge.

Loads, validates, scores, filters, and splits JSONL chat datasets
in OpenAI format (messages array with role/content).
"""

import hashlib
import json
import random
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from rich.console import Console
from rich.table import Table

console = Console()


@dataclass
class DatasetStats:
    """Statistics for a curated dataset.

    Attributes:
        total_raw: Number of samples before filtering.
        total_filtered: Number of samples after filtering.
        duplicates_removed: Number of duplicate samples removed.
        low_quality_removed: Number of low-quality samples removed.
        avg_turns: Average number of turns per conversation.
        avg_user_length: Average character length of user messages.
        avg_assistant_length: Average character length of assistant messages.
        role_distribution: Count of messages per role.
        quality_scores: Distribution of quality score buckets.
    """

    total_raw: int = 0
    total_filtered: int = 0
    duplicates_removed: int = 0
    low_quality_removed: int = 0
    avg_turns: float = 0.0
    avg_user_length: float = 0.0
    avg_assistant_length: float = 0.0
    role_distribution: dict[str, int] = field(default_factory=dict)
    quality_scores: dict[str, int] = field(default_factory=dict)


def load_jsonl(path: str | Path) -> list[dict[str, Any]]:
    """Load a JSONL file containing chat samples.

    Each line must be a JSON object with a "messages" key containing
    a list of message objects with "role" and "content" fields.

    Args:
        path: Path to the JSONL file.

    Returns:
        List of parsed samples (each a dict with "messages" key).

    Raises:
        FileNotFoundError: If the file does not exist.
        ValueError: If the file is empty or contains no valid samples.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Dataset file not found: {path}")

    samples: list[dict[str, Any]] = []
    errors: list[str] = []

    with open(path, "r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
                samples.append(obj)
            except json.JSONDecodeError as e:
                errors.append(f"Line {line_num}: {e}")

    if errors:
        console.print(
            f"[yellow]Warning: {len(errors)} lines had JSON parse errors[/yellow]"
        )
        for err in errors[:5]:
            console.print(f"  [dim]{err}[/dim]")

    if not samples:
        raise ValueError(f"No valid samples found in {path}")

    return samples


def validate_sample(sample: dict[str, Any]) -> list[str]:
    """Validate a single chat sample against the expected format.

    Expected format (OpenAI chat):
        {"messages": [
            {"role": "system", "content": "..."},
            {"role": "user", "content": "..."},
            {"role": "assistant", "content": "..."}
        ]}

    Args:
        sample: A parsed JSON object to validate.

    Returns:
        List of validation error strings. Empty if valid.
    """
    errors: list[str] = []

    if not isinstance(sample, dict):
        return [f"Sample must be a dict, got {type(sample).__name__}"]

    if "messages" not in sample:
        return ["Missing 'messages' key"]

    messages = sample["messages"]
    if not isinstance(messages, list):
        return [f"'messages' must be a list, got {type(messages).__name__}"]

    if len(messages) < 2:
        errors.append(
            f"Conversation must have at least 2 messages, got {len(messages)}"
        )

    valid_roles = {"system", "user", "assistant"}
    has_user = False
    has_assistant = False

    for i, msg in enumerate(messages):
        if not isinstance(msg, dict):
            errors.append(f"Message {i} must be a dict, got {type(msg).__name__}")
            continue

        if "role" not in msg:
            errors.append(f"Message {i} missing 'role'")
        elif msg["role"] not in valid_roles:
            errors.append(
                f"Message {i} has invalid role '{msg['role']}', "
                f"expected one of {valid_roles}"
            )
        else:
            if msg["role"] == "user":
                has_user = True
            elif msg["role"] == "assistant":
                has_assistant = True

        if "content" not in msg:
            errors.append(f"Message {i} missing 'content'")
        elif not isinstance(msg["content"], str):
            errors.append(
                f"Message {i} 'content' must be a string, "
                f"got {type(msg['content']).__name__}"
            )

    if not has_user:
        errors.append("Conversation must contain at least one 'user' message")
    if not has_assistant:
        errors.append("Conversation must contain at least one 'assistant' message")

    return errors


def score_sample(sample: dict[str, Any]) -> float:
    """Score a sample's quality on a 0.0–1.0 scale.

    Heuristics used:
    - Assistant response length (longer = more informative, up to a point)
    - Multi-turn conversations score higher
    - Presence of a system prompt adds a small bonus
    - User message length (too short = low effort)
    - Diversity of vocabulary in assistant responses

    Args:
        sample: A validated chat sample.

    Returns:
        Quality score between 0.0 and 1.0.
    """
    messages = sample.get("messages", [])
    if not messages:
        return 0.0

    score = 0.0

    # Extract messages by role
    user_msgs = [m["content"] for m in messages if m.get("role") == "user"]
    assistant_msgs = [m["content"] for m in messages if m.get("role") == "assistant"]
    has_system = any(m.get("role") == "system" for m in messages)

    if not user_msgs or not assistant_msgs:
        return 0.0

    # 1. Assistant response length (0–0.3)
    avg_assistant_len = sum(len(m) for m in assistant_msgs) / len(assistant_msgs)
    if avg_assistant_len < 20:
        score += 0.0
    elif avg_assistant_len < 100:
        score += 0.1
    elif avg_assistant_len < 500:
        score += 0.2
    else:
        score += 0.3

    # 2. Multi-turn bonus (0–0.2)
    total_turns = len(user_msgs) + len(assistant_msgs)
    if total_turns >= 6:
        score += 0.2
    elif total_turns >= 4:
        score += 0.15
    elif total_turns >= 2:
        score += 0.1

    # 3. System prompt bonus (0–0.1)
    if has_system:
        score += 0.1

    # 4. User message quality (0–0.2)
    avg_user_len = sum(len(m) for m in user_msgs) / len(user_msgs)
    if avg_user_len < 5:
        score += 0.0
    elif avg_user_len < 20:
        score += 0.05
    elif avg_user_len < 100:
        score += 0.1
    else:
        score += 0.2

    # 5. Vocabulary diversity in assistant responses (0–0.2)
    all_assistant_text = " ".join(assistant_msgs).lower()
    words = all_assistant_text.split()
    if words:
        unique_ratio = len(set(words)) / len(words)
        score += min(0.2, unique_ratio * 0.3)

    return min(1.0, round(score, 3))


def _content_hash(sample: dict[str, Any]) -> str:
    """Generate a content hash for deduplication."""
    messages = sample.get("messages", [])
    content_str = json.dumps(
        [(m.get("role", ""), m.get("content", "")) for m in messages],
        sort_keys=True,
    )
    return hashlib.sha256(content_str.encode("utf-8")).hexdigest()


def filter_dataset(
    samples: list[dict[str, Any]],
    min_quality_score: float = 0.3,
    min_assistant_length: int = 10,
) -> tuple[list[dict[str, Any]], DatasetStats]:
    """Filter and deduplicate a dataset.

    Steps:
    1. Validate each sample's format
    2. Remove duplicates (by content hash)
    3. Score quality
    4. Remove samples below the minimum quality threshold
    5. Remove samples with very short assistant responses

    Args:
        samples: Raw loaded samples.
        min_quality_score: Minimum quality score to keep (0.0–1.0).
        min_assistant_length: Minimum average assistant response length.

    Returns:
        Tuple of (filtered_samples, statistics).
    """
    stats = DatasetStats(total_raw=len(samples))
    role_counter: Counter[str] = Counter()
    quality_buckets: Counter[str] = Counter()

    # Step 1: Validate
    valid_samples: list[dict[str, Any]] = []
    for sample in samples:
        errors = validate_sample(sample)
        if not errors:
            valid_samples.append(sample)

    # Step 2: Deduplicate
    seen_hashes: set[str] = set()
    deduped: list[dict[str, Any]] = []
    for sample in valid_samples:
        h = _content_hash(sample)
        if h not in seen_hashes:
            seen_hashes.add(h)
            deduped.append(sample)
    stats.duplicates_removed = len(valid_samples) - len(deduped)

    # Step 3 & 4: Score and filter
    scored: list[tuple[dict[str, Any], float]] = []
    for sample in deduped:
        quality = score_sample(sample)
        scored.append((sample, quality))

    filtered: list[dict[str, Any]] = []
    low_quality_count = 0
    for sample, quality in scored:
        # Bucket the quality score
        if quality >= 0.8:
            quality_buckets["excellent (0.8-1.0)"] += 1
        elif quality >= 0.6:
            quality_buckets["good (0.6-0.8)"] += 1
        elif quality >= 0.4:
            quality_buckets["fair (0.4-0.6)"] += 1
        else:
            quality_buckets["poor (0.0-0.4)"] += 1

        if quality < min_quality_score:
            low_quality_count += 1
            continue

        # Check assistant message length
        assistant_msgs = [
            m["content"]
            for m in sample["messages"]
            if m.get("role") == "assistant"
        ]
        avg_len = (
            sum(len(m) for m in assistant_msgs) / len(assistant_msgs)
            if assistant_msgs
            else 0
        )
        if avg_len < min_assistant_length:
            low_quality_count += 1
            continue

        filtered.append(sample)

        # Count roles
        for msg in sample["messages"]:
            role_counter[msg.get("role", "unknown")] += 1

    stats.low_quality_removed = low_quality_count
    stats.total_filtered = len(filtered)
    stats.role_distribution = dict(role_counter)
    stats.quality_scores = dict(quality_buckets)

    # Compute averages
    if filtered:
        all_user_lens: list[int] = []
        all_assistant_lens: list[int] = []
        all_turn_counts: list[int] = []

        for sample in filtered:
            messages = sample["messages"]
            all_turn_counts.append(len(messages))
            for msg in messages:
                if msg["role"] == "user":
                    all_user_lens.append(len(msg["content"]))
                elif msg["role"] == "assistant":
                    all_assistant_lens.append(len(msg["content"]))

        stats.avg_turns = round(sum(all_turn_counts) / len(all_turn_counts), 1)
        if all_user_lens:
            stats.avg_user_length = round(
                sum(all_user_lens) / len(all_user_lens), 1
            )
        if all_assistant_lens:
            stats.avg_assistant_length = round(
                sum(all_assistant_lens) / len(all_assistant_lens), 1
            )

    return filtered, stats


def split_dataset(
    samples: list[dict[str, Any]],
    eval_ratio: float = 0.1,
    seed: int = 42,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Split a dataset into train and eval sets.

    Args:
        samples: Filtered samples to split.
        eval_ratio: Fraction of samples to use for evaluation (0.0–1.0).
        seed: Random seed for reproducibility.

    Returns:
        Tuple of (train_samples, eval_samples).
    """
    if eval_ratio <= 0.0 or eval_ratio >= 1.0:
        return samples, []

    rng = random.Random(seed)
    shuffled = list(samples)
    rng.shuffle(shuffled)

    split_idx = max(1, int(len(shuffled) * (1 - eval_ratio)))
    return shuffled[:split_idx], shuffled[split_idx:]


def save_jsonl(samples: list[dict[str, Any]], path: str | Path) -> None:
    """Save samples to a JSONL file.

    Args:
        samples: Chat samples to save.
        path: Destination file path.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for sample in samples:
            f.write(json.dumps(sample, ensure_ascii=False) + "\n")


def print_stats(stats: DatasetStats) -> None:
    """Pretty-print dataset statistics using Rich.

    Args:
        stats: The DatasetStats to display.
    """
    table = Table(title="Dataset Statistics", show_header=False)
    table.add_column("Metric", style="bold cyan")
    table.add_column("Value", style="white")

    table.add_row("Raw samples", str(stats.total_raw))
    table.add_row("After filtering", str(stats.total_filtered))
    table.add_row("Duplicates removed", str(stats.duplicates_removed))
    table.add_row("Low quality removed", str(stats.low_quality_removed))
    table.add_row("Avg turns/conversation", str(stats.avg_turns))
    table.add_row("Avg user msg length", f"{stats.avg_user_length} chars")
    table.add_row("Avg assistant msg length", f"{stats.avg_assistant_length} chars")

    console.print(table)

    if stats.role_distribution:
        role_table = Table(title="Role Distribution")
        role_table.add_column("Role", style="cyan")
        role_table.add_column("Count", style="white", justify="right")
        for role, count in sorted(stats.role_distribution.items()):
            role_table.add_row(role, str(count))
        console.print(role_table)

    if stats.quality_scores:
        quality_table = Table(title="Quality Score Distribution")
        quality_table.add_column("Bucket", style="cyan")
        quality_table.add_column("Count", style="white", justify="right")
        for bucket, count in sorted(stats.quality_scores.items()):
            quality_table.add_row(bucket, str(count))
        console.print(quality_table)
