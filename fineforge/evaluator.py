"""Before/after evaluation for FineForge.

Runs a set of test prompts through both the base model and the
fine-tuned model, then compares and scores the results.

All heavy ML imports are lazy-loaded inside methods.
"""

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

import yaml
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

console = Console()


@dataclass
class EvalPrompt:
    """A single evaluation prompt.

    Attributes:
        name: Short identifier for this test case.
        system: Optional system prompt.
        user: The user message to send.
        expected_keywords: Keywords expected in a good response.
        max_tokens: Maximum tokens to generate.
    """

    name: str
    user: str
    system: str = ""
    expected_keywords: list[str] = field(default_factory=list)
    max_tokens: int = 256


@dataclass
class EvalResult:
    """Result of evaluating one prompt.

    Attributes:
        prompt_name: Name of the test case.
        user_input: The user message.
        base_response: Response from the base model.
        tuned_response: Response from the fine-tuned model.
        base_score: Quality score for the base response (0.0-1.0).
        tuned_score: Quality score for the tuned response (0.0-1.0).
        improvement: Score difference (tuned - base).
    """

    prompt_name: str
    user_input: str
    base_response: str
    tuned_response: str
    base_score: float
    tuned_score: float
    improvement: float


def load_prompts(path: str | Path) -> list[EvalPrompt]:
    """Load evaluation prompts from a YAML file.

    Expected format:
        prompts:
          - name: "greeting"
            system: "You are a helpful assistant."
            user: "Hello, how are you?"
            expected_keywords: ["hello", "help"]
            max_tokens: 128

    Args:
        path: Path to the YAML prompts file.

    Returns:
        List of EvalPrompt instances.

    Raises:
        FileNotFoundError: If the file does not exist.
        ValueError: If the file format is invalid.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Prompts file not found: {path}")

    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)

    if not isinstance(data, dict) or "prompts" not in data:
        raise ValueError(
            "Prompts YAML must have a top-level 'prompts' key containing a list"
        )

    prompts_data = data["prompts"]
    if not isinstance(prompts_data, list):
        raise ValueError("'prompts' must be a list")

    prompts: list[EvalPrompt] = []
    for i, item in enumerate(prompts_data):
        if not isinstance(item, dict):
            raise ValueError(f"Prompt {i} must be a dict")
        if "name" not in item or "user" not in item:
            raise ValueError(
                f"Prompt {i} must have 'name' and 'user' keys"
            )
        prompts.append(
            EvalPrompt(
                name=item["name"],
                user=item["user"],
                system=item.get("system", ""),
                expected_keywords=item.get("expected_keywords", []),
                max_tokens=item.get("max_tokens", 256),
            )
        )

    return prompts


def score_response(
    response: str,
    expected_keywords: list[str],
) -> float:
    """Score a model response using simple heuristics.

    Scoring criteria (each 0.0-1.0, averaged):
    - Length: responses between 50-500 chars score highest.
    - Keyword coverage: fraction of expected keywords present.
    - Coherence proxy: ratio of unique words (vocabulary diversity).

    Args:
        response: The model's generated text.
        expected_keywords: Keywords to look for in the response.

    Returns:
        Quality score between 0.0 and 1.0.
    """
    if not response or not response.strip():
        return 0.0

    scores: list[float] = []

    # Length score
    length = len(response.strip())
    if length < 10:
        scores.append(0.1)
    elif length < 50:
        scores.append(0.3)
    elif length < 500:
        scores.append(1.0)
    elif length < 1000:
        scores.append(0.7)
    else:
        scores.append(0.5)

    # Keyword coverage
    if expected_keywords:
        response_lower = response.lower()
        hits = sum(1 for kw in expected_keywords if kw.lower() in response_lower)
        scores.append(hits / len(expected_keywords))
    else:
        scores.append(0.5)  # No keywords defined, neutral score

    # Vocabulary diversity
    words = response.lower().split()
    if words:
        unique_ratio = len(set(words)) / len(words)
        scores.append(min(1.0, unique_ratio * 1.2))
    else:
        scores.append(0.0)

    return round(sum(scores) / len(scores), 3)


def _build_messages(prompt: EvalPrompt) -> list[dict[str, str]]:
    """Build a messages list from an EvalPrompt."""
    messages: list[dict[str, str]] = []
    if prompt.system:
        messages.append({"role": "system", "content": prompt.system})
    messages.append({"role": "user", "content": prompt.user})
    return messages


class Evaluator:
    """Runs before/after comparison of base vs fine-tuned models.

    All heavy imports (torch, transformers) happen lazily when
    evaluate() is called.

    Args:
        base_model: HuggingFace model ID for the base model.
        tuned_model_path: Path to the fine-tuned adapter directory.
        prompts: List of evaluation prompts.
    """

    def __init__(
        self,
        base_model: str,
        tuned_model_path: str | Path,
        prompts: list[EvalPrompt],
        trust_remote_code: bool = False,
    ) -> None:
        self.base_model = base_model
        self.tuned_model_path = Path(tuned_model_path)
        self.prompts = prompts
        self.trust_remote_code = trust_remote_code

    def _generate(
        self,
        model: Any,
        tokenizer: Any,
        messages: list[dict[str, str]],
        max_tokens: int,
    ) -> str:
        """Generate a response from a model.

        Args:
            model: A loaded HuggingFace model.
            tokenizer: The corresponding tokenizer.
            messages: Chat messages to send.
            max_tokens: Maximum tokens to generate.

        Returns:
            The generated text.
        """
        try:
            import torch
        except ImportError as e:
            raise ImportError(
                "torch is required for evaluation. "
                "Install with: pip install fineforge[train]"
            ) from e

        text = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = tokenizer(text, return_tensors="pt").to(model.device)

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_tokens,
                temperature=0.7,
                top_p=0.9,
                do_sample=True,
                pad_token_id=tokenizer.pad_token_id,
            )

        # Decode only the new tokens
        new_tokens = outputs[0][inputs["input_ids"].shape[1]:]
        return tokenizer.decode(new_tokens, skip_special_tokens=True)

    def evaluate(self) -> list[EvalResult]:
        """Run evaluation on all prompts.

        Loads the base model, generates responses, then loads the
        fine-tuned model and generates responses. Scores both.

        Returns:
            List of EvalResult instances.

        Raises:
            ImportError: If required packages are not installed.
        """
        try:
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer
            from peft import PeftModel
        except ImportError as e:
            raise ImportError(
                f"Missing packages for evaluation: {e}. "
                f"Install with: pip install fineforge[train]"
            ) from e

        results: list[EvalResult] = []

        # Load base model
        console.print(f"[bold cyan]Loading base model:[/bold cyan] {self.base_model}")
        tokenizer = AutoTokenizer.from_pretrained(
            self.base_model, trust_remote_code=self.trust_remote_code
        )
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        base_model = AutoModelForCausalLM.from_pretrained(
            self.base_model,
            device_map="auto",
            torch_dtype=torch.float16,
            trust_remote_code=self.trust_remote_code,
        )

        # Generate base responses
        console.print("[bold]Generating base model responses...[/bold]")
        base_responses: dict[str, str] = {}
        for prompt in self.prompts:
            messages = _build_messages(prompt)
            response = self._generate(base_model, tokenizer, messages, prompt.max_tokens)
            base_responses[prompt.name] = response
            console.print(f"  [dim]{prompt.name}: done[/dim]")

        # Free base model memory
        del base_model
        try:
            torch.cuda.empty_cache()
        except Exception:
            pass

        # Load fine-tuned model
        console.print(
            f"[bold cyan]Loading fine-tuned model:[/bold cyan] {self.tuned_model_path}"
        )
        tuned_base = AutoModelForCausalLM.from_pretrained(
            self.base_model,
            device_map="auto",
            torch_dtype=torch.float16,
            trust_remote_code=self.trust_remote_code,
        )
        tuned_model = PeftModel.from_pretrained(tuned_base, str(self.tuned_model_path))

        # Generate tuned responses
        console.print("[bold]Generating fine-tuned model responses...[/bold]")
        for prompt in self.prompts:
            messages = _build_messages(prompt)
            tuned_response = self._generate(
                tuned_model, tokenizer, messages, prompt.max_tokens
            )
            base_response = base_responses[prompt.name]

            base_score = score_response(base_response, prompt.expected_keywords)
            tuned_score = score_response(tuned_response, prompt.expected_keywords)

            results.append(
                EvalResult(
                    prompt_name=prompt.name,
                    user_input=prompt.user,
                    base_response=base_response,
                    tuned_response=tuned_response,
                    base_score=base_score,
                    tuned_score=tuned_score,
                    improvement=round(tuned_score - base_score, 3),
                )
            )
            console.print(f"  [dim]{prompt.name}: done[/dim]")

        return results


def print_results(results: list[EvalResult]) -> None:
    """Pretty-print evaluation results.

    Args:
        results: List of EvalResult instances to display.
    """
    for result in results:
        console.print(
            Panel(
                f"[bold]User:[/bold] {result.user_input}\n\n"
                f"[cyan]Base model[/cyan] (score: {result.base_score}):\n"
                f"{result.base_response[:500]}\n\n"
                f"[green]Fine-tuned[/green] (score: {result.tuned_score}):\n"
                f"{result.tuned_response[:500]}",
                title=f"[bold]{result.prompt_name}[/bold]",
                subtitle=f"Improvement: {result.improvement:+.3f}",
            )
        )

    # Summary table
    table = Table(title="Evaluation Summary")
    table.add_column("Prompt", style="cyan")
    table.add_column("Base Score", justify="right")
    table.add_column("Tuned Score", justify="right")
    table.add_column("Improvement", justify="right")

    total_improvement = 0.0
    for result in results:
        style = "green" if result.improvement > 0 else "red" if result.improvement < 0 else "white"
        table.add_row(
            result.prompt_name,
            f"{result.base_score:.3f}",
            f"{result.tuned_score:.3f}",
            f"[{style}]{result.improvement:+.3f}[/{style}]",
        )
        total_improvement += result.improvement

    avg_improvement = total_improvement / len(results) if results else 0
    table.add_row(
        "[bold]Average[/bold]",
        "",
        "",
        f"[bold]{avg_improvement:+.3f}[/bold]",
    )
    console.print(table)


def save_results(results: list[EvalResult], path: str | Path) -> None:
    """Save evaluation results to a JSON file.

    Args:
        results: EvalResult instances to save.
        path: Destination file path.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    data = []
    for r in results:
        data.append({
            "prompt_name": r.prompt_name,
            "user_input": r.user_input,
            "base_response": r.base_response,
            "tuned_response": r.tuned_response,
            "base_score": r.base_score,
            "tuned_score": r.tuned_score,
            "improvement": r.improvement,
        })

    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
