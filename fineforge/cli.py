"""CLI interface for FineForge.

Provides commands: prepare, train, eval, export.
"""

from pathlib import Path

import click
from rich.console import Console

from fineforge import __version__

console = Console()


@click.group()
@click.version_option(version=__version__, prog_name="fineforge")
def main() -> None:
    """FineForge -- Local LoRA fine-tuning pipeline for consumer GPUs.

    Curate datasets, train QLoRA adapters, evaluate results,
    and export to GGUF for Ollama.
    """


@main.command()
@click.argument("input_file", type=click.Path(exists=True))
@click.option(
    "--output-dir",
    "-o",
    default="./data",
    help="Output directory for curated train/eval JSONL files.",
)
@click.option(
    "--min-quality",
    "-q",
    default=0.3,
    type=float,
    help="Minimum quality score to keep (0.0-1.0).",
)
@click.option(
    "--eval-ratio",
    "-e",
    default=0.1,
    type=float,
    help="Fraction of data to hold out for evaluation.",
)
@click.option(
    "--seed",
    "-s",
    default=42,
    type=int,
    help="Random seed for train/eval split.",
)
def prepare(
    input_file: str,
    output_dir: str,
    min_quality: float,
    eval_ratio: float,
    seed: int,
) -> None:
    """Curate and validate a chat dataset.

    Loads a JSONL file, validates format, scores quality,
    removes duplicates and low-quality samples, then splits
    into train and eval sets.
    """
    from fineforge.dataset import (
        filter_dataset,
        load_jsonl,
        print_stats,
        save_jsonl,
        split_dataset,
    )

    console.print(f"[bold cyan]Loading dataset:[/bold cyan] {input_file}")
    samples = load_jsonl(input_file)
    console.print(f"  Loaded {len(samples)} raw samples")

    console.print("[bold cyan]Filtering and scoring...[/bold cyan]")
    filtered, stats = filter_dataset(samples, min_quality_score=min_quality)

    print_stats(stats)

    if not filtered:
        console.print("[red]No samples passed filtering. Aborting.[/red]")
        raise click.Abort()

    train_set, eval_set = split_dataset(filtered, eval_ratio=eval_ratio, seed=seed)

    output_path = Path(output_dir)
    train_path = output_path / "train.jsonl"
    save_jsonl(train_set, train_path)
    console.print(f"[green]Train set:[/green] {len(train_set)} samples -> {train_path}")

    if eval_set:
        eval_path = output_path / "eval.jsonl"
        save_jsonl(eval_set, eval_path)
        console.print(f"[green]Eval set:[/green] {len(eval_set)} samples -> {eval_path}")

    console.print("[bold green]Dataset preparation complete.[/bold green]")


@main.command()
@click.argument("config_file", type=click.Path(exists=True))
def train(config_file: str) -> None:
    """Run QLoRA fine-tuning.

    Loads a YAML config file and trains a LoRA adapter
    on the specified base model and dataset.
    """
    from fineforge.config import TrainConfig
    from fineforge.trainer import Trainer

    console.print(f"[bold cyan]Loading config:[/bold cyan] {config_file}")
    config = TrainConfig.from_yaml(config_file)

    errors = config.validate()
    if errors:
        for err in errors:
            console.print(f"[red]Config error:[/red] {err}")
        raise click.Abort()

    console.print(f"[bold]Base model:[/bold] {config.base_model}")
    console.print(f"[bold]Dataset:[/bold] {config.dataset_path}")
    console.print(f"[bold]Output:[/bold] {config.output_dir}")
    console.print(
        f"[bold]LoRA:[/bold] r={config.lora_r}, alpha={config.lora_alpha}, "
        f"dropout={config.lora_dropout}"
    )
    console.print(
        f"[bold]Training:[/bold] {config.num_epochs} epochs, "
        f"lr={config.learning_rate}, batch={config.batch_size}"
    )
    console.print()

    trainer = Trainer(config)
    adapter_path = trainer.train()

    console.print(
        f"\n[bold green]Training complete![/bold green] "
        f"Adapter saved to: {adapter_path}"
    )


@main.command(name="eval")
@click.argument("model_path", type=click.Path(exists=True))
@click.option(
    "--prompts",
    "-p",
    required=True,
    type=click.Path(exists=True),
    help="Path to YAML file containing test prompts.",
)
@click.option(
    "--base-model",
    "-b",
    default="unsloth/Qwen2.5-7B",
    help="Base model to compare against.",
)
@click.option(
    "--output",
    "-o",
    default=None,
    help="Save results to a JSON file.",
)
def eval_cmd(
    model_path: str,
    prompts: str,
    base_model: str,
    output: str | None,
) -> None:
    """Evaluate a fine-tuned model against the base model.

    Runs test prompts through both models and compares responses.
    """
    from fineforge.evaluator import (
        Evaluator,
        load_prompts,
        print_results,
        save_results,
    )

    console.print(f"[bold cyan]Loading prompts:[/bold cyan] {prompts}")
    eval_prompts = load_prompts(prompts)
    console.print(f"  {len(eval_prompts)} test prompts loaded")

    evaluator = Evaluator(
        base_model=base_model,
        tuned_model_path=model_path,
        prompts=eval_prompts,
    )
    results = evaluator.evaluate()

    print_results(results)

    if output:
        save_results(results, output)
        console.print(f"[green]Results saved to:[/green] {output}")


@main.command()
@click.argument("model_path", type=click.Path(exists=True))
@click.option(
    "--base-model",
    "-b",
    default="unsloth/Qwen2.5-7B",
    help="Base model the adapter was trained on.",
)
@click.option(
    "--format",
    "-f",
    "output_format",
    default="gguf",
    type=click.Choice(["gguf"]),
    help="Export format.",
)
@click.option(
    "--quantization",
    "-q",
    default="q4_k_m",
    help="Quantization type (e.g., q4_k_m, q5_k_m, q8_0, f16).",
)
@click.option(
    "--ollama-name",
    default=None,
    help="Register in Ollama with this name.",
)
@click.option(
    "--system-prompt",
    default="",
    help="Default system prompt for the Ollama Modelfile.",
)
@click.option(
    "--output-dir",
    "-o",
    default="./output",
    help="Output directory for exported files.",
)
@click.option(
    "--llama-cpp-path",
    default=None,
    help="Path to llama.cpp directory.",
)
def export(
    model_path: str,
    base_model: str,
    output_format: str,
    quantization: str,
    ollama_name: str | None,
    system_prompt: str,
    output_dir: str,
    llama_cpp_path: str | None,
) -> None:
    """Export a fine-tuned model to GGUF and optionally register in Ollama.

    Merges the LoRA adapter into the base model, converts to GGUF,
    and optionally registers in Ollama.
    """
    from fineforge.exporter import (
        ExportError,
        export_gguf,
        merge_adapter,
        register_ollama,
    )

    out_path = Path(output_dir)

    # Step 1: Merge adapter
    console.print("[bold]Step 1/3: Merging adapter into base model...[/bold]")
    merged_dir = merge_adapter(base_model, model_path, out_path)

    # Step 2: Export to GGUF
    console.print(f"\n[bold]Step 2/3: Exporting to GGUF ({quantization})...[/bold]")
    gguf_name = f"fineforge-{quantization}.gguf"
    gguf_path = export_gguf(
        model_dir=merged_dir,
        output_path=out_path / gguf_name,
        quantization=quantization,
        llama_cpp_path=llama_cpp_path,
    )

    # Step 3: Register in Ollama (optional)
    if ollama_name:
        console.print(f"\n[bold]Step 3/3: Registering in Ollama...[/bold]")
        register_ollama(
            gguf_path=gguf_path,
            model_name=ollama_name,
            system_prompt=system_prompt,
        )
    else:
        console.print(
            "\n[dim]Skipping Ollama registration (use --ollama-name to enable)[/dim]"
        )

    console.print(f"\n[bold green]Export complete![/bold green]")
    console.print(f"  GGUF: {gguf_path}")
    if ollama_name:
        console.print(f"  Ollama: ollama run {ollama_name}")


if __name__ == "__main__":
    main()
