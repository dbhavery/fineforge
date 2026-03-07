"""GGUF export and Ollama registration for FineForge.

Handles merging LoRA adapters back into the base model,
exporting to GGUF format via llama.cpp, and registering
the result in Ollama.

All heavy imports (torch, transformers, peft) are lazy-loaded.
"""

import json
import shutil
import subprocess
from pathlib import Path
from typing import Any, Optional

from rich.console import Console

console = Console()


class ExportError(Exception):
    """Raised when export fails."""


def generate_modelfile(
    model_path: str,
    system_prompt: str = "",
    temperature: float = 0.7,
    top_p: float = 0.9,
    stop_tokens: Optional[list[str]] = None,
) -> str:
    """Generate an Ollama Modelfile.

    Args:
        model_path: Path to the GGUF file (relative or absolute).
        system_prompt: Default system prompt for the model.
        temperature: Default sampling temperature.
        top_p: Default top-p sampling value.
        stop_tokens: List of stop token strings.

    Returns:
        The Modelfile content as a string.
    """
    lines: list[str] = []
    lines.append(f"FROM {model_path}")
    lines.append("")

    if system_prompt:
        lines.append(f'SYSTEM """{system_prompt}"""')
        lines.append("")

    lines.append(f"PARAMETER temperature {temperature}")
    lines.append(f"PARAMETER top_p {top_p}")

    if stop_tokens:
        for token in stop_tokens:
            lines.append(f'PARAMETER stop "{token}"')

    lines.append("")
    return "\n".join(lines)


def merge_adapter(
    base_model: str,
    adapter_path: str | Path,
    output_dir: str | Path,
) -> Path:
    """Merge a LoRA adapter into the base model.

    Loads the base model and the LoRA adapter, merges them,
    and saves the full merged model to output_dir.

    Args:
        base_model: HuggingFace model ID for the base model.
        adapter_path: Path to the LoRA adapter directory.
        output_dir: Directory to save the merged model.

    Returns:
        Path to the merged model directory.

    Raises:
        ExportError: If merging fails.
        ImportError: If required packages are not installed.
    """
    try:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
        from peft import PeftModel
    except ImportError as e:
        raise ImportError(
            f"Missing packages for model merging: {e}. "
            f"Install with: pip install fineforge[train]"
        ) from e

    adapter_path = Path(adapter_path)
    output_dir = Path(output_dir)
    merged_dir = output_dir / "merged"
    merged_dir.mkdir(parents=True, exist_ok=True)

    try:
        console.print(f"[bold cyan]Loading base model:[/bold cyan] {base_model}")
        model = AutoModelForCausalLM.from_pretrained(
            base_model,
            device_map="auto",
            torch_dtype=torch.float16,
            trust_remote_code=True,
        )

        console.print(f"[bold cyan]Loading adapter:[/bold cyan] {adapter_path}")
        model = PeftModel.from_pretrained(model, str(adapter_path))

        console.print("[bold cyan]Merging adapter into base model...[/bold cyan]")
        model = model.merge_and_unload()

        console.print(f"[bold cyan]Saving merged model to:[/bold cyan] {merged_dir}")
        model.save_pretrained(str(merged_dir))

        tokenizer = AutoTokenizer.from_pretrained(
            base_model, trust_remote_code=True
        )
        tokenizer.save_pretrained(str(merged_dir))

        console.print("[green]Model merged and saved successfully.[/green]")
        return merged_dir

    except Exception as e:
        raise ExportError(f"Failed to merge adapter: {e}") from e


def export_gguf(
    model_dir: str | Path,
    output_path: str | Path,
    quantization: str = "q4_k_m",
    llama_cpp_path: Optional[str] = None,
) -> Path:
    """Export a HuggingFace model to GGUF format.

    Uses llama.cpp's convert script to perform the conversion.

    Args:
        model_dir: Path to the merged HuggingFace model directory.
        output_path: Desired output GGUF file path.
        quantization: Quantization type (e.g., "q4_k_m", "q5_k_m", "q8_0", "f16").
        llama_cpp_path: Path to llama.cpp directory. If None, attempts
            to find convert scripts on PATH.

    Returns:
        Path to the exported GGUF file.

    Raises:
        ExportError: If the conversion fails.
    """
    model_dir = Path(model_dir)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Find the convert script
    convert_script: Optional[Path] = None

    if llama_cpp_path:
        llama_dir = Path(llama_cpp_path)
        candidates = [
            llama_dir / "convert_hf_to_gguf.py",
            llama_dir / "convert-hf-to-gguf.py",
        ]
        for candidate in candidates:
            if candidate.exists():
                convert_script = candidate
                break

    if convert_script is None:
        # Try to find on PATH
        convert_cmd = shutil.which("convert-hf-to-gguf")
        if convert_cmd:
            convert_script = Path(convert_cmd)

    if convert_script is None:
        # Try python -m approach
        console.print(
            "[yellow]llama.cpp convert script not found. "
            "Attempting conversion via llama-cpp-python...[/yellow]"
        )
        raise ExportError(
            "Could not find llama.cpp convert script. "
            "Please install llama.cpp and provide the path via --llama-cpp-path, "
            "or ensure convert-hf-to-gguf is on your PATH.\n\n"
            "Install llama.cpp:\n"
            "  git clone https://github.com/ggerganov/llama.cpp\n"
            "  cd llama.cpp && pip install -r requirements.txt"
        )

    # Step 1: Convert to f16 GGUF
    f16_path = output_path.parent / f"{output_path.stem}-f16.gguf"
    console.print(f"[bold cyan]Converting to GGUF (f16)...[/bold cyan]")

    cmd_convert = [
        "python",
        str(convert_script),
        str(model_dir),
        "--outfile",
        str(f16_path),
        "--outtype",
        "f16",
    ]

    result = subprocess.run(
        cmd_convert,
        capture_output=True,
        text=True,
        timeout=1800,  # 30 minute timeout
    )

    if result.returncode != 0:
        raise ExportError(
            f"GGUF conversion failed:\n{result.stderr}"
        )

    # Step 2: Quantize if not f16
    if quantization.lower() == "f16":
        if f16_path != output_path:
            shutil.move(str(f16_path), str(output_path))
        console.print(f"[green]GGUF exported (f16):[/green] {output_path}")
        return output_path

    # Find quantize binary
    quantize_cmd: Optional[str] = None
    if llama_cpp_path:
        llama_dir = Path(llama_cpp_path)
        for name in ["llama-quantize", "llama-quantize.exe", "quantize", "quantize.exe"]:
            candidate = llama_dir / name
            if candidate.exists():
                quantize_cmd = str(candidate)
                break

    if quantize_cmd is None:
        quantize_cmd = shutil.which("llama-quantize") or shutil.which("quantize")

    if quantize_cmd is None:
        console.print(
            f"[yellow]Quantize binary not found. "
            f"Keeping f16 GGUF at: {f16_path}[/yellow]"
        )
        return f16_path

    console.print(f"[bold cyan]Quantizing to {quantization}...[/bold cyan]")
    cmd_quantize = [
        quantize_cmd,
        str(f16_path),
        str(output_path),
        quantization,
    ]

    result = subprocess.run(
        cmd_quantize,
        capture_output=True,
        text=True,
        timeout=1800,
    )

    if result.returncode != 0:
        raise ExportError(
            f"Quantization failed:\n{result.stderr}"
        )

    # Clean up f16 intermediate
    f16_path.unlink(missing_ok=True)

    console.print(f"[green]GGUF exported ({quantization}):[/green] {output_path}")
    return output_path


def register_ollama(
    gguf_path: str | Path,
    model_name: str,
    system_prompt: str = "",
    temperature: float = 0.7,
) -> None:
    """Register a GGUF model in Ollama.

    Creates a Modelfile and runs `ollama create` to register it.

    Args:
        gguf_path: Path to the GGUF file.
        model_name: Name to register the model under in Ollama.
        system_prompt: Default system prompt.
        temperature: Default sampling temperature.

    Raises:
        ExportError: If Ollama registration fails.
    """
    gguf_path = Path(gguf_path)

    if not gguf_path.exists():
        raise ExportError(f"GGUF file not found: {gguf_path}")

    # Check if Ollama is available
    ollama_cmd = shutil.which("ollama")
    if ollama_cmd is None:
        raise ExportError(
            "Ollama not found on PATH. Install from https://ollama.com"
        )

    # Generate Modelfile
    modelfile_content = generate_modelfile(
        model_path=str(gguf_path.resolve()),
        system_prompt=system_prompt,
        temperature=temperature,
    )

    modelfile_path = gguf_path.parent / "Modelfile"
    with open(modelfile_path, "w", encoding="utf-8") as f:
        f.write(modelfile_content)

    console.print(f"[bold cyan]Registering in Ollama as:[/bold cyan] {model_name}")

    result = subprocess.run(
        [ollama_cmd, "create", model_name, "-f", str(modelfile_path)],
        capture_output=True,
        text=True,
        timeout=600,
    )

    if result.returncode != 0:
        raise ExportError(
            f"Ollama registration failed:\n{result.stderr}"
        )

    console.print(
        f"[green]Model registered in Ollama![/green]\n"
        f"  Run it with: [bold]ollama run {model_name}[/bold]"
    )
