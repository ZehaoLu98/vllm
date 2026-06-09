"""
Prompt loader module for vLLM profiling.

This module provides functionality to load prompts from external files,
separating prompt data from the main script logic.
"""

import json
import os
from typing import List, Tuple, Union


def _validate_jsonl_file(file_path: str) -> None:
    """
    Validate file existence and format support.

    Only ``.jsonl`` files are supported.
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Prompt file not found: {file_path}")

    _, ext = os.path.splitext(file_path)
    ext = ext.lower()
    if ext in {".json", ".txt"}:
        raise ValueError(
            f"Unsupported prompt file format: '{ext}'. "
            "Only '.jsonl' is supported."
        )
    if ext != ".jsonl":
        raise ValueError(
            f"Unsupported prompt file format: '{ext or '<none>'}'. "
            "Only '.jsonl' is supported."
        )


def load_prompts_from_jsonl(file_path: str) -> List[str]:
    """Load prompts from a JSONL file."""
    _validate_jsonl_file(file_path)

    prompts: List[str] = []
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            prompts.append(obj["prompt"])
    return prompts


def load_prompts(file_path: str, file_type: str = "auto") -> List[str]:
    """
    Load prompts from a JSONL file.

    Args:
        file_path: Path to the file containing prompts
        file_type: Type of file ("jsonl" or "auto")

    Returns:
        List of prompt strings

    Raises:
        FileNotFoundError: If the file doesn't exist
        ValueError: If file type is unsupported
    """
    if file_type not in {"auto", "jsonl"}:
        raise ValueError(
            f"Unsupported file type: {file_type}. Use 'jsonl' or 'auto'."
        )
    return load_prompts_from_jsonl(file_path)


def load_prompts_with_output_tokens(
    file_path: str,
    default_output_tokens: int = 512,
) -> Tuple[List[str], List[int]]:
    """
    Load prompts together with their per-prompt ``output_tokens``.

        Supports:
        - ``.jsonl``: one JSON object per line, each with a ``"prompt"`` key and an
            optional ``"output_tokens"`` key.

    Args:
        file_path: Path to the prompts file.
        default_output_tokens: Value used when a prompt has no ``output_tokens``.

    Returns:
        A tuple ``(prompts, output_tokens)`` where both lists have equal length.

    Raises:
        FileNotFoundError: If the file doesn't exist.
    """
    _validate_jsonl_file(file_path)

    prompts: List[str] = []
    output_tokens: List[int] = []

    def _append(prompt: str, tokens: Union[int, None]) -> None:
        prompts.append(prompt)
        output_tokens.append(int(tokens) if tokens is not None else default_output_tokens)

    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            _append(obj["prompt"], obj.get("output_tokens"))
    return prompts, output_tokens


def get_default_prompts_path() -> str:
    """
    Get the default path for the prompts file.
    
    Returns:
        Path to the default prompts.jsonl file in the same directory as this module
    """
    module_dir = os.path.dirname(os.path.abspath(__file__))
    return os.path.join(module_dir, "prompts.jsonl")
