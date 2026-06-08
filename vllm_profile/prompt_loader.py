"""
Prompt loader module for vLLM profiling.

This module provides functionality to load prompts from external files,
separating prompt data from the main script logic.
"""

import json
import os
from typing import List, Tuple, Union


def load_prompts_from_json(file_path: str) -> List[str]:
    """
    Load prompts from a JSON file.
    
    Args:
        file_path: Path to the JSON file containing prompts as an array
        
    Returns:
        List of prompt strings
        
    Raises:
        FileNotFoundError: If the JSON file doesn't exist
        json.JSONDecodeError: If the JSON file is malformed
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Prompt file not found: {file_path}")
    
    with open(file_path, 'r', encoding='utf-8') as f:
        prompts = json.load(f)
    
    if not isinstance(prompts, list):
        raise ValueError("JSON file must contain an array of prompts")
    
    return prompts


def load_prompts_from_text(file_path: str, delimiter: str = "\n") -> List[str]:
    """
    Load prompts from a text file.
    
    Args:
        file_path: Path to the text file containing prompts
        delimiter: String delimiter separating prompts (default: "\\n", i.e.
            one prompt per line, matching the generator's .txt output)
        
    Returns:
        List of prompt strings
        
    Raises:
        FileNotFoundError: If the text file doesn't exist
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Prompt file not found: {file_path}")
    
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    prompts = [p.strip() for p in content.split(delimiter) if p.strip()]
    return prompts


def load_prompts(file_path: str, file_type: str = "auto") -> List[str]:
    """
    Load prompts from a file. Automatically detects file type if not specified.
    
    Args:
        file_path: Path to the file containing prompts
        file_type: Type of file ("json", "text", or "auto" for auto-detection)
        
    Returns:
        List of prompt strings
        
    Raises:
        FileNotFoundError: If the file doesn't exist
        ValueError: If file type is unsupported
    """
    if file_type == "auto":
        if file_path.endswith('.json'):
            file_type = "json"
        elif file_path.endswith('.txt'):
            file_type = "text"
        else:
            # Try JSON first, then text
            try:
                return load_prompts_from_json(file_path)
            except (json.JSONDecodeError, ValueError):
                return load_prompts_from_text(file_path)
    
    if file_type == "json":
        return load_prompts_from_json(file_path)
    elif file_type == "text":
        return load_prompts_from_text(file_path)
    else:
        raise ValueError(f"Unsupported file type: {file_type}. Use 'json', 'text', or 'auto'.")


def load_prompts_with_output_tokens(
    file_path: str,
    default_output_tokens: int = 512,
) -> Tuple[List[str], List[int]]:
    """
    Load prompts together with their per-prompt ``output_tokens``.

    Supports:
    - ``.jsonl``: one JSON object per line, each with a ``"prompt"`` key and an
      optional ``"output_tokens"`` key.
    - ``.json``: an array of strings, or an array of objects with ``"prompt"``
      and optional ``"output_tokens"``.
    - ``.txt`` (or anything else): one prompt per line; ``output_tokens`` falls
      back to ``default_output_tokens`` for every prompt.

    Args:
        file_path: Path to the prompts file.
        default_output_tokens: Value used when a prompt has no ``output_tokens``.

    Returns:
        A tuple ``(prompts, output_tokens)`` where both lists have equal length.

    Raises:
        FileNotFoundError: If the file doesn't exist.
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Prompt file not found: {file_path}")

    prompts: List[str] = []
    output_tokens: List[int] = []

    def _append(prompt: str, tokens: Union[int, None]) -> None:
        prompts.append(prompt)
        output_tokens.append(int(tokens) if tokens is not None else default_output_tokens)

    if file_path.endswith(".jsonl"):
        with open(file_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                obj = json.loads(line)
                _append(obj["prompt"], obj.get("output_tokens"))
        return prompts, output_tokens

    if file_path.endswith(".json"):
        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        if not isinstance(data, list):
            raise ValueError("JSON file must contain an array of prompts")
        for item in data:
            if isinstance(item, str):
                _append(item, None)
            else:
                _append(item["prompt"], item.get("output_tokens"))
        return prompts, output_tokens

    # Fall back to plain text: one prompt per line, default output_tokens.
    for prompt in load_prompts_from_text(file_path):
        _append(prompt, None)
    return prompts, output_tokens


def get_default_prompts_path() -> str:
    """
    Get the default path for the prompts file.
    
    Returns:
        Path to the default prompts.json file in the same directory as this module
    """
    module_dir = os.path.dirname(os.path.abspath(__file__))
    return os.path.join(module_dir, "prompts.json")
