"""
Prompt loader module for vLLM profiling.

This module provides functionality to load prompts from external files,
separating prompt data from the main script logic.
"""

import json
import os
from typing import List, Union


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


def get_default_prompts_path() -> str:
    """
    Get the default path for the prompts file.
    
    Returns:
        Path to the default prompts.json file in the same directory as this module
    """
    module_dir = os.path.dirname(os.path.abspath(__file__))
    return os.path.join(module_dir, "prompts.json")
