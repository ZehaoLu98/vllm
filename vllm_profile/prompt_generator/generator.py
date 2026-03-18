#!/usr/bin/env python3
"""Prompt generator that produces prompts in JSON format.

Each prompt consists of:
  - A system prompt (picked from a configured list)
  - A descriptive text (random or from history, with configurable length)
  - A query (randomly generated text of given length)

Configuration file (YAML) example:
  system_prompts:
    - "You are a helpful assistant."
    - "You are a knowledgeable expert."
    - "You are a friendly chatbot."
  initial_descriptive_length: 200
  query_length: 50
  system_prompt_length: 100   # max length used when picking system prompts
  warmup_count: 50
  total_count: 500
  pick_from_hist_pctg: 0.3
  output_file: "prompts.json"

Usage:
  python generator.py --config config.yaml
"""

import argparse
import json
import random
import string
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import List

import yaml


@dataclass
class GeneratorConfig:
    system_prompts: List[str] = field(default_factory=lambda: [
        "You are a helpful assistant.",
        "You are a knowledgeable expert.",
        "You are a friendly chatbot.",
    ])
    initial_descriptive_length: int = 200
    max_descriptive_length: int = 0
    query_length: int = 50
    system_prompt_length: int = 0
    warmup_count: int = 50
    total_count: int = 500
    pick_from_hist_pctg: float = 0.3
    output_file: str = "prompts.json"
    output_tokens: int = 100


def load_config(config_path: str) -> GeneratorConfig:
    """Load configuration from a YAML file."""
    path = Path(config_path)
    if not path.exists():
        print(f"Config file not found: {config_path}", file=sys.stderr)
        sys.exit(1)

    with open(path, "r") as f:
        raw = yaml.safe_load(f)

    if raw is None:
        raw = {}

    return GeneratorConfig(**{
        k: v for k, v in raw.items() if k in GeneratorConfig.__dataclass_fields__
    })


def generate_random_text(length: int) -> str:
    """Generate a random text string of approximately the given character length.

    Produces space-separated random words so the text looks more natural.
    """
    if length <= 0:
        return ""

    chars = string.ascii_lowercase + " "
    # Build random text character by character, then clean up double spaces
    text = "".join(random.choice(chars) for _ in range(length))
    # Ensure it doesn't start/end with space and collapse multiple spaces
    text = " ".join(text.split())
    # Trim or pad to exact length
    if len(text) > length:
        text = text[:length].rstrip()
    elif len(text) < length:
        extra = length - len(text)
        text += "".join(random.choice(string.ascii_lowercase) for _ in range(extra))
    return text


def pick_system_prompt(config: GeneratorConfig) -> str:
    """Pick a random system prompt, truncated to system_prompt_length.

    If system_prompt_length is 0, returns an empty string.
    """
    if config.system_prompt_length == 0:
        return ""
    prompt = random.choice(config.system_prompts)
    return prompt[:config.system_prompt_length]


def generate_descriptive_text(config: GeneratorConfig) -> str:
    """Generate a new random descriptive text."""
    return generate_random_text(config.initial_descriptive_length)


def generate_query(config: GeneratorConfig) -> str:
    """Generate a random query of the configured length."""
    return generate_random_text(config.query_length)


def build_prompt(system_prompt: str, descriptive_text: str, query: str) -> dict:
    """Combine the three parts into a single prompt dict."""
    return {
        "system_prompt": system_prompt,
        "descriptive_text": descriptive_text,
        "query": query,
    }


def generate_prompts(config: GeneratorConfig) -> List[dict]:
    """Generate all prompts according to the configuration."""
    # History stores tuples of (system_prompt, descriptive_text, query)
    history: List[tuple[str, str, str]] = []
    prompts: List[dict] = []

    for i in range(config.total_count):
        is_warmup = i < config.warmup_count

        if not is_warmup and history and random.random() < config.pick_from_hist_pctg:
            # Pick a full prompt from history and truncate to a random length,
            # but always keep at least the system prompt portion.
            sys_p, desc_p, query_p = random.choice(history)
            full = sys_p + desc_p + query_p
            min_len = len(sys_p)
            cap = config.max_descriptive_length if config.max_descriptive_length > 0 else len(full)
            rand_len = random.randint(min_len, min(len(full), cap))
            truncated = full[:rand_len]
            prompt = build_prompt("", truncated, "")
        else:
            system_prompt = pick_system_prompt(config)
            descriptive_text = generate_descriptive_text(config)
            query = generate_query(config)
            prompt = build_prompt(system_prompt, descriptive_text, query)

        # Save the three parts into history
        history.append((prompt["system_prompt"], prompt["descriptive_text"], prompt["query"]))
        prompts.append(prompt)

    return prompts


def write_prompts(prompts: List[dict], output_file: str, output_tokens: int) -> None:
    """Write the list of prompts to a JSON file and a JSONL file."""
    path = Path(output_file)
    path.parent.mkdir(parents=True, exist_ok=True)

    with open(path, "w") as f:
        json.dump(prompts, f, indent=2)
    print(f"Wrote {len(prompts)} prompts to {path}")

    # Write JSONL file for vllm bench serve --dataset-name custom
    jsonl_path = path.with_suffix(".jsonl")
    with open(jsonl_path, "w") as f:
        for p in prompts:
            line = p["system_prompt"] + p["descriptive_text"] + p["query"]
            json.dump({"prompt": line, "output_tokens": output_tokens},
                      f, ensure_ascii=False)
            f.write("\n")
    print(f"Wrote {len(prompts)} prompts to {jsonl_path}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate prompts and write them to a JSON file."
    )
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to the YAML configuration file.",
    )
    args = parser.parse_args()

    config = load_config(args.config)
    prompts = generate_prompts(config)
    write_prompts(prompts, config.output_file, config.output_tokens)

    # Write raw text file: one line per prompt, concatenating all three parts
    raw_path = Path(config.output_file).with_suffix(".txt")
    with open(raw_path, "w") as f:
        for p in prompts:
            f.write(p["system_prompt"] + p["descriptive_text"] + p["query"] + "\n")
    print(f"Wrote {len(prompts)} raw prompts to {raw_path}")


if __name__ == "__main__":
    main()
