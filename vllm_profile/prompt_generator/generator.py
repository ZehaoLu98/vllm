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
  descriptive_length: 200
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
    descriptive_length: int = 200
    query_length: int = 50
    warmup_count: int = 50
    total_count: int = 500
    pick_from_hist_pctg: float = 0.3
    output_file: str = "prompts.json"


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
    """Pick a random system prompt from the configured list."""
    return random.choice(config.system_prompts)


def generate_descriptive_text(
    config: GeneratorConfig,
    history: set[str],
    is_warmup: bool,
) -> str:
    """Generate or retrieve a descriptive text.

    During warmup: always generate a new random descriptive text.
    After warmup: with probability `pick_from_hist_pctg`, pick from history
                  (with a random length from 0 to descriptive_length);
                  otherwise generate a new random one.
    History is managed by the caller.
    """
    if is_warmup:
        return generate_random_text(config.descriptive_length)

    if random.random() < config.pick_from_hist_pctg and history:
        # Pick from history and optionally truncate to a random length
        picked = random.choice(history)
        rand_len = random.randint(0, config.descriptive_length)
        return picked[:rand_len]
    else:
        return generate_random_text(config.descriptive_length)


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
    history: List[str] = []
    prompts: List[dict] = []

    for i in range(config.total_count):
        is_warmup = i < config.warmup_count

        system_prompt = pick_system_prompt(config)
        descriptive_text = generate_descriptive_text(config, history, is_warmup)
        query = generate_query(config)

        # Save concatenated descriptive text + query into history
        history.append(descriptive_text + " " + query)

        prompt = build_prompt(system_prompt, descriptive_text, query)
        prompts.append(prompt)

    return prompts


def write_prompts(prompts: List[dict], output_file: str) -> None:
    """Write the list of prompts to a JSON file."""
    path = Path(output_file)
    path.parent.mkdir(parents=True, exist_ok=True)

    with open(path, "w") as f:
        json.dump(prompts, f, indent=2)

    print(f"Wrote {len(prompts)} prompts to {path}")


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
    write_prompts(prompts, config.output_file)


if __name__ == "__main__":
    main()
