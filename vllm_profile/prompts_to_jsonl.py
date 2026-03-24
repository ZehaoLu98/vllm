#!/usr/bin/env python3
"""Convert prompts.txt (one prompt per line) to a JSONL file
suitable for vllm bench serve --dataset-name custom."""

import json
import argparse
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(
        description="Convert prompts.txt to JSONL for vllm benchmark")
    parser.add_argument("--input", default="prompts.txt",
                        help="Input prompts file (one per line)")
    parser.add_argument("--output", default="prompts.jsonl",
                        help="Output JSONL file")
    parser.add_argument("--output-tokens", type=int, default=100,
                        help="Number of output tokens per prompt")
    args = parser.parse_args()

    lines = Path(args.input).read_text(encoding="utf-8").splitlines()
    with open(args.output, "w", encoding="utf-8") as f:
        for line in lines:
            line = line.strip()
            if not line:
                continue
            json.dump({"prompt": line, "output_tokens": args.output_tokens},
                      f, ensure_ascii=False)
            f.write("\n")

    print(f"Wrote {len(lines)} prompts to {args.output}")


if __name__ == "__main__":
    main()
