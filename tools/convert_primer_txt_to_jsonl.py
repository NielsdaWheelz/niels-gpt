#!/usr/bin/env python3
"""Convert primer.txt (plaintext dialogue format) to primer.jsonl (chat JSONL format).

Usage:
    python tools/convert_primer_txt_to_jsonl.py
    python tools/convert_primer_txt_to_jsonl.py --input data/primer.txt --output data/primer.jsonl
    python tools/convert_primer_txt_to_jsonl.py --input data/primer.generated.txt --output data/primer.generated.jsonl
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path


DIALOGUE_DELIM = "\n\n<dialogue>\n\n"

# Regex to parse role: content lines
# Matches "system:", "user:", "assistant:" at start of line, captures the rest
ROLE_PATTERN = re.compile(r"^(system|user|assistant):\s*(.*)$", re.IGNORECASE)


def parse_dialogue_block(block: str) -> list[dict[str, str]] | None:
    """
    Parse a single dialogue block into a list of messages.

    Expected format:
        system: ...
        user: ...
        assistant: ...

    Returns None if block is malformed (missing required roles, etc).
    """
    lines = block.strip().split("\n")
    messages: list[dict[str, str]] = []

    for line in lines:
        line = line.strip()
        if not line:
            continue

        match = ROLE_PATTERN.match(line)
        if not match:
            # Line doesn't match role pattern - could be continuation or malformed
            # For now, skip non-matching lines (primer.txt has single-line messages)
            continue

        role = match.group(1).lower()
        content = match.group(2).strip()
        messages.append({"role": role, "content": content})

    # Validate: must have at least one assistant message
    has_assistant = any(m["role"] == "assistant" for m in messages)
    if not has_assistant:
        return None

    # Validate: should have at least 2 messages (user + assistant minimum)
    if len(messages) < 2:
        return None

    return messages


def convert_primer_txt_to_jsonl(input_path: str, output_path: str) -> tuple[int, int]:
    """
    Convert primer.txt to primer.jsonl.

    Args:
        input_path: Path to input .txt file
        output_path: Path to output .jsonl file

    Returns:
        (num_converted, num_skipped)
    """
    input_file = Path(input_path)
    if not input_file.exists():
        raise FileNotFoundError(f"input file not found: {input_path}")

    text = input_file.read_text(encoding="utf-8")
    blocks = text.split(DIALOGUE_DELIM)

    converted = 0
    skipped = 0

    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)

    with open(output_file, "w", encoding="utf-8") as f:
        for i, block in enumerate(blocks):
            block = block.strip()
            if not block:
                skipped += 1
                continue

            messages = parse_dialogue_block(block)
            if messages is None:
                print(f"warning: skipping malformed block {i + 1}")
                skipped += 1
                continue

            example = {"messages": messages}
            f.write(json.dumps(example, ensure_ascii=False) + "\n")
            converted += 1

    return converted, skipped


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Convert primer.txt to primer.jsonl format"
    )
    parser.add_argument(
        "--input",
        type=str,
        default="data/primer.txt",
        help="Input primer.txt path (default: data/primer.txt)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="data/primer.jsonl",
        help="Output primer.jsonl path (default: data/primer.jsonl)",
    )
    args = parser.parse_args()

    print(f"converting {args.input} → {args.output}")

    try:
        converted, skipped = convert_primer_txt_to_jsonl(args.input, args.output)
        print(f"✓ converted {converted} dialogues")
        if skipped > 0:
            print(f"  skipped {skipped} empty/malformed blocks")
        print(f"  output: {args.output}")
        return 0
    except Exception as e:
        print(f"error: {e}")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())

