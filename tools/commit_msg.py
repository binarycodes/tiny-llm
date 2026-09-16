#!/usr/bin/env python3
"""Reject commit messages that are not a single Conventional Commits header."""

import re
import sys
from pathlib import Path

TYPES = (
    "build",
    "chore",
    "ci",
    "docs",
    "feat",
    "fix",
    "perf",
    "refactor",
    "revert",
    "style",
    "test",
)

HEADER = re.compile(rf"^({'|'.join(TYPES)})(\([^()\s]+\))?!?: \S.*$")
SCISSORS = "# ------------------------ >8 ------------------------"


def message_lines(raw: str) -> list[str]:
    lines: list[str] = []
    for line in raw.splitlines():
        if line == SCISSORS:
            break
        if line.startswith("#"):
            continue
        lines.append(line.rstrip())
    while lines and not lines[-1]:
        lines.pop()
    return lines


def check(raw: str) -> str | None:
    lines = message_lines(raw)
    if not lines:
        return "empty commit message"
    if len(lines) > 1:
        return "commit message must be a single line, no body or footer"
    if not HEADER.match(lines[0]):
        return (
            f"header must look like '<type>[(scope)][!]: <description>' "
            f"with type one of: {', '.join(TYPES)}"
        )
    return None


def main(argv: list[str]) -> int:
    error = check(Path(argv[1]).read_text(encoding="utf-8"))
    if error is None:
        return 0
    print(f"commit rejected: {error}", file=sys.stderr)
    return 1


if __name__ == "__main__":
    sys.exit(main(sys.argv))
