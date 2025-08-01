#!/usr/bin/env python3
from pathlib import Path
import re
import sys

SRC_ROOT = Path("data/topics")
DEST_ROOT = Path("data/cleaned_topics")

SECTION_TITLES = [
    "Review Questions",
    "References",
    "Objectives",
]


def clean_content(text: str) -> str:
    original = text

    # 1. If "Last Update:" line exists, remove everything from start up to and including that line
    last_update_pattern = re.compile(r"^.*?Last Update:.*?\n", re.DOTALL | re.MULTILINE)

    if not last_update_pattern.search(text):
        print(f"[WARNING] No 'Last Update:' line found in {text}")

    text = last_update_pattern.sub("", text, count=1)

    # 2. Remove specified sections - find all sections first, then remove from end to start
    lines = text.split("\n")
    sections_to_remove = []

    # Find all sections to remove
    for title in SECTION_TITLES:
        # Pattern for markdown headers (# Title) or bold text (**Title:**)
        header_pattern = rf"^#{{1,6}}\s*{re.escape(title)}\b"
        bold_pattern = rf"^\*\*{re.escape(title)}:\*\*\s*$"

        for i, line in enumerate(lines):
            line_stripped = line.strip()
            if re.match(header_pattern, line_stripped, re.IGNORECASE) or re.match(
                bold_pattern, line_stripped, re.IGNORECASE
            ):
                # Find where this section ends
                section_end = len(lines)  # Default to end of file
                for j in range(i + 1, len(lines)):
                    next_line = lines[j].strip()
                    # Look for next markdown header or next bold section
                    if (
                        re.match(r"^#{{1,6}}\s*\w", next_line)
                        or re.match(r"^\*\*\w.*:\*\*\s*$", next_line)
                        or re.match(r"^##\s+\w", next_line)
                    ):  # Also match ## headers
                        section_end = j
                        break
                sections_to_remove.append((i, section_end))
                break  # Only remove first occurrence of each title

    # Remove sections from end to start so line numbers don't shift
    for start, end in sorted(sections_to_remove, reverse=True):
        lines = lines[:start] + lines[end:]

    text = "\n".join(lines)

    # 3. Remove disclosure lines
    text = re.sub(r"(?m)^\*\*Disclosure:.*(?:\n|$)", "", text)

    # 4. Remove inline reference citations [x], [x-y], [x,y,z], etc.
    text = re.sub(r"\[[0-9,\-\s]+\]", "", text)

    # 5. Normalize multiple blank lines to at most two
    text = re.sub(r"\n{3,}", "\n\n", text)

    return text.strip() + "\n"


def process_file(src_path: Path, dest_root: Path) -> bool:
    try:
        original = src_path.read_text(encoding="utf-8")
    except Exception as e:
        print(f"[ERROR] Cannot read {src_path}: {e}", file=sys.stderr)
        return False

    cleaned = clean_content(original)
    # compute destination path
    relative = src_path.relative_to(SRC_ROOT)
    dest_path = dest_root / relative
    dest_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        dest_path.write_text(cleaned, encoding="utf-8")
    except Exception as e:
        print(f"[ERROR] Writing {dest_path}: {e}", file=sys.stderr)
        return False
    return True


def main():
    if not SRC_ROOT.is_dir():
        print(f"[FATAL] Source directory {SRC_ROOT} does not exist.", file=sys.stderr)
        sys.exit(1)

    md_files = list(SRC_ROOT.rglob("*.md"))
    if not md_files:
        print("No markdown files found under", SRC_ROOT)
        return

    DEST_ROOT.mkdir(exist_ok=True)
    modified = []
    for f in md_files:
        if process_file(f, DEST_ROOT):
            modified.append(f)

    print(
        f"Processed {len(md_files)} markdown files, wrote cleaned copies to '{DEST_ROOT}/'."
    )


if __name__ == "__main__":
    main()
