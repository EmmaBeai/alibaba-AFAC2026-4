"""Materialize raw HTML/TXT documents into text files.

The parser modules in this package expose ``parse(path) -> str``. This entrypoint
applies them to every raw ``.html/.htm/.txt`` file under ``public_dataset_upload``
and writes the legacy text cache under ``processed_data/text_html_parsed``. The
header-based pipeline primarily uses ``build_parsed_files.py``.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from preprocess import html_parser, txt_parser

SUPPORTED_SUFFIXES = {".html", ".htm", ".txt"}


def materialize_text_html(
    *,
    dataset_root: Path | str = "public_dataset_upload",
    processed_root: Path | str = "processed_data",
) -> list[dict[str, Any]]:
    dataset_root = Path(dataset_root)
    processed_root = Path(processed_root)
    out_dir = processed_root / "text_html_parsed" / "texts"
    out_dir.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, Any]] = []
    for raw_path in _discover_text_html(dataset_root):
        suffix = raw_path.suffix.lower()
        if suffix in {".html", ".htm"}:
            text = html_parser.parse(raw_path)
            title = html_parser.extract_title(raw_path)
            source_kind = "html"
            parser = "html_parser"
        else:
            text = txt_parser.parse(raw_path)
            title = _first_nonempty_line(text)
            source_kind = "txt"
            parser = "txt_parser"

        output_path = out_dir / f"{raw_path.stem}.txt"
        output_path.write_text(text.rstrip() + "\n", encoding="utf-8")
        row: dict[str, Any] = {
            "doc_id": raw_path.stem,
            "source_path": str(raw_path),
            "source_kind": source_kind,
            "parser": parser,
            "text_path": str(output_path),
        }
        if title:
            row["title"] = title
        rows.append(row)

    metadata_path = processed_root / "text_html_parsed" / "documents.jsonl"
    with metadata_path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")
    return rows


def _discover_text_html(dataset_root: Path) -> list[Path]:
    raw_root = dataset_root / "raw"
    if not raw_root.exists():
        return []
    return sorted(
        path
        for path in raw_root.rglob("*")
        if path.is_file() and path.suffix.lower() in SUPPORTED_SUFFIXES
    )


def _first_nonempty_line(text: str) -> str | None:
    for line in text.splitlines():
        stripped = line.strip()
        if stripped:
            return stripped
    return None


def main() -> None:
    parser = argparse.ArgumentParser(description="Materialize raw HTML/TXT files as text.")
    parser.add_argument("--dataset-root", default="public_dataset_upload")
    parser.add_argument("--processed-root", default="processed_data")
    args = parser.parse_args()

    rows = materialize_text_html(
        dataset_root=args.dataset_root,
        processed_root=args.processed_root,
    )
    print(f"wrote {len(rows)} text/html parses under {args.processed_root}/text_html_parsed")


if __name__ == "__main__":
    main()
