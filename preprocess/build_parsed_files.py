"""Materialize allowlisted parsed files used by heading cleaning.

This is the bridge between raw/parser outputs and the latest header-based chunk
pipeline. PDF content uses the production OCR markdown (default: ``glm-ocr``);
HTML/TXT content uses the existing lightweight parsers in this package.
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path
from typing import Any

from preprocess import html_parser, pdf_parser, txt_parser

SOURCE_KIND_BY_SUFFIX = {
    ".pdf": "pdf",
    ".html": "html",
    ".htm": "html",
    ".txt": "txt",
}


def build_parsed_files(
    *,
    document_sources_path: Path | str = "processed_data/document_sources.jsonl",
    processed_root: Path | str = "processed_data",
    output_dir: Path | str = "processed_data/parsed_files",
    pdf_model: str = "glm-ocr",
    overwrite: bool = False,
) -> list[dict[str, Any]]:
    document_sources = _read_jsonl(Path(document_sources_path))
    processed_root = Path(processed_root)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, Any]] = []
    for document in document_sources:
        doc_id = str(document["doc_id"])
        raw_path = Path(str(document["source_path"]))
        if not raw_path.exists():
            raise FileNotFoundError(f"Missing raw source for doc_id={doc_id}: {raw_path}")

        suffix = raw_path.suffix.lower()
        source_kind = str(document.get("source_kind") or SOURCE_KIND_BY_SUFFIX[suffix])
        if source_kind == "pdf":
            text = pdf_parser.parse(
                raw_path,
                model=pdf_model,
                parsed_dir=processed_root / "pdf_parsed",
            )
            output_path = output_dir / f"{doc_id}.md"
            parser = pdf_model
        elif source_kind == "html":
            text = html_parser.parse(raw_path)
            output_path = output_dir / f"{doc_id}.txt"
            parser = "html_parser"
        else:
            text = txt_parser.parse(raw_path)
            output_path = output_dir / f"{doc_id}.txt"
            parser = "txt_parser"

        if output_path.exists() and not overwrite:
            rows.append(_row(document, raw_path, output_path, source_kind, parser, skipped=True))
            continue

        output_path.write_text(text.rstrip() + "\n", encoding="utf-8")
        rows.append(_row(document, raw_path, output_path, source_kind, parser, skipped=False))

    manifest_path = output_dir / "_manifest.jsonl"
    _write_jsonl(manifest_path, rows)
    return rows


def sync_text_html_cache(
    *,
    parsed_files_dir: Path | str = "processed_data/parsed_files",
    processed_root: Path | str = "processed_data",
) -> None:
    """Keep the legacy text/html cache available for tools that still inspect it."""
    parsed_files_dir = Path(parsed_files_dir)
    text_dir = Path(processed_root) / "text_html_parsed" / "texts"
    text_dir.mkdir(parents=True, exist_ok=True)
    for source in parsed_files_dir.glob("*.txt"):
        shutil.copy2(source, text_dir / source.name)


def _row(
    document: dict[str, Any],
    raw_path: Path,
    output_path: Path,
    source_kind: str,
    parser: str,
    *,
    skipped: bool,
) -> dict[str, Any]:
    return {
        "doc_id": document["doc_id"],
        "domain": document.get("domain", ""),
        "title": document.get("title", ""),
        "source_kind": source_kind,
        "parser": parser,
        "source_path": str(raw_path),
        "parsed_path": str(output_path),
        "skipped_existing": skipped,
    }


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Build parsed_files for heading cleaning.")
    parser.add_argument("--document-sources", default="processed_data/document_sources.jsonl")
    parser.add_argument("--processed-root", default="processed_data")
    parser.add_argument("--output-dir", default="processed_data/parsed_files")
    parser.add_argument("--pdf-model", default="glm-ocr")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--sync-text-html-cache", action="store_true")
    args = parser.parse_args()

    rows = build_parsed_files(
        document_sources_path=args.document_sources,
        processed_root=args.processed_root,
        output_dir=args.output_dir,
        pdf_model=args.pdf_model,
        overwrite=args.overwrite,
    )
    if args.sync_text_html_cache:
        sync_text_html_cache(parsed_files_dir=args.output_dir, processed_root=args.processed_root)
    print(f"wrote {len(rows)} parsed files to {args.output_dir}")


if __name__ == "__main__":
    main()
