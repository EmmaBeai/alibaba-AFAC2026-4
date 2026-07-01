from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from preprocess.build_h2_chunks import (
    _content_path,
    _direct_text_until_next_heading,
    _next_section_boundary,
    _page_lines_for,
    _page_range,
    _page_ranges_for_level,
    _parse_lines,
    _read_documents,
)


SCHEMA_VERSION = "structure_nodes.v1"


def build_structure_nodes(
    *,
    documents_path: Path | str = "processed_data/documents.jsonl",
    content_dir: Path | str = "processed_data/cleaned_markdown",
    page_content_dir: Path | str | None = None,
) -> list[dict[str, Any]]:
    """Build one parent-linked document/section node list from cleaned markdown."""
    documents_path = Path(documents_path)
    content_dir = Path(content_dir)
    page_content_dir = Path(page_content_dir) if page_content_dir is not None else None
    rows: list[dict[str, Any]] = []

    for document in _read_documents(documents_path):
        doc_id = str(document["doc_id"])
        domain = str(document.get("domain", ""))
        title = str(document.get("title") or doc_id)
        content_path = _content_path(content_dir, doc_id)
        lines = _parse_lines(content_path.read_text(encoding="utf-8", errors="replace"))
        page_lines = _page_lines_for(
            doc_id=doc_id,
            content_lines=lines,
            page_content_dir=page_content_dir,
        )
        rows.extend(_document_structure_nodes(doc_id, domain, title, lines, page_lines))

    return rows


def write_jsonl(rows: list[dict[str, Any]], output_path: Path | str) -> None:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )


def _document_structure_nodes(doc_id: str, domain: str, title: str, lines, page_lines) -> list[dict[str, Any]]:
    doc_node_id = f"{doc_id}::doc"
    page_start, page_end = _page_range(page_lines)
    doc_node = {
        "schema_version": SCHEMA_VERSION,
        "node_id": doc_node_id,
        "parent_id": None,
        "doc_id": doc_id,
        "domain": domain,
        "kind": "document",
        "heading_level": 1,
        "title": title,
        "path": [title],
        "order": 1,
        "page_start": page_start,
        "page_end": page_end,
        "text": _document_direct_text(lines),
        "child_ids": [],
    }
    rows = [doc_node]
    nodes_by_id = {doc_node_id: doc_node}
    open_node_ids: dict[int, str] = {1: doc_node_id}
    open_paths: dict[int, list[str]] = {1: [title]}
    sibling_orders: dict[str, int] = {doc_node_id: 0}
    level_counts = {level: 0 for level in range(2, 7)}
    page_ranges_by_level = {
        level: _page_ranges_for_level(lines, page_lines, level=level)
        for level in range(2, 7)
    }

    for index, line in enumerate(lines):
        level = line.heading_level
        if level is None or level < 2 or level > 6 or not line.heading:
            continue
        for existing_level in list(open_node_ids):
            if existing_level >= level:
                del open_node_ids[existing_level]
                del open_paths[existing_level]
        parent_level = max((candidate for candidate in open_node_ids if candidate < level), default=1)
        parent_id = open_node_ids.get(parent_level, doc_node_id)
        parent_path = open_paths.get(parent_level, [title])
        level_counts[level] += 1
        node_id = f"{doc_id}::h{level}::{level_counts[level]:04d}"
        sibling_orders[parent_id] = sibling_orders.get(parent_id, 0) + 1
        page_start, page_end = page_ranges_by_level[level].get(index, (None, None))
        next_boundary = _next_section_boundary(lines, index, level=level)
        path = [*parent_path, line.heading]
        node = {
            "schema_version": SCHEMA_VERSION,
            "node_id": node_id,
            "parent_id": parent_id,
            "doc_id": doc_id,
            "domain": domain,
            "kind": "section",
            "heading_level": level,
            "title": line.heading,
            "path": path,
            "order": sibling_orders[parent_id],
            "page_start": page_start,
            "page_end": page_end,
            "text": _direct_text_until_next_heading(lines[index + 1:next_boundary]),
            "child_ids": [],
        }
        nodes_by_id[parent_id]["child_ids"].append(node_id)
        rows.append(node)
        nodes_by_id[node_id] = node
        open_node_ids[level] = node_id
        open_paths[level] = path
        sibling_orders[node_id] = 0

    return rows


def _document_direct_text(lines) -> str:
    for index, line in enumerate(lines):
        if line.heading_level == 1 and line.heading:
            return _direct_text_until_next_heading(lines[index + 1:])
    return _direct_text_until_next_heading(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="Build unified parent-linked structure nodes.")
    parser.add_argument("--documents", default="processed_data/documents.jsonl")
    parser.add_argument("--content-dir", default="processed_data/cleaned_markdown")
    parser.add_argument("--page-content-dir", default="processed_data/pdf_parsed/pypdf")
    parser.add_argument("--output", default="processed_data/structure_nodes.jsonl")
    args = parser.parse_args()

    rows = build_structure_nodes(
        documents_path=args.documents,
        content_dir=args.content_dir,
        page_content_dir=args.page_content_dir or None,
    )
    write_jsonl(rows, args.output)
    print(f"wrote {len(rows)} structure nodes to {args.output}")


if __name__ == "__main__":
    main()
