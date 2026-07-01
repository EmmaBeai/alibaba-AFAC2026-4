from __future__ import annotations

import argparse
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any


"""Build per-heading-level nested JSON.

`--content-dir` is the authoritative content source: heading keys and `text`
come from those cleaned files. `--page-content-dir` is only used to fill
`page_start` and `page_end` by matching the cleaned headings back to a parsed
source that still has `<!-- page:N -->` markers, such as pypdf output.
"""


PAGE_MARKER_RE = re.compile(r"<!--\s*page\s*:\s*(\d+)\s*-->", re.IGNORECASE)
MARKDOWN_HEADING_RE = re.compile(r"^\s{0,3}(#{1,6})\s+(.+?)\s*#*\s*$")


@dataclass(slots=True)
class _Line:
    text: str
    page: int | None
    heading_level: int | None = None
    heading: str = ""


@dataclass(slots=True)
class _HeadingNode:
    heading: str
    start_index: int
    parent_title: str


def build_heading_chunks(
    *,
    level: int,
    documents_path: Path | str = "processed_data/documents.jsonl",
    content_dir: Path | str = "processed_data/cleaned_markdown",
    page_content_dir: Path | str | None = None,
) -> dict[str, Any]:
    """Build nested chunks for one markdown heading level.

    `content_dir` controls headings and text. `page_content_dir`, when supplied,
    only supplies page markers for `page_start`/`page_end`.
    """
    if level < 2 or level > 6:
        raise ValueError(f"level must be between 2 and 6, got {level}")
    documents_path = Path(documents_path)
    content_dir = Path(content_dir)
    page_content_dir = Path(page_content_dir) if page_content_dir is not None else None
    documents = _read_documents(documents_path)
    payload_documents: list[dict[str, Any]] = []

    for document in documents:
        content_path = _content_path(content_dir, str(document["doc_id"]))
        lines = _parse_lines(content_path.read_text(encoding="utf-8", errors="replace"))
        page_lines = _page_lines_for(
            doc_id=str(document["doc_id"]),
            content_lines=lines,
            page_content_dir=page_content_dir,
        )
        title = str(document.get("title") or document["doc_id"])
        payload_documents.extend(
            _document_entries_for_level(
                document=document,
                lines=lines,
                page_lines=page_lines,
                level=level,
                root_title=title,
            )
        )

    return {
        "schema_version": f"h{level}_chunks.v1",
        "documents": payload_documents,
    }


def build_h2_chunks(
    *,
    documents_path: Path | str = "processed_data/documents.jsonl",
    content_dir: Path | str = "processed_data/cleaned_markdown",
    page_content_dir: Path | str | None = None,
) -> dict[str, Any]:
    return build_heading_chunks(
        level=2,
        documents_path=documents_path,
        content_dir=content_dir,
        page_content_dir=page_content_dir,
    )


def write_json(payload: dict[str, Any], output_path: Path | str) -> None:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def _read_documents(path: Path) -> list[dict[str, Any]]:
    documents: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                documents.append(json.loads(line))
    return documents


def _content_path(content_dir: Path, doc_id: str) -> Path:
    for suffix in (".md", ".txt"):
        path = content_dir / f"{doc_id}{suffix}"
        if path.exists() and path.is_file():
            return path
    raise FileNotFoundError(f"Missing cleaned content for {doc_id}: {content_dir / (doc_id + '.md')}")


def _page_lines_for(
    *,
    doc_id: str,
    content_lines: list[_Line],
    page_content_dir: Path | None,
) -> list[_Line]:
    if page_content_dir is None:
        return content_lines
    for suffix in (".md", ".txt"):
        path = page_content_dir / f"{doc_id}{suffix}"
        if path.exists() and path.is_file():
            return _parse_lines(path.read_text(encoding="utf-8", errors="replace"))
    return content_lines


def _parse_lines(markdown: str) -> list[_Line]:
    lines: list[_Line] = []
    current_page: int | None = None
    for raw_line in markdown.replace("\r\n", "\n").replace("\r", "\n").split("\n"):
        page_match = PAGE_MARKER_RE.search(raw_line)
        if page_match:
            current_page = int(page_match.group(1))
            continue
        text = raw_line.rstrip()
        heading_match = MARKDOWN_HEADING_RE.match(text.strip())
        if heading_match:
            level = len(heading_match.group(1))
            heading = _clean_heading(heading_match.group(2))
            lines.append(_Line(text=text, page=current_page, heading_level=level, heading=heading))
        else:
            lines.append(_Line(text=text, page=current_page))
    return lines


def _document_entries_for_level(
    *,
    document: dict[str, Any],
    lines: list[_Line],
    page_lines: list[_Line],
    level: int,
    root_title: str,
) -> list[dict[str, Any]]:
    groups = _heading_groups(lines, level=level, root_title=root_title)
    page_ranges = _page_ranges_for_level(lines, page_lines, level=level)
    entries: list[dict[str, Any]] = []
    for parent_title, nodes in groups:
        tree: dict[str, dict[str, Any]] = {}
        used_keys: dict[str, int] = {}
        for order, node in enumerate(nodes, start=1):
            next_boundary = _next_section_boundary(lines, node.start_index, level=level)
            page_start, page_end = page_ranges.get(node.start_index, (None, None))
            key = _unique_key(node.heading, used_keys)
            tree[key] = {
                "heading": node.heading,
                "level": level,
                "order": order,
                "page_start": page_start,
                "page_end": page_end,
                "text": _direct_text_until_next_heading(lines[node.start_index + 1:next_boundary]),
            }
        if tree:
            entries.append(
                {
                    "doc_id": document["doc_id"],
                    "domain": document.get("domain", ""),
                    "title": parent_title,
                    "tree": {
                        parent_title: tree,
                    },
                }
            )
    return entries


def _heading_groups(
    lines: list[_Line],
    *,
    level: int,
    root_title: str,
) -> list[tuple[str, list[_HeadingNode]]]:
    groups: list[tuple[str, list[_HeadingNode]]] = []
    root_group: list[_HeadingNode] = []
    parent_groups: dict[int, list[_HeadingNode]] = {}
    heading_stack: dict[int, tuple[str, int]] = {}
    next_parent_id = 1

    for index, line in enumerate(lines):
        if line.heading_level is None or not line.heading:
            continue

        for existing_level in list(heading_stack):
            if existing_level >= line.heading_level:
                del heading_stack[existing_level]

        if line.heading_level == level:
            if level == 2:
                root_group.append(_HeadingNode(line.heading, index, root_title))
            else:
                parent = heading_stack.get(level - 1)
                if parent is not None:
                    parent_title, parent_id = parent
                    parent_groups.setdefault(parent_id, []).append(
                        _HeadingNode(line.heading, index, parent_title)
                    )

        heading_stack[line.heading_level] = (line.heading, next_parent_id)
        if line.heading_level == level - 1:
            groups.append((line.heading, parent_groups.setdefault(next_parent_id, [])))
        next_parent_id += 1

    if level == 2:
        groups.append((root_title, root_group))
    return [(title, nodes) for title, nodes in groups if nodes]


def _page_ranges_for_level(
    content_lines: list[_Line],
    page_lines: list[_Line],
    *,
    level: int,
) -> dict[int, tuple[int | None, int | None]]:
    target_indexes = [
        index
        for index, line in enumerate(content_lines)
        if line.heading_level == level and line.heading
    ]
    if content_lines is page_lines:
        return {
            index: _page_range(content_lines[index:_next_section_boundary(content_lines, index, level=level)])
            for index in target_indexes
        }

    boundary_indexes = [
        index
        for index, line in enumerate(content_lines)
        if line.heading_level is not None and line.heading_level <= level and line.heading
    ]
    matched_boundary_indexes: dict[int, int | None] = {}
    search_start = 0
    for index in boundary_indexes:
        match = _find_heading_in_page_lines(page_lines, content_lines[index].heading, search_start)
        matched_boundary_indexes[index] = match
        if match is not None:
            search_start = match + 1

    ranges: dict[int, tuple[int | None, int | None]] = {}
    for index in target_indexes:
        start = matched_boundary_indexes.get(index)
        if start is None:
            ranges[index] = (None, None)
            continue
        next_boundary = next((boundary for boundary in boundary_indexes if boundary > index), None)
        next_start = (
            _next_matched_boundary(boundary_indexes, matched_boundary_indexes, after=index)
            if next_boundary is not None
            else None
        )
        end = next_start if next_start is not None else len(page_lines)
        ranges[index] = _page_range(page_lines[start:end])
    return ranges


def _next_matched_boundary(
    boundary_indexes: list[int],
    matched_boundary_indexes: dict[int, int | None],
    *,
    after: int,
) -> int | None:
    for boundary in boundary_indexes:
        if boundary <= after:
            continue
        match = matched_boundary_indexes.get(boundary)
        if match is not None:
            return match
    return None


def _next_section_boundary(lines: list[_Line], start_index: int, *, level: int) -> int:
    for index in range(start_index + 1, len(lines)):
        line = lines[index]
        if line.heading_level is not None and line.heading_level <= level:
            return index
    return len(lines)


def _find_heading_in_page_lines(page_lines: list[_Line], heading: str, start: int) -> int | None:
    target = _normalize_for_match(heading)
    if not target:
        return None
    for index in range(start, len(page_lines)):
        line = page_lines[index]
        candidate_text = line.heading if line.heading else line.text
        candidate = _normalize_for_match(candidate_text)
        if candidate == target or target in candidate:
            return index
    return None


def _direct_text_until_next_heading(lines: list[_Line]) -> str:
    direct: list[str] = []
    for line in lines:
        if line.heading_level is not None:
            break
        direct.append(line.text)
    return _trim_blank_lines(direct)


def _trim_blank_lines(lines: list[str]) -> str:
    start = 0
    end = len(lines)
    while start < end and not lines[start].strip():
        start += 1
    while end > start and not lines[end - 1].strip():
        end -= 1
    return "\n".join(lines[start:end]).strip()


def _page_range(lines: list[_Line]) -> tuple[int | None, int | None]:
    pages = [line.page for line in lines if line.page is not None]
    if not pages:
        return None, None
    return min(pages), max(pages)


def _clean_heading(value: str) -> str:
    value = re.sub(r"<[^>]+>", "", value).strip()
    value = re.sub(r"!\[[^\]]*]\([^)]*\)", "", value).strip()
    value = value.strip("#").strip()
    return re.sub(r"\s+", " ", value)


def _normalize_for_match(value: str) -> str:
    value = _clean_heading(value)
    value = re.sub(r"\s+", "", value)
    value = value.strip("：:.-—_·•")
    return value


def _unique_key(heading: str, used_keys: dict[str, int]) -> str:
    count = used_keys.get(heading, 0) + 1
    used_keys[heading] = count
    if count == 1:
        return heading
    return f"{heading} #{count}"


def main() -> None:
    parser = argparse.ArgumentParser(description="Build nested heading-level chunk JSON from cleaned markdown files.")
    parser.add_argument("--documents", default="processed_data/documents.jsonl")
    parser.add_argument(
        "--content-dir",
        default="processed_data/cleaned_markdown",
        help="Cleaned files used for headings and direct heading text.",
    )
    parser.add_argument(
        "--page-content-dir",
        default="processed_data/pdf_parsed/pypdf",
        help="Parsed files used only for page_start/page_end markers; empty disables page backfill.",
    )
    parser.add_argument("--level", type=int, default=2, choices=range(2, 7), help="Heading level to export.")
    parser.add_argument("--all-levels", action="store_true", help="Write h2 through h6 chunk JSON files.")
    parser.add_argument("--output", default="", help="Output file for one level. Defaults to processed_data/hN_chunks.json.")
    parser.add_argument("--output-dir", default="processed_data", help="Output directory for --all-levels.")
    args = parser.parse_args()

    if args.all_levels:
        output_dir = Path(args.output_dir)
        for level in range(2, 7):
            payload = build_heading_chunks(
                level=level,
                documents_path=args.documents,
                content_dir=args.content_dir,
                page_content_dir=args.page_content_dir or None,
            )
            output_path = output_dir / f"h{level}_chunks.json"
            write_json(payload, output_path)
            print(f"wrote {len(payload['documents'])} documents to {output_path}")
        return

    output_path = Path(args.output or f"processed_data/h{args.level}_chunks.json")
    payload = build_heading_chunks(
        level=args.level,
        documents_path=args.documents,
        content_dir=args.content_dir,
        page_content_dir=args.page_content_dir or None,
    )
    write_json(payload, output_path)
    print(f"wrote {len(payload['documents'])} documents to {output_path}")


if __name__ == "__main__":
    main()
