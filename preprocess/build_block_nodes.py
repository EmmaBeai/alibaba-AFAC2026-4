from __future__ import annotations

import argparse
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any


SCHEMA_VERSION = "block_nodes.v1"
IMAGE_RE = re.compile(r"^\s*!\[([^\]]*)]\(([^)]+)\)\s*$")
TABLE_START_RE = re.compile(r"^\s*<table\b", re.IGNORECASE)
TABLE_END_RE = re.compile(r"</table\s*>", re.IGNORECASE)
NUMBERED_START_RE = re.compile(
    r"^\s*(?:"
    r"[（(]?[一二三四五六七八九十百千万\d]+[）).、]"
    r"|第[一二三四五六七八九十百千万\d]+[章节条款项]"
    r"|[A-Za-z][.)、]"
    r")"
)
SENTENCE_ENDINGS = set("。！？!?；;：:.)）】]》”’\"'")


@dataclass(slots=True)
class _Block:
    kind: str
    text: str
    image_alt: str = ""
    image_path: str = ""
    image_relative_path: str = ""
    page: int | None = None


@dataclass(frozen=True, slots=True)
class _ImageOccurrence:
    relative_path: str
    absolute_path: str
    page: int


def build_block_nodes(
    *,
    structure_nodes_path: Path | str = "processed_data/structure_nodes.jsonl",
    structure_nodes: list[dict[str, Any]] | None = None,
    raw_glm_ocr_dir: Path | str | None = "processed_data/pdf_parsed/_raw/glm-ocr",
) -> list[dict[str, Any]]:
    """Build paragraph/table/image block nodes under section nodes."""
    rows = structure_nodes if structure_nodes is not None else _read_jsonl(Path(structure_nodes_path))
    block_nodes: list[dict[str, Any]] = []
    image_resolver = _ImageResolver(Path(raw_glm_ocr_dir)) if raw_glm_ocr_dir is not None else None

    for node in rows:
        if node.get("kind") != "section":
            continue
        blocks = _blocks_from_section_text(str(node.get("text") or ""))
        if image_resolver is not None:
            _resolve_block_images(
                str(node.get("doc_id") or ""),
                blocks,
                image_resolver,
                page_start=_int_or_none(node.get("page_start")),
                page_end=_int_or_none(node.get("page_end")),
            )
        for order, block in enumerate(blocks, start=1):
            block_nodes.append(_block_node(node, block, order=order))

    return block_nodes


def write_jsonl(rows: list[dict[str, Any]], output_path: Path | str) -> None:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _blocks_from_section_text(text: str) -> list[_Block]:
    return _merge_continuation_paragraphs(_initial_blocks_from_text(text))


class _ImageResolver:
    def __init__(self, raw_glm_ocr_dir: Path) -> None:
        self.raw_glm_ocr_dir = raw_glm_ocr_dir
        self._occurrences_by_doc: dict[str, list[_ImageOccurrence]] = {}
        self._next_index_by_doc: dict[str, int] = {}
        self._consumed_indexes_by_doc: dict[str, set[int]] = {}

    def next(
        self,
        doc_id: str,
        relative_path: str,
        *,
        page_start: int | None = None,
        page_end: int | None = None,
    ) -> _ImageOccurrence | None:
        occurrences = self._occurrences_for_doc(doc_id)
        start = self._next_index_by_doc.get(doc_id, 0)
        consumed = self._consumed_indexes_by_doc.setdefault(doc_id, set())
        search_start = 0 if page_start is not None or page_end is not None else start
        for index in range(search_start, len(occurrences)):
            if index in consumed:
                continue
            occurrence = occurrences[index]
            if not _same_image_reference(occurrence.relative_path, relative_path):
                continue
            if not _page_in_range(occurrence.page, page_start=page_start, page_end=page_end):
                continue
            consumed.add(index)
            self._next_index_by_doc[doc_id] = max(start, index + 1)
            return occurrence
        for index in range(0, len(occurrences)):
            if index not in consumed:
                continue
            occurrence = occurrences[index]
            if not _same_image_reference(occurrence.relative_path, relative_path):
                continue
            if not _page_in_range(occurrence.page, page_start=page_start, page_end=page_end):
                continue
            return occurrence
        return None

    def _occurrences_for_doc(self, doc_id: str) -> list[_ImageOccurrence]:
        if doc_id not in self._occurrences_by_doc:
            self._occurrences_by_doc[doc_id] = _read_raw_glm_ocr_image_occurrences(
                self.raw_glm_ocr_dir,
                doc_id,
            )
        return self._occurrences_by_doc[doc_id]


def _initial_blocks_from_text(text: str) -> list[_Block]:
    blocks: list[_Block] = []
    paragraph_lines: list[str] = []
    lines = text.replace("\r\n", "\n").replace("\r", "\n").split("\n")
    index = 0

    while index < len(lines):
        line = lines[index].rstrip()
        stripped = line.strip()
        if not stripped:
            _flush_paragraph(blocks, paragraph_lines)
            index += 1
            continue

        image_match = IMAGE_RE.match(stripped)
        if image_match:
            _flush_paragraph(blocks, paragraph_lines)
            blocks.append(
                _Block(
                    kind="image",
                    text=image_match.group(1).strip(),
                    image_alt=image_match.group(1).strip(),
                    image_path="",
                    image_relative_path=image_match.group(2).strip(),
                )
            )
            index += 1
            continue

        if TABLE_START_RE.match(stripped):
            _flush_paragraph(blocks, paragraph_lines)
            table_lines = [stripped]
            index += 1
            while index < len(lines) and not TABLE_END_RE.search(table_lines[-1]):
                table_lines.append(lines[index].strip())
                index += 1
            blocks.append(_Block(kind="table", text="\n".join(line for line in table_lines if line)))
            continue

        paragraph_lines.append(stripped)
        index += 1

    _flush_paragraph(blocks, paragraph_lines)
    return blocks


def _flush_paragraph(blocks: list[_Block], paragraph_lines: list[str]) -> None:
    if not paragraph_lines:
        return
    text = _join_text_lines(paragraph_lines)
    if text:
        blocks.append(_Block(kind="paragraph", text=text))
    paragraph_lines.clear()


def _merge_continuation_paragraphs(blocks: list[_Block]) -> list[_Block]:
    merged: list[_Block] = []
    for block in blocks:
        if (
            block.kind == "paragraph"
            and merged
            and merged[-1].kind == "paragraph"
            and _should_merge_paragraphs(merged[-1].text, block.text)
        ):
            merged[-1].text = _join_text_lines([merged[-1].text, block.text])
            continue
        merged.append(block)
    return merged


def _resolve_block_images(
    doc_id: str,
    blocks: list[_Block],
    image_resolver: _ImageResolver,
    *,
    page_start: int | None,
    page_end: int | None,
) -> None:
    for block in blocks:
        if block.kind != "image":
            continue
        occurrence = image_resolver.next(
            doc_id,
            block.image_relative_path or block.image_path,
            page_start=page_start,
            page_end=page_end,
        )
        if occurrence is None:
            continue
        block.image_path = occurrence.absolute_path
        block.page = occurrence.page


def _read_raw_glm_ocr_image_occurrences(raw_glm_ocr_dir: Path, doc_id: str) -> list[_ImageOccurrence]:
    output_dir = raw_glm_ocr_dir / doc_id / "glmocr_output"
    if not output_dir.exists():
        return []
    occurrences: list[_ImageOccurrence] = []
    for page_dir in sorted(output_dir.glob("page_*"), key=_page_dir_sort_key):
        if not page_dir.is_dir():
            continue
        page = _page_number_from_name(page_dir.name)
        if page is None:
            continue
        page_md = page_dir / f"{page_dir.name}.md"
        if not page_md.exists():
            continue
        for line in page_md.read_text(encoding="utf-8", errors="replace").splitlines():
            image_match = IMAGE_RE.match(line.strip())
            if not image_match:
                continue
            relative_path = image_match.group(2).strip()
            absolute_path = page_dir / relative_path
            if not absolute_path.exists():
                continue
            occurrences.append(
                _ImageOccurrence(
                    relative_path=relative_path,
                    absolute_path=str(absolute_path.resolve()),
                    page=page,
                )
            )
    return occurrences


def _page_dir_sort_key(path: Path) -> tuple[int, str]:
    page = _page_number_from_name(path.name)
    return (page if page is not None else 10**9, path.name)


def _page_number_from_name(value: str) -> int | None:
    match = re.fullmatch(r"page_(\d+)", value)
    if not match:
        return None
    return int(match.group(1))


def _same_image_reference(left: str, right: str) -> bool:
    return left == right or Path(left).name == Path(right).name


def _page_in_range(page: int, *, page_start: int | None, page_end: int | None) -> bool:
    if page_start is not None and page < page_start:
        return False
    if page_end is not None and page > page_end:
        return False
    return True


def _int_or_none(value: Any) -> int | None:
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _should_merge_paragraphs(previous: str, current: str) -> bool:
    previous = previous.strip()
    current = current.strip()
    if not previous or not current:
        return False
    if previous[-1] in SENTENCE_ENDINGS:
        return False
    if NUMBERED_START_RE.match(current):
        return False
    if IMAGE_RE.match(current) or TABLE_START_RE.match(current):
        return False
    return True


def _join_text_lines(lines: list[str]) -> str:
    text = ""
    for line in lines:
        line = line.strip()
        if not line:
            continue
        if not text:
            text = line
        elif _needs_space(text[-1], line[0]):
            text = f"{text} {line}"
        else:
            text = f"{text}{line}"
    return text


def _needs_space(left: str, right: str) -> bool:
    return left.isascii() and right.isascii() and left.isalnum() and right.isalnum()


def _block_node(section: dict[str, Any], block: _Block, *, order: int) -> dict[str, Any]:
    page_start = block.page if block.kind == "image" and block.page is not None else section.get("page_start")
    page_end = block.page if block.kind == "image" and block.page is not None else section.get("page_end")
    node = {
        "schema_version": SCHEMA_VERSION,
        "node_id": f"{section['node_id']}::block::{order:04d}",
        "parent_id": section["node_id"],
        "doc_id": section["doc_id"],
        "domain": section.get("domain", ""),
        "kind": block.kind,
        "path": list(section.get("path") or []),
        "order": order,
        "page_start": page_start,
        "page_end": page_end,
        "text": block.text,
        "child_ids": [],
    }
    if block.kind == "image":
        node["image_alt"] = block.image_alt
        node["image_path"] = block.image_path
        node["image_relative_path"] = block.image_relative_path
    return node


def main() -> None:
    parser = argparse.ArgumentParser(description="Build paragraph/table/image block nodes from structure nodes.")
    parser.add_argument("--structure-nodes", default="processed_data/structure_nodes.jsonl")
    parser.add_argument("--raw-glm-ocr-dir", default="processed_data/pdf_parsed/_raw/glm-ocr")
    parser.add_argument("--output", default="processed_data/block_nodes.jsonl")
    args = parser.parse_args()

    rows = build_block_nodes(
        structure_nodes_path=args.structure_nodes,
        raw_glm_ocr_dir=args.raw_glm_ocr_dir or None,
    )
    write_jsonl(rows, args.output)
    print(f"wrote {len(rows)} block nodes to {args.output}")


if __name__ == "__main__":
    main()
