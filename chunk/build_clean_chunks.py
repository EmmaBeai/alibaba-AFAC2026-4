from __future__ import annotations

import argparse
import html
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable


PAGE_MARKER_RE = re.compile(r"<!--\s*page\s*:\s*(\d+)\s*-->", re.IGNORECASE)
NON_PAGE_COMMENT_RE = re.compile(r"<!--(?!\s*page\s*:).*?-->", re.IGNORECASE | re.DOTALL)
IMAGE_RE = re.compile(r"!\[[^\]]*]\([^)]*\)")
HTML_CELL_RE = re.compile(r"(?i)</\s*(?:td|th)\s*>\s*<\s*(?:td|th)[^>]*>")
HTML_BLOCK_RE = re.compile(r"(?i)</?\s*(?:br|div|p|table|tr|h[1-6]|li|section|article)[^>]*>")
HTML_TAG_RE = re.compile(r"<[^>]+>")
MARKDOWN_HEADING_RE = re.compile(r"^\s{0,3}(#{1,6})\s+(.+?)\s*#*\s*$")
SPACE_RE = re.compile(r"[ \t]+")
CONTROL_RE = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")
SENTENCE_SPLIT_RE = re.compile(r"(?<=[。！？；;.!?])")

SOURCE_KIND_BY_SUFFIX = {
    ".pdf": "pdf",
    ".html": "html",
    ".htm": "html",
    ".txt": "txt",
}


@dataclass(slots=True)
class _Source:
    path: Path
    parser: str
    source_kind: str


@dataclass(slots=True)
class _Segment:
    page: int | None
    text: str


@dataclass(slots=True)
class _Block:
    text: str
    page: int | None
    section_path: list[str]
    char_start: int
    char_end: int
    is_heading: bool = False


def build_clean_chunks(
    *,
    documents_path: Path | str = "processed_data/documents.jsonl",
    processed_root: Path | str = "processed_data",
    dataset_root: Path | str = "public_dataset_upload",
    pdf_parsers: Iterable[str] = ("pypdf", "glm-ocr"),
    target_chars: int = 1600,
    overlap_chars: int = 120,
) -> list[dict[str, Any]]:
    documents_path = Path(documents_path)
    processed_root = Path(processed_root)
    dataset_root = Path(dataset_root)
    documents = _read_documents(documents_path)
    raw_sources = _discover_raw_sources(dataset_root)

    chunks: list[dict[str, Any]] = []
    for document in documents:
        source = _select_source(document["doc_id"], processed_root, raw_sources, tuple(pdf_parsers))
        raw_text = source.path.read_text(encoding="utf-8", errors="replace")
        segments = _segments(raw_text, paged=source.source_kind == "pdf")
        blocks = list(_blocks_from_segments(segments))
        if not blocks:
            continue
        chunks.extend(
            _chunk_blocks(
                document=document,
                source=source,
                blocks=blocks,
                target_chars=target_chars,
                overlap_chars=overlap_chars,
            )
        )
    return chunks


def write_jsonl(rows: list[dict[str, Any]], output_path: Path | str) -> None:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")


def _read_documents(path: Path) -> list[dict[str, Any]]:
    documents: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                documents.append(json.loads(line))
    return documents


def _discover_raw_sources(dataset_root: Path) -> dict[str, Path]:
    raw_root = dataset_root / "raw"
    sources: dict[str, Path] = {}
    if not raw_root.exists():
        return sources
    for path in sorted(raw_root.rglob("*")):
        if not path.is_file() or path.suffix.lower() not in SOURCE_KIND_BY_SUFFIX:
            continue
        if path.stem in sources:
            raise ValueError(f"Duplicate raw document stem is ambiguous: {path.stem}")
        sources[path.stem] = path
    return sources


def _select_source(
    doc_id: str,
    processed_root: Path,
    raw_sources: dict[str, Path],
    pdf_parsers: tuple[str, ...],
) -> _Source:
    raw_path = raw_sources.get(doc_id)
    raw_kind = SOURCE_KIND_BY_SUFFIX.get(raw_path.suffix.lower(), "") if raw_path else ""
    text_path = processed_root / "text_html_parsed" / "texts" / f"{doc_id}.txt"

    if raw_kind in {"html", "txt"}:
        if _has_text(text_path):
            return _Source(path=text_path, parser="text_html", source_kind=raw_kind)
        raise FileNotFoundError(f"Missing parsed text/html file for {doc_id}: {text_path}")

    for parser in pdf_parsers:
        path = processed_root / "pdf_parsed" / parser / f"{doc_id}.md"
        if _has_text(path):
            return _Source(path=path, parser=parser, source_kind="pdf")

    if _has_text(text_path):
        return _Source(path=text_path, parser="text_html", source_kind=raw_kind or "text")

    searched = ", ".join(str(processed_root / "pdf_parsed" / parser / f"{doc_id}.md") for parser in pdf_parsers)
    raise FileNotFoundError(f"Missing parsed content for {doc_id}. Searched: {searched}; {text_path}")


def _has_text(path: Path) -> bool:
    return path.exists() and path.is_file() and bool(path.read_text(encoding="utf-8", errors="replace").strip())


def _segments(raw_text: str, *, paged: bool) -> list[_Segment]:
    if not paged:
        clean = _clean_text(raw_text)
        return [_Segment(page=None, text=clean)] if clean else []

    markers = list(PAGE_MARKER_RE.finditer(raw_text))
    if not markers:
        clean = _clean_text(raw_text)
        return [_Segment(page=None, text=clean)] if clean else []

    segments: list[_Segment] = []
    for index, marker in enumerate(markers):
        start = marker.end()
        end = markers[index + 1].start() if index + 1 < len(markers) else len(raw_text)
        clean = _clean_text(raw_text[start:end])
        if clean:
            segments.append(_Segment(page=int(marker.group(1)), text=clean))
    return segments


def _clean_text(text: str) -> str:
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = NON_PAGE_COMMENT_RE.sub("\n", text)
    text = IMAGE_RE.sub("", text)
    text = HTML_CELL_RE.sub(" | ", text)
    text = HTML_BLOCK_RE.sub("\n", text)
    text = HTML_TAG_RE.sub("", text)
    text = html.unescape(text)
    text = CONTROL_RE.sub("", text)

    cleaned_lines: list[str] = []
    previous_blank = False
    for line in text.splitlines():
        line = SPACE_RE.sub(" ", line).strip()
        if not line:
            if not previous_blank:
                cleaned_lines.append("")
            previous_blank = True
            continue
        cleaned_lines.append(line)
        previous_blank = False
    return "\n".join(cleaned_lines).strip()


def _blocks_from_segments(segments: list[_Segment]) -> Iterable[_Block]:
    section_path: list[str] = []
    cursor = 0
    for segment in segments:
        for paragraph in _paragraphs(segment.text):
            heading = _markdown_heading(paragraph)
            is_heading = heading is not None
            if heading is not None:
                level, title = heading
                section_path = section_path[: max(0, level - 1)] + [title]
            for piece in _split_long_text(paragraph):
                start = cursor
                end = start + len(piece)
                yield _Block(
                    text=piece,
                    page=segment.page,
                    section_path=list(section_path),
                    char_start=start,
                    char_end=end,
                    is_heading=is_heading,
                )
                cursor = end + 2


def _paragraphs(text: str) -> list[str]:
    paragraphs: list[str] = []
    buffer: list[str] = []

    def flush() -> None:
        if buffer:
            paragraphs.append("\n".join(buffer).strip())
            buffer.clear()

    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line:
            flush()
            continue
        if MARKDOWN_HEADING_RE.match(line):
            flush()
            paragraphs.append(line)
            continue
        buffer.append(line)
    flush()
    return [paragraph for paragraph in paragraphs if paragraph]


def _markdown_heading(text: str) -> tuple[int, str] | None:
    match = MARKDOWN_HEADING_RE.match(text)
    if not match:
        return None
    title = match.group(2).strip()
    return len(match.group(1)), title


def _split_long_text(text: str, *, max_chars: int = 2200) -> list[str]:
    if len(text) <= max_chars:
        return [text]
    raw_parts: list[str] = []
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        parts = [part.strip() for part in SENTENCE_SPLIT_RE.split(line) if part.strip()]
        raw_parts.extend(parts or [line])

    pieces: list[str] = []
    current = ""
    for part in raw_parts:
        if not current:
            current = part
            continue
        candidate = current + "\n" + part
        if len(candidate) > max_chars:
            pieces.append(current)
            current = part
        else:
            current = candidate
    if current:
        pieces.append(current)
    return pieces


def _chunk_blocks(
    *,
    document: dict[str, Any],
    source: _Source,
    blocks: list[_Block],
    target_chars: int,
    overlap_chars: int,
) -> list[dict[str, Any]]:
    chunks: list[dict[str, Any]] = []
    current: list[_Block] = []
    ordinal = 1

    def current_len() -> int:
        return sum(len(block.text) for block in current) + max(0, len(current) - 1)

    def flush(*, keep_overlap: bool) -> None:
        nonlocal current, ordinal
        if not current:
            return
        chunks.append(_make_chunk(document, source, current, ordinal))
        ordinal += 1
        if overlap_chars <= 0 or not keep_overlap:
            current = []
            return
        overlap: list[_Block] = []
        total = 0
        for block in reversed(current):
            if total and total + len(block.text) > overlap_chars:
                break
            overlap.insert(0, block)
            total += len(block.text)
        current = overlap

    for block in blocks:
        if block.is_heading and current:
            flush(keep_overlap=False)
        elif current and current_len() + len(block.text) + 1 > target_chars:
            flush(keep_overlap=True)
        current.append(block)
    flush(keep_overlap=False)
    return chunks


def _make_chunk(
    document: dict[str, Any],
    source: _Source,
    blocks: list[_Block],
    ordinal: int,
) -> dict[str, Any]:
    pages = [block.page for block in blocks if block.page is not None]
    page_start = min(pages) if pages else None
    page_end = max(pages) if pages else None
    section_path = next((block.section_path for block in blocks if block.section_path), [])
    text = "\n".join(block.text for block in blocks).strip()
    return {
        "schema_version": "clean_chunks.v1",
        "chunk_id": _chunk_id(str(document["doc_id"]), page_start, ordinal),
        "doc_id": document["doc_id"],
        "domain": document.get("domain", ""),
        "title": document.get("title", ""),
        "source_kind": source.source_kind,
        "parser": source.parser,
        "page_start": page_start,
        "page_end": page_end,
        "section_path": section_path,
        "text": text,
        "char_start": blocks[0].char_start,
        "char_end": blocks[-1].char_end,
    }


def _chunk_id(doc_id: str, page_start: int | None, ordinal: int) -> str:
    if page_start is None:
        return f"{doc_id}::c{ordinal:04d}"
    return f"{doc_id}::p{page_start:04d}::c{ordinal:04d}"


def main() -> None:
    parser = argparse.ArgumentParser(description="Build clean chunks for allowlisted AFAC documents.")
    parser.add_argument("--documents", default="processed_data/documents.jsonl")
    parser.add_argument("--processed-root", default="processed_data")
    parser.add_argument("--dataset-root", default="public_dataset_upload")
    parser.add_argument("--output", default="processed_data/clean_chunks.jsonl")
    parser.add_argument("--pdf-parsers", default="pypdf,glm-ocr")
    parser.add_argument("--target-chars", type=int, default=1600)
    parser.add_argument("--overlap-chars", type=int, default=120)
    args = parser.parse_args()

    chunks = build_clean_chunks(
        documents_path=args.documents,
        processed_root=args.processed_root,
        dataset_root=args.dataset_root,
        pdf_parsers=tuple(part.strip() for part in args.pdf_parsers.split(",") if part.strip()),
        target_chars=args.target_chars,
        overlap_chars=args.overlap_chars,
    )
    write_jsonl(chunks, args.output)
    print(f"wrote {len(chunks)} chunks to {args.output}")


if __name__ == "__main__":
    main()
