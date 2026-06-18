"""Text normalization and chunk construction."""

from __future__ import annotations

import re

from agent.schema import Chunk, ParsedDocument

_CONTROL_CHARS_RE = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")


def normalize_text(text: str) -> str:
    """Normalize parser output without destroying Markdown structure."""
    if not text:
        return ""

    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = _CONTROL_CHARS_RE.sub("", text)
    lines = [line.rstrip() for line in text.split("\n")]
    text = "\n".join(lines).strip()
    text = re.sub(r"\n{4,}", "\n\n\n", text)
    return text


def make_chunks(
    doc: ParsedDocument,
    chunk_chars: int = 1800,
    overlap_chars: int = 240,
) -> list[Chunk]:
    """Split a parsed document into retrieval chunks.

    The splitter is intentionally conservative: it prefers paragraph boundaries,
    keeps Markdown tables intact when they fit, and falls back to character
    windows only for very long paragraphs.
    """
    if chunk_chars <= 0:
        raise ValueError("chunk_chars must be positive")
    if overlap_chars < 0:
        raise ValueError("overlap_chars must be non-negative")
    if overlap_chars >= chunk_chars:
        raise ValueError("overlap_chars must be smaller than chunk_chars")

    text = normalize_text(doc.text)
    if not text:
        return []

    chunks: list[Chunk] = []
    current_parts: list[str] = []
    current_start = 0
    current_len = 0

    for start, part in _iter_blocks(text):
        if len(part) > chunk_chars:
            _flush_chunk(chunks, doc, current_parts, current_start, current_len)
            current_parts = []
            current_len = 0
            for win_start, win_text in _window_text(part, start, chunk_chars, overlap_chars):
                _append_chunk(chunks, doc, win_text, win_start)
            current_start = start + len(part)
            continue

        separator_len = 2 if current_parts else 0
        projected_len = current_len + separator_len + len(part)
        if current_parts and projected_len > chunk_chars:
            _flush_chunk(chunks, doc, current_parts, current_start, current_len)
            tail = _overlap_tail("\n\n".join(current_parts), overlap_chars)
            current_parts = [tail, part] if tail else [part]
            current_start = max(start - len(tail) - 2, 0) if tail else start
            current_len = len("\n\n".join(current_parts))
        else:
            if not current_parts:
                current_start = start
            current_parts.append(part)
            current_len = projected_len

    _flush_chunk(chunks, doc, current_parts, current_start, current_len)
    return chunks


def _iter_blocks(text: str) -> list[tuple[int, str]]:
    blocks: list[tuple[int, str]] = []
    for match in re.finditer(r"\S(?:.*?)(?=\n{2,}\S|\Z)", text, flags=re.S):
        block = match.group(0).strip()
        if block:
            blocks.append((match.start(), block))
    return blocks


def _window_text(
    text: str,
    absolute_start: int,
    chunk_chars: int,
    overlap_chars: int,
) -> list[tuple[int, str]]:
    windows: list[tuple[int, str]] = []
    step = chunk_chars - overlap_chars
    offset = 0
    while offset < len(text):
        piece = text[offset : offset + chunk_chars].strip()
        if piece:
            windows.append((absolute_start + offset, piece))
        if offset + chunk_chars >= len(text):
            break
        offset += step
    return windows


def _overlap_tail(text: str, overlap_chars: int) -> str:
    if overlap_chars <= 0 or not text:
        return ""
    tail = text[-overlap_chars:].strip()
    newline = tail.find("\n")
    if newline > 0:
        tail = tail[newline + 1 :].strip()
    return tail


def _flush_chunk(
    chunks: list[Chunk],
    doc: ParsedDocument,
    parts: list[str],
    start: int,
    current_len: int,
) -> None:
    if not parts or current_len <= 0:
        return
    _append_chunk(chunks, doc, "\n\n".join(parts), start)


def _append_chunk(
    chunks: list[Chunk],
    doc: ParsedDocument,
    text: str,
    start: int,
) -> None:
    text = text.strip()
    if not text:
        return
    idx = len(chunks) + 1
    chunks.append(
        Chunk(
            chunk_id=f"{doc.doc_id}::{idx:04d}",
            doc_id=doc.doc_id,
            title=doc.title,
            domain=doc.domain,
            text=text,
            start=start,
            end=start + len(text),
        )
    )
