"""Text normalization and chunk construction.

STUB: reverted to not-implemented for refactor.
"""

from __future__ import annotations

from agent.schema import Chunk, ParsedDocument


def normalize_text(text: str) -> str:
    raise NotImplementedError


def make_chunks(
    doc: ParsedDocument,
    chunk_chars: int = 1800,
    overlap_chars: int = 240,
) -> list[Chunk]:
    raise NotImplementedError
