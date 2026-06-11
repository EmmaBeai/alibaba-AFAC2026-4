"""Core data structures for the QA agent.

STUB: reverted to not-implemented for refactor. Fields below are a provisional
contract reference only — redesign as needed. All parsing/serialization logic is
gutted.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class DocumentMeta:
    doc_id: str
    path: str
    title: str = ""
    domain: str = ""

    @classmethod
    def from_dict(cls, row: dict[str, Any]) -> "DocumentMeta":
        raise NotImplementedError


@dataclass(frozen=True)
class Question:
    qid: str
    domain: str
    split: str
    question: str
    options: dict[str, str]
    answer_format: str
    qtype: str = ""
    doc_ids: tuple[str, ...] = ()

    @classmethod
    def from_dict(cls, row: dict[str, Any]) -> "Question":
        raise NotImplementedError


@dataclass(frozen=True)
class ParsedDocument:
    doc_id: str
    title: str
    domain: str
    text: str


@dataclass(frozen=True)
class Chunk:
    chunk_id: str
    doc_id: str
    title: str
    domain: str
    text: str
    start: int
    end: int

    @classmethod
    def from_dict(cls, row: dict[str, Any]) -> "Chunk":
        raise NotImplementedError

    def to_dict(self) -> dict[str, Any]:
        raise NotImplementedError
