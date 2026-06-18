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
        return cls(
            doc_id=str(row["doc_id"]),
            path=str(row["path"]),
            title=str(row.get("title") or ""),
            domain=str(row.get("domain") or ""),
        )


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
        doc_ids = row.get("doc_ids") or ()
        return cls(
            qid=str(row["qid"]),
            domain=str(row["domain"]),
            split=str(row.get("split") or ""),
            question=str(row["question"]),
            options={str(k): str(v) for k, v in dict(row["options"]).items()},
            answer_format=str(row["answer_format"]),
            qtype=str(row.get("raw_type") or row.get("type") or row.get("qtype") or ""),
            doc_ids=tuple(str(x) for x in doc_ids),
        )


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
        return cls(
            chunk_id=str(row["chunk_id"]),
            doc_id=str(row["doc_id"]),
            title=str(row.get("title") or ""),
            domain=str(row.get("domain") or ""),
            text=str(row["text"]),
            start=int(row.get("start", 0)),
            end=int(row.get("end", 0)),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "chunk_id": self.chunk_id,
            "doc_id": self.doc_id,
            "title": self.title,
            "domain": self.domain,
            "text": self.text,
            "start": self.start,
            "end": self.end,
        }
