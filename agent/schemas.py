from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any


@dataclass(slots=True)
class Document:
    doc_id: str
    domain: str
    path: Path
    title: str


@dataclass(slots=True)
class Question:
    qid: str
    domain: str
    split: str
    question: str
    options: dict[str, str]
    answer_format: str
    question_type: str
    doc_ids: list[str] = field(default_factory=list)

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "Question":
        return cls(
            qid=value["qid"],
            domain=value["domain"],
            split=value["split"],
            question=value["question"],
            options=value["options"],
            answer_format=value["answer_format"],
            question_type=value.get("type", ""),
            doc_ids=value.get("doc_ids", []),
        )


@dataclass(slots=True)
class Page:
    page_number: int
    text: str


@dataclass(slots=True)
class ExtractedPages:
    pages: list[Page]
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class IndexNode:
    node_id: str
    title: str
    start_page: int
    end_page: int
    children: list["IndexNode"] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "IndexNode":
        return cls(
            node_id=value["node_id"],
            title=value["title"],
            start_page=value["start_page"],
            end_page=value["end_page"],
            children=[cls.from_dict(item) for item in value.get("children", [])],
        )


@dataclass(slots=True)
class TokenUsage:
    prompt_tokens: int = 0
    completion_tokens: int = 0

    @property
    def total_tokens(self) -> int:
        return self.prompt_tokens + self.completion_tokens

    def add(self, prompt_tokens: int, completion_tokens: int) -> None:
        self.prompt_tokens += prompt_tokens
        self.completion_tokens += completion_tokens


@dataclass(slots=True)
class AnswerResult:
    qid: str
    answer: str
    evidence_retrieval: list[dict[str, Any]]
    usage: TokenUsage
