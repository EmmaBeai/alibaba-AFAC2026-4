"""First-pass lexical retrieval.

STUB: reverted to not-implemented for refactor.
"""

from __future__ import annotations

from dataclasses import dataclass

from agent.schema import Chunk, Question


def tokenize(text: str) -> list[str]:
    raise NotImplementedError


@dataclass(frozen=True)
class SearchHit:
    chunk: Chunk
    score: float


class LexicalRetriever:
    def __init__(self, chunks: list[Chunk]) -> None:
        raise NotImplementedError

    def search(
        self,
        question: Question,
        top_k: int,
        allowed_doc_ids: set[str] | None = None,
    ) -> list[SearchHit]:
        raise NotImplementedError

    def candidate_doc_ids(self, question: Question, limit: int) -> list[str]:
        raise NotImplementedError


def build_query_text(question: Question) -> str:
    raise NotImplementedError
