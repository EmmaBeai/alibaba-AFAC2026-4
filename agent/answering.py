"""Single-question answering orchestration.

STUB: reverted to not-implemented for refactor.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from agent.qwen_client import QwenClient
from agent.retrieval import LexicalRetriever, SearchHit
from agent.schema import Question


@dataclass(frozen=True)
class AnswerResult:
    qid: str
    answer: str
    prompt_tokens: int
    completion_tokens: int
    evidence: dict[str, Any]

    @property
    def total_tokens(self) -> int:
        raise NotImplementedError


def answer_question(
    question: Question,
    retriever: LexicalRetriever,
    client: QwenClient,
    top_k: int,
    candidate_docs: int,
    max_context_chars: int,
) -> AnswerResult:
    raise NotImplementedError


def build_evidence_blocks(hits: list[SearchHit], max_context_chars: int) -> list[str]:
    raise NotImplementedError
