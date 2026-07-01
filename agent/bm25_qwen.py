from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from agent.bm25_top1 import (
    BM25Candidate,
    BM25Top1Answerer,
    _BM25Index,
    _exact_overlap_bonus,
    _token_counter,
    _unit_id,
    _unit_text,
)
from agent.qwen_client import QwenCompletion, QwenPlusClient
from agent.schemas import AnswerResult, Document, Question, TokenUsage

ANSWER_SYSTEM = """你是金融长文档选择题助手。只能根据给定 BM25 证据回答。
输出必须是紧凑 JSON object，只包含 answer、reason、evidence_retrieval。
answer 只能由选项字母组成；多选题可以输出多个字母，例如 AC；不要输出选项文本。
reason 不超过 80 个汉字。
evidence_retrieval 最多 3 项，每项只保留 option、unit_id、doc_id、quote。quote 不超过 40 个汉字。"""


class BM25QwenAnswerer:
    def __init__(
        self,
        units_path: Path | str | None = None,
        *,
        qwen_client: QwenPlusClient | Any | None = None,
        chunk_chars: int = 1800,
        k1: float = 1.5,
        b: float = 0.75,
        max_evidence_chars: int = 12000,
        top_k_patches: int = 5,
    ):
        self.retriever = BM25Top1Answerer(
            units_path,
            chunk_chars=chunk_chars,
            k1=k1,
            b=b,
        )
        self.qwen_client = qwen_client or QwenPlusClient()
        self.max_evidence_chars = max_evidence_chars
        self.top_k_patches = top_k_patches

    def answer(self, question: Question, documents: list[Document]) -> AnswerResult:
        patches = self._rank_patches(question, documents)
        bm25_payload = self._bm25_payload(patches)
        fallback = str(bm25_payload.get("selected_option") or next(iter(question.options), ""))
        completion: QwenCompletion = self.qwen_client.json_completion(
            system=ANSWER_SYSTEM,
            prompt=self._prompt(question, bm25_payload),
        )
        answer = normalize_answer(
            completion.payload.get("answer", ""),
            question.answer_format,
            question.options,
            fallback=fallback,
        )
        return AnswerResult(
            qid=question.qid,
            answer=answer,
            evidence_retrieval=[
                {
                    "mode": "bm25_qwen",
                    "selected_option": answer,
                    "bm25": bm25_payload,
                    "qwen_reason": completion.payload.get("reason", ""),
                    "qwen_thinking": completion.reasoning_content,
                    "qwen_raw_answer": completion.payload.get("answer", ""),
                    "qwen_evidence_retrieval": completion.payload.get("evidence_retrieval", []),
                }
            ],
            usage=TokenUsage(
                prompt_tokens=completion.prompt_tokens,
                completion_tokens=completion.completion_tokens,
            ),
        )

    def _rank_patches(
        self,
        question: Question,
        documents: list[Document],
    ) -> list[BM25Candidate]:
        units = self.retriever._units_for_question(question, documents)
        index = _BM25Index(units, k1=self.retriever.k1, b=self.retriever.b)
        query_text = "\n".join(
            [
                question.question,
                *[f"{key}. {value}" for key, value in question.options.items()],
            ]
        )
        query = _token_counter(query_text)
        candidates: list[BM25Candidate] = []
        for unit_index, unit in enumerate(units):
            bm25 = index._score(query, unit_index)
            if bm25 <= 0:
                continue
            best_option = ""
            best_bonus = 0.0
            for option, option_text in question.options.items():
                bonus = _exact_overlap_bonus(option_text, unit)
                if bonus > best_bonus:
                    best_bonus = bonus
                    best_option = option
            candidates.append(
                BM25Candidate(
                    option=best_option,
                    unit=unit,
                    score=bm25 + best_bonus,
                    bm25=bm25,
                )
            )
        ranked = sorted(candidates, key=lambda item: item.score, reverse=True)
        seen: set[str] = set()
        unique: list[BM25Candidate] = []
        for candidate in ranked:
            unit_id = _unit_id(candidate.unit)
            if unit_id in seen:
                continue
            seen.add(unit_id)
            unique.append(candidate)
            if len(unique) == self.top_k_patches:
                break
        return unique

    def _bm25_payload(self, candidates: list[BM25Candidate]) -> dict[str, Any]:
        best = candidates[0] if candidates else BM25Candidate("", {}, 0.0, 0.0)
        evidence_units: list[dict[str, Any]] = []
        remaining = self.max_evidence_chars
        for candidate in candidates[: self.top_k_patches]:
            text = _unit_text(candidate.unit)
            if remaining <= 0:
                break
            if len(text) > remaining:
                text = text[:remaining] + "\n[truncated]"
            remaining -= len(text)
            evidence_units.append(
                {
                    "option": candidate.option,
                    "score": round(candidate.score, 4),
                    "bm25": round(candidate.bm25, 4),
                    "unit_id": _unit_id(candidate.unit),
                    "doc_id": candidate.unit.get("doc_id", ""),
                    "page": candidate.unit.get("page", ""),
                    "text": text,
                }
            )
        return {
            "mode": "bm25",
            "selected_option": best.option,
            "top_score": round(best.score, 4),
            "top_bm25": round(best.bm25, 4),
            "top_unit_id": _unit_id(best.unit),
            "top_doc_id": best.unit.get("doc_id", ""),
            "top_page": best.unit.get("page", ""),
            "evidence_units": evidence_units,
        }

    def _prompt(self, question: Question, bm25_payload: dict[str, Any]) -> str:
        task = "多选题，可以有多个正确选项。" if question.answer_format == "multi" else "单选/判断题，只能有一个正确选项。"
        return "\n".join(
            [
                f"qid: {question.qid}",
                f"answer_format: {question.answer_format}",
                f"question_type: {question.question_type}",
                f"task: {task}",
                f"question: {question.question}",
                f"options: {json.dumps(question.options, ensure_ascii=False)}",
                "BM25 evidence:",
                json.dumps(bm25_payload["evidence_units"], ensure_ascii=False, indent=2),
                (
                    'Return compact JSON only, for example: '
                    '{"answer":"AC","reason":"...","evidence_retrieval":'
                    '[{"option":"A","unit_id":"doc::0001","doc_id":"doc","quote":"..."}]}'
                ),
            ]
        )


def normalize_answer(
    value: Any,
    answer_format: str,
    options: dict[str, str],
    *,
    fallback: str = "",
) -> str:
    allowed = list(options)
    raw = _answer_text(value).upper()
    letters: list[str] = []
    for char in raw:
        if char in allowed and char not in letters:
            letters.append(char)
    if not letters:
        letters = [char for char in fallback.upper() if char in allowed]
    if not letters:
        letters = allowed[:1]
    if answer_format == "multi":
        return "".join(char for char in allowed if char in set(letters))
    return letters[0]


def _answer_text(value: Any) -> str:
    if isinstance(value, list):
        return "".join(str(item) for item in value)
    return str(value)
