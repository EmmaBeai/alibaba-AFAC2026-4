from __future__ import annotations

import hashlib
import json
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

try:
    import numpy as np
except ImportError:  # pragma: no cover - numpy is an optimization, not a contract.
    np = None

from agent.embedding_client import DashScopeEmbeddingClient
from agent.qwen_client import QwenCompletion, QwenPlusClient
from agent.schemas import AnswerResult, Document, Question, TokenUsage


ANSWER_SYSTEM = """你是金融长文档选择题助手。只能根据给定 dense retrieval 证据回答。
输出必须是紧凑 JSON object，只包含 answer、reason、evidence_retrieval。
answer 只能由选项字母组成；多选题可以输出多个字母，例如 AC；不要输出选项文本。
reason 不超过 80 个汉字。
evidence_retrieval 最多 3 项，每项只保留 option、unit_id、doc_id、quote。quote 不超过 40 个汉字。"""

CJK_RE = re.compile(r"[\u4e00-\u9fff]+")
WORD_RE = re.compile(r"[A-Za-z0-9_.%+-]+")


@dataclass(slots=True)
class DenseCandidate:
    option: str
    unit: dict[str, Any]
    score: float


class DenseQwenAnswerer:
    def __init__(
        self,
        units_path: Path | str,
        *,
        embedding_cache_path: Path | str,
        embedding_client: Any | None = None,
        qwen_client: QwenPlusClient | Any | None = None,
        top_k_patches: int = 5,
        max_evidence_chars: int = 12000,
        build_missing_embeddings: bool = False,
    ):
        self.units_path = Path(units_path)
        self.embedding_cache_path = Path(embedding_cache_path)
        self.embedding_client = embedding_client or DashScopeEmbeddingClient()
        self.qwen_client = qwen_client or QwenPlusClient()
        self.top_k_patches = top_k_patches
        self.max_evidence_chars = max_evidence_chars
        self.build_missing_embeddings = build_missing_embeddings
        self._file_units = _load_units(self.units_path)
        self._embedding_rows: dict[str, list[float]] = {}
        self._embedding_matrix: Any | None = None
        self._embedding_unit_ids: list[str] = []

    def answer(self, question: Question, documents: list[Document]) -> AnswerResult:
        patches = self._rank_patches(question, documents)
        dense_payload = self._dense_payload(patches)
        fallback = next(iter(question.options), "")
        completion: QwenCompletion = self.qwen_client.json_completion(
            system=ANSWER_SYSTEM,
            prompt=self._prompt(question, dense_payload),
        )
        answer = _normalize_answer(
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
                    "mode": "dense_qwen",
                    "selected_option": answer,
                    "dense": dense_payload,
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

    def _rank_patches(self, question: Question, documents: list[Document]) -> list[DenseCandidate]:
        units = self._units_for_question(question, documents)
        if not units:
            return []
        unit_vectors = self._unit_vectors(units)
        query_vector = _normalize_vector(
            self.embedding_client.embed_texts([_query_text(question)], input_type="query")[0]
        )
        scores = _similarity_scores(unit_vectors, query_vector)
        ranked: list[DenseCandidate] = []
        for unit, score in zip(units, scores):
            ranked.append(
                DenseCandidate(
                    option=_best_option_for_unit(question.options, unit),
                    unit=unit,
                    score=float(score),
                )
            )
        ranked.sort(key=lambda item: (item.score, _unit_id(item.unit)), reverse=True)
        return ranked[: self.top_k_patches]

    def _units_for_question(self, question: Question, documents: list[Document]) -> list[dict[str, Any]]:
        allowed_doc_ids = {document.doc_id for document in documents}
        return [
            unit
            for unit in self._file_units
            if unit.get("domain") == question.domain and unit.get("doc_id") in allowed_doc_ids
        ]

    def _unit_vectors(self, units: list[dict[str, Any]]) -> list[list[float]]:
        cache = _read_embedding_cache(
            self.embedding_cache_path,
            model=getattr(self.embedding_client, "model", ""),
            dimension=int(getattr(self.embedding_client, "dimension", 0) or 0),
        )
        missing: list[dict[str, Any]] = []
        for unit in units:
            unit_id = _unit_id(unit)
            text_hash = _text_hash(_unit_text(unit))
            if unit_id not in cache or cache[unit_id]["text_sha256"] != text_hash:
                missing.append(unit)
        if missing and not self.build_missing_embeddings:
            preview = ", ".join(_unit_id(unit) for unit in missing[:5])
            raise RuntimeError(
                "Missing prebuilt dense embeddings for "
                f"{len(missing)} units. First missing: {preview}. "
                "Run: python -m script.build_dense_embeddings --config config/dense_qwen.yaml"
            )
        if missing:
            batch_size = int(getattr(self.embedding_client, "batch_size", 10) or 10)
            total_missing = len(missing)
            for start in range(0, len(missing), batch_size):
                batch = missing[start:start + batch_size]
                print(
                    f"[embedding] units {start + 1}-{start + len(batch)}/{total_missing}",
                    flush=True,
                )
                embeddings = self.embedding_client.embed_texts(
                    [_unit_text(unit) for unit in batch],
                    input_type="document",
                )
                for unit, embedding in zip(batch, embeddings):
                    cache[_unit_id(unit)] = {
                        "unit_id": _unit_id(unit),
                        "text_sha256": _text_hash(_unit_text(unit)),
                        "model": getattr(self.embedding_client, "model", ""),
                        "dimension": len(embedding),
                        "embedding": _normalize_vector(embedding),
                    }
                _write_embedding_cache(self.embedding_cache_path, cache)
        return [cache[_unit_id(unit)]["embedding"] for unit in units]

    def _dense_payload(self, candidates: list[DenseCandidate]) -> dict[str, Any]:
        best = candidates[0] if candidates else DenseCandidate("", {}, 0.0)
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
                    "score": round(candidate.score, 6),
                    "unit_id": _unit_id(candidate.unit),
                    "doc_id": candidate.unit.get("doc_id", ""),
                    "page": candidate.unit.get("page", ""),
                    "text": text,
                }
            )
        return {
            "mode": "dense",
            "embedding_model": getattr(self.embedding_client, "model", ""),
            "embedding_dimension": getattr(self.embedding_client, "dimension", ""),
            "selected_option": best.option,
            "top_score": round(best.score, 6),
            "top_unit_id": _unit_id(best.unit),
            "top_doc_id": best.unit.get("doc_id", ""),
            "top_page": best.unit.get("page", ""),
            "evidence_units": evidence_units,
        }

    def _prompt(self, question: Question, dense_payload: dict[str, Any]) -> str:
        task = "多选题，可以有多个正确选项。" if question.answer_format == "multi" else "单选/判断题，只能有一个正确选项。"
        return "\n".join(
            [
                f"qid: {question.qid}",
                f"answer_format: {question.answer_format}",
                f"question_type: {question.question_type}",
                f"task: {task}",
                f"question: {question.question}",
                f"options: {json.dumps(question.options, ensure_ascii=False)}",
                "Dense retrieval evidence:",
                json.dumps(dense_payload["evidence_units"], ensure_ascii=False, indent=2),
                (
                    'Return compact JSON only, for example: '
                    '{"answer":"AC","reason":"...","evidence_retrieval":'
                    '[{"option":"A","unit_id":"doc::0001","doc_id":"doc","quote":"..."}]}'
                ),
            ]
        )


def _load_units(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    units: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                units.append(json.loads(line))
    return units


def _unit_text(unit: dict[str, Any]) -> str:
    return "\n".join(
        str(part)
        for part in [
            unit.get("title", ""),
            unit.get("section_path", ""),
            unit.get("chunk_type", ""),
            unit.get("search_text", ""),
            unit.get("raw_text", ""),
            unit.get("text", ""),
        ]
        if part
    )


def _unit_id(unit: dict[str, Any]) -> str:
    return str(unit.get("unit_id") or unit.get("chunk_id") or "")


def _query_text(question: Question) -> str:
    return "\n".join(
        [
            question.question,
            *[f"{key}. {value}" for key, value in question.options.items()],
        ]
    )


def _text_hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _normalize_vector(vector: list[float]) -> list[float]:
    norm = math.sqrt(sum(float(value) * float(value) for value in vector))
    if norm <= 0:
        return [0.0 for _ in vector]
    return [float(value) / norm for value in vector]


def _similarity_scores(vectors: list[list[float]], query_vector: list[float]) -> list[float]:
    if np is not None:
        matrix = np.asarray(vectors, dtype=np.float32)
        query = np.asarray(query_vector, dtype=np.float32)
        return (matrix @ query).astype(float).tolist()
    return [sum(left * right for left, right in zip(vector, query_vector)) for vector in vectors]


def _read_embedding_cache(path: Path, *, model: str, dimension: int) -> dict[str, dict[str, Any]]:
    if not path.exists():
        return {}
    rows: dict[str, dict[str, Any]] = {}
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            embedding = row.get("embedding")
            if row.get("model") != model or not isinstance(embedding, list):
                continue
            if dimension and len(embedding) != dimension:
                continue
            rows[str(row.get("unit_id", ""))] = row
    return rows


def _write_embedding_cache(path: Path, rows: dict[str, dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    ordered = [rows[key] for key in sorted(rows)]
    path.write_text(
        "\n".join(json.dumps(row, ensure_ascii=False) for row in ordered),
        encoding="utf-8",
    )


def _best_option_for_unit(options: dict[str, str], unit: dict[str, Any]) -> str:
    text = str(unit.get("raw_text") or unit.get("search_text") or unit.get("text") or "")
    best_option = ""
    best_score = 0
    for option, option_text in options.items():
        score = 0
        for token in WORD_RE.findall(option_text):
            if len(token) >= 2 and token in text:
                score += len(token)
        for match in CJK_RE.finditer(option_text):
            value = match.group(0)
            if len(value) >= 2 and value in text:
                score += len(value)
        if score > best_score:
            best_option = option
            best_score = score
    return best_option


def _normalize_answer(
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
