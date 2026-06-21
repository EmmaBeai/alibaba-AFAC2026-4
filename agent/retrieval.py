from __future__ import annotations

import json
import math
import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from agent.schemas import Document, Question
from agent.structured import NUMBER_RE

CJK_RE = re.compile(r"[\u4e00-\u9fff]+")
WORD_RE = re.compile(r"[A-Za-z0-9_.%-]+")


@dataclass(slots=True)
class RetrievedUnit:
    unit: dict[str, Any]
    score: float
    option: str


class StructuredRetriever:
    def __init__(
        self,
        units_path: Path,
        *,
        max_units: int = 18,
        per_option: int = 3,
        per_option_per_doc: int = 1,
        per_doc: int = 8,
        max_evidence_chars: int = 18000,
        min_score: float = 0.1,
    ):
        self.units_path = units_path
        self.max_units = max_units
        self.per_option = per_option
        self.per_option_per_doc = per_option_per_doc
        self.per_doc = per_doc
        self.max_evidence_chars = max_evidence_chars
        self.min_score = min_score
        self.units = _load_units(units_path) if units_path.exists() else []
        self._tokens_by_unit = [_token_counter(unit.get("search_text") or unit.get("raw_text") or "") for unit in self.units]
        self._idf = _idf(self._tokens_by_unit)

    @property
    def available(self) -> bool:
        return bool(self.units)

    def retrieve(self, question: Question, documents: list[Document]) -> str:
        if not self.units:
            return ""
        allowed_docs = {document.doc_id for document in documents}
        candidates = [
            (index, unit)
            for index, unit in enumerate(self.units)
            if unit.get("doc_id") in allowed_docs and unit.get("domain") == question.domain
        ]
        if not candidates:
            return ""

        selected: dict[str, RetrievedUnit] = {}
        option_queries = question.options or {"": ""}
        for option, option_text in option_queries.items():
            query = f"{question.question}\n选项{option}: {option_text}\n{option_text}\n{option_text}"
            ranked = self._rank(query, candidates)
            for item in ranked[: self.per_option]:
                key = f"{item.option}:{item.unit['unit_id']}"
                current = selected.get(key)
                if current is None or item.score > current.score:
                    selected[key] = item
            by_doc: defaultdict[str, list[tuple[int, dict[str, Any]]]] = defaultdict(list)
            for candidate in candidates:
                by_doc[candidate[1].get("doc_id", "")].append(candidate)
            for doc_candidates in by_doc.values():
                for item in self._rank(query, doc_candidates)[: self.per_option_per_doc]:
                    key = f"{item.option}:{item.unit['unit_id']}"
                    current = selected.get(key)
                    if current is None or item.score > current.score:
                        selected[key] = item

        for item in self._rank(question.question, candidates)[: self.per_option]:
            key = f"{item.option}:{item.unit['unit_id']}"
            current = selected.get(key)
            if current is None or item.score > current.score:
                selected[key] = item

        ranked_unique = sorted(selected.values(), key=lambda item: item.score, reverse=True)
        capped = self._cap_per_doc(ranked_unique)
        return self._format_evidence(_sort_for_prompt(capped))

    def _rank(self, query: str, candidates: list[tuple[int, dict[str, Any]]]) -> list[RetrievedUnit]:
        query_tokens = _token_counter(query)
        query_numbers = set(_numbers(query))
        ranked: list[RetrievedUnit] = []
        for index, unit in candidates:
            score = _score(query_tokens, query_numbers, self._tokens_by_unit[index], unit, self._idf)
            if score >= self.min_score:
                ranked.append(RetrievedUnit(unit=unit, score=score, option=_option_from_query(query)))
        return sorted(ranked, key=lambda item: item.score, reverse=True)

    def _cap_per_doc(self, items: list[RetrievedUnit]) -> list[RetrievedUnit]:
        counts: Counter[str] = Counter()
        capped: list[RetrievedUnit] = []
        for item in items:
            doc_id = item.unit.get("doc_id", "")
            if counts[doc_id] >= self.per_doc:
                continue
            capped.append(item)
            counts[doc_id] += 1
            if len(capped) >= self.max_units:
                break
        return capped

    def _format_evidence(self, items: list[RetrievedUnit]) -> str:
        blocks: list[str] = []
        remaining = self.max_evidence_chars
        for item in items:
            unit = item.unit
            numbers = "、".join(unit.get("numbers") or [])
            keywords = "、".join(unit.get("keywords") or [])
            block = (
                f"\n[unit_id={unit.get('unit_id')}; doc_id={unit.get('doc_id')}; "
                f"page={unit.get('page')}; type={unit.get('chunk_type')}; "
                f"option={item.option}; score={item.score:.2f}]\n"
                f"标题：{unit.get('title', '')}\n"
                f"章节：{unit.get('section_path', '')}\n"
                f"条款：{unit.get('clause_no', '')}\n"
                f"关键词：{keywords}\n"
                f"数字：{numbers}\n"
                f"文本：{unit.get('raw_text', '')}\n"
            )
            if len(block) > remaining:
                if remaining > 500:
                    blocks.append(block[:remaining] + "\n[truncated]\n")
                break
            blocks.append(block)
            remaining -= len(block)
        return "".join(blocks)


def _load_units(path: Path) -> list[dict[str, Any]]:
    units: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                units.append(json.loads(line))
    return units


def _token_counter(text: str) -> Counter[str]:
    normalized = text.lower()
    tokens: list[str] = []
    for match in CJK_RE.finditer(normalized):
        value = match.group(0)
        if len(value) == 1:
            tokens.append(value)
        else:
            tokens.extend(value[index : index + 2] for index in range(len(value) - 1))
            if len(value) >= 4:
                tokens.extend(value[index : index + 3] for index in range(len(value) - 2))
    tokens.extend(match.group(0) for match in WORD_RE.finditer(normalized))
    return Counter(token for token in tokens if len(token.strip()) >= 1)


def _idf(counters: list[Counter[str]]) -> dict[str, float]:
    doc_freq: Counter[str] = Counter()
    for counter in counters:
        doc_freq.update(counter.keys())
    total = max(1, len(counters))
    return {token: math.log((total + 1) / (freq + 1)) + 1 for token, freq in doc_freq.items()}


def _score(
    query_tokens: Counter[str],
    query_numbers: set[str],
    unit_tokens: Counter[str],
    unit: dict[str, Any],
    idf: dict[str, float],
) -> float:
    if not query_tokens:
        return 0.0
    score = 0.0
    for token, weight in query_tokens.items():
        if token in unit_tokens:
            score += min(weight, unit_tokens[token]) * idf.get(token, 1.0)
    raw_text = unit.get("raw_text") or ""
    unit_numbers = set(unit.get("numbers") or [])
    number_hits = query_numbers & unit_numbers
    if number_hits:
        score += 8.0 * len(number_hits)
    elif any(number and number in raw_text for number in query_numbers):
        score += 4.0
    keyword_hits = sum(1 for keyword in unit.get("keywords") or [] if keyword in raw_text)
    score += keyword_hits * 0.2
    length_penalty = 1.0 + len(raw_text) / 2600
    return score / length_penalty


def _numbers(text: str) -> list[str]:
    return [match.group(0).strip() for match in NUMBER_RE.finditer(text) if match.group(0).strip()]


def _sort_for_prompt(items: list[RetrievedUnit]) -> list[RetrievedUnit]:
    return sorted(
        items,
        key=lambda item: (
            item.option == "-",
            item.option,
            str(item.unit.get("doc_id", "")),
            int(item.unit.get("page") or 0),
            -item.score,
        ),
    )


def _option_from_query(query: str) -> str:
    match = re.search(r"选项([A-Z])", query)
    return match.group(1) if match else "-"
