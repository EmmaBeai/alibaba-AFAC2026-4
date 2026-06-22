from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from agent.bm25_experiment import DebugCandidate, ExperimentalBM25, ParsedOption
from agent.schemas import Document, Question


@dataclass(slots=True)
class RetrievedUnit:
    unit: dict[str, Any]
    score: float
    option: str
    source: str


@dataclass(slots=True)
class RetrievalResult:
    evidence: str
    items: list[RetrievedUnit]

    @property
    def pages_by_doc(self) -> dict[str, list[int]]:
        pages: dict[str, set[int]] = defaultdict(set)
        for item in self.items:
            doc_id = str(item.unit.get("doc_id", ""))
            page = int(item.unit.get("page") or 0)
            if doc_id and page:
                pages[doc_id].add(page)
        return {doc_id: sorted(values) for doc_id, values in pages.items()}


class StructuredRetriever:
    """Formal field-aware BM25 retriever.

    The old retriever used global keyword matching. This class now wraps the
    experimental field-aware BM25 path: option parser, table-row expansion,
    field/value rerank, and per-doc retrieval for comparison questions.
    """

    def __init__(
        self,
        units_path: Path,
        *,
        max_units: int = 24,
        per_option: int = 3,
        per_option_per_doc: int = 1,
        per_doc: int = 8,
        max_evidence_chars: int = 18000,
        min_score: float = 0.1,
        k1: float = 1.5,
        b: float = 0.75,
        force_number_hits: int = 3,
        force_entity_hits: int = 3,
        force_rating_hits: int = 2,
        number_bonus: float = 14.0,
        organization_bonus: float = 18.0,
        rating_bonus: float = 5.0,
        context_phrase_bonus: float = 20.0,
        noise_penalty: float = 18.0,
        split_tables: bool = True,
        page_filter_mode: str = "hard",
        page_boost: float = 0.0,
    ):
        self.units_path = units_path
        self.max_units = max_units
        self.per_option = per_option
        self.per_option_per_doc = per_option_per_doc
        self.per_doc = per_doc
        self.max_evidence_chars = max_evidence_chars
        self.min_score = min_score
        self.page_filter_mode = page_filter_mode
        self.page_boost = page_boost
        self.engine = ExperimentalBM25(units_path, split_tables=split_tables) if units_path.exists() else None

    @property
    def available(self) -> bool:
        return self.engine is not None and bool(self.engine.units)

    def retrieve(self, question: Question, documents: list[Document]) -> str:
        return self.retrieve_result(question, documents).evidence

    def retrieve_result(
        self,
        question: Question,
        documents: list[Document],
        *,
        pages_by_doc: dict[str, list[int]] | None = None,
    ) -> RetrievalResult:
        if self.engine is None:
            return RetrievalResult(evidence="", items=[])

        option_payloads: list[tuple[ParsedOption, list[DebugCandidate], dict[str, list[DebugCandidate]]]] = []
        selected: list[RetrievedUnit] = []
        for option, option_text in question.options.items():
            parsed = self.engine.parse_option(question, documents, option, option_text)
            if parsed.compare:
                per_doc = self.engine.rank_per_doc(
                    question,
                    documents,
                    parsed,
                    top_k=max(1, self.per_option_per_doc),
                    pages_by_doc=pages_by_doc,
                    page_filter_mode=self.page_filter_mode,
                    page_boost=self.page_boost,
                )
                candidates = _flatten_per_doc(per_doc)
            else:
                per_doc = {}
                candidates = self.engine.rank(
                    question,
                    documents,
                    parsed,
                    top_k=self.per_option,
                    pages_by_doc=pages_by_doc,
                    page_filter_mode=self.page_filter_mode,
                    page_boost=self.page_boost,
                )
            option_payloads.append((parsed, candidates, per_doc))
            selected.extend(
                self._candidate_to_item(candidate, option, "per_doc" if parsed.compare else "bm25")
                for candidate in candidates
            )

        selected = self._cap_items(selected)
        evidence = self._format_evidence(question, option_payloads)
        return RetrievalResult(evidence=evidence, items=selected)

    def _candidate_to_item(self, candidate: DebugCandidate, option: str, source: str) -> RetrievedUnit:
        assert self.engine is not None
        unit = self.engine.unit_by_id.get(candidate.unit_id)
        if unit is None:
            unit = {
                "unit_id": candidate.unit_id,
                "doc_id": candidate.doc_id,
                "page": candidate.page,
                "chunk_type": candidate.chunk_type,
                "raw_text": candidate.text,
            }
        return RetrievedUnit(unit=unit, score=candidate.score, option=option, source=source)

    def _cap_items(self, items: list[RetrievedUnit]) -> list[RetrievedUnit]:
        counts: Counter[str] = Counter()
        capped: list[RetrievedUnit] = []
        for item in sorted(items, key=lambda value: value.score, reverse=True):
            doc_id = str(item.unit.get("doc_id", ""))
            if counts[doc_id] >= self.per_doc:
                continue
            capped.append(item)
            counts[doc_id] += 1
            if len(capped) >= self.max_units:
                break
        return capped

    def _format_evidence(
        self,
        question: Question,
        option_payloads: list[tuple[ParsedOption, list[DebugCandidate], dict[str, list[DebugCandidate]]]],
    ) -> str:
        blocks: list[str] = []
        remaining = self.max_evidence_chars
        for parsed, candidates, per_doc in option_payloads:
            option_header = (
                f"\n### 选项 {parsed.option}\n"
                f"选项文本：{parsed.text}\n"
                f"解析：doc_ids={parsed.doc_ids}; fields={parsed.fields}; "
                f"values={parsed.values}; compare={parsed.compare}\n"
            )
            if len(option_header) > remaining:
                break
            blocks.append(option_header)
            remaining -= len(option_header)
            if per_doc:
                for doc_id, doc_candidates in per_doc.items():
                    doc_header = f"\n#### 文档 {doc_id} 候选证据\n"
                    if len(doc_header) > remaining:
                        return "".join(blocks)
                    blocks.append(doc_header)
                    remaining -= len(doc_header)
                    remaining = self._append_candidates(blocks, remaining, parsed.option, doc_candidates)
                    if remaining <= 0:
                        return "".join(blocks)
            else:
                remaining = self._append_candidates(blocks, remaining, parsed.option, candidates)
                if remaining <= 0:
                    return "".join(blocks)
        return "".join(blocks)

    def _append_candidates(
        self,
        blocks: list[str],
        remaining: int,
        option: str,
        candidates: list[DebugCandidate],
    ) -> int:
        for candidate in candidates:
            block = (
                f"\n[unit_id={candidate.unit_id}; doc_id={candidate.doc_id}; "
                f"page={candidate.page}; type={candidate.chunk_type}; option={option}; "
                f"score={candidate.score:.2f}; bm25={candidate.bm25:.2f}]\n"
                f"字段命中：{'、'.join(candidate.field_hits)}\n"
                f"值命中：{'、'.join(candidate.value_hits)}\n"
                f"召回原因：{'; '.join(candidate.reasons)}\n"
                f"文本：{candidate.text}\n"
            )
            if len(block) > remaining:
                if remaining > 500:
                    blocks.append(block[:remaining] + "\n[truncated]\n")
                return 0
            blocks.append(block)
            remaining -= len(block)
        return remaining


def _flatten_per_doc(per_doc: dict[str, list[DebugCandidate]]) -> list[DebugCandidate]:
    candidates: list[DebugCandidate] = []
    for doc_candidates in per_doc.values():
        candidates.extend(doc_candidates)
    return sorted(candidates, key=lambda item: item.score, reverse=True)
