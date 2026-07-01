from __future__ import annotations

import html
import json
import math
import re
from collections import Counter
from dataclasses import dataclass
from html.parser import HTMLParser
from pathlib import Path
from typing import Any

from pypdf import PdfReader

from agent.schemas import AnswerResult, Document, Question, TokenUsage

CJK_RE = re.compile(r"[\u4e00-\u9fff]+")
WORD_RE = re.compile(r"[A-Za-z0-9_.%+-]+")
SPACE_RE = re.compile(r"[ \t]+")
BLANK_RE = re.compile(r"\n{3,}")
HTML_BLOCK_RE = re.compile(r"(?i)</?\s*(?:br|div|p|table|tr|h[1-6]|li|section|article)[^>]*>")
HTML_CELL_RE = re.compile(r"(?i)</\s*(?:td|th)\s*>\s*<\s*(?:td|th)[^>]*>")
HTML_TAG_RE = re.compile(r"<[^>]+>")
CONTROL_RE = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")
SURROGATE_RE = re.compile(r"[\ud800-\udfff]")


@dataclass(slots=True)
class BM25Candidate:
    option: str
    unit: dict[str, Any]
    score: float
    bm25: float


class BM25Top1Answerer:
    def __init__(
        self,
        units_path: Path | str | None = None,
        *,
        chunk_chars: int = 1800,
        k1: float = 1.5,
        b: float = 0.75,
    ):
        self.units_path = Path(units_path) if units_path else None
        self.chunk_chars = chunk_chars
        self.k1 = k1
        self.b = b
        self._file_units = self._load_units(self.units_path)

    def answer(self, question: Question, documents: list[Document]) -> AnswerResult:
        units = self._units_for_question(question, documents)
        index = _BM25Index(units, k1=self.k1, b=self.b)
        candidates: list[BM25Candidate] = []
        for option, option_text in question.options.items():
            candidate = index.best(option, option_text, question.question)
            if candidate is not None:
                candidates.append(candidate)

        if candidates:
            best = max(candidates, key=lambda item: (item.score, item.option))
            selected = best.option
        else:
            selected = next(iter(question.options), "")
            best = BM25Candidate(selected, {}, 0.0, 0.0)

        return AnswerResult(
            qid=question.qid,
            answer=selected,
            evidence_retrieval=[self._evidence_payload(selected, best, candidates)],
            usage=TokenUsage(),
        )

    def _units_for_question(self, question: Question, documents: list[Document]) -> list[dict[str, Any]]:
        allowed_doc_ids = {document.doc_id for document in documents}
        units = [
            unit
            for unit in self._file_units
            if unit.get("domain") == question.domain and unit.get("doc_id") in allowed_doc_ids
        ]
        if units:
            return units
        return self._raw_units(documents)

    def _raw_units(self, documents: list[Document]) -> list[dict[str, Any]]:
        units: list[dict[str, Any]] = []
        for document in documents:
            for page, text in _read_document_pages(document.path, chunk_chars=self.chunk_chars):
                for index, chunk in enumerate(_split_text(text, self.chunk_chars), start=1):
                    units.append(
                        {
                            "unit_id": f"{document.doc_id}_p{page}_{index:03d}",
                            "doc_id": document.doc_id,
                            "domain": document.domain,
                            "title": document.title,
                            "page": page,
                            "raw_text": chunk,
                            "search_text": f"{document.title}\n{chunk}",
                        }
                    )
        return units

    @staticmethod
    def _load_units(path: Path | None) -> list[dict[str, Any]]:
        if path is None or not path.exists():
            return []
        units: list[dict[str, Any]] = []
        with path.open(encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if line:
                    units.append(json.loads(line))
        return units

    @staticmethod
    def _evidence_payload(
        selected: str,
        best: BM25Candidate,
        candidates: list[BM25Candidate],
    ) -> dict[str, Any]:
        ranked = sorted(candidates, key=lambda item: item.score, reverse=True)
        return {
            "mode": "bm25_top1",
            "selected_option": selected,
            "top_score": round(best.score, 4),
            "top_bm25": round(best.bm25, 4),
            "top_unit_id": _unit_id(best.unit),
            "top_doc_id": best.unit.get("doc_id", ""),
            "top_page": best.unit.get("page", ""),
            "option_scores": [
                {
                    "option": item.option,
                    "score": round(item.score, 4),
                    "bm25": round(item.bm25, 4),
                    "unit_id": _unit_id(item.unit),
                    "doc_id": item.unit.get("doc_id", ""),
                    "page": item.unit.get("page", ""),
                }
                for item in ranked
            ],
        }


class _BM25Index:
    def __init__(self, units: list[dict[str, Any]], *, k1: float, b: float):
        self.units = units
        self.k1 = k1
        self.b = b
        self.counters = [_token_counter(_unit_text(unit)) for unit in units]
        self.lengths = [sum(counter.values()) for counter in self.counters]
        self.avgdl = sum(self.lengths) / len(self.lengths) if self.lengths else 0.0
        self.idf = self._idf(self.counters)

    def best(self, option: str, option_text: str, question_text: str) -> BM25Candidate | None:
        query = _token_counter(f"{option_text}\n{question_text}")
        best: BM25Candidate | None = None
        for index, unit in enumerate(self.units):
            bm25 = self._score(query, index)
            if bm25 <= 0:
                continue
            score = bm25 + _exact_overlap_bonus(option_text, unit)
            candidate = BM25Candidate(option=option, unit=unit, score=score, bm25=bm25)
            if best is None or candidate.score > best.score:
                best = candidate
        return best

    def _score(self, query: Counter[str], index: int) -> float:
        if not query or not self.units:
            return 0.0
        counter = self.counters[index]
        doc_length = self.lengths[index] or 1
        score = 0.0
        for token, query_frequency in query.items():
            frequency = counter.get(token, 0)
            if frequency <= 0:
                continue
            denominator = frequency + self.k1 * (1 - self.b + self.b * doc_length / max(self.avgdl, 1))
            score += self.idf.get(token, 0.0) * frequency * (self.k1 + 1) / denominator * query_frequency
        return score

    @staticmethod
    def _idf(counters: list[Counter[str]]) -> dict[str, float]:
        document_count = len(counters)
        if document_count == 0:
            return {}
        document_frequency: Counter[str] = Counter()
        for counter in counters:
            document_frequency.update(counter.keys())
        return {
            token: math.log(1 + (document_count - frequency + 0.5) / (frequency + 0.5))
            for token, frequency in document_frequency.items()
        }


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


def _token_counter(text: str) -> Counter[str]:
    tokens: list[str] = []
    for match in CJK_RE.finditer(text):
        value = match.group(0)
        tokens.extend(value[index:index + 2] for index in range(max(0, len(value) - 1)))
        tokens.extend(value[index:index + 3] for index in range(max(0, len(value) - 2)))
        if len(value) <= 8:
            tokens.append(value)
    tokens.extend(match.group(0).lower() for match in WORD_RE.finditer(text))
    return Counter(token for token in tokens if token)


def _exact_overlap_bonus(option_text: str, unit: dict[str, Any]) -> float:
    text = str(unit.get("raw_text") or unit.get("search_text") or unit.get("text") or "")
    score = 0.0
    for token in WORD_RE.findall(option_text):
        if len(token) >= 2 and token in text:
            score += 2.0
    for match in CJK_RE.finditer(option_text):
        value = match.group(0)
        if len(value) >= 4 and value in text:
            score += min(len(value), 20) * 1.5
    return score


def _read_document_pages(path: Path, *, chunk_chars: int) -> list[tuple[int, str]]:
    suffix = path.suffix.lower()
    if suffix == ".pdf":
        reader = PdfReader(str(path))
        return [
            (index, _clean_text(page.extract_text() or ""))
            for index, page in enumerate(reader.pages, start=1)
        ]
    if suffix == ".txt":
        return [(1, _clean_text(path.read_text(encoding="utf-8", errors="replace")))]
    if suffix in {".html", ".htm"}:
        return [(1, _clean_text(_html_to_text(path.read_text(encoding="utf-8", errors="replace"))))]
    text = _clean_text(path.read_text(encoding="utf-8", errors="replace"))
    return [(index, chunk) for index, chunk in enumerate(_split_text(text, chunk_chars), start=1)]


class _HTMLTextExtractor(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.parts: list[str] = []
        self.ignored_depth = 0

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag in {"script", "style", "nav"}:
            self.ignored_depth += 1
        elif tag in {"br", "div", "p", "table", "tr", "h1", "h2", "h3", "h4", "h5", "h6", "li"}:
            self.parts.append("\n")

    def handle_endtag(self, tag: str) -> None:
        if tag in {"script", "style", "nav"} and self.ignored_depth:
            self.ignored_depth -= 1
        elif tag in {"div", "p", "table", "tr", "h1", "h2", "h3", "h4", "h5", "h6", "li"}:
            self.parts.append("\n")

    def handle_data(self, data: str) -> None:
        if not self.ignored_depth:
            self.parts.append(data)


def _html_to_text(value: str) -> str:
    parser = _HTMLTextExtractor()
    parser.feed(value)
    text = "".join(parser.parts)
    text = HTML_CELL_RE.sub(" | ", text)
    text = HTML_BLOCK_RE.sub("\n", text)
    text = HTML_TAG_RE.sub("", text)
    return html.unescape(text)


def _clean_text(text: str) -> str:
    text = SURROGATE_RE.sub("", text)
    text = CONTROL_RE.sub("", text)
    lines = [SPACE_RE.sub(" ", line).strip() for line in text.replace("\r", "\n").splitlines()]
    return BLANK_RE.sub("\n\n", "\n".join(line for line in lines if line)).strip()


def _split_text(text: str, target_chars: int) -> list[str]:
    if not text:
        return [""]
    chunks: list[str] = []
    current: list[str] = []
    size = 0
    for paragraph in text.split("\n"):
        if current and size + len(paragraph) > target_chars:
            chunks.append("\n".join(current).strip())
            current = []
            size = 0
        current.append(paragraph)
        size += len(paragraph) + 1
    if current:
        chunks.append("\n".join(current).strip())
    return chunks
