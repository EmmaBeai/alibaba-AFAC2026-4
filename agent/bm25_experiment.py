from __future__ import annotations

import json
import math
import re
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from agent.schemas import Document, Question
from agent.structured import NUMBER_RE

ORG_RE = re.compile(
    r"[\u4e00-\u9fffA-Za-z0-9（）()]{2,60}?"
    r"(?:股份有限公司|有限责任公司|集团有限公司|控股集团有限公司|证券股份有限公司|"
    r"银行股份有限公司|保险股份有限公司|交易所|评级有限公司|会计师事务所|律师事务所)"
)
RATING_RE = re.compile(r"(?<![A-Za-z0-9])(?:AAA|AA\+?|A\+|BBB\+?|BB\+?|B\+)(?![A-Za-z0-9])")
CJK_RE = re.compile(r"[\u4e00-\u9fff]+")
WORD_RE = re.compile(r"[A-Za-z0-9_.%-]+")

FIELD_TERMS: dict[str, list[str]] = {
    "issuer": ["发行人", "发行主体", "主体名称", "公司名称"],
    "issue_size": ["发行规模", "发行金额", "募集规模", "本期债券发行规模"],
    "trustee": ["受托管理人", "债券受托管理人"],
    "lead_underwriter": ["主承销商", "牵头主承销商", "联席主承销商", "承销商"],
    "bookrunner": ["簿记管理人", "簿记管理"],
    "rating_subject": ["主体信用评级", "主体信用等级", "主体评级", "信用等级"],
    "rating_bond": ["债项信用评级", "债项评级"],
    "term": ["债券期限", "期限"],
    "coupon": ["票面利率", "利率"],
    "guarantee": ["增信", "担保", "无增信"],
    "revenue": ["营业收入"],
    "net_profit": ["净利润", "归母净利润", "归属于上市公司股东的净利润"],
    "cashflow": ["现金流", "现金流量净额", "经营活动产生的现金流量"],
    "rd": ["研发投入", "研发费用", "研发人员"],
    "dividend": ["现金分红", "利润分配", "股利"],
    "insurance_liability": ["保险责任", "给付", "保险金"],
    "exclusion": ["责任免除", "不承担", "除外"],
    "deadline": ["期限", "日内", "工作日", "报送"],
    "penalty": ["处罚", "罚款", "责令", "警告"],
}

FIELD_CHUNK_HINTS: dict[str, set[str]] = {
    "issuer": {"issue_terms", "table_row"},
    "issue_size": {"issue_terms", "table_row"},
    "trustee": {"intermediary", "issue_terms", "table_row"},
    "lead_underwriter": {"intermediary", "issue_terms", "table_row"},
    "bookrunner": {"intermediary", "issue_terms", "table_row"},
    "rating_subject": {"rating", "issue_terms", "table_row"},
    "rating_bond": {"rating", "issue_terms", "table_row"},
    "term": {"issue_terms", "table_row"},
    "coupon": {"issue_terms", "table_row"},
    "guarantee": {"guarantee", "issue_terms", "table_row"},
}

FINANCE_TERMS = sorted(
    {term for terms in FIELD_TERMS.values() for term in terms}
    | {
        "受托管理协议",
        "债券持有人会议",
        "募集说明书",
        "评级展望",
        "无增信",
        "本期债券",
        "专业投资者",
    },
    key=len,
    reverse=True,
)

STOP_TOKENS = {
    "下列",
    "哪些",
    "描述",
    "准确",
    "关于",
    "提供",
    "基于",
    "文档",
    "第一",
    "第二",
    "两份",
    "明确",
    "是否",
    "以及",
}

NOISE_MARKERS = ["声明", "签署", "备查文件", "目录", "联系方式", "查阅地点"]


@dataclass(slots=True)
class ParsedOption:
    option: str
    text: str
    doc_ids: list[str]
    fields: list[str]
    values: list[str]
    numbers: list[str]
    organizations: list[str]
    ratings: list[str]
    compare: bool

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class DebugCandidate:
    unit_id: str
    doc_id: str
    page: int
    chunk_type: str
    score: float
    bm25: float
    field_hits: list[str]
    value_hits: list[str]
    reasons: list[str]
    text: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class ExperimentalBM25:
    def __init__(self, units_path: Path, *, split_tables: bool = True):
        self.units = _load_units(units_path)
        if split_tables:
            self.units = _expand_table_rows(self.units)
        self.unit_by_id = {str(unit.get("unit_id", "")): unit for unit in self.units}
        self.token_counters = [_token_counter(_unit_search_text(unit)) for unit in self.units]
        self.doc_lengths = [sum(counter.values()) for counter in self.token_counters]
        self.avgdl = sum(self.doc_lengths) / len(self.doc_lengths) if self.doc_lengths else 0.0
        self.idf = _bm25_idf(self.token_counters)

    def parse_option(self, question: Question, documents: list[Document], option: str, option_text: str) -> ParsedOption:
        doc_ids = _option_doc_ids(question, documents, option_text)
        fields = _fields(option_text) or _fields(question.question)
        numbers = _numbers(option_text)
        organizations = _organizations(option_text)
        ratings = _ratings(option_text)
        values = _dedupe(numbers + organizations + ratings)
        return ParsedOption(
            option=option,
            text=option_text,
            doc_ids=doc_ids,
            fields=fields,
            values=values,
            numbers=numbers,
            organizations=organizations,
            ratings=ratings,
            compare=_is_compare_option(question.question, option_text, len(documents)),
        )

    def rank(
        self,
        question: Question,
        documents: list[Document],
        parsed: ParsedOption,
        *,
        top_k: int = 8,
        pages_by_doc: dict[str, list[int]] | None = None,
    ) -> list[DebugCandidate]:
        allowed_docs = set(parsed.doc_ids)
        allowed_pages = _allowed_pages(pages_by_doc)
        query_text = _query_text(question.question, parsed)
        query_tokens = _token_counter(query_text)
        scored: list[DebugCandidate] = []
        for index, unit in enumerate(self.units):
            if unit.get("domain") != question.domain or unit.get("doc_id") not in allowed_docs:
                continue
            if not _page_allowed(unit, allowed_pages):
                continue
            candidate = self._score_unit(index, unit, query_tokens, parsed)
            if candidate.score > 0:
                scored.append(candidate)
        return sorted(scored, key=lambda item: item.score, reverse=True)[:top_k]

    def rank_per_doc(
        self,
        question: Question,
        documents: list[Document],
        parsed: ParsedOption,
        *,
        top_k: int = 3,
        pages_by_doc: dict[str, list[int]] | None = None,
    ) -> dict[str, list[DebugCandidate]]:
        results: dict[str, list[DebugCandidate]] = {}
        query_text = _query_text(question.question, parsed)
        query_tokens = _token_counter(query_text)
        allowed_pages = _allowed_pages(pages_by_doc)
        for doc_id in parsed.doc_ids:
            scored: list[DebugCandidate] = []
            for index, unit in enumerate(self.units):
                if unit.get("domain") != question.domain or unit.get("doc_id") != doc_id:
                    continue
                if not _page_allowed(unit, allowed_pages):
                    continue
                candidate = self._score_unit(index, unit, query_tokens, parsed)
                if candidate.score > 0:
                    scored.append(candidate)
            results[doc_id] = sorted(scored, key=lambda item: item.score, reverse=True)[:top_k]
        return results

    def _score_unit(
        self,
        index: int,
        unit: dict[str, Any],
        query_tokens: Counter[str],
        parsed: ParsedOption,
    ) -> DebugCandidate:
        text = unit.get("raw_text") or ""
        chunk_type = str(unit.get("chunk_type") or "")
        bm25 = self._bm25_score(query_tokens, index)
        field_hits = _field_hits(text, parsed.fields)
        value_hits = [value for value in parsed.values if value and value in text]
        reasons: list[str] = []
        score = bm25
        if field_hits:
            bonus = 26.0 * len(field_hits)
            score += bonus
            reasons.append(f"field+{bonus:.0f}:{'/'.join(field_hits)}")
        if value_hits:
            bonus = 18.0 * len(value_hits)
            score += bonus
            reasons.append(f"value+{bonus:.0f}:{'/'.join(value_hits[:4])}")
        if field_hits and value_hits:
            score += 70.0
            reasons.append("field_value_same_unit+70")
        if _chunk_type_matches_fields(chunk_type, parsed.fields):
            score += 14.0
            reasons.append(f"chunk+14:{chunk_type}")
        penalty = _noise_penalty(unit)
        if penalty:
            score -= penalty
            reasons.append(f"noise-{penalty:.0f}")
        if bm25:
            reasons.insert(0, f"bm25={bm25:.2f}")
        return DebugCandidate(
            unit_id=str(unit.get("unit_id", "")),
            doc_id=str(unit.get("doc_id", "")),
            page=int(unit.get("page") or 0),
            chunk_type=chunk_type,
            score=max(0.0, score),
            bm25=bm25,
            field_hits=field_hits,
            value_hits=value_hits,
            reasons=reasons,
            text=text,
        )

    def _bm25_score(self, query_tokens: Counter[str], unit_index: int) -> float:
        if not query_tokens or not self.avgdl:
            return 0.0
        counter = self.token_counters[unit_index]
        doc_len = max(1, self.doc_lengths[unit_index])
        k1 = 1.5
        b = 0.75
        score = 0.0
        for token, query_weight in query_tokens.items():
            tf = counter.get(token, 0)
            if not tf:
                continue
            denominator = tf + k1 * (1 - b + b * doc_len / self.avgdl)
            score += self.idf.get(token, 0.0) * (tf * (k1 + 1) / denominator) * min(query_weight, 3)
        return score


def _load_units(path: Path) -> list[dict[str, Any]]:
    units: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                units.append(json.loads(line))
    return units


def _expand_table_rows(units: list[dict[str, Any]]) -> list[dict[str, Any]]:
    expanded: list[dict[str, Any]] = []
    for unit in units:
        raw_text = unit.get("raw_text") or ""
        rows = _table_rows(raw_text)
        if not rows:
            expanded.append(unit)
            continue
        for row_index, row in enumerate(rows, start=1):
            row_unit = dict(unit)
            row_unit["unit_id"] = f"{unit.get('unit_id')}_r{row_index:03d}"
            row_unit["chunk_type"] = "table_row"
            row_unit["raw_text"] = row
            row_unit["search_text"] = "\n".join(
                str(part)
                for part in [
                    unit.get("title", ""),
                    unit.get("section_path", ""),
                    "table_row",
                    row,
                ]
                if part
            )
            row_unit["numbers"] = _numbers(row)
            expanded.append(row_unit)
    return expanded


def _table_rows(text: str) -> list[str]:
    rows: list[str] = []
    for line in text.splitlines():
        stripped = line.strip()
        if "|" not in stripped:
            continue
        cells = [cell.strip() for cell in stripped.split("|") if cell.strip()]
        if len(cells) < 2:
            continue
        if all(set(cell) <= {"-", ":"} for cell in cells):
            continue
        rows.append(" | ".join(cells))
    return rows


def _unit_search_text(unit: dict[str, Any]) -> str:
    return "\n".join(
        str(unit.get(field, ""))
        for field in ("title", "section_path", "clause_no", "chunk_type", "search_text", "raw_text")
    )


def _token_counter(text: str) -> Counter[str]:
    normalized = text.lower()
    tokens: list[str] = []
    for term in FINANCE_TERMS:
        if term.lower() in normalized:
            tokens.append(term.lower())
    for number in _numbers(normalized):
        tokens.append(number.lower())
    for organization in _organizations(text):
        tokens.append(organization.lower())
    for rating in _ratings(text):
        tokens.append(rating.lower())
    for match in CJK_RE.finditer(normalized):
        value = match.group(0)
        if len(value) == 1:
            tokens.append(value)
        else:
            tokens.extend(value[index : index + 2] for index in range(len(value) - 1))
            if len(value) >= 4:
                tokens.extend(value[index : index + 3] for index in range(len(value) - 2))
    tokens.extend(match.group(0) for match in WORD_RE.finditer(normalized))
    return Counter(token for token in tokens if token and token not in STOP_TOKENS)


def _bm25_idf(counters: list[Counter[str]]) -> dict[str, float]:
    doc_freq: Counter[str] = Counter()
    for counter in counters:
        doc_freq.update(counter.keys())
    total = max(1, len(counters))
    return {
        token: math.log(1 + (total - freq + 0.5) / (freq + 0.5))
        for token, freq in doc_freq.items()
    }


def _fields(text: str) -> list[str]:
    return [
        field
        for field, terms in FIELD_TERMS.items()
        if any(term in text for term in terms)
    ]


def _field_hits(text: str, fields: list[str]) -> list[str]:
    hits: list[str] = []
    for field in fields:
        if any(term in text for term in FIELD_TERMS.get(field, [])):
            hits.append(field)
    return hits


def _chunk_type_matches_fields(chunk_type: str, fields: list[str]) -> bool:
    return any(chunk_type in FIELD_CHUNK_HINTS.get(field, set()) for field in fields)


def _query_text(question_text: str, parsed: ParsedOption) -> str:
    field_terms = [term for field in parsed.fields for term in FIELD_TERMS.get(field, [])[:2]]
    return "\n".join([question_text, parsed.text, " ".join(field_terms), " ".join(parsed.values)])


def _numbers(text: str) -> list[str]:
    return _dedupe(match.group(0).strip() for match in NUMBER_RE.finditer(text) if match.group(0).strip())


def _organizations(text: str) -> list[str]:
    return _dedupe(_clean_organization(match.group(0).strip()) for match in ORG_RE.finditer(text) if match.group(0).strip())


def _ratings(text: str) -> list[str]:
    return _dedupe(match.group(0).strip() for match in RATING_RE.finditer(text) if match.group(0).strip())


def _option_doc_ids(question: Question, documents: list[Document], option_text: str) -> list[str]:
    doc_ids = [document.doc_id for document in documents]
    if len(doc_ids) < 2:
        return doc_ids
    has_first = "第一份文档" in option_text or "第一份" in option_text
    has_second = "第二份文档" in option_text or "第二份" in option_text
    if has_first and has_second:
        return doc_ids
    if has_first:
        return [doc_ids[0]]
    if has_second:
        return [doc_ids[1]]
    return doc_ids


def _is_compare_option(question_text: str, option_text: str, doc_count: int) -> bool:
    if doc_count < 2:
        return False
    text = f"{question_text}\n{option_text}"
    markers = [
        "两份文档",
        "两份",
        "第一份",
        "第二份",
        "均",
        "都",
        "低于",
        "高于",
        "不低于",
        "不高于",
        "少于",
        "多于",
        "相同",
        "不同",
        "分别",
        "二者",
        "两者",
    ]
    return any(marker in text for marker in markers)


def _allowed_pages(pages_by_doc: dict[str, list[int]] | None) -> dict[str, set[int]]:
    return {doc_id: set(pages) for doc_id, pages in (pages_by_doc or {}).items()}


def _page_allowed(unit: dict[str, Any], allowed_pages: dict[str, set[int]]) -> bool:
    if not allowed_pages:
        return True
    doc_id = str(unit.get("doc_id", ""))
    return int(unit.get("page") or 0) in allowed_pages.get(doc_id, set())


def _noise_penalty(unit: dict[str, Any]) -> float:
    text = "\n".join(str(unit.get(field, "")) for field in ("section_path", "raw_text"))
    if not any(marker in text for marker in NOISE_MARKERS):
        return 0.0
    if any(marker in text for marker in ("发行概况", "评级", "受托管理", "募集资金")):
        return 8.0
    return 22.0


def _dedupe(values) -> list[str]:
    seen: set[str] = set()
    output: list[str] = []
    for value in values:
        if value and value not in seen:
            seen.add(value)
            output.append(value)
    return output


def _clean_organization(value: str) -> str:
    for marker in ("名称为", "指定", "为", "：", ":"):
        if marker in value:
            value = value.split(marker)[-1]
    return value.strip(" ，。；;、")
