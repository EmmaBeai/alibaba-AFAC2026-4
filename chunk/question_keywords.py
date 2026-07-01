from __future__ import annotations

import json
import re
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from chunk.question_phrase_scores import (
    phraseness_score,
    purity_score,
)


PHRASE_MIN_LEN = 2
GENERIC_PHRASE_MIN_LEN = 3
PHRASE_MAX_LEN = 20
COMPLETENESS_MIN = 0.5
PURITY_ALPHA = 0.5
NON_STRONG_MIN_QID_COUNT = 2


@dataclass(frozen=True)
class _QuestionRecord:
    qid: str
    domain: str
    text: str
    doc_ids: tuple[str, ...]
    phrases: frozenset[str]
    strong_phrases: frozenset[str]


@dataclass(frozen=True)
class _PhraseRank:
    phrase: str
    best_domain: str | None
    domain_counts: dict[str, int]
    qid_count: int
    strong_qid_count: int
    purity: float
    phraseness: float
    completeness: float


@dataclass(frozen=True)
class _KertLiteState:
    phrase_ranks: dict[str, _PhraseRank]
    doc_phrase_qids: dict[str, dict[str, set[str]]]


def build_question_keywords(
    questions_path: Path | str,
    *,
    keyword_limit: int | None = None,
) -> dict[str, list[str]]:
    if keyword_limit is not None and keyword_limit < 1:
        raise ValueError("keyword_limit must be positive when provided")
    state = _build_kert_lite_state(Path(questions_path))
    return _assign_ranked_phrases_to_documents(
        state.doc_phrase_qids,
        state.phrase_ranks,
        keyword_limit=keyword_limit,
    )


def build_question_phrase_vocabulary(questions_path: Path | str) -> list[dict[str, Any]]:
    state = _build_kert_lite_state(Path(questions_path))
    return [
        _phrase_rank_row(rank)
        for rank in _dedupe_near_duplicate_ranks(sorted(state.phrase_ranks.values(), key=_phrase_rank_sort_key))
    ]


def _build_kert_lite_state(questions_path: Path) -> _KertLiteState:
    records = _question_records(questions_path)
    if not records:
        return _KertLiteState(phrase_ranks={}, doc_phrase_qids={})

    phrase_qids: dict[str, set[str]] = defaultdict(set)
    phrase_qids_by_domain: dict[str, dict[str, set[str]]] = defaultdict(lambda: defaultdict(set))
    qids_by_domain: dict[str, set[str]] = defaultdict(set)
    strong_phrase_qids: dict[str, set[str]] = defaultdict(set)
    doc_phrase_qids: dict[str, dict[str, set[str]]] = defaultdict(lambda: defaultdict(set))
    phrases_by_qid: dict[str, set[str]] = {}

    for record in records:
        qids_by_domain[record.domain].add(record.qid)
        phrases_by_qid.setdefault(record.qid, set()).update(record.phrases)
        for phrase in record.phrases:
            phrase_qids[phrase].add(record.qid)
            phrase_qids_by_domain[phrase][record.domain].add(record.qid)
            for doc_id in record.doc_ids:
                doc_phrase_qids[doc_id][phrase].add(record.qid)
        for phrase in record.strong_phrases:
            strong_phrase_qids[phrase].add(record.qid)

    phrase_ranks = _rank_phrase_vocabulary(
        phrase_qids=phrase_qids,
        phrase_qids_by_domain=phrase_qids_by_domain,
        qids_by_domain=qids_by_domain,
        strong_phrase_qids=strong_phrase_qids,
        phrases_by_qid=phrases_by_qid,
    )
    return _KertLiteState(phrase_ranks=phrase_ranks, doc_phrase_qids=doc_phrase_qids)


def remove_cross_document_identity_keywords(documents: list[dict[str, Any]]) -> None:
    identities_by_doc = {
        document["doc_id"]: _document_identity_texts(document)
        for document in documents
    }
    for document in documents:
        current_doc_id = document["doc_id"]
        current_identities = identities_by_doc[current_doc_id]
        kept_keywords: list[str] = []
        for keyword in document["keywords"]:
            if _best_identity_overlap(keyword, current_identities) >= 3:
                kept_keywords.append(keyword)
                continue
            if any(
                _best_identity_overlap(keyword, identities) >= 3
                for doc_id, identities in identities_by_doc.items()
                if doc_id != current_doc_id
            ):
                continue
            kept_keywords.append(keyword)
        document["keywords"] = kept_keywords


def _question_records(questions_path: Path) -> list[_QuestionRecord]:
    files = sorted(questions_path.glob("*.json")) if questions_path.is_dir() else [questions_path]
    records: list[_QuestionRecord] = []
    for file_path in files:
        domain = _domain_from_questions_file(file_path)
        payload = json.loads(file_path.read_text(encoding="utf-8"))
        for index, item in enumerate(payload):
            text = _question_keyword_text(item)
            if not text:
                continue
            phrases, strong_phrases = _candidate_phrases(text)
            if not phrases:
                continue
            qid = str(item.get("qid") or f"{file_path.stem}:{index}")
            records.append(
                _QuestionRecord(
                    qid=qid,
                    domain=str(item.get("domain") or domain),
                    text=text,
                    doc_ids=tuple(str(doc_id) for doc_id in item.get("doc_ids") or ()),
                    phrases=frozenset(phrases),
                    strong_phrases=frozenset(strong_phrases),
                )
            )
    return records


def _domain_from_questions_file(file_path: Path) -> str:
    stem = file_path.stem
    if stem.endswith("_questions"):
        return stem.removesuffix("_questions")
    return stem


def _question_keyword_text(item: dict[str, Any]) -> str:
    question = _drop_question_tail_prompt(str(item.get("question") or ""))
    parts = [question]
    options = item.get("options")
    if isinstance(options, dict):
        parts.extend(str(value) for value in options.values())
    return " ".join(part for part in parts if part).strip()


def _drop_question_tail_prompt(question: str) -> str:
    question = question.strip()
    if not question.endswith(("?", "？")):
        return question
    body = question.rstrip("?？").strip()
    split_at = max(body.rfind(mark) for mark in ("，", ",", "；", ";", "。", ".", "：", ":"))
    if split_at < 0:
        return question
    return body[:split_at].strip()


def _candidate_phrases(text: str) -> tuple[set[str], set[str]]:
    text = _normalize_phrase_text(text)
    candidates: set[str] = set()
    strong_phrases: set[str] = set()

    for phrase in _strong_boundary_phrases(text):
        strong_phrases.add(phrase)
        candidates.add(phrase)

    for match in re.finditer(r"[0-9A-Za-z\u4e00-\u9fff]+", text):
        segment = match.group(0)
        if not segment:
            continue
        max_len = min(PHRASE_MAX_LEN, len(segment))
        for phrase_len in range(GENERIC_PHRASE_MIN_LEN, max_len + 1):
            for start in range(0, len(segment) - phrase_len + 1):
                phrase = _clean_phrase(segment[start : start + phrase_len])
                if _valid_phrase(phrase):
                    candidates.add(phrase)

    return candidates, strong_phrases


def _strong_boundary_phrases(text: str) -> set[str]:
    phrases: set[str] = set()
    for left, right in (("《", "》"), ("「", "」"), ("“", "”"), ('"', '"'), ("'", "'")):
        pattern = rf"{re.escape(left)}([^{re.escape(right)}]{{{PHRASE_MIN_LEN},{PHRASE_MAX_LEN}}}){re.escape(right)}"
        for match in re.finditer(pattern, text):
            phrase = _clean_phrase(match.group(1))
            if _valid_phrase(phrase):
                phrases.add(phrase)
    return phrases


def _rank_phrase_vocabulary(
    *,
    phrase_qids: dict[str, set[str]],
    phrase_qids_by_domain: dict[str, dict[str, set[str]]],
    qids_by_domain: dict[str, set[str]],
    strong_phrase_qids: dict[str, set[str]],
    phrases_by_qid: dict[str, set[str]],
) -> dict[str, _PhraseRank]:
    total_qids = sum(len(qids) for qids in qids_by_domain.values())
    has_domain_contrast = len([qids for qids in qids_by_domain.values() if qids]) > 1
    completeness_by_phrase = _completeness_scores(phrase_qids, phrases_by_qid)
    ranks: dict[str, _PhraseRank] = {}

    for phrase in phrase_qids:
        completeness = completeness_by_phrase.get(phrase, 0.0)
        strong_qid_count = len(strong_phrase_qids.get(phrase, set()))
        if strong_qid_count:
            completeness = max(completeness, strong_qid_count / len(phrase_qids[phrase]))
        elif len(phrase_qids[phrase]) < NON_STRONG_MIN_QID_COUNT:
            continue
        if completeness < COMPLETENESS_MIN:
            continue

        best_domain: str | None = None
        best_purity = 0.0
        domain_counts = {
            domain: len(qids)
            for domain, qids in sorted(phrase_qids_by_domain[phrase].items())
            if qids
        }
        if has_domain_contrast:
            domain_scores = [
                (
                    domain,
                    purity_score(
                        phrase,
                        domain,
                        phrase_qids_by_domain,
                        qids_by_domain,
                        alpha=PURITY_ALPHA,
                    ),
                )
                for domain in qids_by_domain
            ]
            best_domain, best_purity = max(domain_scores, key=lambda item: item[1])
            if best_purity <= 0.0:
                continue

        ranks[phrase] = _PhraseRank(
            phrase=phrase,
            best_domain=best_domain,
            domain_counts=domain_counts,
            qid_count=len(phrase_qids[phrase]),
            strong_qid_count=strong_qid_count,
            purity=best_purity,
            phraseness=phraseness_score(phrase, phrase_qids, total_qids, alpha=PURITY_ALPHA),
            completeness=completeness,
        )

    return ranks


def _phrase_rank_row(rank: _PhraseRank) -> dict[str, Any]:
    return {
        "phrase": rank.phrase,
        "best_domain": rank.best_domain,
        "domain_counts": rank.domain_counts,
        "qid_count": rank.qid_count,
        "strong_qid_count": rank.strong_qid_count,
        "purity": rank.purity,
        "phraseness": rank.phraseness,
        "completeness": rank.completeness,
    }


def _phrase_rank_sort_key(rank: _PhraseRank) -> tuple[Any, ...]:
    return (
        -rank.strong_qid_count,
        -rank.completeness,
        -len(rank.phrase),
        -rank.qid_count,
        -rank.purity,
        -rank.phraseness,
        rank.phrase,
    )


def _dedupe_near_duplicate_ranks(ranks: list[_PhraseRank]) -> list[_PhraseRank]:
    selected: list[_PhraseRank] = []
    for rank in ranks:
        if any(_near_duplicate_phrases(rank.phrase, chosen.phrase) for chosen in selected):
            continue
        selected.append(rank)
    return selected


def _near_duplicate_phrases(left: str, right: str) -> bool:
    shorter_len = min(len(left), len(right))
    longer_len = max(len(left), len(right))
    if shorter_len <= 4 or longer_len - shorter_len > 4:
        return False
    required_overlap = shorter_len - 1
    for length in range(shorter_len, required_overlap - 1, -1):
        for start in range(0, len(left) - length + 1):
            if left[start : start + length] in right:
                return True
    return False


def _completeness_scores(
    phrase_qids: dict[str, set[str]],
    phrases_by_qid: dict[str, set[str]],
) -> dict[str, float]:
    phrase_set = set(phrase_qids)
    pair_counts: dict[tuple[str, str], int] = defaultdict(int)

    for phrases in phrases_by_qid.values():
        qid_pairs: set[tuple[str, str]] = set()
        for longer_phrase in phrases:
            if len(longer_phrase) <= PHRASE_MIN_LEN:
                continue
            for start in range(len(longer_phrase)):
                max_end = min(len(longer_phrase), start + PHRASE_MAX_LEN)
                for end in range(start + PHRASE_MIN_LEN, max_end + 1):
                    if end - start >= len(longer_phrase):
                        continue
                    shorter_phrase = longer_phrase[start:end]
                    if shorter_phrase in phrase_set and shorter_phrase in phrases:
                        qid_pairs.add((shorter_phrase, longer_phrase))
        for pair in qid_pairs:
            pair_counts[pair] += 1

    max_longer_counts: dict[str, int] = defaultdict(int)
    for (phrase, _longer_phrase), count in pair_counts.items():
        max_longer_counts[phrase] = max(max_longer_counts[phrase], count)

    return {
        phrase: 1.0 - (max_longer_counts.get(phrase, 0) / len(qids))
        for phrase, qids in phrase_qids.items()
        if qids
    }


def _assign_ranked_phrases_to_documents(
    doc_phrase_qids: dict[str, dict[str, set[str]]],
    phrase_ranks: dict[str, _PhraseRank],
    *,
    keyword_limit: int | None,
) -> dict[str, list[str]]:
    keywords_by_doc_id: dict[str, list[str]] = {}
    for doc_id, phrase_qids in doc_phrase_qids.items():
        candidates = [
            phrase_ranks[phrase]
            for phrase in phrase_qids
            if phrase in phrase_ranks
        ]
        candidates.sort(
            key=lambda rank: (
                -len(phrase_qids[rank.phrase]),
                -rank.qid_count,
                -rank.strong_qid_count,
                -rank.purity,
                -rank.phraseness,
                -rank.completeness,
                -len(rank.phrase),
                rank.phrase,
            )
        )

        selected: list[str] = []
        for rank in candidates:
            if any(_phrases_overlap(rank.phrase, chosen) for chosen in selected):
                continue
            selected.append(rank.phrase)
            if keyword_limit is not None and len(selected) >= keyword_limit:
                break
        keywords_by_doc_id[doc_id] = selected
    return keywords_by_doc_id


def _phrases_overlap(left: str, right: str) -> bool:
    if left in right or right in left:
        return True
    shorter = min(len(left), len(right))
    if shorter <= 2:
        return False
    required_overlap = shorter - 1
    for length in range(shorter, required_overlap - 1, -1):
        for start in range(0, len(left) - length + 1):
            if left[start : start + length] in right:
                return True
    return False


def _normalize_phrase_text(text: str) -> str:
    return (
        text.replace("（", "(")
        .replace("）", ")")
        .replace("【", "《")
        .replace("】", "》")
    )


def _clean_phrase(phrase: str) -> str:
    return phrase.strip(" ：:-—。,.，；;()（）?？!！")


def _valid_phrase(phrase: str) -> bool:
    if len(phrase) < PHRASE_MIN_LEN or len(phrase) > PHRASE_MAX_LEN:
        return False
    if re.search(r"\d", phrase):
        return False
    return bool(re.search(r"[\u4e00-\u9fffA-Za-z]", phrase))


def _document_identity_texts(document: dict[str, Any]) -> list[str]:
    values = [_compact_identity_text(str(document["title"]))]
    values.extend(
        _compact_identity_text(str(entity["name"]))
        for entity in document["identity_entities"]
    )
    return [value for value in values if value]


def _best_identity_overlap(keyword: str, identity_texts: list[str]) -> int:
    compact_keyword = _compact_identity_text(keyword)
    if not compact_keyword:
        return 0
    best = 0
    for identity_text in identity_texts:
        best = max(best, _longest_common_substring_length(compact_keyword, identity_text))
    return best


def _compact_identity_text(value: str) -> str:
    return re.sub(r"[^0-9A-Za-z\u4e00-\u9fff]", "", value)


def _longest_common_substring_length(left: str, right: str) -> int:
    best = 0
    for start in range(len(left)):
        for end in range(start + best + 1, len(left) + 1):
            if left[start:end] in right:
                best = end - start
    return best
