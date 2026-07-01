from __future__ import annotations

import math
from collections.abc import Collection, Mapping


PhraseQids = Mapping[str, Collection[str]]
PhraseQidsByDomain = Mapping[str, Mapping[str, Collection[str]]]
DomainQids = Mapping[str, Collection[str]]


def purity_score(
    phrase: str,
    domain: str,
    phrase_qids_by_domain: PhraseQidsByDomain,
    qids_by_domain: DomainQids,
    *,
    alpha: float = 0.5,
) -> float:
    phrase_domains = phrase_qids_by_domain.get(phrase, {})
    domain_total = len(qids_by_domain.get(domain, ()))
    if domain_total == 0:
        return 0.0

    domain_count = len(phrase_domains.get(domain, ()))
    domain_prob = _smoothed_probability(domain_count, domain_total, alpha)

    other_probs = [
        _smoothed_probability(len(phrase_domains.get(other_domain, ())), len(other_qids), alpha)
        for other_domain, other_qids in qids_by_domain.items()
        if other_domain != domain and len(other_qids) > 0
    ]
    if not other_probs:
        return 0.0

    return math.log(domain_prob) - math.log(max(other_probs))


def phraseness_score(
    phrase: str,
    phrase_qids: PhraseQids,
    total_qids: int,
    *,
    alpha: float = 0.5,
) -> float:
    if len(phrase) < 2 or total_qids <= 0:
        return 0.0

    phrase_prob = _smoothed_probability(len(phrase_qids.get(phrase, ())), total_qids, alpha)
    split_scores: list[float] = []
    for split_at in range(1, len(phrase)):
        left = phrase[:split_at]
        right = phrase[split_at:]
        left_prob = _smoothed_probability(len(phrase_qids.get(left, ())), total_qids, alpha)
        right_prob = _smoothed_probability(len(phrase_qids.get(right, ())), total_qids, alpha)
        split_scores.append(math.log(phrase_prob) - math.log(left_prob) - math.log(right_prob))

    return min(split_scores)


def completeness_score(phrase: str, phrase_qids: PhraseQids) -> float:
    current_qids = set(phrase_qids.get(phrase, ()))
    if not current_qids:
        return 0.0

    max_longer_count = 0
    for longer_phrase, longer_qids in phrase_qids.items():
        if len(longer_phrase) <= len(phrase) or phrase not in longer_phrase:
            continue
        max_longer_count = max(max_longer_count, len(current_qids.intersection(longer_qids)))

    return 1.0 - (max_longer_count / len(current_qids))


def _smoothed_probability(count: int, total: int, alpha: float) -> float:
    return (count + alpha) / (total + 2 * alpha)
