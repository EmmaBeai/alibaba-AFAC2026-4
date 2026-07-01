from __future__ import annotations

import re
from pathlib import Path
from typing import Any

from preprocess.html_parser import extract_title as extract_html_title


YEAR_RE = re.compile(r"(?<!\d)(20\d{2})(?!\d)")


def build_document_record(
    *,
    doc_id: str,
    source_path: Path,
    source_kind: str,
    domain: str,
    processed_root: Path,
) -> dict[str, Any]:
    title_texts = _title_texts_for_source(
        doc_id,
        source_path,
        source_kind,
        processed_root=processed_root,
    )
    record: dict[str, Any] = {
        "doc_id": doc_id,
        "domain": domain,
    }
    if title_texts:
        record["title"] = " ".join(title_texts)
    period = _period_for(doc_id, title_texts)
    if period:
        record["dates"] = {"period": period}
    return record


def _title_texts_for_source(
    doc_id: str,
    source_path: Path,
    source_kind: str,
    *,
    processed_root: Path,
) -> list[str]:
    if source_kind == "pdf":
        cleaned_path = processed_root / "cleaned_markdown" / f"{doc_id}.md"
        cleaned_titles = _h1_titles_from_markdown_file(
            cleaned_path,
        )
        if cleaned_titles:
            return cleaned_titles
        raise RuntimeError(
            "Cleaned markdown is missing a valid H1 title "
            f"for doc_id={doc_id}: {cleaned_path}. "
            "Re-run markdown cleaning before building documents."
        )
    if source_kind == "html":
        title = extract_html_title(source_path)
        return [_truncate_title(title)] if title else []
    return []


def _h1_titles_from_markdown_file(parsed_path: Path) -> list[str]:
    if not parsed_path.exists():
        return []
    for line in parsed_path.read_text(encoding="utf-8").splitlines():
        match = re.match(r"^(#)\s+(.+?)\s*$", line)
        if not match:
            continue
        title = _clean_markdown_line(match.group(2))
        if title:
            return [_truncate_title(title)]
    return []


def _clean_markdown_line(line: str) -> str:
    line = re.sub(r"<[^>]+>", "", line).strip()
    line = re.sub(r"!\[[^\]]*\]\([^)]*\)", "", line).strip()
    line = line.strip("#").strip()
    return re.sub(r"\s+", " ", line)


def _truncate_title(title: str) -> str:
    return title if len(title) <= 120 else title[:117] + "..."


def _period_for(doc_id: str, title_texts: list[str]) -> str | None:
    return _year_from_text(" ".join([*title_texts, doc_id]))


def _year_from_text(value: str) -> str | None:
    match = YEAR_RE.search(value)
    return match.group(1) if match else None
