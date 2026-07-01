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
    parser_source = _parser_source(doc_id, source_kind, processed_root)
    title_texts = _title_texts_for_source(
        doc_id,
        source_path,
        source_kind,
        processed_root=processed_root,
        parser_source=parser_source,
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


def _parser_source(doc_id: str, source_kind: str, processed_root: Path) -> str:
    if source_kind == "pdf":
        if (processed_root / "pdf_parsed" / "glm-ocr" / f"{doc_id}.md").exists():
            return "glm-ocr"
        if (processed_root / "pdf_parsed" / "pypdf" / f"{doc_id}.md").exists():
            return "pypdf"
        return "raw_pdf"
    if source_kind == "html":
        return "html_parser"
    if source_kind == "txt":
        return "txt_parser"
    return f"raw_{source_kind}"


def _title_texts_for_source(
    doc_id: str,
    source_path: Path,
    source_kind: str,
    *,
    processed_root: Path,
    parser_source: str,
) -> list[str]:
    if source_kind == "pdf":
        cleaned_titles = _h1_titles_from_markdown_file(
            processed_root / "cleaned_markdown_68" / f"{doc_id}.md",
        )
        if cleaned_titles:
            return cleaned_titles
        parsed_titles = _pdf_h1_titles_from_parsed_markdown(
            doc_id,
            processed_root,
            parser_source,
        )
        if parsed_titles:
            return parsed_titles
        for fallback_parser in ("glm-ocr", "pypdf"):
            if fallback_parser == parser_source:
                continue
            parsed_titles = _pdf_h1_titles_from_parsed_markdown(
                doc_id,
                processed_root,
                fallback_parser,
            )
            if parsed_titles:
                return parsed_titles
    if source_kind == "html":
        title = extract_html_title(source_path)
        return [_truncate_title(title)] if title else []
    return []


def _pdf_h1_titles_from_parsed_markdown(
    doc_id: str,
    processed_root: Path,
    parser_source: str,
) -> list[str]:
    if parser_source not in {"glm-ocr", "pypdf"}:
        return []
    parsed_path = processed_root / "pdf_parsed" / parser_source / f"{doc_id}.md"
    return _h1_titles_from_markdown_file(parsed_path)


def _h1_titles_from_markdown_file(parsed_path: Path) -> list[str]:
    if not parsed_path.exists():
        return []
    h1_titles: list[str] = []
    for line in parsed_path.read_text(encoding="utf-8").splitlines():
        match = re.match(r"^(#)\s+(.+?)\s*$", line)
        if not match:
            continue
        title = _clean_markdown_line(match.group(2))
        if title and not _numbered_heading(title):
            return [_truncate_title(title)]
    return []


def _clean_markdown_line(line: str) -> str:
    line = re.sub(r"<[^>]+>", "", line).strip()
    line = re.sub(r"!\[[^\]]*\]\([^)]*\)", "", line).strip()
    line = line.strip("#").strip()
    return re.sub(r"\s+", " ", line)


def _numbered_heading(value: str) -> bool:
    return bool(re.fullmatch(r"\d+(?:\.\d+)*\s*.*", value.strip()))


def _truncate_title(title: str) -> str:
    return title if len(title) <= 120 else title[:117] + "..."


def _period_for(doc_id: str, title_texts: list[str]) -> str | None:
    return _year_from_text(" ".join([*title_texts, doc_id]))


def _year_from_text(value: str) -> str | None:
    match = YEAR_RE.search(value)
    return match.group(1) if match else None
