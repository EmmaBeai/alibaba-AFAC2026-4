from __future__ import annotations

import html
import importlib
import re
from html.parser import HTMLParser
from pathlib import Path

from pypdf import PdfReader

from agent.schemas import Page

SPACE_RE = re.compile(r"[ \t]+")
BLANK_RE = re.compile(r"\n{3,}")
SURROGATE_RE = re.compile(r"[\ud800-\udfff]")
CONTROL_RE = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")
MARKDOWN_IMAGE_RE = re.compile(r"!\[[^\]]*]\([^)]*\)")
HTML_BREAK_RE = re.compile(r"(?i)<\s*br\s*/?\s*>")
HTML_TABLE_CELL_RE = re.compile(r"(?i)</\s*(?:td|th)\s*>\s*<\s*(?:td|th)[^>]*>")
HTML_TABLE_ROW_RE = re.compile(r"(?i)</\s*tr\s*>\s*<\s*tr[^>]*>")
HTML_BLOCK_RE = re.compile(r"(?i)</?\s*(?:div|p|table|tr|h[1-6])[^>]*>")
HTML_TAG_RE = re.compile(r"<[^>]+>")
MD_PAGE_RE = re.compile(
    r"(?im)^\s*(?:<!--\s*)?(?:[-#>*\s]*)?(?:page|页码|第)\s*[:：#-]?\s*(\d+)\s*(?:页)?\s*(?:[-#>*\s]*)?(?:-->)?\s*$"
)


class _HTMLTextExtractor(HTMLParser):
    BLOCK_TAGS = {
        "article", "br", "div", "h1", "h2", "h3", "h4", "h5", "h6",
        "li", "p", "section", "table", "td", "th", "tr",
    }

    def __init__(self) -> None:
        super().__init__()
        self.parts: list[str] = []
        self.ignored_depth = 0

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag in {"script", "style", "nav"}:
            self.ignored_depth += 1
        elif tag in self.BLOCK_TAGS:
            self.parts.append("\n")

    def handle_endtag(self, tag: str) -> None:
        if tag in {"script", "style", "nav"} and self.ignored_depth:
            self.ignored_depth -= 1
        elif tag in self.BLOCK_TAGS:
            self.parts.append("\n")

    def handle_data(self, data: str) -> None:
        if not self.ignored_depth:
            self.parts.append(data)


def extract_pages(
    path: Path,
    text_page_chars: int = 8000,
    pdf_parsed_dir: Path | None = None,
    pdf_model_order: list[str] | tuple[str, ...] | None = None,
) -> list[Page]:
    suffix = path.suffix.lower()
    if suffix == ".pdf":
        markdown_pages = _extract_pdf_markdown_pages(
            path,
            text_page_chars=text_page_chars,
            pdf_parsed_dir=pdf_parsed_dir,
            pdf_model_order=pdf_model_order,
        )
        if markdown_pages is not None:
            return markdown_pages
        reader = PdfReader(str(path))
        return [
            Page(page_number=index, text=clean_text(page.extract_text() or ""))
            for index, page in enumerate(reader.pages, start=1)
        ]
    if suffix == ".txt":
        text = path.read_text(encoding="utf-8", errors="replace")
    elif suffix in {".html", ".htm"}:
        parser = _HTMLTextExtractor()
        parser.feed(path.read_text(encoding="utf-8", errors="replace"))
        text = html.unescape("".join(parser.parts))
    else:
        raise ValueError(f"Unsupported document type: {path}")
    text = clean_text(text)
    chunks = _split_text(text, text_page_chars)
    return [Page(page_number=index, text=chunk) for index, chunk in enumerate(chunks, start=1)]


def clean_text(text: str) -> str:
    text = sanitize_text(text)
    text = clean_markup_noise(text)
    lines = [SPACE_RE.sub(" ", line).strip() for line in text.replace("\r", "\n").splitlines()]
    return BLANK_RE.sub("\n\n", "\n".join(lines)).strip()


def sanitize_text(text: str) -> str:
    text = SURROGATE_RE.sub("", text)
    text = CONTROL_RE.sub("", text)
    return text


def clean_markup_noise(text: str) -> str:
    text = MARKDOWN_IMAGE_RE.sub("", text)
    text = HTML_BREAK_RE.sub("\n", text)
    text = HTML_TABLE_CELL_RE.sub(" | ", text)
    text = HTML_TABLE_ROW_RE.sub("\n", text)
    text = HTML_BLOCK_RE.sub("\n", text)
    text = HTML_TAG_RE.sub("", text)
    return html.unescape(text)


def _split_text(text: str, target_chars: int) -> list[str]:
    if not text:
        return [""]
    paragraphs = text.split("\n")
    chunks: list[str] = []
    current: list[str] = []
    current_size = 0
    for paragraph in paragraphs:
        if current and current_size + len(paragraph) > target_chars:
            chunks.append("\n".join(current).strip())
            current, current_size = [], 0
        current.append(paragraph)
        current_size += len(paragraph) + 1
    if current:
        chunks.append("\n".join(current).strip())
    return chunks


def _extract_pdf_markdown_pages(
    path: Path,
    *,
    text_page_chars: int,
    pdf_parsed_dir: Path | None,
    pdf_model_order: list[str] | tuple[str, ...] | None,
) -> list[Page] | None:
    try:
        pdf_parser = importlib.import_module("preprocess.pdf_parser")
    except ModuleNotFoundError:
        return None

    parsed_dir = pdf_parsed_dir or getattr(pdf_parser, "PDF_PARSED_DIR")
    model_order = pdf_model_order or getattr(pdf_parser, "DEFAULT_MODEL_ORDER")
    for model in model_order:
        try:
            text = pdf_parser.parse(path, model=model, parsed_dir=Path(parsed_dir))
        except FileNotFoundError:
            continue
        text = clean_text(text)
        pages = _split_markdown_pages(text)
        if pages:
            return pages
        chunks = _split_text(text, text_page_chars)
        return [Page(page_number=index, text=chunk) for index, chunk in enumerate(chunks, start=1)]
    return None


def _split_markdown_pages(text: str) -> list[Page]:
    matches = list(MD_PAGE_RE.finditer(text))
    if not matches:
        return []
    pages: list[Page] = []
    for index, match in enumerate(matches):
        start = match.end()
        end = matches[index + 1].start() if index + 1 < len(matches) else len(text)
        page_text = text[start:end].strip()
        if page_text:
            pages.append(Page(page_number=int(match.group(1)), text=page_text))
    return pages
