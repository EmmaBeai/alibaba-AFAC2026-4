from __future__ import annotations

import html
import re
from html.parser import HTMLParser
from pathlib import Path

from pypdf import PdfReader

from agent.schemas import Page

SPACE_RE = re.compile(r"[ \t]+")
BLANK_RE = re.compile(r"\n{3,}")


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


def extract_pages(path: Path, text_page_chars: int = 8000) -> list[Page]:
    suffix = path.suffix.lower()
    if suffix == ".pdf":
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
    lines = [SPACE_RE.sub(" ", line).strip() for line in text.replace("\r", "\n").splitlines()]
    return BLANK_RE.sub("\n\n", "\n".join(lines)).strip()


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

