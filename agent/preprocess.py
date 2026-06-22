from __future__ import annotations

import html
import importlib
import re
from dataclasses import asdict, dataclass
from html.parser import HTMLParser
from pathlib import Path

from pypdf import PdfReader

from agent.schemas import ExtractedPages, Page

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
PAGE_COMMENT_RE = re.compile(r"(?i)<!--\s*page\s*:\s*(\d+)\s*-->")
MD_PAGE_RE = re.compile(
    r"(?im)^\s*(?:<!--\s*)?(?:[-#>*\s]*)?(?:page|页码|第)\s*[:：#-]?\s*(\d+)\s*(?:页)?\s*(?:[-#>*\s]*)?(?:-->)?\s*$"
)

FINANCE_TERMS = (
    "营业收入", "净利润", "现金流", "资产负债率", "资本充足率", "不良贷款",
    "发行规模", "发行金额", "募集资金", "主体信用评级", "债券受托管理人",
    "保险责任", "保险金额", "现金价值", "给付", "赔付",
)
MOJIBAKE_MARKERS = ("å", "æ", "ç", "è", "é", "ä", "ï¼", "ã€")
PYPDF_MODEL = "pypdf"
SUSPICIOUS_UNICODE_RE = re.compile(
    r"[\u0370-\u03ff\u0a00-\u0a7f\u0d00-\u0d7f\u0e00-\u0e7f\u0f00-\u0fff]"
)
CHINESE_RE = re.compile(r"[\u3400-\u9fff]")
ASCII_WORD_RE = re.compile(r"[A-Za-z0-9]")


@dataclass(frozen=True)
class PdfSourceQuality:
    model: str
    chars: int
    chinese_chars: int
    chinese_ratio: float
    suspicious_chars: int
    suspicious_ratio: float
    mojibake_markers: int
    replacement_chars: int
    finance_term_hits: int
    image_markers: int
    score: float
    acceptable: bool


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
    return extract_pages_with_metadata(
        path,
        text_page_chars=text_page_chars,
        pdf_parsed_dir=pdf_parsed_dir,
        pdf_model_order=pdf_model_order,
    ).pages


def extract_pages_with_metadata(
    path: Path,
    text_page_chars: int = 8000,
    pdf_parsed_dir: Path | None = None,
    pdf_model_order: list[str] | tuple[str, ...] | None = None,
) -> ExtractedPages:
    suffix = path.suffix.lower()
    if suffix == ".pdf":
        markdown_extract = _extract_pdf_markdown_pages(
            path,
            text_page_chars=text_page_chars,
            pdf_parsed_dir=pdf_parsed_dir,
            pdf_model_order=pdf_model_order,
        )
        if markdown_extract is not None:
            return markdown_extract
        reader = PdfReader(str(path))
        pages = [
            Page(page_number=index, text=clean_text(page.extract_text() or ""))
            for index, page in enumerate(reader.pages, start=1)
        ]
        return ExtractedPages(
            pages=pages,
            metadata={
                "pdf_source": {
                    "selected_model": "raw_pypdf_fallback",
                    "selection_reason": "no parsed markdown available",
                }
            },
        )
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
    return ExtractedPages(
        pages=[Page(page_number=index, text=chunk) for index, chunk in enumerate(chunks, start=1)]
    )


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
    page_markers: list[str] = []

    def keep_page_marker(match: re.Match[str]) -> str:
        page_markers.append(f"<!-- page:{match.group(1)} -->")
        return f"@@PAGE_MARKER_{len(page_markers) - 1}@@"

    text = PAGE_COMMENT_RE.sub(keep_page_marker, text)
    text = MARKDOWN_IMAGE_RE.sub("", text)
    text = HTML_BREAK_RE.sub("\n", text)
    text = HTML_TABLE_CELL_RE.sub(" | ", text)
    text = HTML_TABLE_ROW_RE.sub("\n", text)
    text = HTML_BLOCK_RE.sub("\n", text)
    text = HTML_TAG_RE.sub("", text)
    for index, marker in enumerate(page_markers):
        text = text.replace(f"@@PAGE_MARKER_{index}@@", marker)
    return html.unescape(text)


def score_pdf_source(model: str, text: str) -> PdfSourceQuality:
    chars = len(text)
    chinese_chars = len(CHINESE_RE.findall(text))
    ascii_chars = len(ASCII_WORD_RE.findall(text))
    suspicious_chars = len(SUSPICIOUS_UNICODE_RE.findall(text))
    mojibake_markers = sum(text.count(marker) for marker in MOJIBAKE_MARKERS)
    replacement_chars = text.count("�")
    finance_term_hits = sum(text.count(term) for term in FINANCE_TERMS)
    image_markers = text.count("![Image")
    denominator = max(1, chars)
    chinese_ratio = chinese_chars / denominator
    suspicious_ratio = suspicious_chars / denominator
    useful_chars = chinese_chars + min(ascii_chars, chinese_chars // 2 + 2000)
    score = (
        useful_chars
        + finance_term_hits * 450.0
        - suspicious_chars * 18.0
        - mojibake_markers * 300.0
        - replacement_chars * 120.0
        - image_markers * 80.0
    )
    if model == PYPDF_MODEL:
        score *= 1.08
    acceptable = (
        chars >= 1000
        and chinese_ratio >= 0.08
        and suspicious_ratio <= 0.03
        and mojibake_markers <= 5
        and replacement_chars <= 5
        and score > 0
    )
    return PdfSourceQuality(
        model=model,
        chars=chars,
        chinese_chars=chinese_chars,
        chinese_ratio=round(chinese_ratio, 4),
        suspicious_chars=suspicious_chars,
        suspicious_ratio=round(suspicious_ratio, 4),
        mojibake_markers=mojibake_markers,
        replacement_chars=replacement_chars,
        finance_term_hits=finance_term_hits,
        image_markers=image_markers,
        score=round(score, 3),
        acceptable=acceptable,
    )


def select_pdf_source(candidates: list[tuple[str, str, PdfSourceQuality]]) -> tuple[str, str, PdfSourceQuality]:
    acceptable = [candidate for candidate in candidates if candidate[2].acceptable]
    pool = acceptable or candidates
    best = max(pool, key=lambda item: item[2].score)
    pypdf = next((item for item in pool if item[0] == PYPDF_MODEL), None)
    if pypdf and pypdf[2].acceptable and pypdf[2].score >= best[2].score * 0.72:
        return pypdf
    return best


def _selection_reason(
    selected_model: str,
    candidates: list[tuple[str, str, PdfSourceQuality]],
) -> str:
    selected = next(quality for model, _, quality in candidates if model == selected_model)
    if selected_model == PYPDF_MODEL and selected.acceptable:
        return "pypdf text layer is healthy and close enough to the best source"
    if not selected.acceptable:
        return "no acceptable parsed source; selected highest score"
    return "selected highest acceptable quality score"


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
) -> ExtractedPages | None:
    try:
        pdf_parser = importlib.import_module("preprocess.pdf_parser")
    except ModuleNotFoundError:
        return None

    parsed_dir = pdf_parsed_dir or getattr(pdf_parser, "PDF_PARSED_DIR")
    model_order = pdf_model_order or getattr(pdf_parser, "DEFAULT_MODEL_ORDER")
    candidates: list[tuple[str, str, PdfSourceQuality]] = []
    for model in model_order:
        try:
            text = pdf_parser.parse(path, model=model, parsed_dir=Path(parsed_dir))
        except FileNotFoundError:
            continue
        text = clean_text(text)
        candidates.append((model, text, score_pdf_source(model, text)))
    if not candidates:
        return None

    selected_model, selected_text, selected_quality = select_pdf_source(candidates)
    pages = _split_markdown_pages(selected_text)
    split_method = "page_markers"
    if not pages:
        chunks = _split_text(selected_text, text_page_chars)
        pages = [Page(page_number=index, text=chunk) for index, chunk in enumerate(chunks, start=1)]
        split_method = "char_chunks"
    return ExtractedPages(
        pages=pages,
        metadata={
            "pdf_source": {
                "selected_model": selected_model,
                "selection_reason": _selection_reason(selected_model, candidates),
                "split_method": split_method,
                "qualities": [asdict(quality) for _, _, quality in candidates],
                "selected_quality": asdict(selected_quality),
            }
        },
    )


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
