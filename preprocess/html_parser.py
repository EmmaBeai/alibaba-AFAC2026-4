"""HTML (.html) extraction.

Covers the 3 regulatory CSRC pages (``csrc_*.html``): full government web pages
that wrap a 行政处罚决定书 body inside ``<div class="detail-news">``.

Two observed quirks drive the approach:
  - The real content lives in ``detail-news``; everything else is site chrome
    (nav menu 首页/机构概况/…, footer, scripts). We target the container, not the
    whole page.
  - Every digit/char is wrapped in its own ``<span>``/``<font>`` (MS-Word paste),
    so a naive ``get_text("\\n")`` shatters ``〔2024〕112号`` into 5 lines. We
    instead pull text per ``<p>`` block, concatenating inline pieces with no
    separator (correct for CJK), which reconstructs whole paragraphs. Extracting
    only ``<p>`` also skips nav/footer, which use ``<li>``/``<a>``.

Uses bs4's built-in ``html.parser`` (no lxml dependency).
"""

from __future__ import annotations

from pathlib import Path

from bs4 import BeautifulSoup

# Tried in order; first match wins. detail-news is the tight body container,
# content/main are looser fallbacks for pages that differ.
_CONTENT_CLASSES = ("detail-news", "content", "main")


def parse(path: Path) -> str:
    """Extract the main-content body of an HTML page as clean paragraph text."""
    # read_bytes + let bs4 detect the encoding from the page's meta charset.
    soup = BeautifulSoup(Path(path).read_bytes(), "html.parser")

    node = None
    for cls in _CONTENT_CLASSES:
        node = soup.find("div", class_=cls)
        if node is not None:
            break
    if node is None:
        node = soup.body or soup

    for junk in node(["script", "style"]):
        junk.decompose()

    # One line per <p> block; inline spans/fonts concatenate (no CJK-breaking spaces).
    paragraphs = [p.get_text("", strip=True) for p in node.find_all("p")]
    paragraphs = [p for p in paragraphs if p]

    # Fallback for pages with no <p> structure: take all text, line by line.
    if not paragraphs:
        paragraphs = [ln.strip() for ln in node.get_text("\n").splitlines() if ln.strip()]

    return "\n".join(paragraphs).strip()


def extract_title(path: Path) -> str | None:
    """Harvest a real document title from CSRC page metadata, if present.

    Prefers the ``ArticleTitle`` meta — a document-specific title like
    ``中国证监会市场禁入决定书（朱要文…）`` that beats the ``doc_id`` placeholder and
    also encodes the doc type (行政处罚决定书 vs 市场禁入决定书). Falls back to the
    ``<title>`` tag with its ``_中国证券监督管理委员会`` site suffix stripped.

    Returns ``None`` when neither yields anything, so the caller can fall back to
    the ``doc_id``. Note this meta only exists for the CSRC ``.html`` source — it
    is a per-source bonus, not a corpus-wide field.
    """
    soup = BeautifulSoup(Path(path).read_bytes(), "html.parser")

    meta = soup.find("meta", attrs={"name": "ArticleTitle"})
    if meta is not None:
        content = (meta.get("content") or "").strip()
        if content:
            return content

    if soup.title and soup.title.string:
        title = soup.title.string.split("_")[0].strip()
        if title:
            return title

    return None
