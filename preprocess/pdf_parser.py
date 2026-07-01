"""PDF markdown consumption for the production preprocessing chain.

PDF conversion is an offline step handled by ``preprocess.pdf_to_markdown``.
That step writes one markdown file per ``(parser, doc_id)`` under:

``processed_data/pdf_parsed/<parser>/<doc_id>.md``

The current production parser set is intentionally small: ``glm-ocr`` first,
then ``pypdf`` as the fallback text-layer parser. This module only reads those
materialized markdown files; it does not run OCR or import GPU libraries.
"""

from __future__ import annotations

from pathlib import Path

PDF_PARSED_DIR = Path("processed_data/pdf_parsed")
DEFAULT_MODEL_ORDER = ("glm-ocr", "pypdf")


def markdown_path(pdf_path: Path, model: str, parsed_dir: Path = PDF_PARSED_DIR) -> Path:
    """Path to a model's markdown for a document."""
    return parsed_dir / model / f"{Path(pdf_path).stem}.md"


def available_models(pdf_path: Path, parsed_dir: Path = PDF_PARSED_DIR) -> list[str]:
    """Models that produced non-empty markdown, in production preference order."""
    found: list[str] = []
    for model in DEFAULT_MODEL_ORDER:
        path = markdown_path(pdf_path, model, parsed_dir)
        if path.exists() and path.read_text(encoding="utf-8", errors="replace").strip():
            found.append(model)
    return found


def parse(
    path: Path,
    model: str | None = None,
    parsed_dir: Path = PDF_PARSED_DIR,
) -> str:
    """Return one PDF's selected pre-converted markdown text."""
    order = (model,) if model is not None else DEFAULT_MODEL_ORDER
    for candidate in order:
        md = markdown_path(path, candidate, parsed_dir)
        if not md.exists():
            continue
        text = md.read_text(encoding="utf-8", errors="replace").strip()
        if text:
            return text
    raise FileNotFoundError(
        f"no non-empty markdown for {Path(path).stem!r} under {parsed_dir} "
        f"(tried {list(order)})"
    )


def extract_title(
    path: Path,
    model: str | None = None,
    parsed_dir: Path = PDF_PARSED_DIR,
) -> str | None:
    """Use the selected markdown's first H1 heading as a lightweight title."""
    try:
        text = parse(path, model=model, parsed_dir=parsed_dir)
    except FileNotFoundError:
        return None
    for line in text.splitlines():
        stripped = line.strip()
        if stripped.startswith("#"):
            title = stripped.lstrip("#").strip()
            if title:
                return title
    return None
