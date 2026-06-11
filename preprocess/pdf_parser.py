"""PDF (.pdf/.PDF) extraction — reads pre-converted markdown.

Conversion happens OFFLINE on a GPU box via ``script/pdf_parse_three.py``, which
runs document parsers (mineru2.5-pro, paddleocr-vl-1.6) and writes
ONE markdown per (model, doc) under ``PDF_PARSED_DIR/<model>/<doc_id>.md``.

This module is the CONSUMPTION layer. We store all parses (audit + later
cross-checking), but the pipeline consumes exactly ONE per doc so retrieval
indexes a single version — picked by ``DEFAULT_MODEL_ORDER``, falling through to
the next model when one's output is missing or empty (the converter records
per-doc failures, so some docs have fewer than three outputs).

No torch here — this only reads committed markdown, keeping the main pipeline
light and GPU-free. Uniform contract with the other parsers: ``parse(path) -> str``,
plus ``extract_title(path) -> str | None``.
"""

from __future__ import annotations

from pathlib import Path

PDF_PARSED_DIR = Path("processed_data/pdf_parsed")

# Consumption order: first non-empty markdown wins. MinerU2.5-Pro first as the
# general-purpose default; the others backstop missing/failed conversions.
DEFAULT_MODEL_ORDER = ("mineru2.5-pro", "paddleocr-vl-1.6")


def markdown_path(pdf_path: Path, model: str, parsed_dir: Path = PDF_PARSED_DIR) -> Path:
    """Path to a given model's markdown for a doc (doc_id == the PDF's stem)."""
    return parsed_dir / model / f"{Path(pdf_path).stem}.md"


def available_models(pdf_path: Path, parsed_dir: Path = PDF_PARSED_DIR) -> list[str]:
    """Models that produced a non-empty markdown for this doc, in default order.

    Useful for the cross-check experiments (compare numbers across parses).
    """
    found: list[str] = []
    for model in DEFAULT_MODEL_ORDER:
        path = markdown_path(pdf_path, model, parsed_dir)
        if path.exists() and path.read_text(encoding="utf-8").strip():
            found.append(model)
    return found


def parse(
    path: Path,
    model: str | None = None,
    parsed_dir: Path = PDF_PARSED_DIR,
) -> str:
    """Return one doc's text from its pre-converted markdown.

    With ``model`` set, reads that model's markdown; otherwise walks
    ``DEFAULT_MODEL_ORDER`` and returns the first non-empty one. Raises if nothing
    is available (i.e. the offline conversion hasn't been run/committed) — no
    silent fallback to degraded text.
    """
    order = (model,) if model is not None else DEFAULT_MODEL_ORDER
    for candidate in order:
        md = markdown_path(path, candidate, parsed_dir)
        if md.exists():
            text = md.read_text(encoding="utf-8").strip()
            if text:
                return text
    raise FileNotFoundError(
        f"no non-empty markdown for {Path(path).stem!r} under {parsed_dir} (tried {list(order)})"
    )


def extract_title(
    path: Path,
    model: str | None = None,
    parsed_dir: Path = PDF_PARSED_DIR,
) -> str | None:
    """Harvest a title from the chosen markdown's first heading, else ``None``.

    Parallels ``html_parser.extract_title`` so the orchestrator fills real PDF
    titles the same way (falling back to ``doc_id``).
    """
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
