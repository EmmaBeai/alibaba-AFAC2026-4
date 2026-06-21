"""Plain-text (.txt) extraction.

Currently covers the 6 regulatory ``strict_v3_*`` files. Observed quirks:
  - UTF-8 **with BOM** -> decode with ``utf-8-sig`` so no stray ``\\ufeff``.
  - **CRLF** line endings -> normalize to ``\\n``.
  - No blank lines; mostly one article (条) per line, which is already good
    paragraph structure for chunking.

Known, deliberately NOT handled here: the leading title is soft-wrapped into
short fragments (e.g. ``《金融机构客户`` / ``受益所有人识别`` / ``管理办法》…``).
Re-joining that is a title-extraction / cleaner concern; this parser's only job
is bytes -> clean text, preserving line structure.
"""

from __future__ import annotations

from pathlib import Path


def parse(path: Path) -> str:
    """Read a ``.txt`` file as clean, newline-normalized text."""
    # utf-8-sig transparently drops a leading BOM and otherwise decodes as utf-8.
    raw = Path(path).read_text(encoding="utf-8-sig")
    raw = raw.replace("\r\n", "\n").replace("\r", "\n")
    lines = [line.rstrip() for line in raw.split("\n")]
    return "\n".join(lines).strip()
