"""Document preprocessing — parse raw files into clean text and chunks.

Kept as a top-level package separate from ``agent/`` on purpose: the competition
allows ANY tool here (non-Qwen OCR / layout analysis / MinerU / …), whereas the
answering stage in ``agent/`` is Qwen-only. The package boundary makes that
compliance line structural, not just a convention.

Grows one format at a time; ``txt_parser`` is the first. Each ``*_parser`` module
exposes ``parse(path) -> str`` (bytes -> clean text), so a dispatcher can route
by extension once more formats land.
"""
