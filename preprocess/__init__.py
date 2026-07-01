"""Self-contained offline preprocessing package.

This directory owns the mixed old/new production path:

1. Use the old PDF OCR/text-layer conversion logic for ``glm-ocr`` and ``pypdf``.
2. Materialize the allowlisted parsed source files used by heading cleanup.
3. Clean heading levels into ``processed_data/cleaned_markdown``.
4. Build parent-linked ``structure_nodes.jsonl``.
5. Build code-generated ``block_nodes.jsonl`` under section nodes.
6. Build header-based ``h2`` through ``h6`` nested chunk JSON files.

The answering/runtime code can consume these artifacts without importing from
the preprocessing implementation.
"""
