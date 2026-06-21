from __future__ import annotations

import argparse
import sys
from collections import Counter

from agent.config import load_config, resolve_path
from agent.structured import (
    build_structured_units,
    iter_processed_document_dirs,
    load_processed_document,
    write_jsonl,
)


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(
        description="Format processed pages into domain-aware structured evidence units."
    )
    parser.add_argument("--config", default="config/default.yaml")
    parser.add_argument("--domain")
    parser.add_argument("--doc-id", action="append", dest="doc_ids")
    parser.add_argument("--out", default="processed_data/structured_units.jsonl")
    parser.add_argument("--max-chars", type=int, default=1400)
    args = parser.parse_args()

    config = load_config(args.config)
    processed_root = resolve_path(config, config["paths"]["processed"])
    selected = list(iter_processed_document_dirs(processed_root))
    if args.domain:
        selected = [path for path in selected if path.parent.name == args.domain]
    if args.doc_ids:
        wanted = set(args.doc_ids)
        selected = [path for path in selected if path.name in wanted]

    rows = []
    domain_counts: Counter[str] = Counter()
    type_counts: Counter[str] = Counter()
    for document_dir in selected:
        metadata, pages = load_processed_document(document_dir)
        units = build_structured_units(metadata, pages, max_chars=args.max_chars)
        for unit in units:
            rows.append(unit.to_dict())
            domain_counts[unit.domain] += 1
            type_counts[unit.chunk_type] += 1

    out_path = resolve_path(config, args.out)
    count = write_jsonl(out_path, rows)
    print(f"[structured] documents={len(selected)} units={count} -> {out_path}")
    print(f"[structured] by_domain={dict(sorted(domain_counts.items()))}")
    print(f"[structured] top_types={dict(type_counts.most_common(12))}")


if __name__ == "__main__":
    main()

