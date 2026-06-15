from __future__ import annotations

import argparse
import sys

from agent.catalog import DatasetCatalog
from agent.config import load_config, resolve_path
from agent.page_index import PageIndexStore, build_page_index
from agent.preprocess import extract_pages


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(description="Build structural PageIndex files.")
    parser.add_argument("--config", default="config/default.yaml")
    parser.add_argument("--domain")
    parser.add_argument("--doc-id", action="append", dest="doc_ids")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    config = load_config(args.config)
    catalog = DatasetCatalog(resolve_path(config, config["paths"]["dataset"]))
    store = PageIndexStore(resolve_path(config, config["paths"]["processed"]))
    selected = list(catalog.documents.values())
    if args.domain:
        selected = [doc for doc in selected if doc.domain == args.domain]
    if args.doc_ids:
        wanted = set(args.doc_ids)
        selected = [doc for doc in selected if doc.doc_id in wanted]

    page_config = config["page_index"]
    for index, document in enumerate(selected, start=1):
        output = store.root / document.domain / document.doc_id / "page_index.json"
        if output.exists() and not args.force:
            print(f"[{index}/{len(selected)}] skip {document.doc_id}")
            continue
        print(f"[{index}/{len(selected)}] extract {document.doc_id}")
        pages = extract_pages(document.path)
        root = build_page_index(
            document,
            pages,
            leaf_pages=page_config["leaf_pages"],
            branch_factor=page_config["branch_factor"],
        )
        store.save(document, pages, root)
    print(f"Built indexes for {len(selected)} document(s).")


if __name__ == "__main__":
    main()
