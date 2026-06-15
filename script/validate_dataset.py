from __future__ import annotations

import argparse
import sys
from collections import Counter

from agent.catalog import DatasetCatalog
from agent.config import load_config, resolve_path


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(description="Validate dataset document references.")
    parser.add_argument("--config", default="config/default.yaml")
    parser.add_argument("--questions")
    args = parser.parse_args()
    config = load_config(args.config)
    catalog = DatasetCatalog(resolve_path(config, config["paths"]["dataset"]))
    question_path = resolve_path(config, args.questions or config["run"]["questions"])
    questions = catalog.load_questions(question_path)
    missing = catalog.validate_questions(questions)
    counts = Counter(question.domain for question in questions)
    print(f"documents: {len(catalog.documents)}")
    print(f"questions: {len(questions)}")
    print(f"domains: {dict(sorted(counts.items()))}")
    print(f"missing doc_ids: {len(missing)}")
    for doc_id in missing:
        print(f"  {doc_id}")
    if missing:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
