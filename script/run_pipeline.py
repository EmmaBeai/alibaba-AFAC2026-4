from __future__ import annotations

import argparse
import sys

from agent.catalog import DatasetCatalog
from agent.config import load_config, resolve_path
from agent.page_index import PageIndexStore
from agent.runner import (
    answer_questions,
    assert_indexes_exist,
    build_missing_indexes,
    create_workflow,
    required_documents,
)


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(description="Run the Qwen PageIndex workflow.")
    parser.add_argument("--config", default="config/default.yaml")
    parser.add_argument("--questions")
    parser.add_argument("--qid", action="append", dest="qids")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--build-missing", action="store_true")
    parser.add_argument("--fresh", action="store_true")
    args = parser.parse_args()

    config = load_config(args.config)
    catalog = DatasetCatalog(resolve_path(config, config["paths"]["dataset"]))
    question_path = resolve_path(config, args.questions or config["run"]["questions"])
    questions = catalog.load_questions(question_path)
    if args.qids:
        wanted = set(args.qids)
        questions = [question for question in questions if question.qid in wanted]
    if args.limit:
        questions = questions[:args.limit]
    missing = catalog.validate_questions(questions)
    if missing:
        raise RuntimeError(f"Missing source documents: {missing}")
    store = PageIndexStore(resolve_path(config, config["paths"]["processed"]))
    documents = required_documents(catalog, questions)
    if args.build_missing:
        build_missing_indexes(documents, store, config["page_index"])
    else:
        assert_indexes_exist(documents, store)
    workflow = create_workflow(config, catalog, store)
    answer_questions(config, questions, workflow, resume=not args.fresh)


if __name__ == "__main__":
    main()
