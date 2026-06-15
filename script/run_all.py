from __future__ import annotations

import argparse
import sys

from agent.catalog import DatasetCatalog
from agent.config import load_config, resolve_path
from agent.page_index import PageIndexStore
from agent.runner import (
    answer_questions,
    build_missing_indexes,
    create_workflow,
    required_documents,
)


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(
        description="Build required PageIndexes, answer every question, and generate answer.csv."
    )
    parser.add_argument("--config", default="config/default.yaml")
    parser.add_argument("--questions")
    parser.add_argument(
        "--fresh",
        action="store_true",
        help="Ignore existing answer.csv and start a new run.",
    )
    args = parser.parse_args()

    config = load_config(args.config)
    catalog = DatasetCatalog(resolve_path(config, config["paths"]["dataset"]))
    question_path = resolve_path(config, args.questions or config["run"]["questions"])
    questions = catalog.load_questions(question_path)
    missing = catalog.validate_questions(questions)
    if missing:
        raise RuntimeError(f"Missing source documents: {missing}")

    store = PageIndexStore(resolve_path(config, config["paths"]["processed"]))
    documents = required_documents(catalog, questions)
    print(f"questions={len(questions)} required_documents={len(documents)}")
    build_missing_indexes(documents, store, config["page_index"])

    workflow = create_workflow(config, catalog, store)
    results = answer_questions(
        config,
        questions,
        workflow,
        checkpoint=True,
        resume=not args.fresh,
    )
    total_prompt = sum(result.usage.prompt_tokens for result in results)
    total_completion = sum(result.usage.completion_tokens for result in results)
    print(
        f"completed={len(results)} prompt_tokens={total_prompt} "
        f"completion_tokens={total_completion} total_tokens={total_prompt + total_completion}"
    )
    print(f"output={resolve_path(config, config['run']['output_csv'])}")


if __name__ == "__main__":
    main()
