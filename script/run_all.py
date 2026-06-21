from __future__ import annotations

import argparse
import sys

from agent.catalog import DatasetCatalog
from agent.config import load_config, resolve_path
from agent.page_index import PageIndexStore
from agent.runner import (
    RETRIEVAL_MODES,
    apply_retrieval_mode,
    answer_questions,
    build_missing_indexes,
    create_workflow,
    override_run_outputs,
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
    parser.add_argument(
        "--retrieval-mode",
        choices=RETRIEVAL_MODES,
        default="config",
        help=(
            "config: use YAML; pageindex: pure PageIndex; "
            "bm25: pure field-aware BM25; pageindex-bm25: PageIndex-first field-aware BM25."
        ),
    )
    parser.add_argument("--output-csv")
    parser.add_argument("--evidence-json")
    args = parser.parse_args()

    config = load_config(args.config)
    apply_retrieval_mode(config, args.retrieval_mode)
    override_run_outputs(config, output_csv=args.output_csv, evidence_json=args.evidence_json)
    catalog = DatasetCatalog(resolve_path(config, config["paths"]["dataset"]))
    question_path = resolve_path(config, args.questions or config["run"]["questions"])
    questions = catalog.load_questions(question_path)
    missing = catalog.validate_questions(questions)
    if missing:
        raise RuntimeError(f"Missing source documents: {missing}")

    store = PageIndexStore(resolve_path(config, config["paths"]["processed"]))
    documents = required_documents(catalog, questions)
    print(
        f"questions={len(questions)} required_documents={len(documents)} "
        f"retrieval_mode={args.retrieval_mode}"
    )
    build_missing_indexes(
        documents,
        store,
        config["page_index"],
        preprocess_config=config.get("preprocess"),
    )

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
    print(f"evidence={resolve_path(config, config['run']['evidence_json'])}")


if __name__ == "__main__":
    main()
