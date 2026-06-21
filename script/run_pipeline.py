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
    assert_indexes_exist,
    build_missing_indexes,
    create_workflow,
    override_run_outputs,
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
        build_missing_indexes(
            documents,
            store,
            config["page_index"],
            preprocess_config=config.get("preprocess"),
        )
    else:
        assert_indexes_exist(documents, store)
    print(f"retrieval_mode={args.retrieval_mode}")
    print(f"output_csv={resolve_path(config, config['run']['output_csv'])}")
    print(f"evidence_json={resolve_path(config, config['run']['evidence_json'])}")
    workflow = create_workflow(config, catalog, store)
    answer_questions(config, questions, workflow, resume=not args.fresh)


if __name__ == "__main__":
    main()
