from __future__ import annotations

import argparse
import sys

from agent.catalog import DatasetCatalog
from agent.config import load_config, resolve_path
from agent.run_artifacts import configure_run_directory, write_run_manifest
from agent.runner import (
    answer_questions,
    create_answerer,
    required_documents,
)
from agent.submission_validator import validate_submission


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(description="Run the BM25 top1 baseline.")
    parser.add_argument("--config", default="config/default.yaml")
    parser.add_argument("--questions")
    parser.add_argument("--run-dir")
    parser.add_argument("--previous-run")
    parser.add_argument("--qid", action="append", dest="qids")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--fresh", action="store_true")
    parser.add_argument("--output-csv")
    parser.add_argument("--evidence-json")
    args = parser.parse_args()

    config = load_config(args.config)
    if args.questions:
        config["run"]["questions"] = args.questions
    if args.previous_run:
        config["run"]["previous_run"] = args.previous_run
    run_dir = configure_run_directory(
        config,
        run_dir=args.run_dir,
        output_csv=args.output_csv,
        evidence_json=args.evidence_json,
    )
    catalog = DatasetCatalog(resolve_path(config, config["paths"]["dataset"]))
    question_path = resolve_path(config, config["run"]["questions"])
    config["run"]["questions"] = str(question_path)
    questions = catalog.load_questions(question_path)
    if args.qids:
        wanted = set(args.qids)
        questions = [question for question in questions if question.qid in wanted]
    if args.limit:
        questions = questions[:args.limit]
    missing = catalog.validate_questions(questions)
    if missing:
        raise RuntimeError(f"Missing source documents: {missing}")
    documents = required_documents(catalog, questions)
    print(f"answer_mode={config.get('answerer', {}).get('type', 'bm25_top1')}")
    print(f"output_csv={resolve_path(config, config['run']['output_csv'])}")
    print(f"evidence_json={resolve_path(config, config['run']['evidence_json'])}")
    answerer = create_answerer(config)
    results = answer_questions(config, questions, documents, answerer, resume=not args.fresh)
    validation_report = validate_submission(
        resolve_path(config, config["run"]["output_csv"]),
        resolve_path(config, config["run"]["evidence_json"]),
        questions,
    )
    previous_run = config["run"].get("previous_run")
    manifest = write_run_manifest(
        config,
        run_dir,
        questions=questions,
        documents=documents,
        results=results,
        previous_run_dir=previous_run,
        validation_report=validation_report,
    )
    print(f"manifest={manifest['outputs']['manifest_json']}")
    print(f"validation_ok={validation_report['ok']}")


if __name__ == "__main__":
    main()
