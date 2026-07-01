from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from agent.catalog import DatasetCatalog
from agent.config import load_config, resolve_path
from agent.submission_validator import validate_submission


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(description="Validate answer.csv and evidence.json.")
    parser.add_argument("--config", default="config/default.yaml")
    parser.add_argument("--run-dir", default=None)
    parser.add_argument("--answer-csv")
    parser.add_argument("--evidence-json")
    parser.add_argument("--questions")
    parser.add_argument("--json-out")
    args = parser.parse_args()

    config = load_config(args.config)
    run_dir = Path(args.run_dir or config["run"].get("output_dir", "runs/00_bm25_top1"))
    if not run_dir.is_absolute():
        run_dir = resolve_path(config, str(run_dir))
    answer_csv = Path(args.answer_csv) if args.answer_csv else run_dir / "answer.csv"
    evidence_json = Path(args.evidence_json) if args.evidence_json else run_dir / "evidence.json"
    if not answer_csv.is_absolute():
        answer_csv = resolve_path(config, str(answer_csv))
    if not evidence_json.is_absolute():
        evidence_json = resolve_path(config, str(evidence_json))

    catalog = DatasetCatalog(resolve_path(config, config["paths"]["dataset"]))
    questions_path = resolve_path(config, args.questions or config["run"]["questions"])
    questions = catalog.load_questions(questions_path)
    report = validate_submission(answer_csv, evidence_json, questions)
    payload = json.dumps(report, ensure_ascii=False, indent=2)
    print(payload)
    if args.json_out:
        output_path = Path(args.json_out)
        if not output_path.is_absolute():
            output_path = resolve_path(config, str(output_path))
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(payload + "\n", encoding="utf-8")
    if not report["ok"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
