import csv
from pathlib import Path

from agent.output import write_outputs
from agent.schemas import AnswerResult, TokenUsage


def test_answer_csv_contains_summary_and_per_question_usage() -> None:
    csv_path = Path("processed_data/test_answer.csv")
    evidence_path = Path("processed_data/test_evidence.json")
    results = [
        AnswerResult("q1", "B", [], TokenUsage(100, 2)),
        AnswerResult("q2", "AC", [], TokenUsage(200, 3)),
    ]

    try:
        write_outputs(results, csv_path, evidence_path)
        with csv_path.open(encoding="utf-8-sig", newline="") as handle:
            rows = list(csv.DictReader(handle))
        assert rows[0] == {
            "qid": "summary",
            "answer": "",
            "prompt_tokens": "300",
            "completion_tokens": "5",
            "total_tokens": "305",
        }
        assert rows[1]["qid"] == "q1"
        assert rows[1]["total_tokens"] == "102"
    finally:
        csv_path.unlink(missing_ok=True)
        evidence_path.unlink(missing_ok=True)
