from pathlib import Path

from agent.output import load_completed_results, write_outputs
from agent.schemas import AnswerResult, TokenUsage


def test_load_completed_results() -> None:
    csv_path = Path("processed_data/test_resume_answer.csv")
    evidence_path = Path("processed_data/test_resume_evidence.json")
    original = [AnswerResult("q1", "BD", [{"doc_id": "doc"}], TokenUsage(12, 3))]
    try:
        write_outputs(original, csv_path, evidence_path)
        loaded = load_completed_results(csv_path, evidence_path)
        assert len(loaded) == 1
        assert loaded[0].qid == "q1"
        assert loaded[0].answer == "BD"
        assert loaded[0].usage.total_tokens == 15
        assert loaded[0].evidence_retrieval == [{"doc_id": "doc"}]
    finally:
        csv_path.unlink(missing_ok=True)
        evidence_path.unlink(missing_ok=True)
