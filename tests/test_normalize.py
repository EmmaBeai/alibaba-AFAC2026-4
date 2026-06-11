from agent.normalize import is_valid_answer, normalize_answer
from agent.schema import Question


def test_normalize_single_choice_uses_first_valid_letter() -> None:
    assert normalize_answer("答案：B，因为...", "mcq") == "B"
    assert normalize_answer("E / C", "mcq") == "C"


def test_normalize_multi_deduplicates_and_sorts() -> None:
    assert normalize_answer("C A A B", "multi") == "ABC"


def test_valid_answer_contract() -> None:
    assert is_valid_answer("A", "mcq")
    assert not is_valid_answer("AB", "mcq")
    assert is_valid_answer("ACD", "multi")
    assert not is_valid_answer("CA", "multi")


def test_tf_question_accepts_ab_options_only() -> None:
    question = Question.from_dict(
        {
            "qid": "tf_001",
            "domain": "regulatory",
            "split": "A",
            "question": "该说法是否正确？",
            "options": {"A": "正确", "B": "错误"},
            "answer_format": "tf",
        }
    )
    assert question.options == {"A": "正确", "B": "错误"}

