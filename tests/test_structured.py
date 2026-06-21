import json

from agent.retrieval import StructuredRetriever
from agent.schemas import Document, Question
from agent.structured import build_structured_units


def test_regulatory_article_units_have_clause_numbers() -> None:
    metadata = {"doc_id": "law", "domain": "regulatory", "title": "测试法规"}
    pages = [
        {
            "page_number": 1,
            "text": "第一章 总则\n第一条 金融机构应当保存资料十年。\n第二条 不得泄露客户信息。",
        }
    ]
    units = build_structured_units(metadata, pages)
    assert [unit.clause_no for unit in units[-2:]] == ["第一条", "第二条"]
    assert units[-2].chunk_type == "clause"
    assert "十年" in units[-2].raw_text


def test_contract_units_extract_numbers_and_type() -> None:
    metadata = {"doc_id": "text01", "domain": "financial_contracts", "title": "募集说明书"}
    pages = [
        {
            "page_number": 3,
            "text": "本期债券发行金额不超过10亿元。主体信用评级为AAA，债项信用评级为-。",
        }
    ]
    units = build_structured_units(metadata, pages)
    assert len(units) == 1
    assert "10亿元" in units[0].numbers
    assert units[0].chunk_type in {"issue_terms", "rating"}


def test_structured_retriever_returns_option_evidence(tmp_path) -> None:
    units_path = tmp_path / "units.jsonl"
    rows = [
        {
            "unit_id": "doc1_p1_001",
            "doc_id": "doc1",
            "domain": "financial_contracts",
            "title": "募集说明书",
            "page": 1,
            "section_path": "发行条款",
            "clause_no": "",
            "chunk_type": "issue_terms",
            "raw_text": "本期债券发行金额不超过10亿元，债券期限为5年。",
            "search_text": "发行金额 不超过10亿元 债券期限 5年",
            "numbers": ["10亿元", "5年"],
            "keywords": ["issue_terms"],
        },
        {
            "unit_id": "doc1_p2_001",
            "doc_id": "doc1",
            "domain": "financial_contracts",
            "title": "募集说明书",
            "page": 2,
            "section_path": "评级情况",
            "clause_no": "",
            "chunk_type": "rating",
            "raw_text": "主体信用评级为AAA。",
            "search_text": "主体信用评级 AAA",
            "numbers": [],
            "keywords": ["rating"],
        },
    ]
    units_path.write_text(
        "\n".join(json.dumps(row, ensure_ascii=False) for row in rows),
        encoding="utf-8",
    )
    retriever = StructuredRetriever(units_path, per_option=1, max_units=4)
    question = Question(
        qid="fc_a_test",
        domain="financial_contracts",
        split="a",
        question="本期债券发行金额是多少？",
        options={"A": "不超过10亿元", "B": "主体信用评级为AAA"},
        answer_format="mcq",
        question_type="",
        doc_ids=["doc1"],
    )
    evidence = retriever.retrieve(
        question,
        [Document("doc1", "financial_contracts", tmp_path / "doc1.txt", "募集说明书")],
    )
    assert "doc1_p1_001" in evidence
    assert "不超过10亿元" in evidence
