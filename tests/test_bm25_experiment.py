import json

from agent.bm25_experiment import ExperimentalBM25, FIELD_TERMS
from agent.schemas import Document, Question


def _write_units(path, rows) -> None:
    path.write_text(
        "\n".join(json.dumps(row, ensure_ascii=False) for row in rows),
        encoding="utf-8",
    )


def test_parse_option_extracts_doc_field_and_value(tmp_path) -> None:
    units_path = tmp_path / "units.jsonl"
    _write_units(units_path, [])
    bm25 = ExperimentalBM25(units_path)
    question = Question(
        qid="q",
        domain="financial_contracts",
        split="a",
        question="关于两份债券募集说明书，下列说法正确的是？",
        options={"D": "第二份文档明确指定国信证券股份有限公司为受托管理人"},
        answer_format="multi",
        question_type="",
        doc_ids=["doc1", "doc2"],
    )
    parsed = bm25.parse_option(
        question,
        [
            Document("doc1", "financial_contracts", tmp_path / "doc1.txt", "doc1"),
            Document("doc2", "financial_contracts", tmp_path / "doc2.txt", "doc2"),
        ],
        "D",
        question.options["D"],
    )
    assert parsed.doc_ids == ["doc2"]
    assert "trustee" in parsed.fields
    assert "国信证券股份有限公司" in parsed.values


def test_table_rows_are_split_for_experiment(tmp_path) -> None:
    units_path = tmp_path / "units.jsonl"
    _write_units(
        units_path,
        [
            {
                "unit_id": "doc_p1_001",
                "doc_id": "doc",
                "domain": "financial_contracts",
                "title": "募集说明书",
                "page": 1,
                "section_path": "发行概况",
                "clause_no": "",
                "chunk_type": "table",
                "raw_text": "项目 | 内容\n发行规模 | 不超过10亿元\n受托管理人 | 国信证券股份有限公司",
                "search_text": "项目 内容 发行规模 不超过10亿元 受托管理人 国信证券股份有限公司",
                "numbers": ["10亿元"],
                "keywords": ["issue_terms", "intermediary"],
            }
        ],
    )
    bm25 = ExperimentalBM25(units_path)
    assert any(unit["unit_id"].endswith("_r002") for unit in bm25.units)
    assert any(unit["raw_text"] == "受托管理人 | 国信证券股份有限公司" for unit in bm25.units)


def test_field_and_value_same_unit_ranks_first(tmp_path) -> None:
    units_path = tmp_path / "units.jsonl"
    _write_units(
        units_path,
        [
            {
                "unit_id": "doc_p1_001",
                "doc_id": "doc",
                "domain": "financial_contracts",
                "title": "募集说明书",
                "page": 1,
                "section_path": "发行概况",
                "clause_no": "",
                "chunk_type": "table",
                "raw_text": "受托管理人 | 国信证券股份有限公司",
                "search_text": "受托管理人 国信证券股份有限公司",
                "numbers": [],
                "keywords": ["intermediary"],
            },
            {
                "unit_id": "doc_p2_001",
                "doc_id": "doc",
                "domain": "financial_contracts",
                "title": "募集说明书",
                "page": 2,
                "section_path": "机构列表",
                "clause_no": "",
                "chunk_type": "table",
                "raw_text": "联席主承销商 | 国信证券股份有限公司",
                "search_text": "联席主承销商 国信证券股份有限公司",
                "numbers": [],
                "keywords": ["intermediary"],
            },
        ],
    )
    bm25 = ExperimentalBM25(units_path)
    question = Question(
        qid="q",
        domain="financial_contracts",
        split="a",
        question="下列说法正确的是？",
        options={"A": "国信证券股份有限公司为受托管理人"},
        answer_format="multi",
        question_type="",
        doc_ids=["doc"],
    )
    parsed = bm25.parse_option(
        question,
        [Document("doc", "financial_contracts", tmp_path / "doc.txt", "doc")],
        "A",
        question.options["A"],
    )
    candidates = bm25.rank(
        question,
        [Document("doc", "financial_contracts", tmp_path / "doc.txt", "doc")],
        parsed,
    )
    assert candidates[0].unit_id.startswith("doc_p1_001")
    assert "field_value_same_unit+70" in candidates[0].reasons


def test_compare_option_returns_per_doc_candidates(tmp_path) -> None:
    units_path = tmp_path / "units.jsonl"
    _write_units(
        units_path,
        [
            {
                "unit_id": "doc1_p1_001",
                "doc_id": "doc1",
                "domain": "financial_contracts",
                "title": "第一份",
                "page": 1,
                "section_path": "发行概况",
                "clause_no": "",
                "chunk_type": "table",
                "raw_text": "发行规模 | 不超过10亿元",
                "search_text": "发行规模 不超过10亿元",
                "numbers": ["10亿元"],
                "keywords": ["issue_terms"],
            },
            {
                "unit_id": "doc2_p1_001",
                "doc_id": "doc2",
                "domain": "financial_contracts",
                "title": "第二份",
                "page": 1,
                "section_path": "发行概况",
                "clause_no": "",
                "chunk_type": "table",
                "raw_text": "发行规模 | 不超过5亿元",
                "search_text": "发行规模 不超过5亿元",
                "numbers": ["5亿元"],
                "keywords": ["issue_terms"],
            },
        ],
    )
    bm25 = ExperimentalBM25(units_path)
    documents = [
        Document("doc1", "financial_contracts", tmp_path / "doc1.txt", "第一份"),
        Document("doc2", "financial_contracts", tmp_path / "doc2.txt", "第二份"),
    ]
    question = Question(
        qid="q",
        domain="financial_contracts",
        split="a",
        question="关于两份债券募集说明书，下列说法正确的是？",
        options={"B": "第二份文档的发行金额上限低于第一份文档"},
        answer_format="multi",
        question_type="",
        doc_ids=["doc1", "doc2"],
    )
    parsed = bm25.parse_option(question, documents, "B", question.options["B"])
    assert parsed.compare is True
    assert parsed.doc_ids == ["doc1", "doc2"]
    per_doc = bm25.rank_per_doc(question, documents, parsed, top_k=1)
    assert per_doc["doc1"][0].unit_id.startswith("doc1_p1_001")
    assert per_doc["doc2"][0].unit_id.startswith("doc2_p1_001")


def test_soft_page_filter_keeps_stronger_non_pageindex_candidate(tmp_path) -> None:
    units_path = tmp_path / "units.jsonl"
    trustee_term = FIELD_TERMS["trustee"][0]
    company = "target_company"
    _write_units(
        units_path,
        [
            {
                "unit_id": "doc_p1_001",
                "doc_id": "doc",
                "domain": "financial_contracts",
                "title": "å‹Ÿé›†è¯´æ˜Žä¹¦",
                "page": 1,
                "section_path": "æ¦‚è¦",
                "clause_no": "",
                "chunk_type": "table",
                "raw_text": company,
                "search_text": company,
                "numbers": [],
                "keywords": [],
            },
            {
                "unit_id": "doc_p2_001",
                "doc_id": "doc",
                "domain": "financial_contracts",
                "title": "å‹Ÿé›†è¯´æ˜Žä¹¦",
                "page": 2,
                "section_path": "å—æ‰˜ç®¡ç†äºº",
                "clause_no": "",
                "chunk_type": "table",
                "raw_text": f"{trustee_term} | {company}",
                "search_text": f"{trustee_term} {company}",
                "numbers": [],
                "keywords": ["intermediary"],
            },
        ],
    )
    bm25 = ExperimentalBM25(units_path)
    question = Question(
        qid="q",
        domain="financial_contracts",
        split="a",
        question="ä¸‹åˆ—è¯´æ³•æ­£ç¡®çš„æ˜¯ï¼Ÿ",
        options={"A": f"{company} ä¸º {trustee_term}"},
        answer_format="multi",
        question_type="",
        doc_ids=["doc"],
    )
    documents = [Document("doc", "financial_contracts", tmp_path / "doc.txt", "doc")]
    parsed = bm25.parse_option(question, documents, "A", question.options["A"])

    hard = bm25.rank(question, documents, parsed, pages_by_doc={"doc": [1]}, page_filter_mode="hard")
    soft = bm25.rank(
        question,
        documents,
        parsed,
        pages_by_doc={"doc": [1]},
        page_filter_mode="soft",
        page_boost=10.0,
    )

    assert hard[0].page == 1
    assert soft[0].page == 2
