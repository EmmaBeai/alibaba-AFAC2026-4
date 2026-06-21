from pathlib import Path

from agent.page_index import build_page_index, flatten_nodes
from agent.preprocess import clean_text, extract_pages
from agent.runner import apply_retrieval_mode, override_run_outputs
from agent.schemas import Document, Page
from agent.workflow import normalize_answer


def test_normalize_multi_answer() -> None:
    assert normalize_answer("答案：C, A, C", "multi", {"A": "", "B": "", "C": "", "D": ""}) == "AC"


def test_normalize_single_answer() -> None:
    assert normalize_answer("B。理由省略", "mcq", {"A": "", "B": "", "C": "", "D": ""}) == "B"


def test_page_index_leaf_ranges_cover_pages() -> None:
    document = Document("doc", "test", Path("doc.pdf"), "doc")
    pages = [Page(number, f"第{number}章 标题\n正文") for number in range(1, 18)]
    root = build_page_index(document, pages, leaf_pages=3, branch_factor=2)
    leaves = flatten_nodes(root, leaves_only=True)
    covered = [
        page
        for node in leaves
        for page in range(node.start_page, node.end_page + 1)
    ]
    assert covered == list(range(1, 18))
    assert root.title.startswith("doc |")


def test_extract_pages_prefers_preconverted_pdf_markdown() -> None:
    parsed_dir = Path("processed_data/pdf_parsed_test")
    md_path = parsed_dir / "glm-ocr" / "sample.md"
    md_path.parent.mkdir(parents=True, exist_ok=True)
    md_path.write_text("## Page 1\n第一页正文\n## Page 2\n第二页正文", encoding="utf-8")
    try:
        pages = extract_pages(
            Path("sample.pdf"),
            pdf_parsed_dir=parsed_dir,
            pdf_model_order=["glm-ocr"],
        )
        assert [page.page_number for page in pages] == [1, 2]
        assert pages[0].text == "第一页正文"
    finally:
        md_path.unlink(missing_ok=True)
        md_path.parent.rmdir()
        parsed_dir.rmdir()


def test_clean_text_removes_invalid_surrogates() -> None:
    assert clean_text("abc\udcb0\ud800def\x00") == "abcdef"


def test_clean_markup_noise_removes_markdown_images_and_keeps_table_text() -> None:
    text = '![Image 0](x.jpg)<div align="center">标题</div><table><tr><td>A</td><td>B</td></tr></table>'
    cleaned = clean_text(text)
    assert "![Image" not in cleaned
    assert "标题" in cleaned
    assert "A | B" in cleaned


def test_apply_retrieval_modes_and_output_overrides() -> None:
    config = {
        "structured_retrieval": {
            "enabled": True,
            "pageindex_first_structured": True,
            "link_page_index_context": True,
        },
        "run": {"output_csv": "answer.csv", "evidence_json": "evidence.json"},
    }
    apply_retrieval_mode(config, "bm25")
    assert config["structured_retrieval"]["enabled"] is True
    assert config["structured_retrieval"]["pageindex_first_structured"] is False
    assert config["structured_retrieval"]["link_page_index_context"] is False
    apply_retrieval_mode(config, "pageindex")
    assert config["structured_retrieval"]["enabled"] is False
    override_run_outputs(
        config,
        output_csv="answer_bm25.csv",
        evidence_json="evidence_bm25.json",
    )
    assert config["run"]["output_csv"] == "answer_bm25.csv"
    assert config["run"]["evidence_json"] == "evidence_bm25.json"
