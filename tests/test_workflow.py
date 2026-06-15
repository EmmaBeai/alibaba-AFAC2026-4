from pathlib import Path

from agent.page_index import build_page_index, flatten_nodes
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
