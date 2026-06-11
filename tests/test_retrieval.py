from agent.retrieval import LexicalRetriever
from agent.schema import Chunk, Question


def test_retriever_prioritizes_matching_financial_terms() -> None:
    chunks = [
        Chunk("d1::1", "d1", "保险条款", "insurance", "身故保险金 等于 已交保费 与 现金价值 较大者", 0, 20),
        Chunk("d2::1", "d2", "年报", "financial_reports", "研发投入 占 营业收入 比例 下降", 0, 20),
    ]
    question = Question(
        qid="q1",
        domain="financial_reports",
        split="A",
        question="研发投入占营业收入的比例是否下降？",
        options={"A": "下降", "B": "上升", "C": "不变", "D": "无法判断"},
        answer_format="mcq",
    )
    hits = LexicalRetriever(chunks).search(question, top_k=1)
    assert hits[0].chunk.doc_id == "d2"

