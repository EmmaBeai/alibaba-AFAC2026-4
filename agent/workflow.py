from __future__ import annotations

import json

from agent.catalog import DatasetCatalog
from agent.llm import QwenClient
from agent.page_index import PageIndexStore, compact_tree, flatten_nodes
from agent.prompts import ANSWER_SYSTEM, ROUTE_SYSTEM, TREE_SYSTEM
from agent.schemas import AnswerResult, Document, Question, TokenUsage


class PageIndexWorkflow:
    def __init__(
        self,
        catalog: DatasetCatalog,
        store: PageIndexStore,
        llm: QwenClient,
        max_selected_nodes: int = 6,
        max_evidence_chars: int = 45000,
    ):
        self.catalog = catalog
        self.store = store
        self.llm = llm
        self.max_selected_nodes = max_selected_nodes
        self.max_evidence_chars = max_evidence_chars

    def answer(self, question: Question) -> AnswerResult:
        before = TokenUsage(self.llm.usage.prompt_tokens, self.llm.usage.completion_tokens)
        documents = self._resolve_documents(question)
        evidence = self._retrieve_evidence(question, documents)
        payload = self.llm.json_completion(
            ANSWER_SYSTEM,
            _answer_prompt(question, evidence),
            purpose="answer",
            qid=question.qid,
        )
        answer = normalize_answer(payload.get("answer", ""), question.answer_format, question.options)
        usage = TokenUsage(
            self.llm.usage.prompt_tokens - before.prompt_tokens,
            self.llm.usage.completion_tokens - before.completion_tokens,
        )
        return AnswerResult(
            qid=question.qid,
            answer=answer,
            evidence_retrieval=payload.get("evidence_retrieval", []),
            usage=usage,
        )

    def _resolve_documents(self, question: Question) -> list[Document]:
        if question.doc_ids:
            return [self.catalog.get_document(doc_id) for doc_id in question.doc_ids]
        candidates = self.catalog.domain_documents(question.domain)
        candidate_lines = []
        for document in candidates:
            try:
                title = self.store.load_index(document.doc_id).title
            except FileNotFoundError:
                title = document.title
            candidate_lines.append(f"- {document.doc_id} | {title}")
        candidate_text = "\n".join(candidate_lines)
        payload = self.llm.json_completion(
            ROUTE_SYSTEM,
            f"问题：{question.question}\n选项：{json.dumps(question.options, ensure_ascii=False)}\n候选文档：\n{candidate_text}",
            purpose="document_route",
            qid=question.qid,
        )
        allowed = {doc.doc_id: doc for doc in candidates}
        selected = [allowed[item] for item in payload.get("doc_ids", []) if item in allowed]
        return selected or candidates[:3]

    def _retrieve_evidence(self, question: Question, documents: list[Document]) -> str:
        blocks: list[str] = []
        remaining = self.max_evidence_chars
        for document in documents:
            root = self.store.load_index(document.doc_id)
            payload = self.llm.json_completion(
                TREE_SYSTEM,
                (
                    f"问题：{question.question}\n"
                    f"选项：{json.dumps(question.options, ensure_ascii=False)}\n"
                    f"文档：{document.doc_id} | {document.title}\n"
                    f"PageIndex：\n{compact_tree(root)}\n"
                    f"最多选择 {self.max_selected_nodes} 个叶节点。"
                ),
                purpose="page_index_select",
                qid=question.qid,
            )
            leaves = {node.node_id: node for node in flatten_nodes(root, leaves_only=True)}
            node_ids = [item for item in payload.get("node_ids", []) if item in leaves][
                :self.max_selected_nodes
            ]
            if not node_ids:
                node_ids = list(leaves)[:1]
            for node_id in node_ids:
                node = leaves[node_id]
                for page in self.store.load_pages(document.doc_id, node.start_page, node.end_page):
                    block = f"\n[doc_id={document.doc_id}; page={page.page_number}]\n{page.text}\n"
                    if len(block) > remaining:
                        return "".join(blocks)
                    blocks.append(block)
                    remaining -= len(block)
        return "".join(blocks)


def normalize_answer(value: str, answer_format: str, options: dict[str, str]) -> str:
    allowed = set(options)
    letters = [char for char in str(value).upper() if char in allowed]
    if answer_format in {"mcq", "tf"}:
        return letters[0] if letters else ""
    return "".join(sorted(set(letters)))


def _answer_prompt(question: Question, evidence: str) -> str:
    return (
        f"题号：{question.qid}\n"
        f"题型：{question.answer_format}\n"
        f"问题：{question.question}\n"
        f"选项：{json.dumps(question.options, ensure_ascii=False)}\n"
        f"证据页：\n{evidence}"
    )
