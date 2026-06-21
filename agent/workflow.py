from __future__ import annotations

import json
from collections import Counter, defaultdict

from agent.catalog import DatasetCatalog
from agent.llm import QwenClient
from agent.page_index import PageIndexStore, compact_tree, flatten_nodes
from agent.prompts import ANSWER_SYSTEM, ROUTE_SYSTEM, TREE_SYSTEM
from agent.retrieval import RetrievalResult, StructuredRetriever
from agent.schemas import AnswerResult, Document, Question, TokenUsage


class PageIndexWorkflow:
    def __init__(
        self,
        catalog: DatasetCatalog,
        store: PageIndexStore,
        llm: QwenClient,
        max_selected_nodes: int = 6,
        max_evidence_chars: int = 45000,
        structured_retriever: StructuredRetriever | None = None,
        link_page_index_context: bool = True,
        pageindex_first_structured: bool = True,
        linked_page_window: int = 0,
        linked_max_pages_per_doc: int = 4,
        linked_max_chars: int = 12000,
    ):
        self.catalog = catalog
        self.store = store
        self.llm = llm
        self.max_selected_nodes = max_selected_nodes
        self.max_evidence_chars = max_evidence_chars
        self.structured_retriever = structured_retriever
        self.link_page_index_context = link_page_index_context
        self.pageindex_first_structured = pageindex_first_structured
        self.linked_page_window = linked_page_window
        self.linked_max_pages_per_doc = linked_max_pages_per_doc
        self.linked_max_chars = linked_max_chars

    def answer(self, question: Question) -> AnswerResult:
        before = TokenUsage(self.llm.usage.prompt_tokens, self.llm.usage.completion_tokens)
        documents = self._resolve_documents(question)
        evidence = self._retrieve_structured_evidence(question, documents)
        evidence_kind = "结构化证据卡"
        if not evidence:
            evidence = self._retrieve_page_evidence(question, documents)
            evidence_kind = "证据页"
        payload = self.llm.json_completion(
            ANSWER_SYSTEM,
            _answer_prompt(question, evidence, evidence_kind),
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

    def _retrieve_structured_evidence(self, question: Question, documents: list[Document]) -> str:
        if self.structured_retriever is None or not self.structured_retriever.available:
            return ""
        selected_pages: dict[str, list[int]] | None = None
        if self.pageindex_first_structured:
            selected_pages = self._select_page_candidates(question, documents)
        result = self.structured_retriever.retrieve_result(
            question,
            documents,
            pages_by_doc=selected_pages,
        )
        if not result.evidence:
            if selected_pages:
                return self._format_selected_page_context(documents, selected_pages)
            return ""
        if not self.link_page_index_context:
            return result.evidence
        page_context = (
            self._format_selected_page_context(documents, selected_pages)
            if selected_pages
            else self._retrieve_linked_page_context(documents, result)
        )
        if not page_context:
            return result.evidence
        return f"{result.evidence}\n\n## PageIndex 命中页上下文\n{page_context}"

    def _select_page_candidates(
        self,
        question: Question,
        documents: list[Document],
    ) -> dict[str, list[int]]:
        pages_by_doc: dict[str, list[int]] = {}
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
            pages: set[int] = set()
            for node_id in node_ids:
                node = leaves[node_id]
                pages.update(range(node.start_page, node.end_page + 1))
            pages_by_doc[document.doc_id] = sorted(pages)
        return pages_by_doc

    def _format_selected_page_context(
        self,
        documents: list[Document],
        pages_by_doc: dict[str, list[int]] | None,
    ) -> str:
        if not pages_by_doc:
            return ""
        document_by_id = {document.doc_id: document for document in documents}
        blocks: list[str] = []
        remaining = self.linked_max_chars
        for doc_id, pages in pages_by_doc.items():
            if doc_id not in document_by_id:
                continue
            for page_number in pages[: self.linked_max_pages_per_doc]:
                for page in self.store.load_pages(doc_id, page_number, page_number):
                    block = f"\n[page_context; doc_id={doc_id}; page={page.page_number}]\n{page.text}\n"
                    if len(block) > remaining:
                        if remaining > 500:
                            blocks.append(block[:remaining] + "\n[truncated]\n")
                        return "".join(blocks)
                    blocks.append(block)
                    remaining -= len(block)
        return "".join(blocks)

    def _retrieve_linked_page_context(
        self,
        documents: list[Document],
        result: RetrievalResult,
    ) -> str:
        document_by_id = {document.doc_id: document for document in documents}
        remaining = self.linked_max_chars
        blocks: list[str] = []
        seen_pages: set[tuple[str, int]] = set()
        pages_by_doc = _rank_pages_by_score(result)
        for doc_id, pages in pages_by_doc.items():
            if doc_id not in document_by_id:
                continue
            try:
                root = self.store.load_index(doc_id)
            except FileNotFoundError:
                continue
            leaves = flatten_nodes(root, leaves_only=True)
            for page_number in pages[: self.linked_max_pages_per_doc]:
                node = _node_covering_page(leaves, page_number)
                start_page = page_number
                end_page = page_number
                if node is not None:
                    start_page = max(node.start_page, page_number - self.linked_page_window)
                    end_page = min(node.end_page, page_number + self.linked_page_window)
                for page in self.store.load_pages(doc_id, start_page, end_page):
                    key = (doc_id, page.page_number)
                    if key in seen_pages:
                        continue
                    seen_pages.add(key)
                    node_label = f"{node.node_id} | {node.title}" if node is not None else "unknown"
                    block = (
                        f"\n[page_context; doc_id={doc_id}; page={page.page_number}; "
                        f"linked_node={node_label}]\n{page.text}\n"
                    )
                    if len(block) > remaining:
                        if remaining > 500:
                            blocks.append(block[:remaining] + "\n[truncated]\n")
                        return "".join(blocks)
                    blocks.append(block)
                    remaining -= len(block)
        return "".join(blocks)

    def _retrieve_page_evidence(self, question: Question, documents: list[Document]) -> str:
        blocks: list[str] = []
        per_doc_budget = max(6000, self.max_evidence_chars // max(1, len(documents)))
        pages_by_doc = self._select_page_candidates(question, documents)
        for document in documents:
            doc_blocks: list[str] = []
            remaining = per_doc_budget
            for page_number in pages_by_doc.get(document.doc_id, []):
                for page in self.store.load_pages(document.doc_id, page_number, page_number):
                    block = f"\n[doc_id={document.doc_id}; page={page.page_number}]\n{page.text}\n"
                    if len(block) > remaining:
                        if remaining > 500:
                            doc_blocks.append(block[:remaining] + "\n[truncated]\n")
                        remaining = 0
                        break
                    doc_blocks.append(block)
                    remaining -= len(block)
                if remaining <= 0:
                    break
            blocks.extend(doc_blocks)
        return "".join(blocks)


def normalize_answer(value: str, answer_format: str, options: dict[str, str]) -> str:
    allowed = set(options)
    letters = [char for char in str(value).upper() if char in allowed]
    if answer_format in {"mcq", "tf"}:
        return letters[0] if letters else ""
    return "".join(sorted(set(letters)))


def _answer_prompt(question: Question, evidence: str, evidence_kind: str) -> str:
    return (
        f"题号：{question.qid}\n"
        f"题型：{question.answer_format}\n"
        f"参考文档ID：{json.dumps(question.doc_ids, ensure_ascii=False)}\n"
        f"问题：{question.question}\n"
        f"选项：{json.dumps(question.options, ensure_ascii=False)}\n"
        "要求：涉及多份文档时必须逐份核对，不能只基于第一份文档推断第二份文档。"
        "对每个选项分别寻找支持或反驳证据；证据不足时不要猜测。\n"
        f"{evidence_kind}：\n{evidence}"
    )


def _rank_pages_by_score(result: RetrievalResult) -> dict[str, list[int]]:
    scores: dict[str, Counter[int]] = defaultdict(Counter)
    for item in result.items:
        doc_id = str(item.unit.get("doc_id", ""))
        page = int(item.unit.get("page") or 0)
        if doc_id and page:
            scores[doc_id][page] += item.score
    return {
        doc_id: [page for page, _ in counter.most_common()]
        for doc_id, counter in scores.items()
    }


def _node_covering_page(nodes, page_number: int):
    for node in nodes:
        if node.start_page <= page_number <= node.end_page:
            return node
    return None
