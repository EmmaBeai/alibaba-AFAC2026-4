from __future__ import annotations

from collections.abc import Callable

from agent.catalog import DatasetCatalog
from agent.config import resolve_path
from agent.llm import QwenClient
from agent.output import load_completed_results, write_outputs
from agent.page_index import PageIndexStore, build_page_index
from agent.preprocess import extract_pages
from agent.schemas import AnswerResult, Document, Question
from agent.workflow import PageIndexWorkflow


def required_documents(
    catalog: DatasetCatalog,
    questions: list[Question],
) -> list[Document]:
    doc_ids: set[str] = set()
    for question in questions:
        if question.doc_ids:
            doc_ids.update(question.doc_ids)
        else:
            doc_ids.update(doc.doc_id for doc in catalog.domain_documents(question.domain))
    return [catalog.get_document(doc_id) for doc_id in sorted(doc_ids)]


def build_missing_indexes(
    documents: list[Document],
    store: PageIndexStore,
    page_config: dict,
    progress: Callable[[str], None] = print,
) -> None:
    missing: list[Document] = []
    for document in documents:
        try:
            store.load_index(document.doc_id)
        except FileNotFoundError:
            missing.append(document)
    for index, document in enumerate(missing, start=1):
        progress(f"[index {index}/{len(missing)}] {document.doc_id}")
        pages = extract_pages(document.path)
        root = build_page_index(
            document,
            pages,
            leaf_pages=page_config["leaf_pages"],
            branch_factor=page_config["branch_factor"],
        )
        store.save(document, pages, root)


def assert_indexes_exist(documents: list[Document], store: PageIndexStore) -> None:
    missing = []
    for document in documents:
        try:
            store.load_index(document.doc_id)
        except FileNotFoundError:
            missing.append(document.doc_id)
    if missing:
        preview = ", ".join(missing[:10])
        suffix = " ..." if len(missing) > 10 else ""
        raise RuntimeError(
            f"Missing PageIndex for {len(missing)} document(s): {preview}{suffix}. "
            "Run python -m script.build_index first."
        )


def create_workflow(config: dict, catalog: DatasetCatalog, store: PageIndexStore) -> PageIndexWorkflow:
    model = config["model"]
    llm = QwenClient(
        model=model["name"],
        api_key_env=model["api_key_env"],
        base_url_env=model["base_url_env"],
        default_base_url=model["default_base_url"],
        temperature=model["temperature"],
        log_dir=resolve_path(config, config["paths"]["logs"]),
    )
    page_config = config["page_index"]
    return PageIndexWorkflow(
        catalog=catalog,
        store=store,
        llm=llm,
        max_selected_nodes=page_config["max_selected_nodes"],
        max_evidence_chars=page_config["max_evidence_chars"],
    )


def answer_questions(
    config: dict,
    questions: list[Question],
    workflow: PageIndexWorkflow,
    *,
    checkpoint: bool = True,
    resume: bool = True,
) -> list[AnswerResult]:
    csv_path = resolve_path(config, config["run"]["output_csv"])
    evidence_path = resolve_path(config, config["run"]["evidence_json"])
    results = load_completed_results(csv_path, evidence_path) if resume else []
    completed = {result.qid for result in results}
    pending = [question for question in questions if question.qid not in completed]
    if completed:
        print(f"resume: completed={len(completed)} pending={len(pending)}")
    for index, question in enumerate(pending, start=1):
        print(f"[answer {index}/{len(pending)}] {question.qid}")
        result = workflow.answer(question)
        results.append(result)
        print(
            f"  answer={result.answer} "
            f"prompt={result.usage.prompt_tokens} "
            f"completion={result.usage.completion_tokens} "
            f"total={result.usage.total_tokens}"
        )
        if checkpoint:
            write_outputs(results, csv_path, evidence_path)
    if not checkpoint:
        write_outputs(results, csv_path, evidence_path)
    return results
