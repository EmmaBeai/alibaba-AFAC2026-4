from __future__ import annotations

from collections.abc import Callable

from agent.catalog import DatasetCatalog
from agent.config import resolve_path
from agent.llm import QwenClient
from agent.output import load_completed_results, write_outputs
from agent.page_index import PageIndexStore, build_page_index
from agent.preprocess import extract_pages_with_metadata
from agent.retrieval import StructuredRetriever
from agent.schemas import AnswerResult, Document, Question
from agent.workflow import PageIndexWorkflow

RETRIEVAL_MODES = ("config", "pageindex", "bm25", "pageindex-bm25")


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


def apply_retrieval_mode(config: dict, mode: str) -> None:
    if mode not in RETRIEVAL_MODES:
        raise ValueError(f"Unknown retrieval mode: {mode}")
    structured_config = config.setdefault("structured_retrieval", {})
    if mode == "config":
        return
    if mode == "pageindex":
        structured_config["enabled"] = False
        return
    structured_config["enabled"] = True
    if mode == "bm25":
        structured_config["pageindex_first_structured"] = False
        structured_config["link_page_index_context"] = False
        structured_config["page_filter_mode"] = "off"
    elif mode == "pageindex-bm25":
        structured_config["pageindex_first_structured"] = True
        structured_config["link_page_index_context"] = False
        structured_config["page_filter_mode"] = "soft"


def override_run_outputs(
    config: dict,
    *,
    output_csv: str | None = None,
    evidence_json: str | None = None,
) -> None:
    if output_csv:
        config["run"]["output_csv"] = output_csv
    if evidence_json:
        config["run"]["evidence_json"] = evidence_json


def build_missing_indexes(
    documents: list[Document],
    store: PageIndexStore,
    page_config: dict,
    preprocess_config: dict | None = None,
    progress: Callable[[str], None] = print,
) -> None:
    preprocess_config = preprocess_config or {}
    missing: list[Document] = []
    for document in documents:
        try:
            store.load_index(document.doc_id)
        except FileNotFoundError:
            missing.append(document)
    for index, document in enumerate(missing, start=1):
        progress(f"[index {index}/{len(missing)}] {document.doc_id}")
        pdf_parsed_dir = preprocess_config.get("pdf_parsed_dir")
        extracted = extract_pages_with_metadata(
            document.path,
            text_page_chars=preprocess_config.get("text_page_chars", 8000),
            pdf_parsed_dir=store.root.parent / pdf_parsed_dir if pdf_parsed_dir else None,
            pdf_model_order=preprocess_config.get("pdf_model_order"),
        )
        root = build_page_index(
            document,
            extracted.pages,
            leaf_pages=page_config["leaf_pages"],
            branch_factor=page_config["branch_factor"],
        )
        store.save(document, extracted.pages, root, extra_metadata=extracted.metadata)


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
        extra_body=model.get("extra_body"),
        max_tokens_by_purpose=model.get("max_tokens"),
        log_dir=resolve_path(config, config["paths"]["logs"]),
    )
    page_config = config["page_index"]
    structured_config = config.get("structured_retrieval", {})
    structured_retriever = None
    if structured_config.get("enabled", True):
        units_path = resolve_path(
            config,
            structured_config.get("units_path", "processed_data/structured_units.jsonl"),
        )
        structured_retriever = StructuredRetriever(
            units_path,
            max_units=structured_config.get("max_units", 18),
            per_option=structured_config.get("per_option", 3),
            per_option_per_doc=structured_config.get("per_option_per_doc", 1),
            per_doc=structured_config.get("per_doc", 8),
            max_evidence_chars=structured_config.get("max_evidence_chars", 18000),
            min_score=structured_config.get("min_score", 0.1),
            k1=structured_config.get("bm25_k1", 1.5),
            b=structured_config.get("bm25_b", 0.75),
            force_number_hits=structured_config.get("force_number_hits", 3),
            force_entity_hits=structured_config.get("force_entity_hits", 3),
            force_rating_hits=structured_config.get("force_rating_hits", 2),
            number_bonus=structured_config.get("number_bonus", 14.0),
            organization_bonus=structured_config.get("organization_bonus", 18.0),
            rating_bonus=structured_config.get("rating_bonus", 5.0),
            context_phrase_bonus=structured_config.get("context_phrase_bonus", 20.0),
            noise_penalty=structured_config.get("noise_penalty", 18.0),
            split_tables=structured_config.get("split_tables", True),
            page_filter_mode=structured_config.get("page_filter_mode", "hard"),
            page_boost=structured_config.get("page_boost", 10.0),
        )
    return PageIndexWorkflow(
        catalog=catalog,
        store=store,
        llm=llm,
        max_selected_nodes=page_config["max_selected_nodes"],
        max_evidence_chars=page_config["max_evidence_chars"],
        structured_retriever=structured_retriever,
        link_page_index_context=structured_config.get("link_page_index_context", True),
        pageindex_first_structured=structured_config.get("pageindex_first_structured", True),
        linked_page_window=structured_config.get("linked_page_window", 0),
        linked_max_pages_per_doc=structured_config.get("linked_max_pages_per_doc", 4),
        linked_max_chars=structured_config.get("linked_max_chars", 12000),
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
