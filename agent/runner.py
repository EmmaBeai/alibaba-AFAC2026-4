from __future__ import annotations

from typing import Any

from agent.bm25_qwen import BM25QwenAnswerer
from agent.bm25_top1 import BM25Top1Answerer
from agent.catalog import DatasetCatalog
from agent.config import resolve_path
from agent.dense_qwen import DenseQwenAnswerer
from agent.embedding_client import DEFAULT_EMBEDDING_MODEL, DashScopeEmbeddingClient
from agent.output import load_completed_results, write_outputs
from agent.qwen_client import DEFAULT_QWEN_MODEL, QwenPlusClient
from agent.schemas import AnswerResult, Document, Question


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


def create_answerer(config: dict[str, Any], *, qwen_client: Any | None = None) -> Any:
    answerer_type = config.get("answerer", {}).get("type", "bm25_top1")
    structured_config = config.get("structured_retrieval", {})
    units_path = resolve_path(
        config,
        structured_config.get("units_path", "processed_data/structured_units.jsonl"),
    )
    common_kwargs = {
        "chunk_chars": structured_config.get("bm25_top1_chunk_chars", 1800),
        "k1": structured_config.get("bm25_k1", 1.5),
        "b": structured_config.get("bm25_b", 0.75),
    }
    if answerer_type == "bm25_top1":
        return BM25Top1Answerer(units_path, **common_kwargs)
    if answerer_type == "dense_qwen":
        model_config = config.get("model", {})
        embedding_config = config.get("embedding", {})
        embedding_client = DashScopeEmbeddingClient(
            model=embedding_config.get("model", DEFAULT_EMBEDDING_MODEL),
            dimension=embedding_config.get("dimension", 1024),
            batch_size=embedding_config.get("batch_size", 10),
            api_key_env=embedding_config.get("api_key_env", "DASHSCOPE_API_KEY"),
            api_key_env_fallbacks=embedding_config.get("api_key_env_fallbacks"),
            base_url_env=embedding_config.get("base_url_env", "DASHSCOPE_EMBEDDING_BASE_URL"),
            default_base_url=embedding_config.get(
                "default_base_url",
                "https://dashscope.aliyuncs.com/compatible-mode/v1",
            ),
            timeout_seconds=embedding_config.get("timeout_seconds", 120),
            max_retries=embedding_config.get("max_retries", 3),
        )
        client = qwen_client or QwenPlusClient(
            model=model_config.get("name", DEFAULT_QWEN_MODEL),
            api_key_env=model_config.get("api_key_env", "DASHSCOPE_API_KEY"),
            api_key_env_fallbacks=model_config.get("api_key_env_fallbacks"),
            base_url_env=model_config.get("base_url_env", "QWEN_BASE_URL"),
            default_base_url=model_config.get(
                "default_base_url",
                "https://dashscope.aliyuncs.com/compatible-mode/v1",
            ),
            temperature=model_config.get("temperature", 0),
            extra_body=model_config.get("extra_body"),
            max_tokens=model_config.get("max_tokens", 768),
            timeout_seconds=model_config.get("timeout_seconds", 120),
        )
        return DenseQwenAnswerer(
            units_path,
            embedding_cache_path=resolve_path(
                config,
                embedding_config.get("cache_path", "processed_data/embeddings/text_embedding_v4_dense1024.jsonl"),
            ),
            embedding_client=embedding_client,
            qwen_client=client,
            max_evidence_chars=structured_config.get("max_evidence_chars", 12000),
            top_k_patches=structured_config.get("top_k_patches", 5),
            build_missing_embeddings=embedding_config.get("build_missing_embeddings", False),
        )
    if answerer_type == "bm25_qwen":
        model_config = config.get("model", {})
        client = qwen_client or QwenPlusClient(
            model=model_config.get("name", DEFAULT_QWEN_MODEL),
            api_key_env=model_config.get("api_key_env", "DASHSCOPE_API_KEY"),
            api_key_env_fallbacks=model_config.get("api_key_env_fallbacks"),
            base_url_env=model_config.get("base_url_env", "QWEN_BASE_URL"),
            default_base_url=model_config.get(
                "default_base_url",
                "https://dashscope.aliyuncs.com/compatible-mode/v1",
            ),
            temperature=model_config.get("temperature", 0),
            extra_body=model_config.get("extra_body"),
            max_tokens=model_config.get("max_tokens", 768),
            timeout_seconds=model_config.get("timeout_seconds", 120),
        )
        return BM25QwenAnswerer(
            units_path,
            qwen_client=client,
            max_evidence_chars=structured_config.get("max_evidence_chars", 12000),
            top_k_patches=structured_config.get("top_k_patches", 5),
            **common_kwargs,
        )
    raise ValueError(f"Answerer type is not implemented: {answerer_type}")


def answer_questions(
    config: dict,
    questions: list[Question],
    documents: list[Document],
    answerer: BM25Top1Answerer,
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
        result = answerer.answer(question, _question_documents(question, documents))
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


def _question_documents(question: Question, documents: list[Document]) -> list[Document]:
    if question.doc_ids:
        by_id = {document.doc_id: document for document in documents}
        selected = [by_id[doc_id] for doc_id in question.doc_ids if doc_id in by_id]
        if selected:
            return selected
    return [document for document in documents if document.domain == question.domain]
