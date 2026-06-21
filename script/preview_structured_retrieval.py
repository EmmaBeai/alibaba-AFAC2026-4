from __future__ import annotations

import argparse
import sys

from agent.catalog import DatasetCatalog
from agent.config import load_config, resolve_path
from agent.retrieval import StructuredRetriever


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(description="Preview structured evidence retrieval without calling Qwen.")
    parser.add_argument("--config", default="config/default.yaml")
    parser.add_argument("--questions")
    parser.add_argument("--qid", action="append", dest="qids")
    parser.add_argument("--limit", type=int, default=1)
    parser.add_argument("--chars", type=int, default=500)
    args = parser.parse_args()

    config = load_config(args.config)
    catalog = DatasetCatalog(resolve_path(config, config["paths"]["dataset"]))
    question_path = resolve_path(config, args.questions or config["run"]["questions"])
    questions = catalog.load_questions(question_path)
    if args.qids:
        wanted = set(args.qids)
        questions = [question for question in questions if question.qid in wanted]
    else:
        questions = questions[: args.limit]

    retrieval_config = config["structured_retrieval"]
    retriever = StructuredRetriever(
        resolve_path(config, retrieval_config["units_path"]),
        max_units=retrieval_config["max_units"],
        per_option=retrieval_config["per_option"],
        per_option_per_doc=retrieval_config["per_option_per_doc"],
        per_doc=retrieval_config["per_doc"],
        max_evidence_chars=retrieval_config["max_evidence_chars"],
        min_score=retrieval_config["min_score"],
        k1=retrieval_config.get("bm25_k1", 1.5),
        b=retrieval_config.get("bm25_b", 0.75),
        force_number_hits=retrieval_config.get("force_number_hits", 3),
        force_entity_hits=retrieval_config.get("force_entity_hits", 3),
        force_rating_hits=retrieval_config.get("force_rating_hits", 2),
        number_bonus=retrieval_config.get("number_bonus", 14.0),
        organization_bonus=retrieval_config.get("organization_bonus", 18.0),
        rating_bonus=retrieval_config.get("rating_bonus", 5.0),
        context_phrase_bonus=retrieval_config.get("context_phrase_bonus", 20.0),
        noise_penalty=retrieval_config.get("noise_penalty", 18.0),
        split_tables=retrieval_config.get("split_tables", True),
    )
    for question in questions:
        documents = (
            [catalog.get_document(doc_id) for doc_id in question.doc_ids]
            if question.doc_ids
            else catalog.domain_documents(question.domain)[:3]
        )
        evidence = retriever.retrieve(question, documents)
        print(f"\n## {question.qid} {question.question}")
        print(f"doc_ids={[document.doc_id for document in documents]}")
        for block in _iter_evidence_blocks(evidence):
            if block.startswith("###"):
                print(block)
                continue
            header, _, body = block.partition("\n")
            print(header)
            print(body[: args.chars].replace("\n", " "))


def _iter_evidence_blocks(evidence: str):
    current: list[str] = []
    for line in evidence.splitlines():
        if line.startswith("### "):
            if current:
                yield "\n".join(current).strip()
                current = []
            yield line
        elif line.startswith("[unit_id="):
            if current:
                yield "\n".join(current).strip()
            current = [line]
        elif current:
            current.append(line)
    if current:
        yield "\n".join(current).strip()


if __name__ == "__main__":
    main()
