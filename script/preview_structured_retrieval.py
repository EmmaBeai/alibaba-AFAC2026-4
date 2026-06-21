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
        for block in evidence.split("\n[unit_id="):
            block = block.strip()
            if not block:
                continue
            block = "[unit_id=" + block if not block.startswith("[unit_id=") else block
            header, _, body = block.partition("\n")
            print(header)
            print(body[: args.chars].replace("\n", " "))


if __name__ == "__main__":
    main()
