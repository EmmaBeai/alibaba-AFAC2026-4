from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from agent.bm25_experiment import ExperimentalBM25
from agent.catalog import DatasetCatalog
from agent.config import load_config, resolve_path


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(
        description="Debug experimental field-aware BM25 retrieval without calling Qwen."
    )
    parser.add_argument("--config", default="config/default.yaml")
    parser.add_argument("--questions")
    parser.add_argument("--qid", action="append", dest="qids")
    parser.add_argument("--limit", type=int, default=1)
    parser.add_argument("--top-k", type=int, default=8)
    parser.add_argument("--chars", type=int, default=260)
    parser.add_argument("--no-split-tables", action="store_true")
    parser.add_argument("--out-json")
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

    units_path = resolve_path(
        config,
        config.get("structured_retrieval", {}).get("units_path", "processed_data/structured_units.jsonl"),
    )
    bm25 = ExperimentalBM25(units_path, split_tables=not args.no_split_tables)
    output = []
    for question in questions:
        documents = (
            [catalog.get_document(doc_id) for doc_id in question.doc_ids]
            if question.doc_ids
            else catalog.domain_documents(question.domain)[:3]
        )
        print(f"\n## {question.qid} {question.question}")
        print(f"doc_ids={[document.doc_id for document in documents]}")
        question_payload = {
            "qid": question.qid,
            "question": question.question,
            "doc_ids": [document.doc_id for document in documents],
            "options": [],
        }
        for option, option_text in question.options.items():
            parsed = bm25.parse_option(question, documents, option, option_text)
            candidates = bm25.rank(question, documents, parsed, top_k=args.top_k)
            per_doc_candidates = (
                bm25.rank_per_doc(question, documents, parsed, top_k=min(3, args.top_k))
                if parsed.compare
                else {}
            )
            print(f"\n### Option {option}: {option_text}")
            print(
                "parsed="
                + json.dumps(
                    {
                        "doc_ids": parsed.doc_ids,
                        "fields": parsed.fields,
                        "values": parsed.values,
                        "numbers": parsed.numbers,
                        "organizations": parsed.organizations,
                        "ratings": parsed.ratings,
                        "compare": parsed.compare,
                    },
                    ensure_ascii=False,
                )
            )
            option_payload = {
                "option": option,
                "text": option_text,
                "parsed": parsed.to_dict(),
                "candidates": [candidate.to_dict() for candidate in candidates],
                "per_doc_candidates": {
                    doc_id: [candidate.to_dict() for candidate in values]
                    for doc_id, values in per_doc_candidates.items()
                },
            }
            question_payload["options"].append(option_payload)
            if per_doc_candidates:
                print("per_doc_candidates:")
                for doc_id, doc_candidates in per_doc_candidates.items():
                    print(f"  [{doc_id}]")
                    for index, candidate in enumerate(doc_candidates, start=1):
                        preview = candidate.text[: args.chars].replace("\n", " ")
                        print(
                            f"    {index}. score={candidate.score:.2f} bm25={candidate.bm25:.2f} "
                            f"page={candidate.page} type={candidate.chunk_type} unit={candidate.unit_id}"
                        )
                        print(f"       field_hits={candidate.field_hits} value_hits={candidate.value_hits}")
                        print(f"       reasons={candidate.reasons}")
                        print(f"       text={preview}")
            for index, candidate in enumerate(candidates, start=1):
                preview = candidate.text[: args.chars].replace("\n", " ")
                print(
                    f"{index}. score={candidate.score:.2f} bm25={candidate.bm25:.2f} "
                    f"doc={candidate.doc_id} page={candidate.page} "
                    f"type={candidate.chunk_type} unit={candidate.unit_id}"
                )
                print(f"   field_hits={candidate.field_hits} value_hits={candidate.value_hits}")
                print(f"   reasons={candidate.reasons}")
                print(f"   text={preview}")
        output.append(question_payload)

    if args.out_json:
        out_path = Path(args.out_json)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"\nwrote {out_path}")


if __name__ == "__main__":
    main()
