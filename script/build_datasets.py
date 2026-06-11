"""Dataset materialization entrypoint.

Adapts the official ``public_dataset_upload/`` into the engineering input
contract under ``data/``:

    questions/group_a/*_questions.json  -> data/questions.jsonl   (step 1)
    referenced doc_ids + raw/<domain>/  -> data/documents.jsonl   (step 2)

Step 2 depends on step 1: A 榜 ``documents.jsonl`` holds only the docs the
questions actually reference, so we collect ``doc_ids`` from the questions and
resolve each to a raw file. See the README "字段一致性审计" section for the field
inconsistencies this script deliberately preserves (e.g. ``raw_type``) and the
per-domain doc_id namespaces it resolves.

Run from the repo root: ``python -m script.build_datasets``.
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

from agent.io_utils import write_jsonl

# Fixed domain order — keeps the outputs byte-stable across rebuilds and matches
# the official five group_a question files.
DOMAIN_ORDER = (
    "insurance",
    "regulatory",
    "financial_contracts",
    "financial_reports",
    "research",
)

VALID_FORMATS = {"multi", "tf", "mcq"}


# --------------------------------------------------------------------------- #
# Step 1: questions
# --------------------------------------------------------------------------- #
def build_questions(group_a_dir: Path) -> list[dict]:
    """Merge the five group_a question files into validated rows.

    The official JSON already matches the question contract, so this is a
    concatenate-and-validate pass with a stable, explicit key order.

    NOTE on ``raw_type``: the official ``type`` field is NOT a uniform taxonomy
    (regulatory restates answer_format in 8 spellings; insurance uses a real
    skill taxonomy; research uses ~1 freeform slug per question). We keep the raw
    value but name it ``raw_type`` so nothing downstream mistakes it for a
    normalized, cross-domain category. A real ``category`` scheme, if ever
    needed, comes later — driven by S0/S1 error analysis.
    """
    rows: list[dict] = []
    seen_qids: set[str] = set()

    for domain in DOMAIN_ORDER:
        path = group_a_dir / f"{domain}_questions.json"
        if not path.exists():
            raise FileNotFoundError(f"missing question file: {path}")

        questions = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(questions, list):
            raise ValueError(f"{path}: expected a JSON list, got {type(questions).__name__}")

        for q in questions:
            qid = q["qid"]
            if qid in seen_qids:
                raise ValueError(f"duplicate qid: {qid}")
            seen_qids.add(qid)

            fmt = q["answer_format"]
            if fmt not in VALID_FORMATS:
                raise ValueError(f"{qid}: unexpected answer_format {fmt!r}")
            if q["domain"] != domain:
                raise ValueError(f"{qid}: domain {q['domain']!r} != file domain {domain!r}")
            if not isinstance(q["options"], dict):
                raise ValueError(f"{qid}: options must be a dict")

            rows.append(
                {
                    "qid": qid,
                    "domain": domain,
                    "split": q.get("split", "A"),
                    "question": q["question"],
                    "options": q["options"],
                    "answer_format": fmt,
                    "raw_type": q.get("type", ""),
                    "doc_ids": list(q.get("doc_ids", [])),
                }
            )

    return rows


# --------------------------------------------------------------------------- #
# Step 2: documents
# --------------------------------------------------------------------------- #
def index_raw_stems(domain_root: Path) -> dict[str, list[Path]]:
    """Map ``filename stem -> [paths]`` for every file under a domain's raw dir.

    Recursive, so regulatory's nested ``txt/ html/ attachments/`` subdirs are
    covered. Matching on the exact stem (not a glob pattern) is robust to the
    full-width ``〔〕()`` in regulatory doc_ids, which we never sanitize.
    """
    index: dict[str, list[Path]] = {}
    for path in sorted(domain_root.rglob("*")):
        if path.is_file() and not path.name.startswith("."):
            index.setdefault(path.stem, []).append(path)
    return index


def _stage(src: Path, dst: Path) -> None:
    """Copy ``src`` -> ``dst`` unless an identical-size copy is already there.

    The 68 referenced docs total ~242MB, so skip-if-same-size keeps reruns cheap
    and idempotent. We never touch the originals.
    """
    if dst.exists() and dst.stat().st_size == src.stat().st_size:
        return
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)


def build_documents(questions: list[dict], raw_root: Path, out_dir: Path) -> list[dict]:
    """Resolve every referenced doc_id, stage its file into ``out_dir/raw/``.

    A 榜 ``documents.jsonl`` is question-driven: it contains only the docs the
    questions reference (the rest of the raw pool is B 榜 blind-retrieval
    distractors). Hard-validates that each doc_id resolves to exactly one
    existing file — ``missing`` and ``ambiguous`` must both be empty.

    Each referenced doc is copied to ``out_dir/raw/<doc_id><ext>`` (flat — the 68
    doc_ids are globally unique) so ``data/`` is self-contained and decoupled
    from the official drop's nested layout. ``path`` is stored relative to
    ``out_dir`` (matches the examples/data fixture convention).

    ``doc_id`` is kept verbatim (it is the official question->doc join key and
    the ``evidence.json`` audit-traceability key — renaming it would break both).
    ``title`` is a placeholder (the doc_id) for now; preprocess fills real,
    semantic titles once documents are parsed.
    """
    # Collect referenced doc_ids per domain, preserving first-appearance order.
    referenced: dict[str, list[str]] = {d: [] for d in DOMAIN_ORDER}
    seen: dict[str, set[str]] = {d: set() for d in DOMAIN_ORDER}
    for q in questions:
        domain = q["domain"]
        for doc_id in q["doc_ids"]:
            if doc_id not in seen[domain]:
                seen[domain].add(doc_id)
                referenced[domain].append(doc_id)

    rows: list[dict] = []
    missing: list[str] = []
    ambiguous: list[str] = []

    for domain in DOMAIN_ORDER:
        domain_root = raw_root / domain
        if not domain_root.is_dir():
            raise FileNotFoundError(f"missing raw dir: {domain_root}")
        index = index_raw_stems(domain_root)

        for doc_id in referenced[domain]:
            matches = index.get(doc_id, [])
            if not matches:
                missing.append(f"{domain}/{doc_id}")
                continue
            if len(matches) > 1:
                ambiguous.append(f"{domain}/{doc_id} -> {[str(p) for p in matches]}")
                continue
            src = matches[0]
            rel_path = Path("raw") / f"{doc_id}{src.suffix}"
            _stage(src, out_dir / rel_path)
            rows.append(
                {
                    "doc_id": doc_id,
                    "path": rel_path.as_posix(),
                    "title": doc_id,  # placeholder; preprocess enriches with real title
                    "domain": domain,
                }
            )

    if missing or ambiguous:
        details = ""
        if missing:
            details += f"\n  unresolved ({len(missing)}): " + ", ".join(missing)
        if ambiguous:
            details += f"\n  ambiguous ({len(ambiguous)}): " + "; ".join(ambiguous)
        raise ValueError(f"doc_id resolution failed:{details}")

    return rows


# --------------------------------------------------------------------------- #
# Entry point
# --------------------------------------------------------------------------- #
def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source",
        type=Path,
        default=Path("public_dataset_upload"),
        help="official dataset root (default: public_dataset_upload)",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("data"),
        help="engineering input output dir (default: data)",
    )
    args = parser.parse_args()

    # Step 1 — questions
    questions = build_questions(args.source / "questions" / "group_a")
    questions_path = args.out / "questions.jsonl"
    write_jsonl(questions_path, questions)

    by_domain: dict[str, int] = {}
    by_format: dict[str, int] = {}
    for q in questions:
        by_domain[q["domain"]] = by_domain.get(q["domain"], 0) + 1
        by_format[q["answer_format"]] = by_format.get(q["answer_format"], 0) + 1

    print(f"[step 1] wrote {len(questions)} questions -> {questions_path}")
    print("         by domain:", ", ".join(f"{d}={by_domain[d]}" for d in DOMAIN_ORDER))
    print("         by format:", ", ".join(f"{k}={by_format[k]}" for k in sorted(by_format)))

    # Step 2 — documents (depends on step 1's doc_ids)
    documents = build_documents(questions, args.source / "raw", args.out)
    documents_path = args.out / "documents.jsonl"
    write_jsonl(documents_path, documents)

    docs_by_domain: dict[str, int] = {}
    for d in documents:
        docs_by_domain[d["domain"]] = docs_by_domain.get(d["domain"], 0) + 1

    print(f"[step 2] wrote {len(documents)} documents -> {documents_path}  (missing=0)")
    print(f"         staged raw files -> {args.out / 'raw'}/")
    print("         by domain:", ", ".join(f"{d}={docs_by_domain.get(d, 0)}" for d in DOMAIN_ORDER))


if __name__ == "__main__":
    main()
