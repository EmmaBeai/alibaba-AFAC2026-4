from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from preprocess.document_metadata import build_document_record


SUPPORTED_SUFFIXES = {".pdf", ".html", ".htm", ".txt"}
SOURCE_KIND_BY_SUFFIX = {
    ".pdf": "pdf",
    ".html": "html",
    ".htm": "html",
    ".txt": "txt",
}


def build_documents(
    *,
    dataset_root: Path | str = "public_dataset_upload",
    questions_path: Path | str = "public_dataset_upload/questions/group_a",
    processed_root: Path | str = "processed_data",
    version: str = "public_dataset_upload_group_a",
) -> list[dict[str, Any]]:
    processed_root = Path(processed_root)
    document_sources = build_document_sources(
        dataset_root=dataset_root,
        questions_path=questions_path,
        version=version,
    )
    return [
        build_document_record(
            doc_id=str(source["doc_id"]),
            source_path=Path(str(source["source_path"])),
            source_kind=str(source["source_kind"]),
            domain=str(source["domain"]),
            processed_root=processed_root,
        )
        for source in document_sources
    ]


def build_document_sources(
    *,
    dataset_root: Path | str = "public_dataset_upload",
    questions_path: Path | str = "public_dataset_upload/questions/group_a",
    version: str = "public_dataset_upload_group_a",
) -> list[dict[str, Any]]:
    dataset_root = Path(dataset_root)
    questions_path = Path(questions_path)
    raw_documents = _discover_raw_documents(dataset_root)
    doc_ids = _question_doc_ids(questions_path)

    missing = [doc_id for doc_id in doc_ids if doc_id not in raw_documents]
    if missing:
        raise ValueError(f"Question doc_ids not found under {dataset_root / 'raw'}: {', '.join(missing)}")

    return [
        {
            "doc_id": doc_id,
            "domain": _domain_from_source_path(raw_documents[doc_id], dataset_root),
            "source_kind": SOURCE_KIND_BY_SUFFIX[raw_documents[doc_id].suffix.lower()],
            "source_path": str(raw_documents[doc_id]),
        }
        for doc_id in doc_ids
    ]


def write_documents_jsonl(documents: list[dict[str, Any]], output_path: Path | str) -> None:
    _write_jsonl(documents, output_path)


def write_document_sources_jsonl(document_sources: list[dict[str, Any]], output_path: Path | str) -> None:
    _write_jsonl(document_sources, output_path)


def _write_jsonl(rows: list[dict[str, Any]], output_path: Path | str) -> None:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")


def _discover_raw_documents(dataset_root: Path) -> dict[str, Path]:
    raw_root = dataset_root / "raw"
    documents: dict[str, Path] = {}
    duplicates: dict[str, list[Path]] = {}
    for path in sorted(raw_root.rglob("*")):
        if not path.is_file() or path.suffix.lower() not in SUPPORTED_SUFFIXES:
            continue
        current = documents.get(path.stem)
        if current is not None:
            duplicates.setdefault(path.stem, [current]).append(path)
            continue
        documents[path.stem] = path
    if duplicates:
        details = "; ".join(
            f"{doc_id}: {', '.join(str(path) for path in paths)}"
            for doc_id, paths in sorted(duplicates.items())
        )
        raise ValueError(f"Duplicate raw document stems are ambiguous: {details}")
    return documents


def _question_doc_ids(questions_path: Path) -> list[str]:
    files = sorted(questions_path.glob("*.json")) if questions_path.is_dir() else [questions_path]
    doc_ids: list[str] = []
    seen: set[str] = set()
    for file_path in files:
        payload = json.loads(file_path.read_text(encoding="utf-8"))
        for item in payload:
            for doc_id in item.get("doc_ids") or []:
                if doc_id not in seen:
                    seen.add(doc_id)
                    doc_ids.append(doc_id)
    return doc_ids


def _domain_from_source_path(source_path: Path, dataset_root: Path) -> str:
    relative = source_path.relative_to(dataset_root / "raw")
    return relative.parts[0]


def main() -> None:
    parser = argparse.ArgumentParser(description="Build canonical processed_data/documents.jsonl.")
    parser.add_argument("--dataset-root", default="public_dataset_upload")
    parser.add_argument("--questions", default="public_dataset_upload/questions/group_a")
    parser.add_argument("--processed-root", default="processed_data")
    parser.add_argument("--output", default="processed_data/documents.jsonl")
    parser.add_argument("--version", default="public_dataset_upload_group_a")
    args = parser.parse_args()

    documents = build_documents(
        dataset_root=args.dataset_root,
        questions_path=args.questions,
        processed_root=args.processed_root,
        version=args.version,
    )
    write_documents_jsonl(documents, args.output)
    print(f"wrote {len(documents)} documents to {args.output}")


if __name__ == "__main__":
    main()
