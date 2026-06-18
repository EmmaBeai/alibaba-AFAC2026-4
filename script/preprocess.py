"""Document parsing and chunk construction entrypoint."""

from __future__ import annotations

import argparse
from pathlib import Path

from agent.chunking import make_chunks, normalize_text
from agent.io_utils import read_jsonl, write_jsonl
from agent.schema import DocumentMeta, ParsedDocument
from preprocess import html_parser, pdf_parser, txt_parser


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Parse documents into normalized text and retrieval chunks."
    )
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--processed-dir", type=Path, default=Path("processed_data"))
    parser.add_argument("--chunk-chars", type=int, default=1800)
    parser.add_argument("--overlap-chars", type=int, default=240)
    parser.add_argument(
        "--pdf-model",
        default=None,
        help="Optional explicit PDF markdown model, e.g. mineru2.5-pro.",
    )
    args = parser.parse_args()

    documents_path = args.data_dir / "documents.jsonl"
    metas = [DocumentMeta.from_dict(row) for row in read_jsonl(documents_path)]

    parsed_docs: list[ParsedDocument] = []
    chunk_rows: list[dict] = []

    for meta in metas:
        source_path = args.data_dir / meta.path
        parsed = parse_document(
            meta,
            source_path=source_path,
            pdf_model=args.pdf_model,
            pdf_parsed_dir=args.processed_dir / "pdf_parsed",
        )
        parsed_docs.append(parsed)

        for chunk in make_chunks(
            parsed,
            chunk_chars=args.chunk_chars,
            overlap_chars=args.overlap_chars,
        ):
            chunk_rows.append(chunk.to_dict())

    parsed_rows = [
        {
            "doc_id": doc.doc_id,
            "title": doc.title,
            "domain": doc.domain,
            "text": doc.text,
        }
        for doc in parsed_docs
    ]

    parsed_path = args.processed_dir / "parsed_documents.jsonl"
    chunks_path = args.processed_dir / "chunks.jsonl"
    write_jsonl(parsed_path, parsed_rows)
    write_jsonl(chunks_path, chunk_rows)

    print(f"[preprocess] parsed {len(parsed_docs)} documents -> {parsed_path}")
    print(f"[preprocess] wrote {len(chunk_rows)} chunks -> {chunks_path}")


def extract_text(path: Path) -> str:
    suffix = Path(path).suffix.lower()
    if suffix == ".txt":
        return txt_parser.parse(path)
    if suffix in {".html", ".htm"}:
        return html_parser.parse(path)
    if suffix == ".pdf":
        return pdf_parser.parse(path)
    raise ValueError(f"unsupported document type: {path}")


def parse_document(
    meta: DocumentMeta,
    source_path: Path,
    pdf_model: str | None = None,
    pdf_parsed_dir: Path = pdf_parser.PDF_PARSED_DIR,
) -> ParsedDocument:
    suffix = source_path.suffix.lower()
    if suffix == ".txt":
        text = txt_parser.parse(source_path)
        title = meta.title or _first_nonempty_line(text) or meta.doc_id
    elif suffix in {".html", ".htm"}:
        text = html_parser.parse(source_path)
        title = html_parser.extract_title(source_path) or meta.title or meta.doc_id
    elif suffix == ".pdf":
        text = pdf_parser.parse(
            source_path,
            model=pdf_model,
            parsed_dir=pdf_parsed_dir,
        )
        title = (
            pdf_parser.extract_title(
                source_path,
                model=pdf_model,
                parsed_dir=pdf_parsed_dir,
            )
            or meta.title
            or meta.doc_id
        )
    else:
        raise ValueError(f"unsupported document type: {source_path}")

    return ParsedDocument(
        doc_id=meta.doc_id,
        title=title,
        domain=meta.domain,
        text=normalize_text(text),
    )


def _first_nonempty_line(text: str) -> str | None:
    for line in text.splitlines():
        stripped = line.strip().lstrip("#").strip()
        if stripped:
            return stripped
    return None


if __name__ == "__main__":
    main()
