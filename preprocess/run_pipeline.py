"""Run the header-based production preprocessing artifact builders.

PDF markdown should already exist under ``processed_data/pdf_parsed``. Use
``preprocess.pdf_to_markdown`` first when those OCR/text-layer outputs are
missing.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from preprocess.build_block_nodes import (
    build_block_nodes,
    write_jsonl as write_block_nodes_jsonl,
)
from preprocess.build_h2_chunks import build_heading_chunks, write_json
from preprocess.build_structure_nodes import (
    build_structure_nodes,
    write_jsonl as write_structure_nodes_jsonl,
)
from preprocess.build_parsed_files import build_parsed_files
from preprocess.clean_markdown_with_openai import (
    DEFAULT_MODEL,
    DEFAULT_PROMPT_PATH,
    clean_directory,
)
from preprocess.documents import (
    build_document_sources,
    build_documents,
    write_document_sources_jsonl,
    write_documents_jsonl,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Build header-based preprocessing artifacts.")
    parser.add_argument("--dataset-root", default="public_dataset_upload")
    parser.add_argument("--questions", default="public_dataset_upload/questions/group_a")
    parser.add_argument("--processed-root", default="processed_data")
    parser.add_argument("--document-sources-output", default="processed_data/document_sources.jsonl")
    parser.add_argument("--documents-output", default="processed_data/documents.jsonl")
    parser.add_argument("--structure-nodes-output", default="processed_data/structure_nodes.jsonl")
    parser.add_argument("--block-nodes-output", default="processed_data/block_nodes.jsonl")
    parser.add_argument("--parsed-output-dir", default="processed_data/parsed_files")
    parser.add_argument("--cleaned-output-dir", default="processed_data/cleaned_markdown")
    parser.add_argument("--heading-output-dir", default="processed_data")
    parser.add_argument("--pdf-content-model", default="glm-ocr")
    parser.add_argument("--page-content-dir", default="processed_data/pdf_parsed/pypdf")
    parser.add_argument("--raw-glm-ocr-dir", default="processed_data/pdf_parsed/_raw/glm-ocr")
    parser.add_argument("--skip-document-sources", action="store_true")
    parser.add_argument("--skip-documents", action="store_true")
    parser.add_argument("--skip-parsed-files", action="store_true")
    parser.add_argument("--skip-clean", action="store_true")
    parser.add_argument("--overwrite-parsed", action="store_true")
    parser.add_argument("--overwrite-clean", action="store_true")
    parser.add_argument("--prompt", type=Path, default=DEFAULT_PROMPT_PATH)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--api-key-env", default="OPENAI_API_KEY")
    parser.add_argument("--base-url", default="https://api.openai.com/v1")
    parser.add_argument("--chunk-chars", type=int, default=0)
    parser.add_argument("--max-output-tokens", type=int, default=50000)
    parser.add_argument("--timeout-seconds", type=int, default=300)
    parser.add_argument("--max-retries", type=int, default=3)
    parser.add_argument("--retry-sleep-seconds", type=float, default=2.0)
    parser.add_argument("--request-sleep-seconds", type=float, default=0.0)
    args = parser.parse_args()

    dataset_root = Path(args.dataset_root)
    processed_root = Path(args.processed_root)

    if not args.skip_document_sources:
        document_sources = build_document_sources(
            dataset_root=dataset_root,
            questions_path=Path(args.questions),
        )
        write_document_sources_jsonl(document_sources, Path(args.document_sources_output))
        print(f"wrote {len(document_sources)} document sources to {args.document_sources_output}")

    if not args.skip_parsed_files:
        parsed_rows = build_parsed_files(
            document_sources_path=Path(args.document_sources_output),
            processed_root=processed_root,
            output_dir=Path(args.parsed_output_dir),
            pdf_model=args.pdf_content_model,
            overwrite=args.overwrite_parsed,
        )
        print(f"wrote {len(parsed_rows)} parsed files to {args.parsed_output_dir}")

    if not args.skip_clean:
        stats = clean_directory(
            input_dir=Path(args.parsed_output_dir),
            output_dir=Path(args.cleaned_output_dir),
            prompt_path=args.prompt,
            model=args.model,
            api_key_env=args.api_key_env,
            base_url=args.base_url,
            chunk_chars=args.chunk_chars,
            max_output_tokens=args.max_output_tokens,
            timeout_seconds=args.timeout_seconds,
            max_retries=args.max_retries,
            retry_sleep_seconds=args.retry_sleep_seconds,
            request_sleep_seconds=args.request_sleep_seconds,
            overwrite=args.overwrite_clean,
            dry_run=False,
            limit=None,
            only=None,
        )
        print(f"heading clean stats: {stats}")

    if not args.skip_documents:
        documents = build_documents(
            dataset_root=dataset_root,
            questions_path=Path(args.questions),
            processed_root=processed_root,
        )
        write_documents_jsonl(documents, Path(args.documents_output))
        print(f"wrote {len(documents)} documents to {args.documents_output}")

    structure_nodes = build_structure_nodes(
        documents_path=Path(args.documents_output),
        content_dir=Path(args.cleaned_output_dir),
        page_content_dir=Path(args.page_content_dir) if args.page_content_dir else None,
    )
    write_structure_nodes_jsonl(structure_nodes, Path(args.structure_nodes_output))
    print(f"wrote {len(structure_nodes)} structure nodes to {args.structure_nodes_output}")

    block_nodes = build_block_nodes(
        structure_nodes=structure_nodes,
        raw_glm_ocr_dir=Path(args.raw_glm_ocr_dir) if args.raw_glm_ocr_dir else None,
    )
    write_block_nodes_jsonl(block_nodes, Path(args.block_nodes_output))
    print(f"wrote {len(block_nodes)} block nodes to {args.block_nodes_output}")

    heading_output_dir = Path(args.heading_output_dir)
    for level in range(2, 7):
        payload = build_heading_chunks(
            level=level,
            documents_path=Path(args.documents_output),
            content_dir=Path(args.cleaned_output_dir),
            page_content_dir=Path(args.page_content_dir) if args.page_content_dir else None,
        )
        output_path = heading_output_dir / f"h{level}_chunks.json"
        write_json(payload, output_path)
        print(f"wrote {len(payload['documents'])} documents to {output_path}")


if __name__ == "__main__":
    main()
