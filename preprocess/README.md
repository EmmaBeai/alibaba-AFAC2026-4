# Preprocess Pipeline

This directory owns the header-based preprocessing pipeline. It turns the raw
competition files into cleaned markdown, canonical document metadata, one
unified parent-linked structure node JSONL, code-generated block nodes, and one
nested chunk JSON per markdown heading level.

The pipeline is intentionally split into two document manifests:

- `processed_data/document_sources.jsonl`: the upstream source manifest. It is
  built from question `doc_ids` and raw files, and contains only `doc_id`,
  `domain`, `source_kind`, and `source_path`.
- `processed_data/documents.jsonl`: the downstream metadata manifest. It is
  built after markdown cleaning, and PDF titles must come from the cleaned
  markdown H1.

Do not use corpus-size suffixes in generated path names. The active document set
comes from the question files, not from a hard-coded corpus size.

## Main Entry Point

Run PDF conversion first when `processed_data/pdf_parsed` is missing:

```bash
PYTHONPATH=. python -m preprocess.pdf_to_markdown public_dataset_upload/raw --models glm-ocr pypdf
```

Then run the header-based pipeline:

```bash
PYTHONPATH=. python -m preprocess.run_pipeline
```

Default outputs:

- `processed_data/document_sources.jsonl`
- `processed_data/parsed_files/`
- `processed_data/cleaned_markdown/`
- `processed_data/documents.jsonl`
- `processed_data/structure_nodes.jsonl`
- `processed_data/block_nodes.jsonl`
- `processed_data/h2_chunks.json`
- `processed_data/h3_chunks.json`
- `processed_data/h4_chunks.json`
- `processed_data/h5_chunks.json`
- `processed_data/h6_chunks.json`

`preprocess.run_pipeline` calls the stages in this order:

1. Build `document_sources.jsonl` from question `doc_ids` and raw files.
2. Materialize parsed files into `processed_data/parsed_files`.
3. Clean markdown heading levels into `processed_data/cleaned_markdown`.
4. Build final `documents.jsonl` from cleaned metadata.
5. Build `structure_nodes.jsonl` as the unified document/section tree.
6. Build `block_nodes.jsonl` as code-generated paragraph/table/image children.
7. Build H2 through H6 nested chunk JSON files.

## Stage Details

### PDF parser cache

`preprocess/pdf_to_markdown.py` runs the production PDF parsers and writes:

- `processed_data/pdf_parsed/glm-ocr/{doc_id}.md`
- `processed_data/pdf_parsed/pypdf/{doc_id}.md`

`glm-ocr` provides the markdown structure used for heading cleaning. `pypdf`
provides a page-marked text layer used later for `page_start` and `page_end`
backfill.

### Source manifest

`document_sources.jsonl` is an allowlist/source manifest. It should stay small
and structural:

```json
{"doc_id":"text01","domain":"financial_contracts","source_kind":"pdf","source_path":"public_dataset_upload/raw/.../text01.pdf"}
```

It is safe to build before markdown cleaning because it does not need titles.

### Parsed files

`preprocess/build_parsed_files.py` reads `document_sources.jsonl` and writes one
parsed file per source document:

- PDFs become `.md` files using the selected PDF parser output, default
  `glm-ocr`.
- HTML and TXT sources become `.txt` files using the local parsers.
- `_manifest.jsonl` records parser/source details for the materialized files.

Standalone command:

```bash
PYTHONPATH=. python -m preprocess.build_parsed_files \
  --document-sources processed_data/document_sources.jsonl \
  --output-dir processed_data/parsed_files
```

### Markdown cleaning

`preprocess/clean_markdown_with_openai.py` reads `processed_data/parsed_files`
and writes `processed_data/cleaned_markdown`.

- `.txt` files are copied unchanged.
- `.md` files are not rewritten wholesale. The cleaner extracts existing
  markdown heading candidate lines, asks the OpenAI Responses API to label them
  as `body` or `h1` through `h6`, validates the labels, then applies those labels
  back to the original markdown.
- Each cleaned markdown document must have exactly one H1.

This stage requires `OPENAI_API_KEY` unless all needed cleaned files already
exist and the stage is skipped.

Standalone command:

```bash
PYTHONPATH=. python -m preprocess.clean_markdown_with_openai \
  --input-dir processed_data/parsed_files \
  --output-dir processed_data/cleaned_markdown
```

### Final document metadata

`preprocess/documents.py` builds `processed_data/documents.jsonl`.

For PDFs, the title must come from `processed_data/cleaned_markdown/{doc_id}.md`
as the first H1. There is no parser-output fallback. If a PDF has no cleaned H1,
the builder fails with an English error asking you to rerun markdown cleaning.

Standalone command:

```bash
PYTHONPATH=. python -m preprocess.documents --output processed_data/documents.jsonl
```

### Structure nodes

`preprocess/build_structure_nodes.py` reads:

- final metadata from `processed_data/documents.jsonl`
- headings/text from `processed_data/cleaned_markdown`
- page markers from `processed_data/pdf_parsed/pypdf`

It writes `processed_data/structure_nodes.jsonl`, one row per document or
section node. This is the canonical tree-shaped structure layer:

```json
{"node_id":"text01::h3::0001","parent_id":"text01::h2::0001","kind":"section","heading_level":3,"path":["募集说明书","重大事项提示","一、发行人基本财务情况"],"child_ids":[]}
```

Each node has a stable `node_id`, `parent_id`, `child_ids`, heading `path`,
direct `text`, and optional `page_start` / `page_end`. The H2 through H6 files
below remain derived exports for per-level inspection and compatibility.

Standalone command:

```bash
PYTHONPATH=. python -m preprocess.build_structure_nodes \
  --documents processed_data/documents.jsonl \
  --content-dir processed_data/cleaned_markdown \
  --page-content-dir processed_data/pdf_parsed/pypdf \
  --output processed_data/structure_nodes.jsonl
```

### Block nodes

`preprocess/build_block_nodes.py` reads `processed_data/structure_nodes.jsonl`
and writes `processed_data/block_nodes.jsonl`. This is the generated block layer
under section nodes. It must be produced by code; do not hand-edit paragraph,
table, image, `node_id`, or `parent_id` values in the output.

Block semantics:

- `paragraph` nodes come from section direct text, split on markdown block
  boundaries.
- `table` nodes keep full `<table ...>...</table>` markdown/HTML blocks.
- `image` nodes keep `image_alt`, `image_relative_path`, and an absolute
  `image_path`.
- Image paths are resolved from the raw GLM OCR page output, for example
  `processed_data/pdf_parsed/_raw/glm-ocr/{doc_id}/glmocr_output/page_0001/imgs/...`.
  Section `page_start` / `page_end` constrains lookup so repeated filenames such
  as `cropped_page0_idx0.jpg` map to the correct page.
- If the referenced image file does not exist under raw GLM OCR output,
  `image_path` stays empty. The original markdown reference remains only in
  `image_relative_path`.
- PDF/OCR false paragraph breaks are merged when the previous paragraph does
  not end with sentence punctuation and the next block is not a new numbered
  item, table, or image.

Standalone command:

```bash
PYTHONPATH=. python -m preprocess.build_block_nodes \
  --structure-nodes processed_data/structure_nodes.jsonl \
  --raw-glm-ocr-dir processed_data/pdf_parsed/_raw/glm-ocr \
  --output processed_data/block_nodes.jsonl
```

### Heading chunks

`preprocess/build_h2_chunks.py` reads:

- final metadata from `processed_data/documents.jsonl`
- headings/text from `processed_data/cleaned_markdown`
- page markers from `processed_data/pdf_parsed/pypdf`

It writes one nested JSON per heading level:

```bash
PYTHONPATH=. python -m preprocess.build_h2_chunks --all-levels
```

Chunk semantics:

- H2 chunks are grouped under the final document title from `documents.jsonl`.
- H3 chunks are grouped under their parent H2.
- H4 chunks are grouped under their parent H3, and so on.
- A node's `text` is the direct body text after that heading, stopping at the
  next heading of any level.
- A node's page range runs from the current heading to the next same-or-higher
  heading, using pypdf page markers when available.

## Common Workflows

Regenerate chunks from existing cleaned markdown without calling the API:

```bash
PYTHONPATH=. python -m preprocess.run_pipeline --skip-parsed-files --skip-clean
```

Regenerate parsed files and chunks, but reuse existing cleaned markdown:

```bash
PYTHONPATH=. python -m preprocess.run_pipeline --skip-clean --overwrite-parsed
```

Run only one heading level:

```bash
PYTHONPATH=. python -m preprocess.build_h2_chunks \
  --level 3 \
  --output processed_data/h3_chunks.json
```

## Legacy Helpers

`preprocess/text_html_to_text.py` maintains the older
`processed_data/text_html_parsed` cache. The header-based pipeline primarily
uses `preprocess/build_parsed_files.py`.
