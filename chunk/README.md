# `documents.jsonl` Schema

`chunk.documents` builds the canonical `processed_data/documents.jsonl` allowlist. This file is intentionally conservative: it records only document identity fields that can be produced without dataset-specific keyword lists, company/product maps, or title heuristics.

Generation command:

```bash
PYTHONPATH=. python -m chunk.documents
```

## Module Boundaries

- `chunk/documents.py`: discovers raw files, preserves the first-seen question `doc_ids` order, calls the metadata builder, and writes JSONL.
- `chunk/document_metadata.py`: extracts only no-magic metadata: `doc_id`, `title`, `domain`, and optional `dates.period`.
- `chunk/question_keywords.py`: remains a separate question phrase module. Its output must not be written into `documents.jsonl`.

## Current Record Format

Each line is one JSON object:

```jsonc
{
  "doc_id": "annual_byd_2024_report",
  // Stable logical document id. It stays aligned with question doc_ids and raw file stems.

  "title": "Some Parsed Title",
  // Prefer H1 headings from cleaned markdown output for PDF sources.
  // If there are multiple H1 headings, use the first one.
  // If there is no H1, omit this field.
  // HTML sources may use their page title metadata.
  // If no title source exists, omit this field.

  "domain": "financial_reports",
  // The first directory under public_dataset_upload/raw.

  "dates": {
    "period": "2024"
  }
  // Optional. Emitted only when a generic 20xx year pattern is found in title or doc_id.
}
```

Records omit fields that require magic to fill. In particular, this canonical file does not include `keywords`, `aliases`, `document_type`, `identity_entities`, `family_id`, parser provenance, hashes, or source paths.

## No-Magic Contract

`document_metadata.py` must not contain:

- hardcoded company, issuer, or product maps
- handwritten document-type keyword lists
- Chinese or English title blacklists
- domain-specific phrase checks such as matching insurance product words or regulatory decision names
- question-derived routing keywords

Allowed logic is limited to generic structure:

- raw path and suffix handling
- parser output location handling
- HTML title extraction
- Markdown title extraction from `processed_data/cleaned_markdown_68/{doc_id}.md`
  when available, falling back to parser markdown only if cleaned output is
  missing: the first H1 wins, and no H1 means the title is omitted
- simple numbered-heading rejection
- generic whitespace/tag/image cleanup
- generic `20xx` year extraction

If a field cannot be filled from those sources, omit the field instead of guessing.

## Verification

After editing the document builder, run:

```bash
PYTHONPATH=. pytest -q tests/test_chunk_documents.py
PYTHONPATH=. python -m chunk.documents
```

Useful artifact check:

```bash
python -c 'import json,pathlib; rows=[json.loads(l) for l in pathlib.Path("processed_data/documents.jsonl").open(encoding="utf-8")]; print(len(rows), len({r["doc_id"] for r in rows}), sorted({k for r in rows for k in r}), sum("keywords" in r for r in rows))'
```
