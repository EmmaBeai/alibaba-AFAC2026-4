from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

from agent.preprocess import extract_pages_with_metadata


DEFAULT_TERMS = [
    "广东",
    "发行规模",
    "发行金额",
    "主体信用评级",
    "债券受托管理人",
    "AAA",
]


def read_text(path: Path) -> str:
    if not path.exists():
        return ""
    return path.read_text(encoding="utf-8", errors="replace")


def inspect_text(label: str, path: Path, terms: list[str]) -> None:
    text = read_text(path)
    if not text:
        print(f"\n[{label}] MISSING {path}")
        return
    print(f"\n[{label}] {path}")
    print(
        " ".join(
            [
                f"chars={len(text)}",
                f"has_guangdong={'广东' in text}",
                f"mojibake_marker={'å¹¿' in text}",
                f"replacement={'�' in text}",
            ]
        )
    )
    print("terms=" + ", ".join(f"{term}:{text.count(term)}" for term in terms))
    print("head=" + repr(text[:220]))


def inspect_selector(doc_id: str, pdf_path: Path, processed_dir: Path) -> None:
    if not pdf_path.exists():
        print(f"\n[selector] MISSING {pdf_path}")
        return
    extracted = extract_pages_with_metadata(
        pdf_path,
        pdf_parsed_dir=processed_dir / "pdf_parsed",
        pdf_model_order=["glm-ocr", "mineru2.5-pro", "paddleocr-vl-1.6", "pypdf"],
    )
    source = extracted.metadata.get("pdf_source", {})
    print(f"\n[selector] {doc_id} -> {source.get('selected_model')}")
    print(f"reason={source.get('selection_reason')} split={source.get('split_method')} pages={len(extracted.pages)}")
    for quality in source.get("qualities", []):
        print(
            "quality="
            + json.dumps(
                {
                    key: quality.get(key)
                    for key in [
                        "model",
                        "chars",
                        "chinese_ratio",
                        "suspicious_ratio",
                        "finance_term_hits",
                        "score",
                        "acceptable",
                    ]
                },
                ensure_ascii=False,
            )
        )


def inspect_pages(doc_id: str, processed_dir: Path, terms: list[str]) -> None:
    matches = list(processed_dir.glob(f"*/{doc_id}/pages.jsonl"))
    if not matches:
        print(f"\n[pages] MISSING */{doc_id}/pages.jsonl")
        return
    path = matches[0]
    rows = [
        json.loads(line)
        for line in path.read_text(encoding="utf-8", errors="replace").splitlines()
        if line.strip()
    ]
    text = "\n".join(row.get("text", "") for row in rows)
    print(f"\n[pages] {path}")
    print(
        " ".join(
            [
                f"pages={len(rows)}",
                f"chars={len(text)}",
                f"has_guangdong={'广东' in text}",
                f"mojibake_marker={'å¹¿' in text}",
                f"replacement={'�' in text}",
            ]
        )
    )
    print("terms=" + ", ".join(f"{term}:{text.count(term)}" for term in terms))
    if rows:
        print("page1=" + repr(rows[0].get("text", "")[:300]))


def inspect_structured(doc_id: str, units_path: Path, terms: list[str]) -> None:
    if not units_path.exists():
        print(f"\n[structured] MISSING {units_path}")
        return
    units = []
    for line in units_path.read_text(encoding="utf-8", errors="replace").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        if row.get("doc_id") == doc_id:
            units.append(row)
    text = "\n".join(row.get("raw_text", "") for row in units)
    types = Counter(row.get("chunk_type") for row in units)
    print(f"\n[structured] {units_path}")
    print(
        " ".join(
            [
                f"units={len(units)}",
                f"types={dict(types)}",
                f"chars={len(text)}",
                f"has_guangdong={'广东' in text}",
                f"mojibake_marker={'å¹¿' in text}",
                f"replacement={'�' in text}",
            ]
        )
    )
    print("terms=" + ", ".join(f"{term}:{text.count(term)}" for term in terms))
    for row in units[:3]:
        print("unit=" + repr((row.get("raw_text") or "")[:260]))


def find_pdf(doc_id: str) -> Path | None:
    candidates = list(Path("Dataset/raw").glob(f"*/{doc_id}.pdf"))
    candidates.extend(Path("Dataset/raw").glob(f"*/{doc_id}.PDF"))
    return candidates[0] if candidates else None


def main() -> None:
    parser = argparse.ArgumentParser(description="Inspect PDF source, pages, and structured-unit text quality.")
    parser.add_argument("--doc-id", default="text01")
    parser.add_argument("--pdf-stem", help="PDF markdown stem if different from doc_id, e.g. 1 for insurance/1.pdf.")
    parser.add_argument("--processed-dir", default="processed_data")
    parser.add_argument("--terms", nargs="*", default=DEFAULT_TERMS)
    args = parser.parse_args()

    processed_dir = Path(args.processed_dir)
    stem = args.pdf_stem or args.doc_id
    for model in ("glm-ocr", "pypdf"):
        inspect_text(f"pdf:{model}", processed_dir / "pdf_parsed" / model / f"{stem}.md", args.terms)
    pdf_path = find_pdf(args.doc_id)
    if pdf_path:
        inspect_selector(args.doc_id, pdf_path, processed_dir)
    inspect_pages(args.doc_id, processed_dir, args.terms)
    inspect_structured(args.doc_id, processed_dir / "structured_units.jsonl", args.terms)


if __name__ == "__main__":
    main()
