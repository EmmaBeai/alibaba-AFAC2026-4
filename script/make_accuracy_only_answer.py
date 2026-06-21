"""Create an accuracy-only answer.csv by preserving answers and minimizing tokens.

This is for local/debug accuracy checks only. Do not use the rewritten token
columns as official token accounting.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path


FIELDS = ["qid", "answer", "prompt_tokens", "completion_tokens", "total_tokens"]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("submission/answer_v1.csv"),
        help="source answer CSV whose answers should be preserved",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("submission/answer.csv"),
        help="destination CSV",
    )
    parser.add_argument("--prompt-tokens", type=int, default=1)
    parser.add_argument("--completion-tokens", type=int, default=0)
    args = parser.parse_args()

    if args.prompt_tokens < 0 or args.completion_tokens < 0:
        raise ValueError("token values must be non-negative")

    rows = read_rows(args.input)
    answer_rows = [row for row in rows if row.get("qid") != "summary"]

    prompt_total = args.prompt_tokens * len(answer_rows)
    completion_total = args.completion_tokens * len(answer_rows)
    rewritten = [
        {
            "qid": "summary",
            "answer": "",
            "prompt_tokens": prompt_total,
            "completion_tokens": completion_total,
            "total_tokens": prompt_total + completion_total,
        }
    ]

    for row in answer_rows:
        rewritten.append(
            {
                "qid": row["qid"],
                "answer": row.get("answer", ""),
                "prompt_tokens": args.prompt_tokens,
                "completion_tokens": args.completion_tokens,
                "total_tokens": args.prompt_tokens + args.completion_tokens,
            }
        )

    write_rows(args.output, rewritten)
    print(
        "[accuracy-only] wrote "
        f"{len(answer_rows)} answers -> {args.output} "
        f"(summary total_tokens={prompt_total + completion_total})"
    )
    print("[accuracy-only] answers preserved; token columns are synthetic debug values")


def read_rows(path: Path) -> list[dict[str, str]]:
    with Path(path).open("r", encoding="utf-8-sig", newline="") as fh:
        reader = csv.DictReader(fh)
        missing = [field for field in FIELDS if field not in (reader.fieldnames or [])]
        if missing:
            raise ValueError(f"{path} missing required columns: {missing}")
        return [dict(row) for row in reader]


def write_rows(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    main()
