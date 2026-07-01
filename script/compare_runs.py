from __future__ import annotations

import argparse
import sys
from pathlib import Path

from agent.run_compare import compare_answer_files, write_comparison_csv


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(description="Compare two runs by qid.")
    parser.add_argument("left", help="Left run directory or answer.csv.")
    parser.add_argument("right", help="Right run directory or answer.csv.")
    parser.add_argument("--out", help="Optional CSV path for changed qids.")
    parser.add_argument("--all", action="store_true", help="Include unchanged qids in --out.")
    args = parser.parse_args()

    left = _answer_path(Path(args.left))
    right = _answer_path(Path(args.right))
    comparison = compare_answer_files(left, right)
    print(f"left={left} rows={len(comparison.same_qids) + len(comparison.changed_qids) - len(comparison.missing_left)}")
    print(f"right={right} rows={len(comparison.same_qids) + len(comparison.changed_qids) - len(comparison.missing_right)}")
    print(f"same_qids={len(comparison.same_qids)}")
    print(f"changed_qids={len(comparison.changed_qids)}")
    if comparison.missing_left:
        print(f"missing_left={len(comparison.missing_left)} preview={comparison.missing_left[:10]}")
    if comparison.missing_right:
        print(f"missing_right={len(comparison.missing_right)} preview={comparison.missing_right[:10]}")
    print("qid,left_answer,right_answer,changed,left_total,right_total,total_delta")
    for row in comparison.rows:
        if not row.changed:
            continue
        print(
            f"{row.qid},{row.left_answer},{row.right_answer},{row.changed},"
            f"{row.left_total_tokens},{row.right_total_tokens},"
            f"{row.right_total_tokens - row.left_total_tokens:+d}"
        )
    if args.out:
        write_comparison_csv(args.out, comparison, include_all=args.all)
        print(f"wrote={args.out}")


def _answer_path(path: Path) -> Path:
    return path / "answer.csv" if path.is_dir() else path


if __name__ == "__main__":
    main()
