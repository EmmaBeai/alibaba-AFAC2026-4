from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any


DEFAULT_INPUT_DIR = Path("processed_data/parsed_files")
DEFAULT_OUTPUT_DIR = Path("processed_data/cleaned_markdown")
DEFAULT_PROMPT_PATH = Path("prompts/clean_markdown_structure.md")
DEFAULT_MODEL = "gpt-5.4-mini"
RETRYABLE_HTTP_STATUS = {408, 409, 429, 500, 502, 503, 504}
HEADING_LINE_RE = re.compile(r"^(#{1,6})\s+(.*)$")
VALID_HEADING_LABELS = {"body", "h1", "h2", "h3", "h4", "h5", "h6"}


class OpenAIResponsesClient:
    def __init__(
        self,
        *,
        model: str,
        api_key_env: str,
        base_url: str,
        timeout_seconds: int,
        max_retries: int,
        retry_sleep_seconds: float,
        max_output_tokens: int,
    ) -> None:
        api_key = os.getenv(api_key_env)
        if not api_key:
            raise RuntimeError(f"Missing API key: set {api_key_env} before cleaning markdown files.")
        self.model = model
        self.api_key = api_key
        self.base_url = base_url.rstrip("/")
        self.timeout_seconds = timeout_seconds
        self.max_retries = max_retries
        self.retry_sleep_seconds = retry_sleep_seconds
        self.max_output_tokens = max_output_tokens

    def classify_headings(self, *, system_prompt: str, candidates: list[dict[str, Any]]) -> dict[int, str]:
        payload: dict[str, Any] = {
            "model": self.model,
            "input": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": _user_message(candidates)},
            ],
        }
        if self.max_output_tokens > 0:
            payload["max_output_tokens"] = self.max_output_tokens

        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        request = urllib.request.Request(
            f"{self.base_url}/responses",
            data=body,
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            },
            method="POST",
        )

        for attempt in range(self.max_retries + 1):
            try:
                with urllib.request.urlopen(request, timeout=self.timeout_seconds) as response:
                    response_payload = json.loads(response.read().decode("utf-8"))
                return parse_heading_label_response(_extract_response_text(response_payload))
            except urllib.error.HTTPError as exc:
                error_body = exc.read().decode("utf-8", errors="replace")
                if exc.code not in RETRYABLE_HTTP_STATUS or attempt >= self.max_retries:
                    raise RuntimeError(f"OpenAI request failed: HTTP {exc.code}: {error_body}") from exc
                _sleep_before_retry(attempt, self.retry_sleep_seconds)
            except urllib.error.URLError as exc:
                if attempt >= self.max_retries:
                    raise RuntimeError(f"OpenAI request failed: {exc.reason}") from exc
                _sleep_before_retry(attempt, self.retry_sleep_seconds)

        raise RuntimeError("OpenAI request failed after retries.")


def clean_directory(
    *,
    input_dir: Path,
    output_dir: Path,
    prompt_path: Path,
    model: str,
    api_key_env: str,
    base_url: str,
    chunk_chars: int,
    max_output_tokens: int,
    timeout_seconds: int,
    max_retries: int,
    retry_sleep_seconds: float,
    request_sleep_seconds: float,
    overwrite: bool,
    dry_run: bool,
    limit: int | None,
    only: set[str] | None,
) -> dict[str, int]:
    if not input_dir.is_dir():
        raise FileNotFoundError(f"Input directory not found: {input_dir}")
    system_prompt = prompt_path.read_text(encoding="utf-8")
    if not dry_run:
        output_dir.mkdir(parents=True, exist_ok=True)

    files = _input_files(input_dir, only=only)
    if limit is not None:
        files = files[:limit]

    stats = {
        "total": 0,
        "copied_txt": 0,
        "cleaned_md": 0,
        "skipped_existing": 0,
        "skipped_suffix": 0,
        "api_chunks": 0,
        "api_batches": 0,
        "heading_candidates": 0,
    }
    client: OpenAIResponsesClient | None = None

    for source_path in files:
        stats["total"] += 1
        target_path = output_dir / source_path.name
        suffix = source_path.suffix.lower()

        if target_path.exists() and not overwrite:
            print(f"[skip] exists {target_path}", flush=True)
            stats["skipped_existing"] += 1
            continue

        if suffix == ".txt":
            print(f"[copy] {source_path} -> {target_path}", flush=True)
            if not dry_run:
                shutil.copy2(source_path, target_path)
            stats["copied_txt"] += 1
            continue

        if suffix != ".md":
            print(f"[skip] unsupported suffix {source_path}", flush=True)
            stats["skipped_suffix"] += 1
            continue

        markdown = source_path.read_text(encoding="utf-8", errors="replace")
        candidates = extract_heading_candidates(markdown, doc_id=source_path.stem)
        batches = split_heading_candidates(candidates, max_chars=chunk_chars)
        stats["api_chunks"] += len(batches)
        stats["api_batches"] += len(batches)
        stats["heading_candidates"] += len(candidates)
        print(
            f"[classify] {source_path} headings={len(candidates)} batches={len(batches)} -> {target_path}",
            flush=True,
        )
        if dry_run:
            stats["cleaned_md"] += 1
            continue

        if client is None and batches:
            client = OpenAIResponsesClient(
                model=model,
                api_key_env=api_key_env,
                base_url=base_url,
                timeout_seconds=timeout_seconds,
                max_retries=max_retries,
                retry_sleep_seconds=retry_sleep_seconds,
                max_output_tokens=max_output_tokens,
            )

        labels: dict[int, str] = {}
        for index, batch in enumerate(batches, start=1):
            assert client is not None
            print(f"[api] {source_path.name} batch {index}/{len(batches)} headings={len(batch)}", flush=True)
            batch_labels = client.classify_headings(system_prompt=system_prompt, candidates=batch)
            _validate_batch_labels(batch_labels, batch)
            overlap = labels.keys() & batch_labels.keys()
            if overlap:
                raise RuntimeError(f"Duplicate line numbers returned for {source_path}: {sorted(overlap)}")
            labels.update(batch_labels)
            if request_sleep_seconds > 0 and index < len(batches):
                time.sleep(request_sleep_seconds)

        validate_document_labels(labels, source_path.stem)
        _write_text_atomic(target_path, apply_heading_labels(markdown, labels))
        stats["cleaned_md"] += 1

    return stats


def extract_heading_candidates(markdown: str, *, doc_id: str | None = None) -> list[dict[str, Any]]:
    lines = markdown.splitlines()
    candidates: list[dict[str, Any]] = []
    for index, line in enumerate(lines):
        match = HEADING_LINE_RE.match(line)
        if not match:
            continue
        marker, text = match.groups()
        candidate = {
            "line_no": index + 1,
            "current_marker": marker,
            "current_label": f"h{len(marker)}",
            "text": text,
            "previous_line": _context_line(lines, index - 1),
            "next_line": _context_line(lines, index + 1),
        }
        if doc_id is not None:
            candidate["doc_id"] = doc_id
        candidates.append(candidate)
    return candidates


def split_heading_candidates(candidates: list[dict[str, Any]], *, max_chars: int) -> list[list[dict[str, Any]]]:
    if not candidates:
        return []
    if max_chars <= 0:
        return [candidates]

    batches: list[list[dict[str, Any]]] = []
    current: list[dict[str, Any]] = []
    current_chars = 0
    for candidate in candidates:
        candidate_chars = len(json.dumps(candidate, ensure_ascii=False)) + 2
        if current and current_chars + candidate_chars > max_chars:
            batches.append(current)
            current = []
            current_chars = 0
        current.append(candidate)
        current_chars += candidate_chars
    if current:
        batches.append(current)
    return batches


def parse_heading_label_response(response_text: str) -> dict[int, str]:
    payload_text = _json_payload_text(response_text)
    payload = json.loads(payload_text)
    if not isinstance(payload, list):
        raise ValueError("Heading classifier response must be a JSON array.")

    labels: dict[int, str] = {}
    for item in payload:
        if not isinstance(item, dict):
            raise ValueError("Each heading classifier item must be a JSON object.")
        try:
            line_no = int(item["line_no"])
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(f"Invalid line_no in heading classifier item: {item}") from exc
        label = item.get("label")
        if label not in VALID_HEADING_LABELS:
            raise ValueError(f"Invalid heading label for line {line_no}: {label}")
        if line_no <= 0:
            raise ValueError(f"Invalid non-positive line number: {line_no}")
        if line_no in labels:
            raise ValueError(f"Duplicate heading label for line {line_no}")
        labels[line_no] = label
    return labels


def apply_heading_labels(markdown: str, labels: dict[int, str]) -> str:
    lines = markdown.splitlines(keepends=True)
    seen: set[int] = set()
    for index, line in enumerate(lines, start=1):
        if index not in labels:
            continue
        body, ending = _split_line_ending(line)
        match = HEADING_LINE_RE.match(body)
        if not match:
            raise ValueError(f"Cannot apply heading label to non-heading line {index}: {body[:120]}")
        _, text = match.groups()
        label = labels[index]
        if label not in VALID_HEADING_LABELS:
            raise ValueError(f"Invalid heading label for line {index}: {label}")
        if label == "body":
            lines[index - 1] = f"{text}{ending}"
        else:
            level = int(label[1:])
            lines[index - 1] = f"{'#' * level} {text}{ending}"
        seen.add(index)

    missing = set(labels) - seen
    if missing:
        raise ValueError(f"Heading labels reference missing lines: {sorted(missing)}")
    return "".join(lines)


def validate_document_labels(labels: dict[int, str], doc_id: str) -> None:
    h1_lines = sorted(line_no for line_no, label in labels.items() if label == "h1")
    if len(h1_lines) != 1:
        raise ValueError(f"Expected exactly one h1 for doc_id={doc_id}, found {len(h1_lines)} at lines {h1_lines}")


def _input_files(input_dir: Path, *, only: set[str] | None) -> list[Path]:
    files = [
        path
        for path in sorted(input_dir.iterdir())
        if path.is_file() and path.suffix.lower() in {".md", ".txt"}
    ]
    if only is None:
        return files
    return [path for path in files if path.stem in only or path.name in only]


def _context_line(lines: list[str], index: int) -> str:
    if index < 0 or index >= len(lines):
        return ""
    return lines[index].strip()[:300]


def _user_message(candidates: list[dict[str, Any]]) -> str:
    return (
        "Classify the following OCR-parsed markdown heading candidate lines according to the system rules. "
        "Return only a JSON array with one object per input line_no.\n\n"
        f"{json.dumps(candidates, ensure_ascii=False, indent=2)}"
    )


def _validate_batch_labels(labels: dict[int, str], batch: list[dict[str, Any]]) -> None:
    expected = {int(candidate["line_no"]) for candidate in batch}
    returned = set(labels)
    missing = sorted(expected - returned)
    unexpected = sorted(returned - expected)
    if missing or unexpected:
        raise RuntimeError(f"Heading classifier returned mismatched lines: missing={missing}, unexpected={unexpected}")


def _json_payload_text(response_text: str) -> str:
    text = response_text.strip()
    if text.startswith("```"):
        lines = text.splitlines()
        if lines and lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].startswith("```"):
            lines = lines[:-1]
        text = "\n".join(lines).strip()
    if text.startswith("["):
        return text
    start = text.find("[")
    end = text.rfind("]")
    if start == -1 or end == -1 or end < start:
        raise ValueError(f"Heading classifier response did not contain a JSON array: {response_text[:500]}")
    return text[start : end + 1]


def _split_line_ending(line: str) -> tuple[str, str]:
    if line.endswith("\r\n"):
        return line[:-2], "\r\n"
    if line.endswith("\n"):
        return line[:-1], "\n"
    if line.endswith("\r"):
        return line[:-1], "\r"
    return line, ""


def _extract_response_text(payload: dict[str, Any]) -> str:
    output_text = payload.get("output_text")
    if isinstance(output_text, str) and output_text.strip():
        return output_text

    pieces: list[str] = []
    for item in payload.get("output") or []:
        if not isinstance(item, dict):
            continue
        for content in item.get("content") or []:
            if not isinstance(content, dict):
                continue
            if content.get("type") in {"output_text", "text"} and isinstance(content.get("text"), str):
                pieces.append(content["text"])
    if pieces:
        return "\n".join(pieces)

    choices = payload.get("choices") or []
    if choices:
        message = choices[0].get("message") or {}
        content = message.get("content")
        if isinstance(content, str):
            return content

    raise RuntimeError(f"OpenAI response did not contain output text: {json.dumps(payload)[:1000]}")


def _sleep_before_retry(attempt: int, base_seconds: float) -> None:
    if base_seconds <= 0:
        return
    time.sleep(base_seconds * (2 ** attempt))


def _write_text_atomic(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_name(f"{path.name}.tmp")
    tmp_path.write_text(text, encoding="utf-8")
    tmp_path.replace(path)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Classify and repair parsed markdown heading levels with OpenAI; copy parsed .txt files unchanged.",
    )
    parser.add_argument("--input-dir", type=Path, default=DEFAULT_INPUT_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--prompt", type=Path, default=DEFAULT_PROMPT_PATH)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--api-key-env", default="OPENAI_API_KEY")
    parser.add_argument("--base-url", default="https://api.openai.com/v1")
    parser.add_argument("--chunk-chars", type=int, default=0, help="Maximum serialized heading-candidate chars per API batch; 0 keeps each document together.")
    parser.add_argument("--max-output-tokens", type=int, default=50000)
    parser.add_argument("--timeout-seconds", type=int, default=300)
    parser.add_argument("--max-retries", type=int, default=3)
    parser.add_argument("--retry-sleep-seconds", type=float, default=2.0)
    parser.add_argument("--request-sleep-seconds", type=float, default=0.0)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--only", nargs="*", default=None, help="Optional doc_id or filename allowlist.")
    args = parser.parse_args()

    stats = clean_directory(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
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
        overwrite=args.overwrite,
        dry_run=args.dry_run,
        limit=args.limit,
        only=set(args.only) if args.only else None,
    )
    print(json.dumps(stats, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
