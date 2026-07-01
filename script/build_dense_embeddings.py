from __future__ import annotations

import argparse
import hashlib
import json
import math
import time
from pathlib import Path
from typing import Any

import yaml

from agent.embedding_client import DEFAULT_EMBEDDING_MODEL, DashScopeEmbeddingClient


def build_dense_embeddings(
    *,
    units_path: Path | str,
    cache_path: Path | str,
    embedding_client: Any,
) -> dict[str, int]:
    units_path = Path(units_path)
    cache_path = Path(cache_path)
    units = _load_units(units_path)
    cache = _read_cache(
        cache_path,
        model=getattr(embedding_client, "model", ""),
        dimension=int(getattr(embedding_client, "dimension", 0) or 0),
    )

    missing: list[dict[str, Any]] = []
    for unit in units:
        unit_id = _unit_id(unit)
        text_hash = _text_hash(_unit_text(unit))
        if unit_id not in cache or cache[unit_id].get("text_sha256") != text_hash:
            missing.append(unit)

    batch_size = int(getattr(embedding_client, "batch_size", 10) or 10)
    total_missing = len(missing)
    for start in range(0, total_missing, batch_size):
        batch = missing[start:start + batch_size]
        print(f"[embedding-index] units {start + 1}-{start + len(batch)}/{total_missing}", flush=True)
        new_rows, embed_seconds = _embed_units_with_fallback(batch, embedding_client)
        for row in new_rows:
            cache[str(row["unit_id"])] = row
        write_started = time.perf_counter()
        _append_cache_rows(cache_path, new_rows)
        write_seconds = time.perf_counter() - write_started
        print(
            f"[embedding-index] batch_done embed_seconds={embed_seconds:.2f} "
            f"write_seconds={write_seconds:.2f} cache_rows={len(cache)}",
            flush=True,
        )

    return {
        "total_units": len(units),
        "cached_units": len(units) - len(missing),
        "built_units": len(missing),
        "cache_rows": len(cache),
    }


def _embed_units_with_fallback(
    units: list[dict[str, Any]],
    embedding_client: Any,
) -> tuple[list[dict[str, Any]], float]:
    started = time.perf_counter()
    try:
        embeddings = embedding_client.embed_texts(
            [_unit_text(unit) for unit in units],
            input_type="document",
        )
    except RuntimeError as exc:
        failed_seconds = time.perf_counter() - started
        print(
            f"[embedding-index] batch_failed size={len(units)} "
            f"seconds={failed_seconds:.2f} error={exc}",
            flush=True,
        )
        if len(units) <= 1:
            raise
        midpoint = len(units) // 2
        left_rows, left_seconds = _embed_units_with_fallback(units[:midpoint], embedding_client)
        right_rows, right_seconds = _embed_units_with_fallback(units[midpoint:], embedding_client)
        return left_rows + right_rows, failed_seconds + left_seconds + right_seconds
    rows = []
    for unit, embedding in zip(units, embeddings):
        rows.append(
            {
                "unit_id": _unit_id(unit),
                "text_sha256": _text_hash(_unit_text(unit)),
                "model": getattr(embedding_client, "model", ""),
                "dimension": len(embedding),
                "embedding": _normalize_vector(embedding),
            }
        )
    return rows, time.perf_counter() - started


def _load_units(path: Path) -> list[dict[str, Any]]:
    units: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                units.append(json.loads(line))
    return units


def _unit_text(unit: dict[str, Any]) -> str:
    return "\n".join(
        str(part)
        for part in [
            unit.get("title", ""),
            unit.get("section_path", ""),
            unit.get("chunk_type", ""),
            unit.get("search_text", ""),
            unit.get("raw_text", ""),
            unit.get("text", ""),
        ]
        if part
    )


def _unit_id(unit: dict[str, Any]) -> str:
    return str(unit.get("unit_id") or unit.get("chunk_id") or "")


def _text_hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _normalize_vector(vector: list[float]) -> list[float]:
    norm = math.sqrt(sum(float(value) * float(value) for value in vector))
    if norm <= 0:
        return [0.0 for _ in vector]
    return [float(value) / norm for value in vector]


def _read_cache(path: Path, *, model: str, dimension: int) -> dict[str, dict[str, Any]]:
    if not path.exists():
        return {}
    rows: dict[str, dict[str, Any]] = {}
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            embedding = row.get("embedding")
            if row.get("model") != model or not isinstance(embedding, list):
                continue
            if dimension and len(embedding) != dimension:
                continue
            rows[str(row.get("unit_id", ""))] = row
    return rows


def _append_cache_rows(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def _config_paths(config_path: Path) -> tuple[Path, Path, dict[str, Any]]:
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    root = config_path.resolve().parent.parent
    units_path = root / config.get("structured_retrieval", {}).get("units_path", "processed_data/clean_chunks.jsonl")
    embedding_config = config.get("embedding", {})
    cache_path = root / embedding_config.get("cache_path", "processed_data/embeddings/text_embedding_v4_dense1024.jsonl")
    return units_path, cache_path, embedding_config


def main() -> None:
    parser = argparse.ArgumentParser(description="Build dense document embeddings for clean chunk units.")
    parser.add_argument("--config", default="")
    parser.add_argument("--units", default="")
    parser.add_argument("--cache", default="")
    parser.add_argument("--model", default=DEFAULT_EMBEDDING_MODEL)
    parser.add_argument("--dimension", type=int, default=1024)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--timeout-seconds", type=int, default=120)
    args = parser.parse_args()

    embedding_config: dict[str, Any] = {}
    if args.config:
        units_path, cache_path, embedding_config = _config_paths(Path(args.config))
    else:
        units_path = Path(args.units or "processed_data/clean_chunks.jsonl")
        cache_path = Path(args.cache or "processed_data/embeddings/text_embedding_v4_clean_dense1024.jsonl")

    client = DashScopeEmbeddingClient(
        model=embedding_config.get("model", args.model),
        dimension=embedding_config.get("dimension", args.dimension),
        batch_size=embedding_config.get("batch_size", args.batch_size),
        api_key_env=embedding_config.get("api_key_env", "DASHSCOPE_API_KEY"),
        api_key_env_fallbacks=embedding_config.get("api_key_env_fallbacks"),
        base_url_env=embedding_config.get("base_url_env", "DASHSCOPE_EMBEDDING_BASE_URL"),
        default_base_url=embedding_config.get(
            "default_base_url",
            "https://dashscope.aliyuncs.com/compatible-mode/v1",
        ),
        timeout_seconds=embedding_config.get("timeout_seconds", args.timeout_seconds),
        max_retries=embedding_config.get("max_retries", 3),
    )
    stats = build_dense_embeddings(
        units_path=units_path,
        cache_path=cache_path,
        embedding_client=client,
    )
    print(json.dumps(stats, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
