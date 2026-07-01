from __future__ import annotations

import argparse
import hashlib
import json
import os
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np


RUN_INSTRUCTIONS = """
Two-file handoff layout:
  build_qwen3_embedding8b_chunks.py
  clean_chunks.jsonl

Install runtime dependencies once if needed:
  pip install -U "transformers>=4.51.0" "sentence-transformers>=2.7.0" torch numpy
  pip install -U faiss-cpu  # optional; skip if you only need embeddings.npy

No separate model download command is required. This script defaults to
HF_ENDPOINT=https://hf-mirror.com, so the first run downloads the model automatically
through the mirror into the normal Hugging Face cache.

Smoke test:
  python build_qwen3_embedding8b_chunks.py --limit 20 --batch-size 2 --text-mode search --overwrite

Full run:
  python build_qwen3_embedding8b_chunks.py --batch-size 2 --text-mode search --overwrite
""".strip()

DEFAULT_MODEL_ID = "Qwen/Qwen3-Embedding-8B"
DEFAULT_UNITS_PATH = "clean_chunks.jsonl"
DEFAULT_OUT_DIR = "."
DEFAULT_HF_ENDPOINT = "https://hf-mirror.com"
ARTIFACT_FILENAMES = {
    "embeddings.npy",
    "metadata.jsonl",
    "manifest.json",
    "index_flat_ip.faiss",
}


def load_units(path: Path, *, limit: int = 0) -> list[dict[str, Any]]:
    units: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            units.append(json.loads(line))
            if limit and len(units) >= limit:
                break
    return units


def build_unit_text(unit: dict[str, Any], *, mode: str = "full", max_chars: int = 0) -> str:
    if mode == "search":
        text = str(unit.get("search_text") or unit.get("raw_text") or unit.get("text") or "")
    elif mode == "raw":
        text = str(unit.get("raw_text") or unit.get("text") or unit.get("search_text") or "")
    elif mode == "full":
        text = "\n".join(
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
    else:
        raise ValueError(f"Unsupported text mode: {mode}")
    if max_chars > 0:
        return text[:max_chars]
    return text


def normalize_matrix(matrix: np.ndarray) -> np.ndarray:
    values = np.asarray(matrix, dtype=np.float32)
    norms = np.linalg.norm(values, axis=1, keepdims=True)
    norms[norms <= 0] = 1.0
    return values / norms


def text_hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def unit_id(unit: dict[str, Any]) -> str:
    return str(unit.get("unit_id") or unit.get("chunk_id") or "")


def configure_hf_endpoint(endpoint: str) -> str:
    if endpoint:
        os.environ["HF_ENDPOINT"] = endpoint
    return os.environ.get("HF_ENDPOINT", "")


def encode_texts(
    texts: list[str],
    *,
    model_id: str,
    batch_size: int,
    max_seq_length: int,
    device: str,
    dtype: str,
    attn_implementation: str,
    device_map: str,
) -> np.ndarray:
    try:
        import torch
        from sentence_transformers import SentenceTransformer
    except ImportError as exc:  # pragma: no cover - exercised on teammate GPU boxes.
        raise RuntimeError(
            "Missing local embedding dependencies. Install at least: "
            "pip install 'transformers>=4.51.0' 'sentence-transformers>=2.7.0' torch numpy"
        ) from exc

    dtype_map = {
        "auto": None,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }
    if dtype not in dtype_map:
        raise ValueError(f"Unsupported dtype: {dtype}")

    model_kwargs: dict[str, Any] = {}
    if dtype_map[dtype] is not None:
        model_kwargs["torch_dtype"] = dtype_map[dtype]
    if attn_implementation:
        model_kwargs["attn_implementation"] = attn_implementation
    if device_map:
        model_kwargs["device_map"] = device_map

    st_kwargs: dict[str, Any] = {
        "tokenizer_kwargs": {"padding_side": "left"},
    }
    if model_kwargs:
        st_kwargs["model_kwargs"] = model_kwargs
    if device and not device_map:
        st_kwargs["device"] = device

    model = SentenceTransformer(model_id, **st_kwargs)
    if max_seq_length > 0:
        model.max_seq_length = max_seq_length

    embeddings = model.encode(
        texts,
        batch_size=batch_size,
        show_progress_bar=True,
        normalize_embeddings=True,
        convert_to_numpy=True,
    )
    return normalize_matrix(np.asarray(embeddings, dtype=np.float32))


def write_embedding_artifacts(
    *,
    units: list[dict[str, Any]],
    texts: list[str],
    embeddings: np.ndarray,
    out_dir: Path,
    model_id: str,
    text_mode: str,
    max_seq_length: int,
    batch_size: int,
    dtype: str,
    device: str,
    faiss_written: bool,
) -> dict[str, Any]:
    out_dir.mkdir(parents=True, exist_ok=True)
    embeddings = normalize_matrix(embeddings)
    np.save(out_dir / "embeddings.npy", embeddings)

    metadata_path = out_dir / "metadata.jsonl"
    with metadata_path.open("w", encoding="utf-8") as handle:
        for row, (unit, text) in enumerate(zip(units, texts)):
            metadata = {
                "row": row,
                "unit_id": unit_id(unit),
                "text_sha256": text_hash(text),
                "model_id": model_id,
                "dimension": int(embeddings.shape[1]),
                "text_chars": len(text),
                "doc_id": unit.get("doc_id", ""),
                "domain": unit.get("domain", ""),
                "title": unit.get("title", ""),
                "page": unit.get("page", ""),
                "section_path": unit.get("section_path", ""),
                "chunk_type": unit.get("chunk_type", ""),
            }
            handle.write(json.dumps(metadata, ensure_ascii=False) + "\n")

    manifest = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "model_id": model_id,
        "embedding_count": int(embeddings.shape[0]),
        "dimension": int(embeddings.shape[1]),
        "dtype": "float32",
        "source_dtype": dtype,
        "device": device,
        "text_mode": text_mode,
        "max_seq_length": max_seq_length,
        "batch_size": batch_size,
        "normalized": True,
        "similarity": "cosine_via_inner_product",
        "faiss_index": "index_flat_ip.faiss" if faiss_written else "",
        "files": {
            "embeddings": "embeddings.npy",
            "metadata": "metadata.jsonl",
            "manifest": "manifest.json",
        },
    }
    (out_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return manifest


def try_write_faiss_index(embeddings: np.ndarray, out_dir: Path, *, require: bool) -> bool:
    try:
        import faiss  # type: ignore[import-not-found]
    except ImportError:
        if require:
            raise RuntimeError(
                "FAISS is not installed. Install faiss-cpu/faiss-gpu, or rerun without --require-faiss."
            )
        print("[qwen3-embed] FAISS not installed; kept embeddings.npy + metadata.jsonl only.", flush=True)
        return False

    values = normalize_matrix(embeddings).astype("float32", copy=False)
    index = faiss.IndexFlatIP(values.shape[1])
    index.add(values)
    faiss.write_index(index, str(out_dir / "index_flat_ip.faiss"))
    return True


def resolve_artifact_path(path: str, *, script_path: Path) -> Path:
    value = Path(path)
    if value.is_absolute():
        return value
    return script_path.resolve().parent / value


def prepare_out_dir(out_dir: Path, *, overwrite: bool) -> None:
    if out_dir.exists() and not out_dir.is_dir():
        raise RuntimeError(f"Output path is not a directory: {out_dir}")
    out_dir.mkdir(parents=True, exist_ok=True)
    existing_artifacts = [out_dir / name for name in ARTIFACT_FILENAMES if (out_dir / name).exists()]
    if existing_artifacts and not overwrite:
        names = ", ".join(path.name for path in sorted(existing_artifacts))
        raise RuntimeError(f"Output artifacts already exist in {out_dir}: {names}. Pass --overwrite to replace them.")
    if overwrite:
        for path in existing_artifacts:
            path.unlink()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build local Qwen3-Embedding-8B chunk embeddings from a sibling clean_chunks.jsonl "
            "into flat npy/metadata/manifest files and an optional FAISS index."
        ),
        epilog=RUN_INSTRUCTIONS,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--units",
        default=DEFAULT_UNITS_PATH,
        help="Input clean chunk JSONL. Relative paths are resolved next to this script.",
    )
    parser.add_argument(
        "--out-dir",
        default=DEFAULT_OUT_DIR,
        help="Output directory for flat embedding artifacts. Relative paths are resolved next to this script.",
    )
    parser.add_argument("--model", default=DEFAULT_MODEL_ID, help="Local/Hugging Face model id or path.")
    parser.add_argument(
        "--hf-endpoint",
        default=DEFAULT_HF_ENDPOINT,
        help='Hugging Face endpoint for automatic model download. Default is China mirror. Pass "" to leave unchanged.',
    )
    parser.add_argument("--batch-size", type=int, default=2, help="SentenceTransformer encode batch size.")
    parser.add_argument("--max-seq-length", type=int, default=4096, help="Tokenizer max sequence length.")
    parser.add_argument("--device", default="cuda", help="SentenceTransformer device, e.g. cuda, cuda:0, cpu.")
    parser.add_argument("--device-map", default="", help='Optional Transformers device_map, e.g. "auto".')
    parser.add_argument(
        "--dtype",
        default="float16",
        choices=["auto", "float16", "bfloat16", "float32"],
        help="Model load dtype.",
    )
    parser.add_argument(
        "--attn-implementation",
        default="",
        help='Optional attention implementation, e.g. "flash_attention_2" if installed.',
    )
    parser.add_argument(
        "--text-mode",
        default="full",
        choices=["full", "search", "raw"],
        help="Text assembled for each chunk. full matches the current dense cache script.",
    )
    parser.add_argument("--max-chars", type=int, default=0, help="Optional pre-tokenization character cap.")
    parser.add_argument("--limit", type=int, default=0, help="Optional smoke-test limit.")
    parser.add_argument("--overwrite", action="store_true", help="Replace existing output artifact files.")
    parser.add_argument("--no-faiss", action="store_true", help="Skip writing index_flat_ip.faiss.")
    parser.add_argument("--require-faiss", action="store_true", help="Fail if FAISS cannot be written.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    script_path = Path(__file__)
    units_path = resolve_artifact_path(args.units, script_path=script_path)
    out_dir = resolve_artifact_path(args.out_dir, script_path=script_path)
    prepare_out_dir(out_dir, overwrite=args.overwrite)
    hf_endpoint = configure_hf_endpoint(args.hf_endpoint)

    units = load_units(units_path, limit=args.limit)
    texts = [build_unit_text(unit, mode=args.text_mode, max_chars=args.max_chars) for unit in units]
    if not units:
        raise RuntimeError(f"No units loaded from {units_path}")

    print(
        json.dumps(
            {
                "event": "start",
                "units": len(units),
                "model": args.model,
                "batch_size": args.batch_size,
                "max_seq_length": args.max_seq_length,
                "text_mode": args.text_mode,
                "out_dir": str(out_dir),
                "hf_endpoint": hf_endpoint,
            },
            ensure_ascii=False,
        ),
        flush=True,
    )
    started = time.time()
    embeddings = encode_texts(
        texts,
        model_id=args.model,
        batch_size=args.batch_size,
        max_seq_length=args.max_seq_length,
        device=args.device,
        dtype=args.dtype,
        attn_implementation=args.attn_implementation,
        device_map=args.device_map,
    )
    if embeddings.shape[0] != len(units):
        raise RuntimeError(f"Embedding row count mismatch: expected {len(units)}, got {embeddings.shape[0]}")

    faiss_written = False
    if not args.no_faiss:
        faiss_written = try_write_faiss_index(embeddings, out_dir, require=args.require_faiss)

    manifest = write_embedding_artifacts(
        units=units,
        texts=texts,
        embeddings=embeddings,
        out_dir=out_dir,
        model_id=args.model,
        text_mode=args.text_mode,
        max_seq_length=args.max_seq_length,
        batch_size=args.batch_size,
        dtype=args.dtype,
        device=args.device,
        faiss_written=faiss_written,
    )
    manifest["elapsed_seconds"] = round(time.time() - started, 3)
    (out_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({"event": "done", **manifest}, ensure_ascii=False, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
