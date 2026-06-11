# script/pdf_parse_three.py — offline PDF→markdown converter (runs on the GPU box)
from __future__ import annotations

import argparse
import gc
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


MODELS = (
    "paddleocr-vl-1.6",
    "mineru2.5-pro",
)

PADDLE_MODEL = "paddleocr-vl-1.6"
MINERU_MODEL_ID = "opendatalab/MinerU2.5-Pro-2604-1.2B"

_CONTROL_CHARS_RE = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")
CACHE_CLEAR_INTERVAL_PAGES = 32

# Must be set before torch is imported in the child process. Helps reduce CUDA
# allocator fragmentation on long multi-PDF runs; safe if torch is never loaded.
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")


@dataclass(frozen=True)
class ParseResult:
    markdown: str
    meta: dict


def main() -> None:
    args = parse_args()

    if args._child_model:
        run_one_model_child(args)
        return

    run_parent(args)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run offline PDF parsers and save markdown outputs."
    )

    parser.add_argument(
        "inputs",
        nargs="+",
        help="PDF file(s) or directory/directories containing PDFs.",
    )
    parser.add_argument(
        "--out-root",
        type=Path,
        default=Path("processed_data/pdf_parsed"),
    )
    parser.add_argument(
        "--models",
        nargs="+",
        choices=MODELS,
        default=list(MODELS),
    )
    parser.add_argument("--dpi", type=int, default=220)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--keep-page-images", action="store_true")
    parser.add_argument("--mineru-image-analysis", action="store_true")

    # Internal: parent launches one child process per model so GPU memory is freed
    # between models and heavy imports do not happen in the orchestrator process.
    parser.add_argument("--_child-model", choices=MODELS, default=None)

    return parser.parse_args()


def run_parent(args: argparse.Namespace) -> None:
    inputs = [str(x) for x in args.inputs]

    for model_name in args.models:
        python_bin = python_for_model(model_name)

        cmd = [
            python_bin,
            str(Path(__file__).resolve()),
            *inputs,
            "--out-root",
            str(args.out_root),
            "--dpi",
            str(args.dpi),
            "--_child-model",
            model_name,
        ]

        if args.overwrite:
            cmd.append("--overwrite")
        if args.keep_page_images:
            cmd.append("--keep-page-images")
        if args.mineru_image_analysis:
            cmd.append("--mineru-image-analysis")

        print(f"\n=== Running {model_name} ===")
        subprocess.run(cmd, check=True)


def python_for_model(model_name: str) -> str:
    """Allow separate virtualenvs while keeping one parent command.

    Optional env vars:
      PADDLE_PYTHON=/path/to/paddle-env/bin/python
      TORCH_PYTHON=/path/to/torch-env/bin/python
    """
    if model_name == "paddleocr-vl-1.6":
        return os.environ.get("PADDLE_PYTHON", sys.executable)

    return os.environ.get("TORCH_PYTHON", sys.executable)


def run_one_model_child(args: argparse.Namespace) -> None:
    pdfs = discover_pdfs([Path(x) for x in args.inputs])
    if not pdfs:
        raise SystemExit("No PDF files found.")

    model_name = args._child_model
    assert model_name is not None

    runner = build_runner(model_name, args)

    model_out_dir = args.out_root / model_name
    raw_root = args.out_root / "_raw" / model_name
    model_out_dir.mkdir(parents=True, exist_ok=True)
    raw_root.mkdir(parents=True, exist_ok=True)

    try:
        for pdf_path in pdfs:
            doc_id = pdf_path.stem
            md_path = model_out_dir / f"{doc_id}.md"
            meta_path = model_out_dir / f"{doc_id}.meta.json"
            err_path = model_out_dir / f"{doc_id}.error.txt"

            if md_path.exists() and not args.overwrite:
                print(f"[skip] {model_name} {pdf_path} -> {md_path}")
                continue

            started = time.time()
            raw_dir = raw_root / doc_id

            if args.overwrite and raw_dir.exists():
                shutil.rmtree(raw_dir)

            raw_dir.mkdir(parents=True, exist_ok=True)

            try:
                print(f"[parse] {model_name} {pdf_path}")
                result = runner.parse_pdf(pdf_path, raw_dir=raw_dir)
                markdown = normalize_markdown(result.markdown)

                if not markdown:
                    raise RuntimeError("Parser returned empty markdown.")

                md_path.write_text(markdown + "\n", encoding="utf-8")

                meta = {
                    "ok": True,
                    "model": model_name,
                    "doc_id": doc_id,
                    "pdf_path": str(pdf_path),
                    "markdown_path": str(md_path),
                    "seconds": round(time.time() - started, 3),
                    **result.meta,
                }
                meta_path.write_text(
                    json.dumps(meta, ensure_ascii=False, indent=2),
                    encoding="utf-8",
                )

                if err_path.exists():
                    err_path.unlink()

                print(f"[ok] {model_name} {pdf_path} -> {md_path}")

            except Exception as exc:
                err_path.write_text(repr(exc) + "\n", encoding="utf-8")

                meta = {
                    "ok": False,
                    "model": model_name,
                    "doc_id": doc_id,
                    "pdf_path": str(pdf_path),
                    "seconds": round(time.time() - started, 3),
                    "error": repr(exc),
                }
                meta_path.write_text(
                    json.dumps(meta, ensure_ascii=False, indent=2),
                    encoding="utf-8",
                )

                print(f"[error] {model_name} {pdf_path}: {exc}", file=sys.stderr)

            finally:
                clear_runtime_caches()

    finally:
        runner.unload()
        clear_runtime_caches()


def discover_pdfs(inputs: Iterable[Path]) -> list[Path]:
    pdfs: list[Path] = []

    for item in inputs:
        item = item.expanduser()

        if item.is_file():
            if item.suffix.lower() == ".pdf":
                pdfs.append(item)
            continue

        if item.is_dir():
            for path in item.rglob("*"):
                if path.is_file() and path.suffix.lower() == ".pdf":
                    pdfs.append(path)

    return sorted(set(path.resolve() for path in pdfs))


def build_runner(model_name: str, args: argparse.Namespace):
    if model_name == "paddleocr-vl-1.6":
        return PaddleOCRVLRunner(args)

    if model_name == "mineru2.5-pro":
        return MinerU25ProRunner(args)

    raise ValueError(f"Unknown model: {model_name}")


def clear_runtime_caches() -> None:
    """Best-effort Python/GPU cache cleanup after heavy page/doc work."""
    gc.collect()

    torch = sys.modules.get("torch")
    if torch is not None:
        try:
            cuda = getattr(torch, "cuda", None)
            if cuda is not None and cuda.is_available():
                if hasattr(cuda, "synchronize"):
                    cuda.synchronize()
                cuda.empty_cache()
                if hasattr(cuda, "ipc_collect"):
                    cuda.ipc_collect()
        except Exception:
            pass

        try:
            mps = getattr(torch, "mps", None)
            if mps is not None and hasattr(mps, "empty_cache"):
                mps.empty_cache()
        except Exception:
            pass

    paddle = sys.modules.get("paddle")
    if paddle is not None:
        try:
            cuda = getattr(getattr(paddle, "device", None), "cuda", None)
            if cuda is not None and hasattr(cuda, "empty_cache"):
                cuda.empty_cache()
        except Exception:
            pass


def offload_model_to_cpu(model) -> None:
    """Best-effort model offload before dropping references."""
    if model is None or not hasattr(model, "to"):
        return
    try:
        model.to("cpu")
    except Exception:
        pass


class PaddleOCRVLRunner:
    """PaddleOCR-VL-1.6 PDF/image -> Markdown runner."""

    def __init__(self, args: argparse.Namespace):
        self.pipeline = None

    def _ensure_loaded(self) -> None:
        if self.pipeline is not None:
            return
        from paddleocr import PaddleOCRVL

        self.pipeline = PaddleOCRVL(pipeline_version="v1.6")

    def parse_pdf(self, pdf_path: Path, raw_dir: Path) -> ParseResult:
        self._ensure_loaded()
        assert self.pipeline is not None

        md_dir = raw_dir / "markdown"
        json_dir = raw_dir / "json"
        md_dir.mkdir(parents=True, exist_ok=True)
        json_dir.mkdir(parents=True, exist_ok=True)

        output = self.pipeline.predict(str(pdf_path))

        result_count = 0
        for res in output:
            result_count += 1
            res.save_to_json(save_path=str(json_dir))
            res.save_to_markdown(save_path=str(md_dir))
            if result_count % CACHE_CLEAR_INTERVAL_PAGES == 0:
                clear_runtime_caches()
        del output
        clear_runtime_caches()

        md_files = sorted(md_dir.rglob("*.md"))
        if not md_files:
            raise RuntimeError(
                f"PaddleOCR-VL produced no markdown files under {md_dir}"
            )

        markdown = combine_markdown_files(md_files)

        return ParseResult(
            markdown=markdown,
            meta={
                "backend": "paddleocr",
                "result_count": result_count,
                "raw_dir": str(raw_dir),
            },
        )

    def unload(self) -> None:
        self.pipeline = None
        clear_runtime_caches()


class MinerU25ProRunner:
    """MinerU2.5-Pro page-image -> JSON -> Markdown runner."""

    def __init__(self, args: argparse.Namespace):
        self.args = args
        self.json2md = None
        self.model = None
        self.processor = None
        self.client = None

    def _ensure_loaded(self) -> None:
        if self.client is not None:
            return
        from transformers import AutoProcessor, Qwen2VLForConditionalGeneration
        from mineru_vl_utils import MinerUClient
        from mineru_vl_utils.post_process import json2md

        self.json2md = json2md

        self.model = Qwen2VLForConditionalGeneration.from_pretrained(
            MINERU_MODEL_ID,
            dtype="auto",
            device_map="auto",
        )
        self.processor = AutoProcessor.from_pretrained(
            MINERU_MODEL_ID,
            use_fast=True,
        )
        self.client = MinerUClient(
            backend="transformers",
            model=self.model,
            processor=self.processor,
            image_analysis=self.args.mineru_image_analysis,
        )

    def parse_pdf(self, pdf_path: Path, raw_dir: Path) -> ParseResult:
        from PIL import Image

        page_paths = render_pdf_pages(
            pdf_path=pdf_path,
            cache_root=self.args.out_root / "_page_cache",
            dpi=self.args.dpi,
            overwrite=self.args.overwrite,
        )

        page_markdowns: list[str] = []
        page_json_dir = raw_dir / "pages_json"
        page_json_dir.mkdir(parents=True, exist_ok=True)

        self._ensure_loaded()
        assert self.client is not None
        assert self.json2md is not None

        try:
            for idx, page_path in enumerate(page_paths, start=1):
                print(f"  [mineru2.5-pro] page {idx}/{len(page_paths)}")

                with Image.open(page_path) as image:
                    content_list = self.client.two_step_extract(image.convert("RGB"))

                (page_json_dir / f"page_{idx:04d}.json").write_text(
                    json.dumps(content_list, ensure_ascii=False, indent=2),
                    encoding="utf-8",
                )

                page_md = self.json2md(content_list)
                page_markdowns.append(page_md)
                if idx % CACHE_CLEAR_INTERVAL_PAGES == 0:
                    clear_runtime_caches()

        finally:
            if not self.args.keep_page_images:
                cleanup_page_cache(pdf_path, self.args.out_root / "_page_cache")
            clear_runtime_caches()

        return ParseResult(
            markdown=join_page_markdowns(page_markdowns),
            meta={
                "backend": "mineru-vl-utils-transformers",
                "model_id": MINERU_MODEL_ID,
                "pages": len(page_paths),
                "dpi": self.args.dpi,
                "image_analysis": bool(self.args.mineru_image_analysis),
                "raw_dir": str(raw_dir),
            },
        )

    def unload(self) -> None:
        offload_model_to_cpu(self.model)
        self.client = None
        self.processor = None
        self.model = None
        self.json2md = None
        clear_runtime_caches()


def render_pdf_pages(
    pdf_path: Path,
    cache_root: Path,
    dpi: int,
    overwrite: bool,
) -> list[Path]:
    """Render PDF pages to PNG for page-image VLM parsers."""
    import fitz  # PyMuPDF

    doc_id = pdf_path.stem
    out_dir = cache_root / doc_id
    out_dir.mkdir(parents=True, exist_ok=True)

    doc = fitz.open(pdf_path)
    zoom = dpi / 72.0
    matrix = fitz.Matrix(zoom, zoom)

    page_paths: list[Path] = []

    try:
        for page_index in range(len(doc)):
            page_no = page_index + 1
            page_path = out_dir / f"page_{page_no:04d}.png"
            page_paths.append(page_path)

            if page_path.exists() and not overwrite:
                continue

            page = doc.load_page(page_index)
            pix = page.get_pixmap(matrix=matrix, alpha=False)
            pix.save(str(page_path))
            del pix
            del page
    finally:
        doc.close()

    return page_paths


def cleanup_page_cache(pdf_path: Path, cache_root: Path) -> None:
    cache_dir = cache_root / pdf_path.stem
    if cache_dir.exists():
        shutil.rmtree(cache_dir)


def combine_markdown_files(md_files: list[Path]) -> str:
    chunks: list[str] = []

    for md_file in md_files:
        text = md_file.read_text(encoding="utf-8").strip()
        if text:
            chunks.append(text)

    return "\n\n".join(chunks)


def join_page_markdowns(page_markdowns: list[str]) -> str:
    chunks: list[str] = []

    for idx, text in enumerate(page_markdowns, start=1):
        text = normalize_markdown(text)
        if not text:
            continue

        chunks.append(f"<!-- page:{idx} -->\n\n{text}")

    return "\n\n".join(chunks)


def normalize_markdown(text: str) -> str:
    if not text:
        return ""

    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = _CONTROL_CHARS_RE.sub("", text)
    text = strip_markdown_fence(text)
    text = re.sub(r"\n{4,}", "\n\n\n", text)

    return text.strip()


def strip_markdown_fence(text: str) -> str:
    text = text.strip()

    if text.startswith("```markdown") and text.endswith("```"):
        return text[len("```markdown") : -3].strip()

    if text.startswith("```md") and text.endswith("```"):
        return text[len("```md") : -3].strip()

    if text.startswith("```") and text.endswith("```"):
        return text[3:-3].strip()

    return text


if __name__ == "__main__":
    main()
