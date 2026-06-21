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
    "glm-ocr",
    "pypdf",
    "paddleocr-vl-1.6",
    "mineru2.5-pro",
)

GLM_OCR_MODEL = "glm-ocr"
PYPDF_MODEL = "pypdf"
PADDLE_MODEL = "paddleocr-vl-1.6"
MINERU_MODEL_ID = "opendatalab/MinerU2.5-Pro-2604-1.2B"

_CONTROL_CHARS_RE = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")
CACHE_CLEAR_INTERVAL_PAGES = 32
DEFAULT_PDF_TIMEOUT_SECONDS = 30 * 60

# Must be set before torch is imported in the child process. Helps reduce CUDA
# allocator fragmentation on long multi-PDF runs; safe if torch is never loaded.
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")


@dataclass(frozen=True)
class ParseResult:
    markdown: str
    meta: dict


@dataclass(frozen=True)
class ExistingGlmOcrOutput:
    markdown: str
    source: str
    md_count: int
    json_count: int


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
    parser.add_argument(
        "--glmocr-config",
        type=Path,
        default=None,
        help="Optional GLM-OCR SDK config.yaml for MaaS or self-hosted mode.",
    )
    parser.add_argument(
        "--glmocr-layout-device",
        default=None,
        help="Optional layout device passed to `glmocr parse`, e.g. cpu or cuda:1.",
    )
    parser.add_argument(
        "--glmocr-env-file",
        type=Path,
        default=None,
        help="Optional .env file loaded by `glmocr parse`, usually for ZHIPU_API_KEY.",
    )
    parser.add_argument(
        "--glmocr-mode",
        choices=("maas", "selfhosted"),
        default=None,
        help="Optional GLM-OCR mode passed to `glmocr parse`.",
    )
    parser.add_argument(
        "--glmocr-set",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help=(
            "Repeatable dotted config override passed through to `glmocr parse --set`. "
            "Use KEY=VALUE; the wrapper expands it to `--set KEY VALUE`."
        ),
    )
    parser.add_argument(
        "--pdf-timeout-seconds",
        type=int,
        default=DEFAULT_PDF_TIMEOUT_SECONDS,
        help=(
            "Max seconds for one PDF on heavy parsers. "
            "Set 0 to disable per-PDF subprocess isolation/timeouts."
        ),
    )

    # Internal: parent launches one child process per model so GPU memory is freed
    # between models and heavy imports do not happen in the orchestrator process.
    parser.add_argument("--_child-model", choices=MODELS, default=None)
    parser.add_argument("--_child-pdf", type=Path, default=None)

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
            "--pdf-timeout-seconds",
            str(args.pdf_timeout_seconds),
            "--_child-model",
            model_name,
        ]

        if args.overwrite:
            cmd.append("--overwrite")
        if args.keep_page_images:
            cmd.append("--keep-page-images")
        if args.mineru_image_analysis:
            cmd.append("--mineru-image-analysis")
        if args.glmocr_config is not None:
            cmd.extend(["--glmocr-config", str(args.glmocr_config)])
        if args.glmocr_layout_device:
            cmd.extend(["--glmocr-layout-device", args.glmocr_layout_device])
        if args.glmocr_env_file is not None:
            cmd.extend(["--glmocr-env-file", str(args.glmocr_env_file)])
        if args.glmocr_mode:
            cmd.extend(["--glmocr-mode", args.glmocr_mode])
        for setting in args.glmocr_set:
            cmd.extend(["--glmocr-set", setting])

        print(f"\n=== Running {model_name} ===", flush=True)
        subprocess.run(cmd, check=True)


def python_for_model(model_name: str) -> str:
    """Allow separate virtualenvs while keeping one parent command.

    Optional env vars:
      GLMOCR_PYTHON=/path/to/glmocr-env/bin/python
      PADDLE_PYTHON=/path/to/paddle-env/bin/python
      TORCH_PYTHON=/path/to/torch-env/bin/python
    """
    if model_name == GLM_OCR_MODEL:
        return os.environ.get("GLMOCR_PYTHON", sys.executable)

    if model_name == "paddleocr-vl-1.6":
        return os.environ.get("PADDLE_PYTHON", sys.executable)

    if model_name == "pypdf":
        return sys.executable

    return os.environ.get("TORCH_PYTHON", sys.executable)


def run_one_model_child(args: argparse.Namespace) -> None:
    if args._child_pdf is not None:
        run_single_pdf_child(args)
        return

    pdfs = discover_pdfs([Path(x) for x in args.inputs])
    if not pdfs:
        raise SystemExit("No PDF files found.")

    model_name = args._child_model
    assert model_name is not None

    model_out_dir = args.out_root / model_name
    raw_root = args.out_root / "_raw" / model_name
    model_out_dir.mkdir(parents=True, exist_ok=True)
    raw_root.mkdir(parents=True, exist_ok=True)

    if should_isolate_pdf_processes(model_name, args):
        run_model_child_with_pdf_timeouts(args, model_name, pdfs, model_out_dir)
        return

    runner = build_runner(model_name, args)

    try:
        for pdf_path in pdfs:
            parse_pdf_to_outputs(
                args,
                model_name,
                pdf_path,
                runner,
                model_out_dir,
                raw_root,
            )

    finally:
        runner.unload()
        clear_runtime_caches()


def run_single_pdf_child(args: argparse.Namespace) -> None:
    model_name = args._child_model
    assert model_name is not None

    pdf_path = args._child_pdf
    assert pdf_path is not None
    pdf_path = pdf_path.expanduser().resolve()
    if not pdf_path.is_file() or pdf_path.suffix.lower() != ".pdf":
        raise SystemExit(f"not a PDF file: {pdf_path}")

    model_out_dir = args.out_root / model_name
    raw_root = args.out_root / "_raw" / model_name
    model_out_dir.mkdir(parents=True, exist_ok=True)
    raw_root.mkdir(parents=True, exist_ok=True)

    runner = build_runner(model_name, args)
    try:
        parse_pdf_to_outputs(
            args,
            model_name,
            pdf_path,
            runner,
            model_out_dir,
            raw_root,
        )
    finally:
        runner.unload()
        clear_runtime_caches()


def should_isolate_pdf_processes(model_name: str, args: argparse.Namespace) -> bool:
    return model_name != PYPDF_MODEL and args.pdf_timeout_seconds > 0


def run_model_child_with_pdf_timeouts(
    args: argparse.Namespace,
    model_name: str,
    pdfs: list[Path],
    model_out_dir: Path,
) -> None:
    python_bin = python_for_model(model_name)

    for pdf_path in pdfs:
        doc_id = pdf_path.stem
        md_path = model_out_dir / f"{doc_id}.md"

        if md_path.exists() and not args.overwrite:
            print(f"[skip] {model_name} {pdf_path} -> {md_path}", flush=True)
            continue

        cmd = [
            python_bin,
            str(Path(__file__).resolve()),
            str(pdf_path),
            "--out-root",
            str(args.out_root),
            "--dpi",
            str(args.dpi),
            "--pdf-timeout-seconds",
            "0",
            "--_child-model",
            model_name,
            "--_child-pdf",
            str(pdf_path),
        ]

        if args.overwrite:
            cmd.append("--overwrite")
        if args.keep_page_images:
            cmd.append("--keep-page-images")
        if args.mineru_image_analysis:
            cmd.append("--mineru-image-analysis")
        if args.glmocr_config is not None:
            cmd.extend(["--glmocr-config", str(args.glmocr_config)])
        if args.glmocr_layout_device:
            cmd.extend(["--glmocr-layout-device", args.glmocr_layout_device])
        if args.glmocr_env_file is not None:
            cmd.extend(["--glmocr-env-file", str(args.glmocr_env_file)])
        if args.glmocr_mode:
            cmd.extend(["--glmocr-mode", args.glmocr_mode])
        for setting in args.glmocr_set:
            cmd.extend(["--glmocr-set", setting])

        started = time.time()
        try:
            subprocess.run(cmd, check=True, timeout=args.pdf_timeout_seconds)
        except subprocess.TimeoutExpired as exc:
            error = (
                f"TimeoutExpired({args.pdf_timeout_seconds}s) while parsing "
                f"{pdf_path}"
            )
            write_failure_outputs(
                model_name=model_name,
                pdf_path=pdf_path,
                model_out_dir=model_out_dir,
                seconds=round(time.time() - started, 3),
                error=error,
            )
            print(f"[timeout] {model_name} {pdf_path}: {error}", file=sys.stderr)
        except subprocess.CalledProcessError as exc:
            error = f"CalledProcessError(returncode={exc.returncode}, cmd={exc.cmd!r})"
            write_failure_outputs(
                model_name=model_name,
                pdf_path=pdf_path,
                model_out_dir=model_out_dir,
                seconds=round(time.time() - started, 3),
                error=error,
            )
            print(f"[error] {model_name} {pdf_path}: {error}", file=sys.stderr)
        finally:
            clear_runtime_caches()


def parse_pdf_to_outputs(
    args: argparse.Namespace,
    model_name: str,
    pdf_path: Path,
    runner,
    model_out_dir: Path,
    raw_root: Path,
) -> bool:
    doc_id = pdf_path.stem
    md_path = model_out_dir / f"{doc_id}.md"
    meta_path = model_out_dir / f"{doc_id}.meta.json"
    err_path = model_out_dir / f"{doc_id}.error.txt"

    if md_path.exists() and not args.overwrite:
        print(f"[skip] {model_name} {pdf_path} -> {md_path}", flush=True)
        return True

    started = time.time()
    raw_dir = raw_root / doc_id

    if args.overwrite and raw_dir.exists():
        shutil.rmtree(raw_dir)

    raw_dir.mkdir(parents=True, exist_ok=True)

    try:
        print(f"[parse] {model_name} {pdf_path}", flush=True)
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

        print(f"[ok] {model_name} {pdf_path} -> {md_path}", flush=True)
        return True

    except Exception as exc:
        write_failure_outputs(
            model_name=model_name,
            pdf_path=pdf_path,
            model_out_dir=model_out_dir,
            seconds=round(time.time() - started, 3),
            error=repr(exc),
        )
        print(f"[error] {model_name} {pdf_path}: {exc}", file=sys.stderr)
        return False

    finally:
        clear_runtime_caches()


def write_failure_outputs(
    model_name: str,
    pdf_path: Path,
    model_out_dir: Path,
    seconds: float,
    error: str,
) -> None:
    doc_id = pdf_path.stem
    err_path = model_out_dir / f"{doc_id}.error.txt"
    meta_path = model_out_dir / f"{doc_id}.meta.json"

    err_path.write_text(error + "\n", encoding="utf-8")
    meta = {
        "ok": False,
        "model": model_name,
        "doc_id": doc_id,
        "pdf_path": str(pdf_path),
        "seconds": seconds,
        "error": error,
    }
    meta_path.write_text(
        json.dumps(meta, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


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
    if model_name == GLM_OCR_MODEL:
        return GlmOcrRunner(args)

    if model_name == "pypdf":
        return PyPDFRunner(args)

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


class PyPDFRunner:
    """Fast text extraction runner for text-backed PDFs.

    This is especially useful for regulatory attachments whose vertical layout
    can make OCR/layout models slow or brittle while still exposing usable text
    through the PDF text layer.
    """

    def __init__(self, args: argparse.Namespace):
        self.args = args

    def parse_pdf(self, pdf_path: Path, raw_dir: Path) -> ParseResult:
        from pypdf import PdfReader

        reader = PdfReader(str(pdf_path))
        page_markdowns: list[str] = []
        blank_pages: list[int] = []

        raw_dir.mkdir(parents=True, exist_ok=True)
        text_dir = raw_dir / "pages_text"
        text_dir.mkdir(parents=True, exist_ok=True)

        for idx, page in enumerate(reader.pages, start=1):
            page_text = normalize_pypdf_text(page.extract_text() or "")
            (text_dir / f"page_{idx:04d}.txt").write_text(
                page_text + "\n",
                encoding="utf-8",
            )
            if page_text:
                page_markdowns.append(page_text)
            else:
                blank_pages.append(idx)

        markdown = join_page_markdowns(page_markdowns)
        if not markdown:
            raise RuntimeError("pypdf extracted no text from any page.")

        return ParseResult(
            markdown=markdown,
            meta={
                "backend": "pypdf",
                "pages": len(reader.pages),
                "blank_pages": blank_pages,
                "raw_dir": str(raw_dir),
            },
        )

    def unload(self) -> None:
        clear_runtime_caches()


class PaddleOCRVLRunner:
    """PaddleOCR-VL-1.6 PDF/image -> Markdown runner."""

    def __init__(self, args: argparse.Namespace):
        self.args = args
        self.pipeline = None

    def _ensure_loaded(self) -> None:
        if self.pipeline is not None:
            return
        from paddleocr import PaddleOCRVL

        self.pipeline = PaddleOCRVL(pipeline_version="v1.6")

    def parse_pdf(self, pdf_path: Path, raw_dir: Path) -> ParseResult:
        self._ensure_loaded()
        assert self.pipeline is not None

        page_inputs = materialize_paddle_page_inputs(
            pdf_path=pdf_path,
            cache_root=self.args.out_root / "_page_cache" / PADDLE_MODEL,
            dpi=self.args.dpi,
            overwrite=self.args.overwrite,
        )

        page_markdowns: list[str] = []
        blank_pages: list[int] = []
        result_count = 0

        try:
            for idx, page_input in enumerate(page_inputs, start=1):
                print(f"  [paddleocr-vl-1.6] page {idx}/{len(page_inputs)}")

                page_raw_dir = raw_dir / "pages" / f"page_{idx:04d}"
                md_dir = page_raw_dir / "markdown"
                json_dir = page_raw_dir / "json"
                md_dir.mkdir(parents=True, exist_ok=True)
                json_dir.mkdir(parents=True, exist_ok=True)

                output = self.pipeline.predict(str(page_input))

                for res in output:
                    result_count += 1
                    res.save_to_json(save_path=str(json_dir))
                    res.save_to_markdown(save_path=str(md_dir))
                del output
                clear_runtime_caches()

                md_files = list(md_dir.rglob("*.md"))
                if not md_files:
                    blank_pages.append(idx)
                    continue

                page_md = combine_markdown_files(md_files)
                if page_md.strip():
                    page_markdowns.append(page_md)
                else:
                    blank_pages.append(idx)

                if result_count % CACHE_CLEAR_INTERVAL_PAGES == 0:
                    clear_runtime_caches()

        finally:
            if not self.args.keep_page_images:
                cleanup_page_cache(
                    pdf_path,
                    self.args.out_root / "_page_cache" / PADDLE_MODEL,
                )
            clear_runtime_caches()

        markdown = join_page_markdowns(page_markdowns)
        if not markdown:
            raise RuntimeError(
                f"PaddleOCR-VL produced no markdown for any page under {raw_dir}"
            )

        return ParseResult(
            markdown=markdown,
            meta={
                "backend": "paddleocr",
                "result_count": result_count,
                "pages": len(page_inputs),
                "blank_pages": blank_pages,
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


class GlmOcrRunner:
    """GLM-OCR SDK/CLI PDF -> Markdown runner.

    GLM-OCR can run against Zhipu MaaS, a self-hosted SDK server, or local
    vLLM/SGLang depending on its config.yaml. This runner only orchestrates the
    offline conversion and normalizes the SDK output into this repo's standard
    ``ParseResult`` contract.
    """

    def __init__(self, args: argparse.Namespace):
        self.args = args

    def parse_pdf(self, pdf_path: Path, raw_dir: Path) -> ParseResult:
        page_paths = render_pdf_pages(
            pdf_path=pdf_path,
            cache_root=self.args.out_root / "_page_cache" / GLM_OCR_MODEL,
            dpi=self.args.dpi,
            overwrite=self.args.overwrite,
        )

        output_dir = raw_dir / "glmocr_output"
        if self.args.overwrite and output_dir.exists():
            shutil.rmtree(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        if not self.args.overwrite:
            existing = collect_glmocr_markdown(output_dir)
            if existing.markdown:
                print(
                    f"[recover] glm-ocr {pdf_path.stem}: using existing "
                    f"{existing.md_count} md / {existing.json_count} json page outputs",
                    flush=True,
                )
                return self._parse_result(
                    pdf_path=pdf_path,
                    raw_dir=raw_dir,
                    page_count=len(page_paths),
                    markdown=existing.markdown,
                    source=existing.source,
                    recovered=True,
                    page_output_counts={
                        "markdown_pages": existing.md_count,
                        "json_pages": existing.json_count,
                    },
                )

        cmd = self._build_command(input_path=page_paths[0].parent, output_dir=output_dir)
        (raw_dir / "command.json").write_text(
            json.dumps(cmd, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

        try:
            self._run_glmocr_with_progress(
                cmd=cmd,
                output_dir=output_dir,
                raw_dir=raw_dir,
                doc_id=pdf_path.stem,
                total_pages=len(page_paths),
            )
        finally:
            if not self.args.keep_page_images:
                cleanup_page_cache(
                    pdf_path,
                    self.args.out_root / "_page_cache" / GLM_OCR_MODEL,
                )

        existing = collect_glmocr_markdown(output_dir)
        markdown = existing.markdown
        source = existing.source

        markdown = normalize_markdown(markdown)
        if not markdown:
            raise RuntimeError(
                f"GLM-OCR produced no markdown under {output_dir}; "
                "check stdout.txt/stderr.txt and SDK config."
            )

        return self._parse_result(
            pdf_path=pdf_path,
            raw_dir=raw_dir,
            page_count=len(page_paths),
            markdown=markdown,
            source=source,
            recovered=False,
            page_output_counts={
                "markdown_pages": existing.md_count,
                "json_pages": existing.json_count,
            },
        )

    def _parse_result(
        self,
        pdf_path: Path,
        raw_dir: Path,
        page_count: int,
        markdown: str,
        source: str,
        recovered: bool,
        page_output_counts: dict[str, int],
    ) -> ParseResult:
        return ParseResult(
            markdown=markdown,
            meta={
                "backend": "glmocr-cli",
                "model_id": "zai-org/GLM-OCR",
                "pages": page_count,
                "dpi": self.args.dpi,
                "output_dir": str(raw_dir / "glmocr_output"),
                "markdown_source": source,
                "recovered_existing_output": recovered,
                **page_output_counts,
                "config": str(self.args.glmocr_config)
                if self.args.glmocr_config is not None
                else None,
                "layout_device": self.args.glmocr_layout_device,
                "env_file": str(self.args.glmocr_env_file)
                if self.args.glmocr_env_file is not None
                else None,
                "mode": self.args.glmocr_mode,
                "set_overrides": list(self.args.glmocr_set),
            },
        )

    def _build_command(self, input_path: Path, output_dir: Path) -> list[str]:
        executable = shutil.which("glmocr")
        if executable:
            cmd = [executable]
        else:
            cmd = [sys.executable, "-m", "glmocr"]

        cmd.extend(["parse", str(input_path), "--output", str(output_dir)])

        if self.args.glmocr_config is not None:
            cmd.extend(["--config", str(self.args.glmocr_config)])
        if self.args.glmocr_layout_device:
            cmd.extend(["--layout-device", self.args.glmocr_layout_device])
        if self.args.glmocr_env_file is not None:
            cmd.extend(["--env-file", str(self.args.glmocr_env_file)])
        if self.args.glmocr_mode:
            cmd.extend(["--mode", self.args.glmocr_mode])
        for setting in self.args.glmocr_set:
            key, sep, value = setting.partition("=")
            if not sep or not key or not value:
                raise ValueError(f"--glmocr-set expects KEY=VALUE, got {setting!r}")
            cmd.extend(["--set", key, value])

        return cmd

    def _run_glmocr_with_progress(
        self,
        cmd: list[str],
        output_dir: Path,
        raw_dir: Path,
        doc_id: str,
        total_pages: int,
    ) -> None:
        stdout_path = raw_dir / "stdout.txt"
        stderr_path = raw_dir / "stderr.txt"

        with stdout_path.open("w", encoding="utf-8") as stdout_fh, stderr_path.open(
            "w",
            encoding="utf-8",
        ) as stderr_fh:
            process = subprocess.Popen(
                cmd,
                stdout=stdout_fh,
                stderr=stderr_fh,
                text=True,
                encoding="utf-8",
                errors="replace",
            )

            started = time.time()
            last_done = -1
            last_printed = 0.0
            while process.poll() is None:
                done, md_count, json_count = count_glmocr_completed_pages(output_dir)
                now = time.time()
                if done != last_done or now - last_printed >= 10:
                    print_glmocr_progress(
                        doc_id=doc_id,
                        done=done,
                        total=total_pages,
                        md_count=md_count,
                        json_count=json_count,
                        elapsed=now - started,
                    )
                    last_done = done
                    last_printed = now
                time.sleep(2)

            done, md_count, json_count = count_glmocr_completed_pages(output_dir)
            print_glmocr_progress(
                doc_id=doc_id,
                done=done,
                total=total_pages,
                md_count=md_count,
                json_count=json_count,
                elapsed=time.time() - started,
                final=True,
            )

            if process.returncode != 0:
                raise subprocess.CalledProcessError(process.returncode, cmd)

    def unload(self) -> None:
        clear_runtime_caches()


def count_glmocr_completed_pages(output_dir: Path) -> tuple[int, int, int]:
    """Return (page_count_with_any_output, md_count, json_count)."""
    page_ids: set[str] = set()
    md_count = 0
    json_count = 0

    if not output_dir.exists():
        return 0, 0, 0

    for path in output_dir.rglob("*"):
        if not path.is_file():
            continue
        suffix = path.suffix.lower()
        if suffix == ".md":
            md_count += 1
        elif suffix == ".json":
            json_count += 1
        else:
            continue

        page_id = path.parent.name if path.parent != output_dir else path.stem
        page_ids.add(page_id)

    return len(page_ids), md_count, json_count


def collect_glmocr_markdown(output_dir: Path) -> ExistingGlmOcrOutput:
    md_files = [
        path
        for path in output_dir.rglob("*.md")
        if path.is_file() and path.read_text(encoding="utf-8").strip()
    ]
    json_files = [path for path in output_dir.rglob("*.json") if path.is_file()]

    if md_files:
        markdown = combine_markdown_files(md_files)
        source = "markdown_files"
    else:
        markdown = combine_markdown_from_json_files(json_files)
        source = "json_fields"

    return ExistingGlmOcrOutput(
        markdown=normalize_markdown(markdown),
        source=source,
        md_count=len(md_files),
        json_count=len(json_files),
    )


def print_glmocr_progress(
    doc_id: str,
    done: int,
    total: int,
    md_count: int,
    json_count: int,
    elapsed: float,
    final: bool = False,
) -> None:
    total = max(total, 1)
    done = min(done, total)
    ratio = done / total
    width = 24
    filled = min(width, int(ratio * width))
    bar = "#" * filled + "-" * (width - filled)
    minutes = int(elapsed // 60)
    seconds = int(elapsed % 60)
    prefix = "[glm-ocr]"
    status = "done" if final else "run "
    print(
        f"{prefix} {status} {doc_id} [{bar}] "
        f"{done}/{total} pages {ratio * 100:5.1f}% "
        f"json={json_count} md={md_count} elapsed={minutes:02d}:{seconds:02d}",
        flush=True,
    )


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


def materialize_paddle_page_inputs(
    pdf_path: Path,
    cache_root: Path,
    dpi: int,
    overwrite: bool,
) -> list[Path]:
    """Create page-sized inputs for PaddleOCR-VL.

    Prefer page images because they avoid the whole-PDF parser path. If PyMuPDF
    is unavailable in the Paddle environment, split to one-page PDFs so large
    files still do not get submitted as a single document.
    """
    try:
        return render_pdf_pages(
            pdf_path=pdf_path,
            cache_root=cache_root,
            dpi=dpi,
            overwrite=overwrite,
        )
    except ModuleNotFoundError as exc:
        if exc.name != "fitz":
            raise
        return split_pdf_pages(
            pdf_path=pdf_path,
            cache_root=cache_root,
            overwrite=overwrite,
        )


def split_pdf_pages(
    pdf_path: Path,
    cache_root: Path,
    overwrite: bool,
) -> list[Path]:
    """Split a PDF into one-page PDFs for parser environments without PyMuPDF."""
    from pypdf import PdfReader, PdfWriter

    doc_id = pdf_path.stem
    out_dir = cache_root / doc_id
    out_dir.mkdir(parents=True, exist_ok=True)

    reader = PdfReader(str(pdf_path))
    page_paths: list[Path] = []

    for page_index, page in enumerate(reader.pages):
        page_no = page_index + 1
        page_path = out_dir / f"page_{page_no:04d}.pdf"
        page_paths.append(page_path)

        if page_path.exists() and not overwrite:
            continue

        writer = PdfWriter()
        writer.add_page(page)
        with page_path.open("wb") as fh:
            writer.write(fh)

    return page_paths


def cleanup_page_cache(pdf_path: Path, cache_root: Path) -> None:
    cache_dir = cache_root / pdf_path.stem
    if cache_dir.exists():
        shutil.rmtree(cache_dir)


def _page_sort_key(md_file: Path) -> list:
    """Natural sort key so page files order numerically, not lexically.

    PaddleOCR writes per-page files like ``8_0.md … 8_38.md`` with NON-padded
    indices, so a plain ``sorted()`` gives 8_1, 8_10, 8_2, … and scrambles the
    page order of any doc with >9 pages. Splitting on digit runs and comparing
    the numeric chunks as ints fixes that.
    """
    return [
        int(tok) if tok.isdigit() else tok
        for tok in re.split(r"(\d+)", md_file.stem)
    ]


def combine_markdown_files(md_files: list[Path]) -> str:
    chunks: list[str] = []

    for md_file in sorted(md_files, key=_page_sort_key):
        text = md_file.read_text(encoding="utf-8").strip()
        if text:
            chunks.append(text)

    return "\n\n".join(chunks)


def combine_markdown_from_json_files(json_files: list[Path]) -> str:
    """Extract markdown-like fields from SDK JSON outputs as a fallback."""
    chunks: list[str] = []

    for json_file in sorted(json_files, key=_page_sort_key):
        try:
            payload = json.loads(json_file.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            continue

        for text in iter_markdown_strings(payload):
            text = normalize_markdown(text)
            if text:
                chunks.append(text)

    return "\n\n".join(chunks)


def iter_markdown_strings(value) -> Iterable[str]:
    """Yield strings from likely markdown fields in nested JSON payloads."""
    if isinstance(value, dict):
        for key, child in value.items():
            key_norm = str(key).lower()
            if isinstance(child, str) and key_norm in {
                "markdown",
                "markdown_text",
                "md",
                "md_text",
            }:
                yield child
            else:
                yield from iter_markdown_strings(child)
        return

    if isinstance(value, list):
        for child in value:
            yield from iter_markdown_strings(child)


def join_page_markdowns(page_markdowns: list[str]) -> str:
    chunks: list[str] = []

    for idx, text in enumerate(page_markdowns, start=1):
        text = normalize_markdown(text)
        if not text:
            continue

        chunks.append(f"<!-- page:{idx} -->\n\n{text}")

    return "\n\n".join(chunks)


def normalize_pypdf_text(text: str) -> str:
    """Normalize text-layer extraction while repairing vertical CJK PDFs."""
    text = normalize_markdown(text)
    if not text:
        return ""

    lines = [line.strip() for line in text.splitlines() if line.strip()]
    if not lines:
        return ""

    short_lines = sum(1 for line in lines if len(line) <= 2)
    if short_lines / len(lines) < 0.65:
        return "\n".join(lines)

    joined = "".join(lines)
    joined = re.sub(r"\s+", "", joined)
    joined = re.sub(r"(第[一二三四五六七八九十百千万零〇0-9]+章)", r"\n\1", joined)
    joined = re.sub(r"(第[一二三四五六七八九十百千万零〇0-9]+节)", r"\n\1", joined)
    joined = re.sub(r"(第[一二三四五六七八九十百千万零〇0-9]+条)", r"\n\1", joined)
    joined = re.sub(r"(附件[一二三四五六七八九十百千万零〇0-9]*)", r"\n\1", joined)

    return "\n".join(line.strip() for line in joined.splitlines() if line.strip())


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
