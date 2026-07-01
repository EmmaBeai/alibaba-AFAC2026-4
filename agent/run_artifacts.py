from __future__ import annotations

import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml

from agent.config import resolve_path
from agent.run_compare import compare_run_dirs
from agent.schemas import AnswerResult, Document, Question


def configure_run_directory(
    config: dict[str, Any],
    *,
    run_dir: Path | str | None = None,
    output_csv: Path | str | None = None,
    evidence_json: Path | str | None = None,
) -> Path:
    resolved_run_dir = _resolve_config_path(
        config,
        run_dir or config.get("run", {}).get("output_dir") or "runs/adhoc",
    )
    resolved_run_dir.mkdir(parents=True, exist_ok=True)
    run_config = config.setdefault("run", {})
    run_config["output_dir"] = str(resolved_run_dir)
    run_config["output_csv"] = str(
        _resolve_config_path(config, output_csv) if output_csv else resolved_run_dir / "answer.csv"
    )
    run_config["evidence_json"] = str(
        _resolve_config_path(config, evidence_json) if evidence_json else resolved_run_dir / "evidence.json"
    )
    Path(run_config["output_csv"]).parent.mkdir(parents=True, exist_ok=True)
    Path(run_config["evidence_json"]).parent.mkdir(parents=True, exist_ok=True)
    write_config_snapshot(config, resolved_run_dir)
    return resolved_run_dir


def write_config_snapshot(config: dict[str, Any], run_dir: Path | str) -> Path:
    path = Path(run_dir) / "config.yaml"
    path.write_text(
        yaml.safe_dump(_public_config(config), allow_unicode=True, sort_keys=False),
        encoding="utf-8",
    )
    return path


def write_run_manifest(
    config: dict[str, Any],
    run_dir: Path | str,
    *,
    questions: list[Question],
    documents: list[Document],
    results: list[AnswerResult],
    previous_run_dir: Path | str | None = None,
    validation_report: dict[str, Any] | None = None,
) -> dict[str, Any]:
    resolved_run_dir = Path(run_dir)
    previous_path = _resolve_config_path(config, previous_run_dir) if previous_run_dir else None
    comparison = None
    if previous_path and (previous_path / "answer.csv").exists() and (resolved_run_dir / "answer.csv").exists():
        comparison = compare_run_dirs(previous_path, resolved_run_dir)
    manifest = {
        "schema_version": 1,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "run_id": resolved_run_dir.name,
        "git": _git_info(Path(config.get("_root", "."))),
        "data": {
            "dataset_path": str(resolve_path(config, config["paths"]["dataset"])),
            "questions_path": str(resolve_path(config, config["run"]["questions"])),
            "data_version": config.get("paths", {}).get("data_version", "unknown"),
            "question_count": len(questions),
            "document_count": len(documents),
        },
        "answerer": config.get("answerer", {"type": "bm25_top1"}),
        "chunk_index_config": config.get("structured_retrieval", {}),
        "embedding": config.get("embedding", {}),
        "model": config.get("model", {"name": "none", "prompt_version": "none"}),
        "token_usage": _token_usage(results),
        "outputs": {
            "answer_csv": str(resolve_path(config, config["run"]["output_csv"])),
            "evidence_json": str(resolve_path(config, config["run"]["evidence_json"])),
            "config_yaml": str(resolved_run_dir / "config.yaml"),
            "manifest_json": str(resolved_run_dir / "manifest.json"),
        },
        "changes": {
            "previous_run": str(previous_path) if previous_path else None,
            "changed_qids": comparison.changed_qids if comparison else [],
            "same_qids": comparison.same_qids if comparison else [],
            "missing_left": comparison.missing_left if comparison else [],
            "missing_right": comparison.missing_right if comparison else [],
        },
        "output_checks": validation_report or {},
    }
    (resolved_run_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return manifest


def _resolve_config_path(config: dict[str, Any], value: Path | str | None) -> Path:
    if value is None:
        return resolve_path(config, "")
    path = Path(value)
    return path if path.is_absolute() else resolve_path(config, str(path))


def _public_config(config: dict[str, Any]) -> dict[str, Any]:
    return {
        key: _jsonable(value)
        for key, value in config.items()
        if key != "_root"
    }


def _jsonable(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {key: _jsonable(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_jsonable(item) for item in value]
    return value


def _git_info(root: Path) -> dict[str, Any]:
    commit = _git_output(root, ["git", "rev-parse", "HEAD"]) or "unknown"
    short_commit = _git_output(root, ["git", "rev-parse", "--short", "HEAD"]) or "unknown"
    status = _git_output(root, ["git", "status", "--short"]) or ""
    return {
        "commit": commit,
        "short_commit": short_commit,
        "dirty": bool(status.strip()),
        "status_short": status.splitlines(),
    }


def _git_output(root: Path, args: list[str]) -> str | None:
    try:
        completed = subprocess.run(
            args,
            cwd=root,
            check=False,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            text=True,
        )
    except OSError:
        return None
    if completed.returncode != 0:
        return None
    return completed.stdout.strip()


def _token_usage(results: list[AnswerResult]) -> dict[str, int]:
    prompt = sum(result.usage.prompt_tokens for result in results)
    completion = sum(result.usage.completion_tokens for result in results)
    return {
        "prompt_tokens": prompt,
        "completion_tokens": completion,
        "total_tokens": prompt + completion,
    }
