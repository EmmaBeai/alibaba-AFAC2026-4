"""Configuration loading.

STUB: reverted to not-implemented for refactor.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Config:
    model: str
    base_url: str
    temperature: float
    max_context_chars: int
    top_k: int
    candidate_docs: int
    data_dir: Path
    processed_dir: Path
    submission_dir: Path

    @classmethod
    def load(cls, root: Path, config_path: Path | None = None) -> "Config":
        raise NotImplementedError
