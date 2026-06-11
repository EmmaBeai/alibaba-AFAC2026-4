"""Qwen API wrapper and token accounting.

STUB: reverted to not-implemented for refactor.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class LLMResult:
    content: str
    prompt_tokens: int
    completion_tokens: int

    @property
    def total_tokens(self) -> int:
        raise NotImplementedError


class QwenClient:
    def __init__(self, model: str, base_url: str, temperature: float = 0) -> None:
        raise NotImplementedError

    def chat_json(self, system_prompt: str, user_prompt: str) -> LLMResult:
        raise NotImplementedError


def parse_json_object(text: str) -> dict[str, Any]:
    raise NotImplementedError
