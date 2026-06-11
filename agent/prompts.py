"""Qwen prompts.

STUB: reverted to not-implemented for refactor.
"""

from __future__ import annotations

from agent.schema import Question


# TODO: redesign system prompt during refactor.
SYSTEM_PROMPT = ""


def build_user_prompt(question: Question, evidence_blocks: list[str]) -> str:
    raise NotImplementedError
