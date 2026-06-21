from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Any

from openai import (
    APIConnectionError,
    APITimeoutError,
    AuthenticationError,
    InternalServerError,
    OpenAI,
    RateLimitError,
)

from agent.schemas import TokenUsage


class QwenClient:
    def __init__(
        self,
        model: str,
        api_key_env: str,
        base_url_env: str,
        default_base_url: str,
        log_dir: Path,
        temperature: float = 0,
        extra_body: dict[str, Any] | None = None,
        max_tokens_by_purpose: dict[str, int] | None = None,
    ):
        api_key = (os.getenv(api_key_env) or "").strip()
        if not api_key:
            raise RuntimeError(f"Missing required environment variable: {api_key_env}")
        base_url = (os.getenv(base_url_env) or default_base_url).strip().rstrip("/")
        self.model = os.getenv("QWEN_MODEL", model)
        self.api_key_env = api_key_env
        self.base_url = base_url
        self.key_hint = f"{api_key[:4]}...{api_key[-4:]} (length={len(api_key)})"
        self.client = OpenAI(
            api_key=api_key,
            base_url=base_url,
            max_retries=0,
        )
        self.temperature = temperature
        self.extra_body = extra_body or {}
        self.max_tokens_by_purpose = max_tokens_by_purpose or {}
        self.usage = TokenUsage()
        self.log_path = log_dir / "llm_calls.jsonl"
        self.log_path.parent.mkdir(parents=True, exist_ok=True)

    def json_completion(
        self,
        system: str,
        user: str,
        *,
        purpose: str,
        qid: str,
    ) -> dict[str, Any]:
        response = self._request(system, user, purpose=purpose, qid=qid)
        usage = response.usage
        prompt_tokens = usage.prompt_tokens if usage else 0
        completion_tokens = usage.completion_tokens if usage else 0
        self.usage.add(prompt_tokens, completion_tokens)
        content = response.choices[0].message.content or "{}"
        record = {
            "qid": qid,
            "purpose": purpose,
            "model": self.model,
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": prompt_tokens + completion_tokens,
        }
        with self.log_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
        try:
            return json.loads(content)
        except json.JSONDecodeError as exc:
            self._log_failure(
                qid,
                purpose,
                1,
                "json_decode_error",
                f"{exc}; content={content[:2000]}",
            )
            raise RuntimeError(
                f"Qwen returned invalid or truncated JSON for qid={qid}, purpose={purpose}. "
                "Check logs/llm_calls.jsonl. If completion_tokens hit the configured max_tokens, "
                "increase model.max_tokens for that purpose."
            ) from exc

    def _request(self, system: str, user: str, *, purpose: str, qid: str):
        attempts = 3
        for attempt in range(1, attempts + 1):
            try:
                request_args: dict[str, Any] = {
                    "model": self.model,
                    "temperature": self.temperature,
                    "response_format": {"type": "json_object"},
                    "messages": [
                        {"role": "system", "content": system},
                        {"role": "user", "content": user},
                    ],
                }
                max_tokens = self.max_tokens_by_purpose.get(purpose)
                if max_tokens:
                    request_args["max_tokens"] = max_tokens
                if self.extra_body:
                    request_args["extra_body"] = self.extra_body
                return self.client.chat.completions.create(
                    **request_args,
                )
            except AuthenticationError as exc:
                self._log_failure(qid, purpose, attempt, "authentication_error", str(exc))
                if attempt == attempts:
                    raise RuntimeError(
                        "Qwen authentication failed after successful startup or retries. "
                        f"Check {self.api_key_env}={self.key_hint}, base_url={self.base_url}, "
                        "and whether the Alibaba Cloud Model Studio API key is still enabled. "
                        "The completed questions remain in answer.csv; rerun the same command "
                        "after fixing the key to resume."
                    ) from exc
                time.sleep(2 ** (attempt - 1))
            except (RateLimitError, APIConnectionError, APITimeoutError, InternalServerError) as exc:
                self._log_failure(qid, purpose, attempt, type(exc).__name__, str(exc))
                if attempt == attempts:
                    raise
                time.sleep(2 ** (attempt - 1))
        raise AssertionError("unreachable")

    def _log_failure(
        self,
        qid: str,
        purpose: str,
        attempt: int,
        error_type: str,
        message: str,
    ) -> None:
        record = {
            "qid": qid,
            "purpose": purpose,
            "model": self.model,
            "status": "failed",
            "attempt": attempt,
            "error_type": error_type,
            "message": message[:1000],
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "total_tokens": 0,
        }
        with self.log_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
