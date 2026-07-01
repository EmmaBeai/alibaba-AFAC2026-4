from __future__ import annotations

import json
import os
import re
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import Any


JSON_BLOCK_RE = re.compile(r"```(?:json)?\s*(.*?)```", re.DOTALL | re.IGNORECASE)
DEFAULT_QWEN_MODEL = "qwen3.7-max"


@dataclass(slots=True)
class QwenCompletion:
    payload: dict[str, Any]
    prompt_tokens: int
    completion_tokens: int
    raw_content: str
    reasoning_content: str = ""


class QwenPlusClient:
    def __init__(
        self,
        *,
        model: str = DEFAULT_QWEN_MODEL,
        api_key_env: str = "DASHSCOPE_API_KEY",
        api_key_env_fallbacks: list[str] | None = None,
        base_url_env: str = "QWEN_BASE_URL",
        default_base_url: str = "https://dashscope.aliyuncs.com/compatible-mode/v1",
        temperature: float = 0,
        extra_body: dict[str, Any] | None = None,
        max_tokens: int | None = None,
        timeout_seconds: int = 120,
    ):
        self.model = model
        self.api_key_env = api_key_env
        self.api_key_env_fallbacks = api_key_env_fallbacks or ["QWEN_API_KEY"]
        self.base_url_env = base_url_env
        self.default_base_url = default_base_url.rstrip("/")
        self.temperature = temperature
        self.extra_body = extra_body or {}
        self.max_tokens = max_tokens
        self.timeout_seconds = timeout_seconds

    def json_completion(self, *, system: str, prompt: str) -> QwenCompletion:
        completion = self._completion(system=system, prompt=prompt, json_response=True)
        return QwenCompletion(
            payload=_parse_json_content(completion.raw_content),
            prompt_tokens=completion.prompt_tokens,
            completion_tokens=completion.completion_tokens,
            raw_content=completion.raw_content,
            reasoning_content=completion.reasoning_content,
        )

    def raw_completion(self, *, system: str, prompt: str) -> QwenCompletion:
        completion = self._completion(system=system, prompt=prompt, json_response=False)
        payload: dict[str, Any]
        try:
            payload = _parse_json_content(completion.raw_content)
        except RuntimeError:
            payload = {}
        return QwenCompletion(
            payload=payload,
            prompt_tokens=completion.prompt_tokens,
            completion_tokens=completion.completion_tokens,
            raw_content=completion.raw_content,
            reasoning_content=completion.reasoning_content,
        )

    def _completion(self, *, system: str, prompt: str, json_response: bool) -> QwenCompletion:
        api_key_env, api_key = self._api_key()
        if not api_key:
            raise RuntimeError(
                "Missing Qwen API key: set one of "
                f"{', '.join(self._api_key_env_names())} before running bm25_qwen."
            )
        base_url = os.getenv(self.base_url_env, self.default_base_url).rstrip("/")
        request_payload: dict[str, Any] = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": prompt},
            ],
            "temperature": self.temperature,
        }
        if json_response:
            request_payload["response_format"] = {"type": "json_object"}
        if self.max_tokens:
            request_payload["max_tokens"] = self.max_tokens
        request_payload.update(self.extra_body)
        request = urllib.request.Request(
            f"{base_url}/chat/completions",
            data=json.dumps(request_payload, ensure_ascii=False).encode("utf-8"),
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            },
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=self.timeout_seconds) as response:
                response_payload = json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            body = exc.read().decode("utf-8", errors="replace")
            raise RuntimeError(f"Qwen request failed: HTTP {exc.code}: {body}") from exc
        except urllib.error.URLError as exc:
            raise RuntimeError(f"Qwen request failed: {exc.reason}") from exc
        message = response_payload["choices"][0]["message"]
        content = str(message["content"])
        usage = response_payload.get("usage") or {}
        return QwenCompletion(
            payload={},
            prompt_tokens=int(usage.get("prompt_tokens") or 0),
            completion_tokens=int(usage.get("completion_tokens") or 0),
            raw_content=content,
            reasoning_content=_extract_reasoning_content(message),
        )

    def _api_key(self) -> tuple[str | None, str | None]:
        for name in self._api_key_env_names():
            value = os.getenv(name)
            if value:
                return name, value
        return None, None

    def _api_key_env_names(self) -> list[str]:
        names = [self.api_key_env, *self.api_key_env_fallbacks]
        output: list[str] = []
        for name in names:
            if name and name not in output:
                output.append(name)
        return output


def _parse_json_content(content: str) -> dict[str, Any]:
    text = content.strip()
    block = JSON_BLOCK_RE.search(text)
    if block:
        text = block.group(1).strip()
    try:
        payload = json.loads(text)
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"Qwen did not return valid JSON: {content[:500]}") from exc
    if isinstance(payload, list):
        return {"items": payload}
    if not isinstance(payload, dict):
        raise RuntimeError(f"Qwen JSON response must be an object: {content[:500]}")
    return payload


def _extract_reasoning_content(message: dict[str, Any]) -> str:
    for key in ("reasoning_content", "reasoning"):
        value = message.get(key)
        if value:
            return value if isinstance(value, str) else json.dumps(value, ensure_ascii=False)
    return ""
