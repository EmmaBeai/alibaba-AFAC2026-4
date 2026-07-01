from __future__ import annotations

import json
import os
import http.client
import urllib.error
import urllib.request
from typing import Any


DEFAULT_EMBEDDING_MODEL = "text-embedding-v4"


class DashScopeEmbeddingClient:
    def __init__(
        self,
        *,
        model: str = DEFAULT_EMBEDDING_MODEL,
        dimension: int = 1024,
        batch_size: int = 10,
        api_key_env: str = "DASHSCOPE_API_KEY",
        api_key_env_fallbacks: list[str] | None = None,
        base_url_env: str = "DASHSCOPE_EMBEDDING_BASE_URL",
        default_base_url: str = "https://dashscope.aliyuncs.com/compatible-mode/v1",
        timeout_seconds: int = 120,
        max_retries: int = 3,
    ):
        self.model = model
        self.dimension = dimension
        self.batch_size = min(max(batch_size, 1), 10)
        self.api_key_env = api_key_env
        self.api_key_env_fallbacks = api_key_env_fallbacks or ["QWEN_API_KEY"]
        self.base_url_env = base_url_env
        self.default_base_url = default_base_url.rstrip("/")
        self.timeout_seconds = timeout_seconds
        self.max_retries = max(1, max_retries)

    def embed_texts(self, texts: list[str], *, input_type: str = "document") -> list[list[float]]:
        if not texts:
            return []
        api_key = self._api_key()
        if not api_key:
            raise RuntimeError(
                "Missing embedding API key: set one of "
                f"{', '.join(self._api_key_env_names())} before running dense_qwen."
            )
        base_url = os.getenv(self.base_url_env, self.default_base_url).rstrip("/")
        request_payload: dict[str, Any] = {
            "model": self.model,
            "input": texts,
            "dimensions": self.dimension,
        }
        payload = self._post_embeddings(base_url, api_key, request_payload)
        data = sorted(payload.get("data") or [], key=lambda item: int(item.get("index", 0)))
        embeddings = [item.get("embedding") for item in data]
        if len(embeddings) != len(texts) or any(not isinstance(item, list) for item in embeddings):
            raise RuntimeError(f"Embedding response shape mismatch: expected {len(texts)}, got {len(embeddings)}")
        return [[float(value) for value in embedding] for embedding in embeddings]

    def _post_embeddings(self, base_url: str, api_key: str, request_payload: dict[str, Any]) -> dict[str, Any]:
        request_data = json.dumps(request_payload, ensure_ascii=False).encode("utf-8")
        for attempt in range(1, self.max_retries + 1):
            request = urllib.request.Request(
                f"{base_url}/embeddings",
                data=request_data,
                headers={
                    "Authorization": f"Bearer {api_key}",
                    "Content-Type": "application/json",
                },
                method="POST",
            )
            try:
                with urllib.request.urlopen(request, timeout=self.timeout_seconds) as response:
                    return json.loads(response.read().decode("utf-8"))
            except urllib.error.HTTPError as exc:
                body = exc.read().decode("utf-8", errors="replace")
                raise RuntimeError(f"Embedding request failed: HTTP {exc.code}: {body}") from exc
            except (http.client.IncompleteRead, http.client.RemoteDisconnected, TimeoutError) as exc:
                if attempt >= self.max_retries:
                    raise RuntimeError("Embedding request failed: incomplete, disconnected, or timed out response") from exc
            except urllib.error.URLError as exc:
                if attempt >= self.max_retries:
                    raise RuntimeError(f"Embedding request failed: {exc.reason}") from exc
        raise RuntimeError("Embedding request failed after retries")

    def _api_key(self) -> str | None:
        for name in self._api_key_env_names():
            value = os.getenv(name)
            if value:
                return value
        return None

    def _api_key_env_names(self) -> list[str]:
        names = [self.api_key_env, *self.api_key_env_fallbacks]
        output: list[str] = []
        for name in names:
            if name and name not in output:
                output.append(name)
        return output
