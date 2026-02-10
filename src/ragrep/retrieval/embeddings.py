"""Embedding client support for local Ollama models."""

from __future__ import annotations

import json
import os
from typing import Iterable, List
from urllib import error, request


class EmbeddingError(RuntimeError):
    """Raised when embeddings cannot be generated."""


class OllamaEmbedder:
    """Generate embeddings via a local Ollama server."""

    def __init__(
        self,
        model: str = "mxbai-embed-large",
        base_url: str | None = None,
        timeout_seconds: float = 120.0,
    ) -> None:
        self.model = model
        self.base_url = (base_url or os.getenv("OLLAMA_BASE_URL", "http://127.0.0.1:11434")).rstrip("/")
        self.timeout_seconds = timeout_seconds

    def embed_texts(self, texts: Iterable[str], batch_size: int = 32) -> List[List[float]]:
        items = list(texts)
        if not items:
            return []

        vectors: List[List[float]] = []
        for index in range(0, len(items), batch_size):
            batch = items[index:index + batch_size]
            vectors.extend(self._embed_batch(batch))
        return vectors

    def embed_query(self, query: str) -> List[float]:
        vectors = self.embed_texts([query], batch_size=1)
        return vectors[0]

    def _embed_batch(self, batch: List[str]) -> List[List[float]]:
        payload = {"model": self.model, "input": batch}

        try:
            data = self._post_json("/api/embed", payload)
            if isinstance(data.get("embeddings"), list):
                return [self._coerce_vector(vector) for vector in data["embeddings"]]
        except EmbeddingError as exc:
            # Older Ollama versions exposed /api/embeddings only.
            message = str(exc)
            if "model" in message and "not found" in message:
                raise EmbeddingError(
                    f"Ollama model '{self.model}' is not available. "
                    f"Run: ollama pull {self.model}"
                ) from exc
            if "HTTP 404" not in message:
                raise

        # Compatibility fallback for older Ollama API.
        vectors: List[List[float]] = []
        for text in batch:
            legacy_data = self._post_json("/api/embeddings", {"model": self.model, "prompt": text})
            if not isinstance(legacy_data.get("embedding"), list):
                raise EmbeddingError("Ollama did not return an embedding vector")
            vectors.append(self._coerce_vector(legacy_data["embedding"]))

        return vectors

    def _post_json(self, path: str, payload: dict) -> dict:
        body = json.dumps(payload).encode("utf-8")
        req = request.Request(
            url=f"{self.base_url}{path}",
            data=body,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with request.urlopen(req, timeout=self.timeout_seconds) as response:
                response_data = response.read().decode("utf-8")
        except error.HTTPError as exc:
            details = exc.read().decode("utf-8", errors="replace") if hasattr(exc, "read") else str(exc)
            if "not found, try pulling it first" in details:
                raise EmbeddingError(
                    f"Ollama model '{self.model}' is not available. Run: ollama pull {self.model}"
                ) from exc
            raise EmbeddingError(f"Ollama request failed ({path}) HTTP {exc.code}: {details}") from exc
        except error.URLError as exc:
            raise EmbeddingError(
                "Could not connect to Ollama at "
                f"{self.base_url}. Start Ollama and run: ollama pull {self.model}"
            ) from exc

        try:
            return json.loads(response_data)
        except json.JSONDecodeError as exc:
            raise EmbeddingError(f"Invalid JSON response from Ollama ({path})") from exc

    @staticmethod
    def _coerce_vector(vector: list) -> List[float]:
        return [float(value) for value in vector]
