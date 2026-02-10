"""Local in-process embedding support for RAGrep."""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List


_MODEL_ALIASES = {
    "mxbai-embed-large": "mixedbread-ai/mxbai-embed-large-v1",
}


class EmbeddingError(RuntimeError):
    """Raised when embeddings cannot be generated."""


def resolve_embedding_model(model: str) -> str:
    """Resolve a short model alias into a full model identifier."""
    return _MODEL_ALIASES.get(model, model)


def resolve_runtime_device(requested: str | None = None) -> str:
    """Resolve the embedding runtime device.

    Supported values: ``auto``, ``cpu``, ``cuda``, ``mps`` (or explicit torch
    devices such as ``cuda:0``).
    """
    return str(get_runtime_device_info(requested)["resolved_device"])


def get_runtime_device_info(requested: str | None = None) -> Dict[str, Any]:
    """Return detailed runtime device support information."""
    value = (requested or os.getenv("RAGREP_DEVICE", "auto")).strip().lower() or "auto"
    explicit = value != "auto"

    info: Dict[str, Any] = {
        "requested_device": value,
        "explicit_request": explicit,
        "torch_available": False,
        "cuda_available": False,
        "cuda_device_count": 0,
        "cuda_devices": [],
        "mps_available": False,
        "resolved_device": value if explicit else "cpu",
    }

    try:
        import torch  # type: ignore
    except Exception:
        return info

    if torch is None:  # pragma: no cover - defensive guard
        return info

    info["torch_available"] = True

    cuda_available = bool(hasattr(torch, "cuda") and torch.cuda.is_available())
    info["cuda_available"] = cuda_available
    if cuda_available and hasattr(torch.cuda, "device_count"):
        count = int(torch.cuda.device_count())
        info["cuda_device_count"] = count
        names: List[str] = []
        if hasattr(torch.cuda, "get_device_name"):
            for index in range(count):
                try:
                    names.append(str(torch.cuda.get_device_name(index)))
                except Exception:
                    names.append(f"cuda:{index}")
        info["cuda_devices"] = names

    backends = getattr(torch, "backends", None)
    mps = getattr(backends, "mps", None) if backends is not None else None
    mps_available = bool(mps is not None and hasattr(mps, "is_available") and mps.is_available())
    info["mps_available"] = mps_available

    if explicit:
        info["resolved_device"] = value
    elif cuda_available:
        info["resolved_device"] = "cuda"
    elif mps_available:
        info["resolved_device"] = "mps"
    else:
        info["resolved_device"] = "cpu"

    return info


def default_model_dir() -> Path:
    """Return the default local model storage directory.

    Linux:  ~/.config/ragrep/models
    macOS:  ~/Library/Application Support/ragrep/models
    Windows: %APPDATA%/ragrep/models
    """
    configured = os.getenv("RAGREP_MODEL_DIR")
    if configured:
        return Path(configured).expanduser().resolve()

    home = Path.home()

    if sys.platform == "darwin":
        base = home / "Library" / "Application Support" / "ragrep"
    elif os.name == "nt":
        appdata = os.getenv("APPDATA")
        if appdata:
            base = Path(appdata) / "ragrep"
        else:
            base = home / "AppData" / "Roaming" / "ragrep"
    else:
        base = home / ".config" / "ragrep"

    return base / "models"


class LocalEmbedder:
    """Generate embeddings in-process using sentence-transformers."""

    def __init__(
        self,
        model: str = "mxbai-embed-large",
        model_dir: str | Path | None = None,
        device: str | None = None,
    ) -> None:
        self.model = model
        self.resolved_model = resolve_embedding_model(model)
        self.model_dir = Path(model_dir).expanduser().resolve() if model_dir else default_model_dir()
        self.requested_device = device or os.getenv("RAGREP_DEVICE", "auto")
        self.device = resolve_runtime_device(self.requested_device)
        self.model_dir.mkdir(parents=True, exist_ok=True)

        try:
            from sentence_transformers import SentenceTransformer
        except Exception as exc:  # pragma: no cover - import path depends on environment
            raise EmbeddingError(
                "sentence-transformers is required for local embeddings. "
                "Install with: pip install sentence-transformers"
            ) from exc

        try:
            self._model = SentenceTransformer(
                self.resolved_model,
                cache_folder=str(self.model_dir),
                device=self.device,
            )
        except Exception as exc:  # pragma: no cover - model download/load depends on environment
            raise EmbeddingError(
                f"Failed to load embedding model '{self.resolved_model}'. "
                f"Model directory: {self.model_dir}."
            ) from exc

    def embed_texts(self, texts: Iterable[str], batch_size: int = 32) -> List[List[float]]:
        items = list(texts)
        if not items:
            return []

        vectors = self._model.encode(
            items,
            batch_size=batch_size,
            show_progress_bar=False,
            normalize_embeddings=False,
            convert_to_numpy=True,
        )
        return vectors.tolist()

    def embed_query(self, query: str) -> List[float]:
        vectors = self.embed_texts([query], batch_size=1)
        return vectors[0]


# Backward-compatible alias.
OllamaEmbedder = LocalEmbedder
