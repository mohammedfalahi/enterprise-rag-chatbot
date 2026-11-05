"""Local Ollama embeddings shared by ingestion and query retrieval."""

from __future__ import annotations

import json
import math
import random
import time
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from app.core.config import settings

_RATE_LIMIT_RETRIES = 5
_INFERENCE_RETRIES = 2
_NETWORK_RETRIES = 5
_BACKOFF_BASE_SECONDS = 1.0
_BACKOFF_MAX_SECONDS = 30.0


class EmbeddingServiceError(RuntimeError):
    def __init__(
        self,
        message: str,
        *,
        status_code: int | None = None,
        retry_after: float | None = None,
        retryable: bool = False,
    ) -> None:
        super().__init__(message)
        self.status_code = status_code
        self.retry_after = retry_after
        self.retryable = retryable


def _retry_after_seconds(value: str | None) -> float | None:
    if not value:
        return None
    try:
        delay = float(value)
    except (TypeError, ValueError, OverflowError):
        try:
            retry_at = parsedate_to_datetime(value)
        except (TypeError, ValueError, OverflowError):
            return None
        if retry_at.tzinfo is None:
            retry_at = retry_at.replace(tzinfo=timezone.utc)
        delay = (retry_at - datetime.now(timezone.utc)).total_seconds()
    return max(0.0, delay) if math.isfinite(delay) else None


def _request_embeddings(texts: list[str]) -> list[list[float]]:
    payload = json.dumps(
        {
            "model": settings.ollama_embedding_model,
            "input": texts,
            "dimensions": 1024,
            "truncate": False,
        }
    ).encode("utf-8")
    request = Request(
        f"{settings.ollama_base_url.rstrip('/')}/api/embed",
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urlopen(request, timeout=settings.ollama_timeout_seconds) as response:
            result: Any = json.load(response)
    except HTTPError as error:
        raise EmbeddingServiceError(
            "Ollama embedding request failed",
            status_code=error.code,
            retry_after=_retry_after_seconds(error.headers.get("Retry-After")),
            retryable=error.code == 429 or error.code in {500, 502, 503, 504},
        ) from None
    except (URLError, TimeoutError, ConnectionError):
        raise EmbeddingServiceError(
            "Could not reach the local Ollama embedding service",
            retryable=True,
        ) from None

    vectors = result.get("embeddings") if isinstance(result, dict) else None
    if not isinstance(vectors, list) or len(vectors) != len(texts):
        raise EmbeddingServiceError(
            "Ollama returned an unexpected number of embeddings"
        )

    checked: list[list[float]] = []
    for vector in vectors:
        if not isinstance(vector, list) or len(vector) != 1024:
            actual_dimension = len(vector) if isinstance(vector, list) else None
            raise EmbeddingServiceError(
                f"Ollama returned an embedding with dimension {actual_dimension}; expected 1024"
            )
        if any(
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(float(value))
            for value in vector
        ):
            raise EmbeddingServiceError("Ollama returned non-finite embedding values")
        checked.append([float(value) for value in vector])
    return checked


def _embed_batch(texts: list[str]) -> tuple[list[list[float]], int]:
    rate_limit_attempt = 0
    inference_attempt = 0
    network_attempt = 0
    while True:
        try:
            return _request_embeddings(texts), len(texts)
        except EmbeddingServiceError as error:
            if error.status_code == 429:
                if rate_limit_attempt >= _RATE_LIMIT_RETRIES:
                    raise
                backoff = min(
                    _BACKOFF_BASE_SECONDS * (2**rate_limit_attempt),
                    _BACKOFF_MAX_SECONDS,
                )
                delay = max(backoff, error.retry_after or 0.0)
                print(
                    f"[Ollama embeddings] HTTP 429; retry "
                    f"{rate_limit_attempt + 1}/{_RATE_LIMIT_RETRIES} "
                    f"in {delay:.1f}s"
                )
                time.sleep(delay)
                rate_limit_attempt += 1
                continue

            if error.status_code in {502, 503, 504}:
                if inference_attempt < _INFERENCE_RETRIES:
                    ceiling = min(
                        _BACKOFF_BASE_SECONDS * (2**inference_attempt),
                        _BACKOFF_MAX_SECONDS,
                    )
                    delay = random.uniform(0.0, ceiling)
                    print(
                        f"[Ollama embeddings] HTTP {error.status_code}; retry "
                        f"{inference_attempt + 1}/{_INFERENCE_RETRIES} "
                        f"in {delay:.1f}s"
                    )
                    time.sleep(delay)
                    inference_attempt += 1
                    continue
                if len(texts) > 1:
                    midpoint = len(texts) // 2
                    print(
                        "[Ollama embeddings] Repeated inference failure; "
                        f"reducing batch size from {len(texts)} to {midpoint}"
                    )
                    first, first_size = _embed_batch(texts[:midpoint])
                    second, second_size = _embed_batch(texts[midpoint:])
                    return first + second, min(midpoint, first_size, second_size)
                raise

            if error.retryable:
                if network_attempt >= _NETWORK_RETRIES:
                    raise
                ceiling = min(
                    _BACKOFF_BASE_SECONDS * (2**network_attempt),
                    _BACKOFF_MAX_SECONDS,
                )
                delay = random.uniform(0.0, ceiling)
                print(
                    "[Ollama embeddings] Local service unavailable; retry "
                    f"{network_attempt + 1}/{_NETWORK_RETRIES} "
                    f"in {delay:.1f}s"
                )
                time.sleep(delay)
                network_attempt += 1
                continue
            raise


def embed_texts(texts: list[str], batch_size: int = 32) -> list[list[float]]:
    """Embed text without truncation and verify each vector has exactly 1024 values."""
    if batch_size < 1:
        raise ValueError("batch_size must be greater than zero")
    if any(not isinstance(text, str) or not text.strip() for text in texts):
        raise ValueError("embedding inputs must be non-empty strings")
    if not texts:
        return []

    vectors: list[list[float]] = []
    next_batch_size = batch_size
    start = 0
    while start < len(texts):
        batch = texts[start : start + next_batch_size]
        batch_vectors, suggested_size = _embed_batch(batch)
        vectors.extend(batch_vectors)
        start += len(batch)
        next_batch_size = min(next_batch_size, suggested_size)
    return vectors


async def require_active_embedding_schema(cursor: Any) -> None:
    await cursor.execute(
        """
        SELECT format_type(a.atttypid, a.atttypmod)
        FROM pg_attribute a
        JOIN pg_class c ON c.oid = a.attrelid
        JOIN pg_namespace n ON n.oid = c.relnamespace
        WHERE n.nspname = 'public'
          AND c.relname = 'document_chunks'
          AND a.attname = 'embedding'
          AND NOT a.attisdropped;
        """
    )
    row = await cursor.fetchone()
    if not row or row[0] != "vector(1024)":
        actual = row[0] if row else "missing"
        raise RuntimeError(
            "Embedding migration is not active: expected "
            f"public.document_chunks.embedding vector(1024), found {actual}"
        )
