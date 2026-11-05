from __future__ import annotations

import json
import sys
import unittest
from unittest.mock import patch
from urllib.error import HTTPError
from io import BytesIO
from pathlib import Path

BACKEND_DIR = Path(__file__).resolve().parent.parent
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

from app.rag import embeddings


class FakeResponse:
    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        return False


class TestLocalEmbeddingClient(unittest.TestCase):
    def test_sends_qwen_1024_request_without_truncation(self):
        response = FakeResponse()
        vectors = [[0.25] * 1024]
        with patch.object(embeddings, "urlopen", return_value=response) as urlopen, patch.object(
            embeddings.json, "load", return_value={"embeddings": vectors}
        ):
            result = embeddings._request_embeddings(["document text"])

        payload = json.loads(urlopen.call_args.args[0].data)
        self.assertEqual(payload["model"], "qwen3-embedding:0.6b")
        self.assertEqual(payload["dimensions"], 1024)
        self.assertIs(payload["truncate"], False)
        self.assertEqual(payload["input"], ["document text"])
        self.assertEqual(result, vectors)

    def test_rejects_non_1024_embeddings(self):
        with patch.object(embeddings, "urlopen", return_value=FakeResponse()), patch.object(
            embeddings.json, "load", return_value={"embeddings": [[0.25] * 768]}
        ):
            with self.assertRaisesRegex(embeddings.EmbeddingServiceError, "expected 1024"):
                embeddings._request_embeddings(["document text"])

    def test_rate_limit_respects_retry_after(self):
        error = embeddings.EmbeddingServiceError(
            "rate limited", status_code=429, retry_after=3.0, retryable=True
        )
        with patch.object(embeddings, "_request_embeddings", side_effect=[error, [[0.1] * 1024]]), patch.object(
            embeddings.time, "sleep"
        ) as sleep:
            result = embeddings.embed_texts(["text"])

        self.assertEqual(len(result), 1)
        sleep.assert_called_once_with(3.0)

    def test_inference_failure_retries_with_jitter_then_splits_batches(self):
        error = embeddings.EmbeddingServiceError(
            "upstream inference failed", status_code=502, retryable=True
        )
        request_sizes: list[int] = []

        def request(texts):
            request_sizes.append(len(texts))
            if len(texts) == 4 and request_sizes.count(4) <= 3:
                raise error
            return [[0.1] * 1024 for _ in texts]

        with patch.object(embeddings, "_request_embeddings", side_effect=request), patch.object(
            embeddings.time, "sleep"
        ) as sleep, patch.object(embeddings.random, "uniform", return_value=0.25) as jitter:
            result = embeddings.embed_texts([str(i) for i in range(6)], batch_size=4)

        self.assertEqual(len(result), 6)
        self.assertEqual(request_sizes, [4, 4, 4, 2, 2, 2])
        self.assertEqual(jitter.call_count, 2)
        self.assertEqual(sleep.call_count, 2)

    def test_http_502_is_not_reclassified_as_429(self):
        error = HTTPError(
            "http://localhost/api/embed",
            502,
            "inference failed",
            {},
            BytesIO(b""),
        )
        with patch.object(embeddings, "urlopen", side_effect=error):
            with self.assertRaises(embeddings.EmbeddingServiceError) as raised:
                embeddings._request_embeddings(["text"])
        self.assertEqual(raised.exception.status_code, 502)
        self.assertNotEqual(raised.exception.status_code, 429)


if __name__ == "__main__":
    unittest.main()
