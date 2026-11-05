from __future__ import annotations

import asyncio
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import httpx
from openai import APITimeoutError, RateLimitError

from scripts.generate_ragas_snapshot import (
    _generate_with_groq,
    _load_snapshot_checkpoint,
    _rate_limit_retry_delay,
    _write_snapshot_checkpoint,
)


class TestRagasSnapshotResilience(unittest.TestCase):
    def test_rate_limit_delay_uses_retry_after_header(self):
        delay = _rate_limit_retry_delay(
            {"Retry-After": "12.5"},
            "rate limited",
            retry_number=0,
        )
        self.assertEqual(delay, 12.5)

    def test_rate_limit_delay_reads_provider_message(self):
        delay = _rate_limit_retry_delay(
            {},
            "Please try again in 15.945 seconds.",
            retry_number=0,
        )
        self.assertEqual(delay, 15.945)

    def test_groq_generation_retries_rate_limit(self):
        rate_limit = RateLimitError(
            "rate limited",
            response=httpx.Response(
                429,
                request=httpx.Request("POST", "https://example.test"),
                headers={"retry-after": "0.01"},
            ),
            body={"error": "rate limited"},
        )
        completion = SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content="answer"))]
        )
        create = AsyncMock(side_effect=[rate_limit, completion])
        client = SimpleNamespace(
            chat=SimpleNamespace(
                completions=SimpleNamespace(create=create),
            )
        )

        with patch(
            "scripts.generate_ragas_snapshot.asyncio.sleep",
            new_callable=AsyncMock,
        ) as sleep:
            result = asyncio.run(_generate_with_groq(client, "prompt"))

        self.assertEqual(result, "answer")
        self.assertEqual(create.await_count, 2)
        sleep.assert_awaited_once_with(0.01)

    def test_groq_generation_retries_connection_timeout(self):
        timeout = APITimeoutError(
            request=httpx.Request("POST", "https://example.test")
        )
        completion = SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content="answer"))]
        )
        create = AsyncMock(side_effect=[timeout, completion])
        client = SimpleNamespace(
            chat=SimpleNamespace(
                completions=SimpleNamespace(create=create),
            )
        )

        with patch(
            "scripts.generate_ragas_snapshot.asyncio.sleep",
            new_callable=AsyncMock,
        ) as sleep:
            result = asyncio.run(_generate_with_groq(client, "prompt"))

        self.assertEqual(result, "answer")
        self.assertEqual(create.await_count, 2)
        sleep.assert_awaited_once_with(5.0)

    def test_snapshot_checkpoint_round_trips_and_validates_model(self):
        selected = [
            {
                "test_id": "case-1",
                "question": "Question",
                "answer_facts": ["Grounded fact"],
                "role": "employee",
            }
        ]
        row = {
            "test_id": "case-1",
            "user_input": "Question",
            "reference": "- Grounded fact",
            "role": "employee",
            "generation_provider": "groq",
            "generation_model": "model",
            "response": "Answer",
            "retrieved_contexts": ["Context"],
        }

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "snapshot.jsonl"
            _write_snapshot_checkpoint(path, selected, {"case-1": row})

            loaded = _load_snapshot_checkpoint(path, selected, "groq", "model")
            self.assertEqual(loaded, {"case-1": row})
            with self.assertRaisesRegex(ValueError, "generation model"):
                _load_snapshot_checkpoint(path, selected, "gemini", "other-model")


if __name__ == "__main__":
    unittest.main()
