from __future__ import annotations

import unittest
from unittest.mock import MagicMock, patch

from app.observability.tracing import TraceSpanContext, update_current_generation


class TestLangfuseTracing(unittest.TestCase):
    @patch("app.observability.tracing.is_langfuse_enabled", return_value=True)
    @patch("app.observability.tracing.get_langfuse_client")
    def test_trace_inputs_and_status_messages_are_redacted(
        self, get_client, _is_enabled
    ):
        client = MagicMock()
        observation = MagicMock()
        context_manager = MagicMock()
        context_manager.__enter__.return_value = observation
        client.start_as_current_observation.return_value = context_manager
        get_client.return_value = client

        with TraceSpanContext(
            name="test",
            input={"api_key": "do-not-record", "question_chars": 12},
        ):
            pass

        kwargs = client.start_as_current_observation.call_args.kwargs
        self.assertEqual(kwargs["input"]["api_key"], "[REDACTED]")
        self.assertEqual(kwargs["input"]["question_chars"], 12)
        update_current_generation(status_message="Bearer abc.def.ghi")
        status = client.update_current_generation.call_args.kwargs["status_message"]
        self.assertNotIn("abc.def.ghi", status)

    @patch("app.observability.tracing.get_langfuse_client")
    def test_generation_status_redacts_bearer_values(self, get_client):
        client = MagicMock()
        get_client.return_value = client
        update_current_generation(status_message="provider rejected Bearer abc123")
        status = client.update_current_generation.call_args.kwargs["status_message"]
        self.assertNotIn("abc123", status)


if __name__ == "__main__":
    unittest.main()
