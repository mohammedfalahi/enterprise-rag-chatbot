"""
Langfuse Tracing and Observability Client for Enterprise Chatbot.
Implements safe no-op fallback when credentials are missing, OpenTelemetry context propagation,
data scrubbing, and support for all observation types (retriever, agent, generation, guardrail, tool).
Reused from DEV_AI (packages/observability/tracing.py).
"""

from __future__ import annotations

import asyncio
import functools
import logging
import os
import types
from collections.abc import Callable
from typing import Any, Literal, Self, TypeVar

from app.core.config import settings
from app.observability.redaction import scrub_sensitive_data

logger = logging.getLogger("app.observability")

_CLIENT_INSTANCE: Any = None
_CLIENT_INITIALIZED = False

ObservationType = Literal[
    "generation",
    "span",
    "agent",
    "tool",
    "chain",
    "retriever",
    "evaluator",
    "guardrail",
    "embedding",
]

F = TypeVar("F", bound=Callable[..., Any])


def is_langfuse_enabled() -> bool:
    """
    Returns True only if both public and secret keys are provided in environment or settings.
    Ensures safe no-op fallback in test environments and offline operations.
    """
    public_key = os.environ.get("LANGFUSE_PUBLIC_KEY") or settings.langfuse_public_key
    secret_key = os.environ.get("LANGFUSE_SECRET_KEY") or settings.langfuse_secret_key
    return bool(public_key and str(public_key).strip() and secret_key and str(secret_key).strip())


def get_langfuse_client() -> Any | None:
    """
    Returns the initialized Langfuse client singleton, or None if disabled.
    """
    global _CLIENT_INSTANCE, _CLIENT_INITIALIZED
    if _CLIENT_INITIALIZED:
        return _CLIENT_INSTANCE

    if not is_langfuse_enabled():
        _CLIENT_INSTANCE = None
        _CLIENT_INITIALIZED = True
        return None

    try:
        from langfuse import Langfuse

        public_key = os.environ.get("LANGFUSE_PUBLIC_KEY") or settings.langfuse_public_key
        secret_key = os.environ.get("LANGFUSE_SECRET_KEY") or settings.langfuse_secret_key
        base_url = (
            os.environ.get("LANGFUSE_BASE_URL")
            or os.environ.get("LANGFUSE_HOST")
            or settings.langfuse_base_url
        )

        _CLIENT_INSTANCE = Langfuse(
            public_key=str(public_key).strip(),
            secret_key=str(secret_key).strip(),
            host=str(base_url).strip(),
        )
        _CLIENT_INITIALIZED = True
        logger.info("Langfuse observability client initialized successfully.")
        return _CLIENT_INSTANCE
    except Exception as exc:
        logger.warning(
            "Failed to initialize Langfuse client (%s). Falling back to no-op.",
            type(exc).__name__,
        )
        _CLIENT_INSTANCE = None
        _CLIENT_INITIALIZED = True
        return None


def reset_client_state() -> None:
    """Reset the client singleton state (primarily for testing)."""
    global _CLIENT_INSTANCE, _CLIENT_INITIALIZED
    _CLIENT_INSTANCE = None
    _CLIENT_INITIALIZED = False


class NoOpSpan:
    """Safe no-op replacement when Langfuse is disabled or offline."""

    def __init__(self, name: str = "noop") -> None:
        self.name = name

    def update(self, *args: Any, **kwargs: Any) -> Self:
        return self

    def end(self, *args: Any, **kwargs: Any) -> None:
        pass

    def score(self, *args: Any, **kwargs: Any) -> None:
        pass

    def score_trace(self, *args: Any, **kwargs: Any) -> None:
        pass

    def __enter__(self) -> Self:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: types.TracebackType | None,
    ) -> None:
        pass

    async def __aenter__(self) -> Self:
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: types.TracebackType | None,
    ) -> None:
        pass


class TraceSpanContext:
    """
    Context manager supporting both sync and async execution for Langfuse observations.
    Automatically handles OpenTelemetry context propagation, redaction, and error capture.
    """

    def __init__(
        self,
        name: str,
        as_type: ObservationType = "span",
        input: Any = None,
        output: Any = None,
        metadata: Any = None,
        model: str | None = None,
        usage_details: dict[str, int] | None = None,
    ) -> None:
        self.name = name
        self.as_type = as_type
        self.input = scrub_sensitive_data(input) if input is not None else None
        self.output = scrub_sensitive_data(output) if output is not None else None
        self.metadata = scrub_sensitive_data(metadata) if metadata is not None else None
        self.model = model
        self.usage_details = usage_details
        self._cm: Any = None
        self._span: Any = None

    def __enter__(self) -> Any:
        client = get_langfuse_client()
        if client is None:
            return NoOpSpan(self.name)

        try:
            kwargs: dict[str, Any] = {
                "name": self.name,
                "as_type": self.as_type,
            }
            if self.input is not None:
                kwargs["input"] = self.input
            if self.output is not None:
                kwargs["output"] = self.output
            if self.metadata is not None:
                kwargs["metadata"] = self.metadata
            if self.model is not None:
                kwargs["model"] = self.model
            if self.usage_details is not None:
                kwargs["usage_details"] = self.usage_details

            self._cm = client.start_as_current_observation(**kwargs)
            self._span = self._cm.__enter__()
            return self._span
        except Exception as exc:
            logger.debug(
                "Langfuse observation start failed (%s).",
                type(exc).__name__,
            )
            return NoOpSpan(self.name)

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: types.TracebackType | None,
    ) -> None:
        if self._cm is not None:
            try:
                if exc_val is not None and hasattr(self._span, "update"):
                    self._span.update(
                        level="ERROR",
                        status_message=type(exc_val).__name__,
                    )
                self._cm.__exit__(exc_type, exc_val, exc_tb)
            except Exception as exc:
                logger.debug("Langfuse observation close failed (%s).", type(exc).__name__)

    async def __aenter__(self) -> Any:
        return self.__enter__()

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: types.TracebackType | None,
    ) -> None:
        self.__exit__(exc_type, exc_val, exc_tb)


def trace_span(
    name: str,
    as_type: ObservationType = "span",
    input: Any = None,
    output: Any = None,
    metadata: Any = None,
    model: str | None = None,
    usage_details: dict[str, int] | None = None,
) -> Any:
    """
    Creates a trace or child observation span.
    Safe to use as either `with trace_span(...)` or `async with trace_span(...)`.
    """
    if not is_langfuse_enabled():
        return NoOpSpan(name)
    return TraceSpanContext(
        name=name,
        as_type=as_type,
        input=input,
        output=output,
        metadata=metadata,
        model=model,
        usage_details=usage_details,
    )


def observe(
    name: str | None = None,
    as_type: ObservationType = "span",
    capture_input: bool = True,
    capture_output: bool = True,
) -> Callable[[F], F]:
    """
    Safe decorator for functions or coroutines.
    If Langfuse is not enabled, transparently executes without overhead or errors.
    """

    def decorator(fn: F) -> F:
        span_name = name or fn.__name__

        if asyncio.iscoroutinefunction(fn):

            @functools.wraps(fn)
            async def async_wrapper(*args: Any, **kwargs: Any) -> Any:
                if not is_langfuse_enabled():
                    return await fn(*args, **kwargs)

                call_input = None
                if capture_input:
                    call_input = scrub_sensitive_data(kwargs if kwargs else args)

                async with trace_span(name=span_name, as_type=as_type, input=call_input) as span:
                    result = await fn(*args, **kwargs)
                    if capture_output and hasattr(span, "update"):
                        span.update(output=scrub_sensitive_data(result))
                    return result

            return async_wrapper
        else:

            @functools.wraps(fn)
            def sync_wrapper(*args: Any, **kwargs: Any) -> Any:
                if not is_langfuse_enabled():
                    return fn(*args, **kwargs)

                call_input = None
                if capture_input:
                    call_input = scrub_sensitive_data(kwargs if kwargs else args)

                with trace_span(name=span_name, as_type=as_type, input=call_input) as span:
                    result = fn(*args, **kwargs)
                    if capture_output and hasattr(span, "update"):
                        span.update(output=scrub_sensitive_data(result))
                    return result

            return sync_wrapper

    return decorator


def update_current_generation(
    *,
    model: str | None = None,
    input: Any = None,
    output: Any = None,
    usage_details: dict[str, int] | None = None,
    metadata: Any = None,
    level: Literal["DEBUG", "DEFAULT", "WARNING", "ERROR"] | None = None,
    status_message: str | None = None,
) -> None:
    """Update active generation observation with model, tokens, and outputs."""
    client = get_langfuse_client()
    if client is None:
        return
    try:
        kwargs: dict[str, Any] = {}
        if model is not None:
            kwargs["model"] = model
        if input is not None:
            kwargs["input"] = scrub_sensitive_data(input)
        if output is not None:
            kwargs["output"] = scrub_sensitive_data(output)
        if usage_details is not None:
            kwargs["usage_details"] = usage_details
        if metadata is not None:
            kwargs["metadata"] = scrub_sensitive_data(metadata)
        if level is not None:
            kwargs["level"] = level
        if status_message is not None:
            kwargs["status_message"] = scrub_sensitive_data(status_message)

        client.update_current_generation(**kwargs)
    except Exception as exc:
        logger.debug("Failed to update current generation (%s).", type(exc).__name__)


def update_current_span(
    *,
    name: str | None = None,
    input: Any = None,
    output: Any = None,
    metadata: Any = None,
    level: Literal["DEBUG", "DEFAULT", "WARNING", "ERROR"] | None = None,
    status_message: str | None = None,
) -> None:
    """Update active span observation with outputs and metadata."""
    client = get_langfuse_client()
    if client is None:
        return
    try:
        kwargs: dict[str, Any] = {}
        if name is not None:
            kwargs["name"] = name
        if input is not None:
            kwargs["input"] = scrub_sensitive_data(input)
        if output is not None:
            kwargs["output"] = scrub_sensitive_data(output)
        if metadata is not None:
            kwargs["metadata"] = scrub_sensitive_data(metadata)
        if level is not None:
            kwargs["level"] = level
        if status_message is not None:
            kwargs["status_message"] = scrub_sensitive_data(status_message)

        client.update_current_span(**kwargs)
    except Exception as exc:
        logger.debug("Failed to update current span (%s).", type(exc).__name__)


def record_score(
    *,
    name: str,
    value: float | str,
    comment: str | None = None,
    metadata: Any = None,
    trace_id: str | None = None,
    observation_id: str | None = None,
    session_id: str | None = None,
    dataset_run_id: str | None = None,
    data_type: Literal["NUMERIC", "CATEGORICAL", "BOOLEAN", "TEXT", "CORRECTION"] | None = None,
) -> None:
    """
    Record an evaluation score to Langfuse.
    Automatically resolves the active trace_id from OpenTelemetry context or generates
    a valid trace identifier if not provided.
    """
    client = get_langfuse_client()
    if client is None:
        return
    try:
        resolved_trace_id = trace_id
        if not resolved_trace_id and not session_id and not dataset_run_id:
            resolved_trace_id = client.get_current_trace_id() or client.create_trace_id()

        resolved_data_type = data_type
        if resolved_data_type is None:
            if isinstance(value, bool):
                resolved_data_type = "BOOLEAN"
            elif isinstance(value, (int, float)):
                resolved_data_type = "NUMERIC"
            elif isinstance(value, str):
                resolved_data_type = "CATEGORICAL"

        kwargs: dict[str, Any] = {
            "name": name,
            "value": value,
        }
        if resolved_trace_id is not None:
            kwargs["trace_id"] = resolved_trace_id
        if observation_id is not None:
            kwargs["observation_id"] = observation_id
        if session_id is not None:
            kwargs["session_id"] = session_id
        if dataset_run_id is not None:
            kwargs["dataset_run_id"] = dataset_run_id
        if resolved_data_type is not None:
            kwargs["data_type"] = resolved_data_type
        if comment is not None:
            kwargs["comment"] = scrub_sensitive_data(comment)
        if metadata is not None:
            kwargs["metadata"] = scrub_sensitive_data(metadata)

        client.create_score(**kwargs)
    except Exception as exc:
        logger.debug("Failed to record score %s (%s).", name, type(exc).__name__)


def flush_tracing() -> None:
    """Flush pending events to Langfuse in non-blocking fashion."""
    client = get_langfuse_client()
    if client is None:
        return
    try:
        client.flush()
    except Exception as exc:
        logger.debug("Langfuse flush failed (%s).", type(exc).__name__)


def shutdown_tracing() -> None:
    """Shutdown Langfuse client gracefully."""
    client = get_langfuse_client()
    if client is None:
        return
    try:
        client.shutdown()
    except Exception as exc:
        logger.debug("Langfuse shutdown failed (%s).", type(exc).__name__)
