from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

from openai import OpenAI

BACKEND_DIR = Path(__file__).resolve().parents[1]
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

from app.core.config import Settings

MODEL = "nvidia/nemotron-3-embed-1b:free"
OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
MAX_RECORDS = 250
BATCH_SIZE = 125
EXPECTED_DIMENSIONS = 2048


@dataclass(frozen=True)
class CorpusRecord:
    line_number: int
    text: str | None
    error: str | None = None


@dataclass(frozen=True)
class BenchmarkSummary:
    records_read: int
    successful_records: int
    failed_records: int
    prompt_tokens: int
    total_tokens: int
    elapsed_seconds: float


class EmbeddingAPI(Protocol):
    def create(self, *, model: str, input: list[str]) -> Any: ...


class OpenRouterClient(Protocol):
    embeddings: EmbeddingAPI


def read_corpus_records(path: Path, limit: int = MAX_RECORDS) -> list[CorpusRecord]:
    if limit < 0:
        raise ValueError("limit cannot be negative")

    records: list[CorpusRecord] = []
    with path.open(encoding="utf-8") as corpus_file:
        for line_number, line in enumerate(corpus_file, start=1):
            if not line.strip():
                continue
            if len(records) >= limit:
                break

            try:
                value = json.loads(line)
            except json.JSONDecodeError:
                records.append(CorpusRecord(line_number, None, "invalid JSON"))
                continue

            if not isinstance(value, dict):
                records.append(
                    CorpusRecord(line_number, None, "record is not a JSON object")
                )
                continue

            text = value.get("text")
            if not isinstance(text, str) or not text.strip():
                records.append(
                    CorpusRecord(line_number, None, "missing or empty string field 'text'")
                )
                continue

            records.append(CorpusRecord(line_number, text))

    return records


def _reported_token_count(usage: Any, name: str) -> int:
    value = getattr(usage, name, 0) if usage is not None else 0
    return value if type(value) is int and value >= 0 else 0


def _request_error_summary(error: Exception) -> str:
    status_code = getattr(error, "status_code", None)
    response = getattr(error, "response", None)

    print(f"Full error: {error}")

    if response is not None:
        try:
            print(f"Response body: {response.text}")
        except Exception:
            pass

    if type(status_code) is int:
        return f"{type(error).__name__} (HTTP {status_code})"

    return type(error).__name__

def run_benchmark(
    corpus_path: Path,
    client: OpenRouterClient,
    *,
    limit: int = MAX_RECORDS,
    batch_size: int = BATCH_SIZE,
) -> BenchmarkSummary:
    if batch_size <= 0:
        raise ValueError("batch_size must be greater than zero")

    started_at = time.perf_counter()
    records = read_corpus_records(corpus_path, limit=limit)
    successful_records = 0
    failed_records = 0
    prompt_tokens = 0
    total_tokens = 0

    for record in records:
        if record.error:
            failed_records += 1
            print(f"Record at line {record.line_number} failed: {record.error}")

    valid_records = [record for record in records if record.text is not None]
    batches = [
        valid_records[index : index + batch_size]
        for index in range(0, len(valid_records), batch_size)
    ]

    for batch_number, batch in enumerate(batches, start=1):
        batch_texts: list[str] = []
        for record in batch:
            if record.text is None:
                raise RuntimeError("Invalid corpus record reached an embedding batch")
            batch_texts.append(record.text)

        batch_started_at = time.perf_counter()
        try:
            response = client.embeddings.create(
                model=MODEL,
                input=batch_texts,
            )
        except Exception as error:
            latency = time.perf_counter() - batch_started_at
            failed_records += len(batch)
            print(
                f"Batch {batch_number}/{len(batches)}: {latency:.3f}s; "
                f"{len(batch)} records failed ({_request_error_summary(error)})"
            )
            continue

        latency = time.perf_counter() - batch_started_at
        usage = getattr(response, "usage", None)
        batch_prompt_tokens = _reported_token_count(usage, "prompt_tokens")
        batch_total_tokens = _reported_token_count(usage, "total_tokens")
        prompt_tokens += batch_prompt_tokens
        total_tokens += batch_total_tokens

        response_data = getattr(response, "data", None)
        indexed_embeddings: dict[int, Any] = {}
        invalid_indices = not isinstance(response_data, (list, tuple))
        if not invalid_indices:
            for item in response_data:
                index = getattr(item, "index", None)
                if (
                    type(index) is not int
                    or not 0 <= index < len(batch)
                    or index in indexed_embeddings
                ):
                    invalid_indices = True
                    break
                indexed_embeddings[index] = getattr(item, "embedding", None)

        if invalid_indices:
            failed_records += len(batch)
            print(
                f"Batch {batch_number}/{len(batches)}: {latency:.3f}s; "
                f"{len(batch)} records failed (invalid response indices)"
            )
            continue

        batch_successes = 0
        batch_failures = 0
        for index in range(len(batch)):
            embedding = indexed_embeddings.get(index)
            if isinstance(embedding, list) and len(embedding) == EXPECTED_DIMENSIONS:
                batch_successes += 1
            else:
                batch_failures += 1

        successful_records += batch_successes
        failed_records += batch_failures
        print(
            f"Batch {batch_number}/{len(batches)}: {latency:.3f}s; "
            f"{batch_successes} successful, {batch_failures} failed; "
            f"tokens prompt={batch_prompt_tokens}, total={batch_total_tokens}"
        )

    elapsed_seconds = time.perf_counter() - started_at
    summary = BenchmarkSummary(
        records_read=len(records),
        successful_records=successful_records,
        failed_records=failed_records,
        prompt_tokens=prompt_tokens,
        total_tokens=total_tokens,
        elapsed_seconds=elapsed_seconds,
    )
    print("\n=== OpenRouter Embedding Benchmark ===")
    print(f"Model:               {MODEL}")
    print(f"Records read:        {summary.records_read}")
    print(f"Successful records:  {summary.successful_records}")
    print(f"Failed records:      {summary.failed_records}")
    print(f"Prompt tokens:       {summary.prompt_tokens}")
    print(f"Total tokens:        {summary.total_tokens}")
    print(
        "Estimated cost:      $0.00 (model is free; embedding usage does not "
        "include billed cost)"
    )
    print(f"Total execution:     {summary.elapsed_seconds:.3f}s")
    print("======================================")
    return summary


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Benchmark OpenRouter embeddings for a JSONL corpus."
    )
    parser.add_argument(
        "--corpus-path",
        type=Path,
        required=True,
        help="Path to the JSONL corpus file",
    )
    args = parser.parse_args(argv)

    if not args.corpus_path.is_file():
        parser.error(f"corpus file does not exist: {args.corpus_path}")

    settings = Settings(_env_file=BACKEND_DIR / ".env")
    if not settings.OPENROUTER_API_KEY.strip():
        parser.error(
            "OPENROUTER_API_KEY is missing from the environment and backend/.env"
        )

    client = OpenAI(
        api_key=settings.OPENROUTER_API_KEY,
        base_url=OPENROUTER_BASE_URL,
    )
    try:
        summary = run_benchmark(args.corpus_path, client)
    finally:
        client.close()
    return 0 if summary.failed_records == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
