from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

BACKEND_DIR = Path(__file__).resolve().parents[1]
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

from app.core.config import settings
from app.rag.embeddings import EmbeddingServiceError, embed_texts

MAX_RECORDS = 250
BATCH_SIZE = 32
EXPECTED_DIMENSIONS = 1024


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
    elapsed_seconds: float


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
                records.append(CorpusRecord(line_number, None, "record is not a JSON object"))
                continue
            text = value.get("text")
            if not isinstance(text, str) or not text.strip():
                records.append(
                    CorpusRecord(line_number, None, "missing or empty string field 'text'")
                )
                continue
            records.append(CorpusRecord(line_number, text))
    return records


def run_benchmark(
    corpus_path: Path,
    embed: Callable[..., list[list[float]]] = embed_texts,
    *,
    limit: int = MAX_RECORDS,
    batch_size: int = BATCH_SIZE,
) -> BenchmarkSummary:
    if batch_size <= 0:
        raise ValueError("batch_size must be greater than zero")

    started_at = time.perf_counter()
    records = read_corpus_records(corpus_path, limit=limit)
    successful_records = 0
    failed_records = sum(record.error is not None for record in records)
    for record in records:
        if record.error:
            print(f"Record at line {record.line_number} failed: {record.error}")

    valid_records = [record for record in records if record.text is not None]
    batches = [
        valid_records[index : index + batch_size]
        for index in range(0, len(valid_records), batch_size)
    ]
    for batch_number, batch in enumerate(batches, start=1):
        texts = [record.text for record in batch]
        batch_started_at = time.perf_counter()
        try:
            vectors = embed(texts, batch_size=len(texts))
            if len(vectors) != len(batch) or any(
                len(vector) != EXPECTED_DIMENSIONS for vector in vectors
            ):
                raise EmbeddingServiceError(
                    "Embedding response did not contain exactly 1024 dimensions per text"
                )
        except EmbeddingServiceError as error:
            failed_records += len(batch)
            status = f"HTTP {error.status_code}" if error.status_code else type(error).__name__
            print(
                f"Batch {batch_number}/{len(batches)}: "
                f"{time.perf_counter() - batch_started_at:.3f}s; "
                f"{len(batch)} records failed ({status})"
            )
            continue

        successful_records += len(batch)
        print(
            f"Batch {batch_number}/{len(batches)}: "
            f"{time.perf_counter() - batch_started_at:.3f}s; "
            f"{len(batch)} records embedded"
        )

    summary = BenchmarkSummary(
        records_read=len(records),
        successful_records=successful_records,
        failed_records=failed_records,
        elapsed_seconds=time.perf_counter() - started_at,
    )
    print("\n=== Local Ollama Embedding Benchmark ===")
    print(f"Model:               {settings.ollama_embedding_model}")
    print(f"Dimensions:          {EXPECTED_DIMENSIONS}")
    print(f"Records read:        {summary.records_read}")
    print(f"Successful records:  {summary.successful_records}")
    print(f"Failed records:      {summary.failed_records}")
    print(f"Total execution:     {summary.elapsed_seconds:.3f}s")
    print("========================================")
    return summary


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Benchmark local Ollama embeddings for a JSONL corpus."
    )
    parser.add_argument("--corpus-path", type=Path, required=True)
    args = parser.parse_args(argv)
    if not args.corpus_path.is_file():
        parser.error(f"corpus file does not exist: {args.corpus_path}")
    summary = run_benchmark(args.corpus_path)
    return 0 if summary.failed_records == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
