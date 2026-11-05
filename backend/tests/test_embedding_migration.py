from __future__ import annotations

import asyncio
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import AsyncMock

BACKEND_DIR = Path(__file__).resolve().parent.parent
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

from scripts.ingest_benchmark import (
    doc_id_to_uuid,
    preflight_corpus,
    require_staging_schema,
)
from app.rag.embeddings import require_active_embedding_schema


class TestEmbeddingMigrationPreflight(unittest.TestCase):
    def _write_jsonl(self, records: list[dict]) -> Path:
        file = tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", delete=False)
        with file:
            for record in records:
                file.write(json.dumps(record) + "\n")
        path = Path(file.name)
        self.addCleanup(path.unlink, missing_ok=True)
        return path

    def test_preflight_returns_expected_chunks_and_stable_document_ids(self):
        path = self._write_jsonl(
            [
                {"doc_id": "one", "text": "first document"},
                {"doc_id": "two", "text": "second document"},
            ]
        )
        count, chunks, fingerprint, document_ids = preflight_corpus(
            path, "tenant-id"
        )
        self.assertEqual(count, 2)
        self.assertEqual(chunks, 2)
        self.assertEqual(len(fingerprint), 64)
        self.assertEqual(document_ids, {doc_id_to_uuid("one"), doc_id_to_uuid("two")})

    def test_preflight_rejects_duplicate_document_ids_before_writes(self):
        path = self._write_jsonl(
            [
                {"doc_id": "duplicate", "text": "first"},
                {"doc_id": "duplicate", "text": "second"},
            ]
        )
        with self.assertRaisesRegex(ValueError, "duplicate doc_id"):
            preflight_corpus(path, "tenant-id")


class TestStagingSchemaGuard(unittest.TestCase):
    def test_allows_only_vector_1024_staging_column(self):
        cursor = AsyncMock()
        cursor.fetchone = AsyncMock(return_value=("vector(1024)",))
        asyncio.run(require_staging_schema(cursor))
        cursor.execute.assert_awaited_once()

    def test_refuses_missing_or_wrong_dimension_staging_column(self):
        for result in (None, ("halfvec(2048)",)):
            with self.subTest(result=result):
                cursor = AsyncMock()
                cursor.fetchone = AsyncMock(return_value=result)
                with self.assertRaisesRegex(RuntimeError, "Staging schema is not ready"):
                    asyncio.run(require_staging_schema(cursor))


class TestActiveSchemaGuard(unittest.TestCase):
    def test_blocks_retrieval_until_active_column_is_vector_1024(self):
        cursor = AsyncMock()
        cursor.fetchone = AsyncMock(return_value=("halfvec(2048)",))
        with self.assertRaisesRegex(RuntimeError, "Embedding migration is not active"):
            asyncio.run(require_active_embedding_schema(cursor))

    def test_allows_active_vector_1024_column(self):
        cursor = AsyncMock()
        cursor.fetchone = AsyncMock(return_value=("vector(1024)",))
        asyncio.run(require_active_embedding_schema(cursor))


if __name__ == "__main__":
    unittest.main()
