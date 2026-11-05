"""
Ingestion script for the primary EnterpriseRAG-Bench dataset (/Users/falahi/datasets-test).
Uses the source-aware chunker and local Qwen3 embeddings with tenant isolation.

Schema Mapping:
- doc_id -> deterministic UUID (uuid.uuid5(uuid.NAMESPACE_OID, doc_id))
- title -> documents.title
- source -> documents.source_type
- allowed_roles -> documents.access_roles & document_chunks.access_roles
- text -> parsed & chunked via chunk_document_text (headings, code-fences, breadcrumbs)
- tenant_name -> "redwood_inference" (deterministic UUID)
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import sys
import uuid
from pathlib import Path
from typing import Any

# Ensure backend root is on sys.path
BACKEND_DIR = Path(__file__).resolve().parent.parent
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

from app.core.config import settings
from app.db.connection import close_db_pool, get_db_connection, init_db_pool
from app.rag.chunker import DocumentChunk, chunk_document_text
from app.rag.embeddings import EmbeddingServiceError
from app.rag.ingest import generate_batch_embeddings

DEFAULT_CORPUS_PATH = "/Users/falahi/datasets-test/data/corpus.jsonl"
DEFAULT_TENANT_NAME = "redwood_inference"


def doc_id_to_uuid(doc_id: str) -> str:
    """Map string doc_id (e.g. dsid_...) deterministically to a valid UUID string."""
    return str(uuid.uuid5(uuid.NAMESPACE_OID, doc_id))


def preflight_corpus(
    path: Path, target_tenant_id: str
) -> tuple[int, int, str, set[str]]:
    content = path.read_bytes()
    fingerprint = hashlib.sha256(content).hexdigest()
    records = 0
    expected_chunks = 0
    seen_ids: set[str] = set()

    for line_number, line in enumerate(content.splitlines(), start=1):
        if not line.strip():
            continue
        try:
            record = json.loads(line)
        except json.JSONDecodeError as error:
            raise ValueError(f"{path}:{line_number}: invalid JSON") from error
        if not isinstance(record, dict):
            raise ValueError(f"{path}:{line_number}: expected a JSON object")

        raw_doc_id = record.get("doc_id")
        text = record.get("text")
        if not isinstance(raw_doc_id, str) or not raw_doc_id.strip():
            raise ValueError(f"{path}:{line_number}: doc_id must be non-empty text")
        if raw_doc_id in seen_ids:
            raise ValueError(f"{path}:{line_number}: duplicate doc_id {raw_doc_id}")
        if not isinstance(text, str) or not text.strip():
            raise ValueError(f"{path}:{line_number}: text must be non-empty")
        seen_ids.add(raw_doc_id)

        raw_tenant = record.get("tenant_id")
        if raw_tenant:
            try:
                record_tenant_id = str(uuid.UUID(str(raw_tenant)))
            except ValueError:
                record_tenant_id = str(uuid.uuid5(uuid.NAMESPACE_OID, str(raw_tenant)))
        else:
            record_tenant_id = target_tenant_id
        if record_tenant_id != target_tenant_id:
            raise ValueError(
                f"{path}:{line_number}: record tenant does not match the requested tenant"
            )

        doc_uuid = doc_id_to_uuid(raw_doc_id)
        chunks = chunk_document_text(
            text=text,
            document_id=doc_uuid,
            title=record.get("title") or f"Doc {raw_doc_id}",
            source_type=record.get("source", "confluence"),
        )
        if not chunks:
            raise ValueError(
                f"{path}:{line_number}: non-empty document produced no chunks"
            )
        records += 1
        expected_chunks += len(chunks)

    if records == 0:
        raise ValueError(f"{path}: no corpus records found")
    return records, expected_chunks, fingerprint, {
        doc_id_to_uuid(doc_id) for doc_id in seen_ids
    }


async def ensure_tenant(tenant_name: str) -> str:
    """Resolve the target tenant without changing tenant data."""
    tenant_uuid = str(uuid.uuid5(uuid.NAMESPACE_OID, tenant_name))
    async with get_db_connection() as conn:
        async with conn.cursor() as cur:
            await cur.execute(
                "SELECT id FROM tenants WHERE id = %s AND name = %s;",
                (tenant_uuid, tenant_name),
            )
            row = await cur.fetchone()
            if not row:
                raise RuntimeError(f"Target tenant {tenant_name!r} does not exist")
            return str(row[0])


async def require_staging_schema(cursor: Any) -> None:
    await cursor.execute(
        """
        SELECT format_type(a.atttypid, a.atttypmod)
        FROM pg_attribute a
        JOIN pg_class c ON c.oid = a.attrelid
        JOIN pg_namespace n ON n.oid = c.relnamespace
        WHERE n.nspname = 'public'
          AND c.relname = 'document_chunks'
          AND a.attname = 'embedding_qwen'
          AND NOT a.attisdropped;
        """
    )
    row = await cursor.fetchone()
    if not row or row[0] != "vector(1024)":
        actual = row[0] if row else "missing"
        raise RuntimeError(
            "Staging schema is not ready: expected "
            f"public.document_chunks.embedding_qwen vector(1024), found {actual}"
        )


async def ingest_corpus(
    corpus_path: str | Path,
    tenant_name: str = DEFAULT_TENANT_NAME,
    limit: int | None = None,
    batch_size: int = 30,
    dry_run: bool = False,
) -> dict[str, Any]:
    path = Path(corpus_path)
    if not path.exists():
        raise FileNotFoundError(f"Corpus file not found: {path}")
    if limit is not None and limit < 1:
        raise ValueError("limit must be greater than zero")
    if batch_size < 1:
        raise ValueError("batch_size must be greater than zero")

    tenant_id = str(uuid.uuid5(uuid.NAMESPACE_OID, tenant_name))
    (
        total_corpus_records,
        expected_chunks,
        corpus_fingerprint,
        corpus_document_ids,
    ) = preflight_corpus(path, tenant_id)

    print(f"=== Enterprise Chatbot Corpus Ingestion ===")
    print(f"Source file:  {path}")
    print(f"Tenant:       {tenant_name}")
    print(f"Limit:        {limit or 'All'}")
    print(f"Batch size:   {batch_size}")
    print(f"Dry run:      {dry_run}")
    print(f"Corpus records: {total_corpus_records}; expected chunks: {expected_chunks}")
    print("===========================================")

    docs_processed = 0
    successful_docs = 0
    failed_docs = 0
    total_chunks = 0
    skipped_docs = 0
    duplicate_records = 0
    failed_embeddings = 0
    records_seen = 0
    total_db_chunks = 0
    staged_chunk_count = 0
    invalid_dimensions = 0
    unembedded_chunk_count = 0
    target_document_count = 0
    target_chunk_count = 0
    completed_checkpoint_count = 0

    try:
        if not dry_run:
            await init_db_pool()
            tenant_id = await ensure_tenant(tenant_name)
            async with get_db_connection() as conn:
                async with conn.cursor() as cur:
                    await require_staging_schema(cur)
                    await cur.execute(
                        """
                        SELECT id, tenant_id
                        FROM public.documents
                        WHERE id = ANY(%s::uuid[]) OR tenant_id = %s;
                        """,
                        (list(corpus_document_ids), tenant_id),
                    )
                    existing_documents = await cur.fetchall()
                    existing_ids = {str(row[0]) for row in existing_documents}
                    conflicting_tenants = [
                        str(row[0])
                        for row in existing_documents
                        if str(row[0]) in corpus_document_ids
                        and str(row[1]) != tenant_id
                    ]
                    extra_documents = existing_ids - corpus_document_ids
                    if conflicting_tenants or extra_documents:
                        raise RuntimeError(
                            "Existing document IDs do not match the target corpus tenant; "
                            "refusing to alter cross-tenant or out-of-corpus data"
                        )
                    await cur.execute(
                        """
                        SELECT count(*)
                        FROM public.document_chunks dc
                        JOIN public.documents d ON d.id = dc.document_id
                        WHERE d.tenant_id = %s AND dc.tenant_id <> d.tenant_id;
                        """,
                        (tenant_id,),
                    )
                    if (await cur.fetchone())[0]:
                        raise RuntimeError(
                            "Existing chunk tenant metadata is inconsistent; refusing migration"
                        )

        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                if limit is not None and records_seen >= limit:
                    break

                records_seen += 1

                record = json.loads(line)
                raw_doc_id = record["doc_id"]
                doc_uuid = doc_id_to_uuid(raw_doc_id)
                title = record.get("title") or f"Doc {raw_doc_id}"
                source_type = record.get("source", "confluence")
                allowed_roles = record.get("allowed_roles", ["employee"])
                text = record.get("text", "")

                # 1. Source-aware chunking (Slack, Gmail, Fireflies, Markdown)
                chunks: list[DocumentChunk] = chunk_document_text(
                    text=text,
                    document_id=doc_uuid,
                    title=title,
                    source_type=source_type,
                )

                if not chunks:
                    skipped_docs += 1
                    continue

                total_chunks += len(chunks)
                docs_processed += 1
                chunk_ids = [chunk.chunk_id for chunk in chunks]
                source_hash = hashlib.sha256(
                    json.dumps(record, sort_keys=True, separators=(",", ":")).encode(
                        "utf-8"
                    )
                ).hexdigest()

                if dry_run:
                    if docs_processed <= 5 or docs_processed % 50 == 0:
                        print(
                            f"[DRY-RUN {docs_processed}] {raw_doc_id} -> {doc_uuid} "
                            f"| source: {source_type} | roles: {allowed_roles} | chunks: {len(chunks)}"
                        )
                    continue

                # Resume only when the committed checkpoint and staged vectors agree.
                async with get_db_connection() as conn:
                    async with conn.cursor() as cur:
                        await cur.execute(
                            """
                            SELECT source_hash, embedding_model, embedding_dim,
                                   expected_chunk_count, status
                            FROM private.embedding_migration_checkpoints
                            WHERE tenant_id = %s AND document_id = %s;
                            """,
                            (tenant_id, doc_uuid),
                        )
                        checkpoint = await cur.fetchone()
                        if (
                            checkpoint
                            and checkpoint[0] == source_hash
                            and checkpoint[1] == settings.ollama_embedding_model
                            and checkpoint[2] == settings.ollama_embedding_dimensions
                            and checkpoint[3] == len(chunks)
                            and checkpoint[4] == "completed"
                        ):
                            await cur.execute(
                                """
                                SELECT count(*)
                                FROM public.document_chunks
                                WHERE document_id = %s
                                  AND id = ANY(%s)
                                  AND embedding_qwen IS NOT NULL;
                                """,
                                (doc_uuid, chunk_ids),
                            )
                            stored_chunks = (await cur.fetchone())[0]
                            if stored_chunks == len(chunks):
                                successful_docs += 1
                                if successful_docs % 100 == 0:
                                    print(
                                        f"Progress: {successful_docs}/{total_corpus_records} "
                                        "documents complete"
                                    )
                                continue

                        if checkpoint and checkpoint[0] != source_hash:
                            await cur.execute(
                                """
                                UPDATE public.document_chunks
                                SET embedding_qwen = NULL
                                WHERE document_id = %s;
                                """,
                                (doc_uuid,),
                            )
                        await cur.execute(
                            """
                            INSERT INTO private.embedding_migration_checkpoints
                                (tenant_id, document_id, source_hash, embedding_model,
                                 embedding_dim, expected_chunk_count,
                                 completed_chunk_count, status, error_type, updated_at)
                            VALUES (%s, %s, %s, %s, %s, %s, 0, 'in_progress', NULL, NOW())
                            ON CONFLICT (tenant_id, document_id) DO UPDATE SET
                                source_hash = EXCLUDED.source_hash,
                                embedding_model = EXCLUDED.embedding_model,
                                embedding_dim = EXCLUDED.embedding_dim,
                                expected_chunk_count = EXCLUDED.expected_chunk_count,
                                completed_chunk_count = 0,
                                status = 'in_progress',
                                error_type = NULL,
                                updated_at = NOW();
                            """,
                            (
                                tenant_id,
                                doc_uuid,
                                source_hash,
                                settings.ollama_embedding_model,
                                settings.ollama_embedding_dimensions,
                                len(chunks),
                            ),
                        )
                        await conn.commit()

                # 2. Generate and verify local Ollama embeddings.
                search_texts = [c.search_text for c in chunks]
                try:
                    embeddings = generate_batch_embeddings(
                        search_texts, batch_size=batch_size
                    )
                except EmbeddingServiceError as error:
                    status = (
                        f"HTTP {error.status_code}"
                        if error.status_code
                        else type(error).__name__
                    )
                    print(
                        f"Failed embeddings for doc {raw_doc_id} ({title}): {status}"
                    )
                    failed_docs += 1
                    failed_embeddings += len(chunks)
                    async with get_db_connection() as conn:
                        async with conn.cursor() as cur:
                            await cur.execute(
                                """
                                UPDATE private.embedding_migration_checkpoints
                                SET status = 'failed', error_type = %s, updated_at = NOW()
                                WHERE tenant_id = %s AND document_id = %s;
                                """,
                                (type(error).__name__, tenant_id, doc_uuid),
                            )
                            await conn.commit()
                    continue

                # 3. Commit content, staged vectors, and checkpoint atomically.
                async with get_db_connection() as conn:
                    async with conn.cursor() as cur:
                        await require_staging_schema(cur)
                        await cur.execute(
                            """
                            INSERT INTO documents (id, tenant_id, title, source_type, access_roles)
                            VALUES (%s, %s, %s, %s, %s)
                            ON CONFLICT (id) DO NOTHING;
                            """,
                            (doc_uuid, tenant_id, title, source_type, allowed_roles),
                        )
                        await cur.execute(
                            "SELECT tenant_id, access_roles FROM documents WHERE id = %s;",
                            (doc_uuid,),
                        )
                        document_row = await cur.fetchone()
                        if not document_row or str(document_row[0]) != tenant_id:
                            raise RuntimeError(
                                f"Document {raw_doc_id} is not owned by the requested tenant"
                            )
                        persisted_roles = document_row[1]
                        await cur.execute(
                            """
                            SELECT id
                            FROM document_chunks
                            WHERE id = ANY(%s) AND document_id <> %s
                            LIMIT 1;
                            """,
                            (chunk_ids, doc_uuid),
                        )
                        conflicting_chunk = await cur.fetchone()
                        if conflicting_chunk:
                            raise RuntimeError(
                                f"Chunk ID {conflicting_chunk[0]} belongs to another document"
                            )
                        await cur.executemany(
                            """
                            INSERT INTO document_chunks
                                (id, document_id, tenant_id, access_roles, heading_path,
                                 content, token_count, embedding_qwen)
                            VALUES (%s, %s, %s, %s, %s, %s, %s, %s::vector(1024))
                            ON CONFLICT (id) DO UPDATE SET
                                document_id = EXCLUDED.document_id,
                                tenant_id = EXCLUDED.tenant_id,
                                access_roles = EXCLUDED.access_roles,
                                heading_path = EXCLUDED.heading_path,
                                content = EXCLUDED.content,
                                token_count = EXCLUDED.token_count,
                                embedding_qwen = EXCLUDED.embedding_qwen;
                            """,
                            [
                                (
                                    chunk.chunk_id,
                                    doc_uuid,
                                    tenant_id,
                                    persisted_roles,
                                    chunk.heading_path,
                                    chunk.content,
                                    chunk.token_count,
                                    f"[{','.join(map(str, embedding))}]",
                                )
                                for chunk, embedding in zip(
                                    chunks, embeddings, strict=True
                                )
                            ],
                        )
                        await cur.execute(
                            """
                            UPDATE private.embedding_migration_checkpoints
                            SET completed_chunk_count = %s, status = 'completed',
                                error_type = NULL, updated_at = NOW()
                            WHERE tenant_id = %s AND document_id = %s;
                            """,
                            (len(chunks), tenant_id, doc_uuid),
                        )
                        await conn.commit()
                successful_docs += 1

                if successful_docs % 25 == 0:
                    print(
                        f"Progress: {successful_docs}/{total_corpus_records} documents "
                        f"complete ({total_chunks} source chunks)"
                    )

        if not dry_run:
            async with get_db_connection() as conn:
                async with conn.cursor() as cur:
                    await cur.execute(
                        """
                        SELECT count(*), count(embedding_qwen),
                               count(*) FILTER (
                                   WHERE embedding_qwen IS NOT NULL
                                     AND vector_dims(embedding_qwen) <> 1024
                               )
                        FROM public.document_chunks;
                        """
                    )
                    total_db_chunks, staged_chunk_count, invalid_dimensions = (
                        await cur.fetchone()
                    )
                    unembedded_chunk_count = total_db_chunks - staged_chunk_count
                    await cur.execute(
                        """
                        SELECT count(*)
                        FROM public.documents
                        WHERE tenant_id = %s;
                        """,
                        (tenant_id,),
                    )
                    target_document_count = (await cur.fetchone())[0]
                    await cur.execute(
                        """
                        SELECT count(*)
                        FROM public.document_chunks
                        WHERE tenant_id = %s;
                        """,
                        (tenant_id,),
                    )
                    target_chunk_count = (await cur.fetchone())[0]
                    await cur.execute(
                        """
                        SELECT count(*)
                        FROM private.embedding_migration_checkpoints
                        WHERE tenant_id = %s
                          AND embedding_model = %s
                          AND embedding_dim = 1024
                          AND status = 'completed'
                          AND completed_chunk_count = expected_chunk_count;
                        """,
                        (tenant_id, settings.ollama_embedding_model),
                    )
                    completed_checkpoint_count = (await cur.fetchone())[0]

    finally:
        if not dry_run:
            await close_db_pool()

    pending_docs = (
        total_corpus_records
        if dry_run
        else max(0, total_corpus_records - records_seen)
    )
    summary = {
        "docs_processed": docs_processed,
        "successful_docs": successful_docs,
        "failed_docs": failed_docs,
        "total_chunks": total_chunks,
        "skipped_docs": skipped_docs,
        "duplicate_records": duplicate_records,
        "failed_embeddings": failed_embeddings,
        "pending_docs": pending_docs,
        "expected_chunks": expected_chunks,
        "staged_chunks": staged_chunk_count,
        "database_chunks": total_db_chunks,
        "unembedded_chunks": unembedded_chunk_count,
        "invalid_dimensions": invalid_dimensions,
        "target_documents": target_document_count,
        "target_chunks": target_chunk_count,
        "completed_checkpoints": completed_checkpoint_count,
        "corpus_fingerprint": corpus_fingerprint,
        "migration_ready": (
            not dry_run
            and successful_docs == total_corpus_records
            and failed_docs == 0
            and pending_docs == 0
            and unembedded_chunk_count == 0
            and invalid_dimensions == 0
            and target_document_count == total_corpus_records
            and target_chunk_count == expected_chunks
            and completed_checkpoint_count == total_corpus_records
        ),
        "tenant_id": tenant_id,
        "tenant_name": tenant_name,
        "dry_run": dry_run,
    }

    print("\n=== Ingestion Complete ===")
    print(f"Successful Documents: {successful_docs}")
    print(f"Failed Documents:     {failed_docs}")
    print(f"Processed Documents:  {docs_processed}")
    print(f"Generated Chunks:    {total_chunks}")
    print(f"Skipped Documents:   {skipped_docs}")
    print(f"Duplicate Records:   {duplicate_records}")
    print(f"Failed Chunks:       {failed_embeddings}")
    print(f"Pending Documents:    {pending_docs}")
    if not dry_run:
        print(f"Staged Chunks:       {staged_chunk_count}/{total_db_chunks}")
        print(f"Unembedded Chunks:   {unembedded_chunk_count}")
        print(f"Invalid Dimensions:  {invalid_dimensions}")
        print(f"Target Documents:    {target_document_count}/{total_corpus_records}")
        print(f"Target Chunks:       {target_chunk_count}/{expected_chunks}")
        print(f"Completed Checkpoints: {completed_checkpoint_count}/{total_corpus_records}")
        print(f"Migration Ready:     {summary['migration_ready']}")
        print(f"Corpus Fingerprint:  {corpus_fingerprint}")
    print("==========================\n")
    return summary


def main():
    parser = argparse.ArgumentParser(description="Ingest primary EnterpriseRAG corpus into Postgres + pgvector")
    parser.add_argument(
        "--corpus-path",
        type=str,
        default=DEFAULT_CORPUS_PATH,
        help="Path to corpus.jsonl file",
    )
    parser.add_argument(
        "--tenant-name",
        type=str,
        default=DEFAULT_TENANT_NAME,
        help="Tenant name to isolate corpus documents under",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Maximum number of documents to ingest (useful for testing)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=30,
        help="Embedding generation batch size",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Parse and validate chunking without contacting database or embedding API",
    )

    args = parser.parse_args()
    asyncio.run(
        ingest_corpus(
            corpus_path=args.corpus_path,
            tenant_name=args.tenant_name,
            limit=args.limit,
            batch_size=args.batch_size,
            dry_run=args.dry_run,
        )
    )


if __name__ == "__main__":
    main()
