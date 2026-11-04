"""
Ingestion script for the primary EnterpriseRAG-Bench dataset (/Users/falahi/datasets-test).
Reuses DEV_AI's structural chunker, Gemini embedding-001 (768 dimensions), and tenant isolation.

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
import json
import os
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
from app.rag.ingest import generate_batch_embeddings

DEFAULT_CORPUS_PATH = "/Users/falahi/datasets-test/data/corpus.jsonl"
DEFAULT_TENANT_NAME = "redwood_inference"


def doc_id_to_uuid(doc_id: str) -> str:
    """Map string doc_id (e.g. dsid_...) deterministically to a valid UUID string."""
    return str(uuid.uuid5(uuid.NAMESPACE_OID, doc_id))


async def ensure_tenant(tenant_name: str) -> str:
    """Get or create tenant record deterministically."""
    tenant_uuid = str(uuid.uuid5(uuid.NAMESPACE_OID, tenant_name))
    async with get_db_connection() as conn:
        async with conn.cursor() as cur:
            await cur.execute(
                """
                INSERT INTO tenants (id, name)
                VALUES (%s, %s)
                ON CONFLICT (id) DO UPDATE SET name = EXCLUDED.name
                RETURNING id;
                """,
                (tenant_uuid, tenant_name),
            )
            row = await cur.fetchone()
            await conn.commit()
            return str(row[0])


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

    print(f"=== Enterprise Chatbot Corpus Ingestion ===")
    print(f"Source file:  {path}")
    print(f"Tenant:       {tenant_name}")
    print(f"Limit:        {limit or 'All'}")
    print(f"Batch size:   {batch_size}")
    print(f"Dry run:      {dry_run}")
    print("===========================================")

    if not dry_run:
        await init_db_pool()
        tenant_id = await ensure_tenant(tenant_name)
    else:
        tenant_id = str(uuid.uuid5(uuid.NAMESPACE_OID, tenant_name))

    docs_processed = 0
    total_chunks = 0
    skipped_docs = 0
    duplicate_records = 0
    failed_embeddings = 0
    seen_raw_ids: set[str] = set()

    try:
        with open(path, "r", encoding="utf-8") as f:
            for line_idx, line in enumerate(f, start=1):
                if limit and docs_processed >= limit:
                    break

                line = line.strip()
                if not line:
                    continue

                record = json.loads(line)
                raw_doc_id = record["doc_id"]

                # Check for duplicate records
                if raw_doc_id in seen_raw_ids:
                    duplicate_records += 1
                else:
                    seen_raw_ids.add(raw_doc_id)

                doc_uuid = doc_id_to_uuid(raw_doc_id)
                title = record.get("title") or f"Doc {raw_doc_id}"
                source_type = record.get("source", "confluence")
                allowed_roles = record.get("allowed_roles", ["employee"])
                text = record.get("text", "")

                # Preserve tenant ID if specified in record, otherwise use default
                raw_tenant = record.get("tenant_id")
                if raw_tenant:
                    try:
                        current_tenant_id = str(uuid.UUID(str(raw_tenant)))
                    except ValueError:
                        current_tenant_id = str(uuid.uuid5(uuid.NAMESPACE_OID, str(raw_tenant)))
                    if not dry_run:
                        await ensure_tenant(str(raw_tenant))
                else:
                    current_tenant_id = tenant_id

                # 1. Structural chunking (code fence preserved, 600 words sub-chunks, breadcrumbs)
                chunks: list[DocumentChunk] = chunk_document_text(
                    text=text,
                    document_id=doc_uuid,
                    title=title,
                )

                if not chunks:
                    skipped_docs += 1
                    continue

                total_chunks += len(chunks)
                docs_processed += 1

                if dry_run:
                    if docs_processed <= 5 or docs_processed % 50 == 0:
                        print(
                            f"[DRY-RUN {docs_processed}] {raw_doc_id} -> {doc_uuid} "
                            f"| source: {source_type} | roles: {allowed_roles} | chunks: {len(chunks)}"
                        )
                    continue

                # 2. Batch embeddings via Gemini embedding-001 (768 dimensions)
                search_texts = [c.search_text for c in chunks]
                try:
                    embeddings = generate_batch_embeddings(
                        search_texts, batch_size=batch_size
                    )
                except Exception as e:
                    print(f"🚨 Failed embeddings for doc {raw_doc_id} ({title}): {e}")
                    failed_embeddings += len(chunks)
                    continue

                # 3. Database upsert
                async with get_db_connection() as conn:
                    async with conn.cursor() as cur:
                        # Upsert document
                        await cur.execute(
                            """
                            INSERT INTO documents (id, tenant_id, title, source_type, access_roles)
                            VALUES (%s, %s, %s, %s, %s)
                            ON CONFLICT (id) DO UPDATE SET
                                title = EXCLUDED.title,
                                source_type = EXCLUDED.source_type,
                                access_roles = EXCLUDED.access_roles;
                            """,
                            (doc_uuid, current_tenant_id, title, source_type, allowed_roles),
                        )

                        # Upsert chunks
                        for chunk, emb in zip(chunks, embeddings, strict=True):
                            await cur.execute(
                                """
                                INSERT INTO document_chunks 
                                (id, document_id, tenant_id, access_roles, heading_path, content, token_count, embedding)
                                VALUES (%s, %s, %s, %s, %s, %s, %s, %s::vector(768))
                                ON CONFLICT (id) DO UPDATE SET
                                    heading_path = EXCLUDED.heading_path,
                                    content = EXCLUDED.content,
                                    token_count = EXCLUDED.token_count,
                                    embedding = EXCLUDED.embedding,
                                    access_roles = EXCLUDED.access_roles;
                                """,
                                (
                                    chunk.chunk_id,
                                    doc_uuid,
                                    current_tenant_id,
                                    allowed_roles,
                                    chunk.heading_path,
                                    chunk.content,
                                    chunk.token_count,
                                    str(emb),
                                ),
                            )
                        await conn.commit()

                if docs_processed % 25 == 0 or docs_processed == limit:
                    print(
                        f"[{docs_processed}] Ingested {title[:40]}... "
                        f"({len(chunks)} chunks, cumulative: {total_chunks})"
                    )

    finally:
        if not dry_run:
            await close_db_pool()

    summary = {
        "docs_processed": docs_processed,
        "total_chunks": total_chunks,
        "skipped_docs": skipped_docs,
        "duplicate_records": duplicate_records,
        "failed_embeddings": failed_embeddings,
        "tenant_id": tenant_id,
        "tenant_name": tenant_name,
        "dry_run": dry_run,
    }

    print("\n=== Ingestion Complete ===")
    print(f"Processed Documents: {docs_processed}")
    print(f"Generated Chunks:    {total_chunks}")
    print(f"Skipped Documents:   {skipped_docs}")
    print(f"Duplicate Records:   {duplicate_records}")
    print(f"Failed Embeddings:   {failed_embeddings}")
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
        help="Parse and validate chunking without contacting database or Gemini API",
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
