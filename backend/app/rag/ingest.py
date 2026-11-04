"""
Document Ingestion with Gemini Embedding-001.
Reused and adapted from DEV_AI (packages/knowledge/ingest.py).
Integrates multi-tenant isolation, document access roles, and batch embedding.
"""

from __future__ import annotations

import os
import uuid
from pathlib import Path
from typing import Any
import psycopg
from google import genai
from google.genai import types

from app.core.config import settings
from app.db.connection import get_db_connection
from app.rag.chunker import DocumentChunk, chunk_document_text
from app.rag.parsers import parse_document


def generate_batch_embeddings(
    texts: list[str],
    client: genai.Client | None = None,
    batch_size: int = 30,
) -> list[list[float]]:
    """
    Batch-embed texts using the configured Gemini embedding model and dimensions.
    Exact implementation from DEV_AI (packages/knowledge/ingest.py).
    """
    if not texts:
        return []

    if client is None:
        client = genai.Client()

    import re
    import time

    embeddings: list[list[float]] = []

    for i in range(0, len(texts), batch_size):
        batch = texts[i : i + batch_size]
        response = None
        for attempt in range(6):
            try:
                response = client.models.embed_content(
                    model=settings.embedding_model,
                    contents=batch,
                    config=types.EmbedContentConfig(
                        output_dimensionality=settings.embedding_dim,
                        task_type="RETRIEVAL_DOCUMENT",
                    ),
                )
                break
            except Exception as e:
                err_str = str(e)
                if ("429" in err_str or "RESOURCE_EXHAUSTED" in err_str) and attempt < 5:
                    delay = 10.0 * (attempt + 1)
                    match = re.search(r"retry in (\d+(?:\.\d+)?)s", err_str)
                    if match:
                        delay = float(match.group(1)) + 1.0
                    else:
                        delay_match = re.search(r"'retryDelay':\s*'(\d+)s'", err_str)
                        if delay_match:
                            delay = float(delay_match.group(1)) + 1.0
                    print(f"⏳ [Quota Backoff] Rate limit reached. Sleeping {delay:.1f}s before retry {attempt + 1}/5...")
                    time.sleep(delay)
                    continue

                if attempt == 5:
                    raise
                time.sleep(2.0 * (attempt + 1))

        if not response or not response.embeddings:
            raise RuntimeError(
                f"No embeddings returned for batch starting at index {i}"
            )

        for item in response.embeddings:
            if not item.values:
                raise RuntimeError("Empty embedding values returned from Gemini API")
            embeddings.append(list(item.values))

    return embeddings


async def ingest_document_file(
    file_path: str | Path,
    tenant_id: str,
    title: str,
    access_roles: list[str],
    client: genai.Client | None = None,
    batch_size: int = 30,
    document_id: str | None = None,
) -> str | None:
    """
    Ingest a local document file into PostgreSQL with multi-tenant and role metadata.
    Guarantees atomic ingestion: parsing and embeddings are completed BEFORE
    persisting records to PostgreSQL, preventing orphaned document rows on failure.
    """
    path = Path(file_path)
    source_type = path.suffix.lstrip(".").lower() or "txt"
    doc_id = document_id or str(uuid.uuid4())

    # 1. Parse into structured DocumentChunk records first
    chunks: list[DocumentChunk] = parse_document(path, document_id=doc_id, title=title)
    if not chunks:
        return None

    # 2. Generate batch embeddings for search_text before opening database connection
    search_texts = [c.search_text for c in chunks]
    embeddings = generate_batch_embeddings(
        search_texts, client=client, batch_size=batch_size
    )

    # 3. Persist document and chunks atomically within a single database transaction
    async with get_db_connection() as conn:
        async with conn.cursor() as cur:
            await cur.execute(
                """
                INSERT INTO documents (id, tenant_id, title, source_type, access_roles)
                VALUES (%s, %s, %s, %s, %s)
                ON CONFLICT (id) DO UPDATE SET
                    title = EXCLUDED.title,
                    source_type = EXCLUDED.source_type,
                    access_roles = EXCLUDED.access_roles;
                """,
                (doc_id, tenant_id, title, source_type, access_roles),
            )

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
                        doc_id,
                        tenant_id,
                        access_roles,
                        chunk.heading_path,
                        chunk.content,
                        chunk.token_count,
                        str(emb),
                    ),
                )
            await conn.commit()
            return doc_id


async def ingest_document_text(
    content: str,
    tenant_id: str,
    title: str,
    source_type: str,
    access_roles: list[str],
    document_id: str | None = None,
    client: genai.Client | None = None,
    batch_size: int = 30,
) -> str:
    """Ingest raw text/markdown into PostgreSQL with multi-tenant and role metadata."""
    async with get_db_connection() as conn:
        async with conn.cursor() as cur:
            if document_id:
                await cur.execute(
                    """
                    INSERT INTO documents (id, tenant_id, title, source_type, access_roles)
                    VALUES (%s, %s, %s, %s, %s)
                    ON CONFLICT (id) DO UPDATE SET
                        title = EXCLUDED.title,
                        source_type = EXCLUDED.source_type,
                        access_roles = EXCLUDED.access_roles
                    RETURNING id;
                    """,
                    (document_id, tenant_id, title, source_type, access_roles),
                )
                doc_id = str((await cur.fetchone())[0])
            else:
                await cur.execute(
                    """
                    INSERT INTO documents (tenant_id, title, source_type, access_roles)
                    VALUES (%s, %s, %s, %s)
                    RETURNING id;
                    """,
                    (tenant_id, title, source_type, access_roles),
                )
                doc_id = str((await cur.fetchone())[0])

            chunks: list[DocumentChunk] = chunk_document_text(
                content, document_id=doc_id, title=title
            )
            if not chunks:
                await conn.commit()
                return doc_id

            search_texts = [c.search_text for c in chunks]
            embeddings = generate_batch_embeddings(
                search_texts, client=client, batch_size=batch_size
            )

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
                        doc_id,
                        tenant_id,
                        access_roles,
                        chunk.heading_path,
                        chunk.content,
                        chunk.token_count,
                        str(emb),
                    ),
                )
            await conn.commit()
            return doc_id

