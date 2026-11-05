"""Document ingestion with local Qwen3 embeddings and tenant/role metadata."""

from __future__ import annotations

import uuid
from pathlib import Path

from app.core.config import settings
from app.db.connection import get_db_connection
from app.rag.chunker import DocumentChunk, chunk_document_text
from app.rag.embeddings import embed_texts, require_active_embedding_schema
from app.rag.parsers import parse_document


def generate_batch_embeddings(
    texts: list[str],
    batch_size: int = 32,
) -> list[list[float]]:
    return embed_texts(texts, batch_size=batch_size)


async def ingest_document_file(
    file_path: str | Path,
    tenant_id: str,
    title: str,
    access_roles: list[str],
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
    embeddings = generate_batch_embeddings(search_texts, batch_size=batch_size)

    # 3. Persist document and chunks atomically within a single database transaction
    async with get_db_connection() as conn:
        async with conn.cursor() as cur:
            await require_active_embedding_schema(cur)
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

            # Transactionally delete existing chunks for this document to prevent duplicate/orphaned chunks upon re-ingestion
            await cur.execute("DELETE FROM document_chunks WHERE document_id = %s;", (doc_id,))

            for chunk, emb in zip(chunks, embeddings, strict=True):
                await cur.execute(
                    f"""
                    INSERT INTO document_chunks 
                    (id, document_id, tenant_id, access_roles, heading_path, content, token_count, embedding)
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s::vector({settings.ollama_embedding_dimensions}))
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
    batch_size: int = 30,
) -> str:
    """Ingest raw text/markdown into PostgreSQL with multi-tenant and role metadata."""
    doc_id = document_id or str(uuid.uuid4())
    chunks: list[DocumentChunk] = chunk_document_text(
        content, document_id=doc_id, title=title, source_type=source_type
    )
    embeddings = generate_batch_embeddings(
        [chunk.search_text for chunk in chunks], batch_size=batch_size
    )

    async with get_db_connection() as conn:
        async with conn.cursor() as cur:
            await require_active_embedding_schema(cur)
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
            await cur.execute("DELETE FROM document_chunks WHERE document_id = %s;", (doc_id,))

            for chunk, emb in zip(chunks, embeddings, strict=True):
                await cur.execute(
                    f"""
                    INSERT INTO document_chunks 
                    (id, document_id, tenant_id, access_roles, heading_path, content, token_count, embedding)
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s::vector({settings.ollama_embedding_dimensions}))
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
