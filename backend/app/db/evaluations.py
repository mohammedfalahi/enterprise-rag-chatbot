from __future__ import annotations

from datetime import datetime
from typing import Any

from psycopg.types.json import Jsonb

from app.db.connection import get_db_connection


_RUN_COLUMNS = """
    id, tenant_id, requested_by, status, sample_size, mode, top_k,
    candidate_limit, rerank_pool_size, error_message, created_at,
    started_at, completed_at
"""


def _serialize_run(row: tuple[Any, ...], report: dict[str, Any] | None = None) -> dict[str, Any]:
    result = {
        "id": str(row[0]),
        "tenant_id": str(row[1]),
        "requested_by": str(row[2]),
        "status": row[3],
        "sample_size": row[4],
        "mode": row[5],
        "top_k": row[6],
        "candidate_limit": row[7],
        "rerank_pool_size": row[8],
        "error_message": row[9],
        "created_at": row[10].isoformat() if isinstance(row[10], datetime) else str(row[10]),
        "started_at": row[11].isoformat() if row[11] else None,
        "completed_at": row[12].isoformat() if row[12] else None,
    }
    if report is not None:
        result["report"] = report
    return result


async def create_evaluation_run(
    *,
    tenant_id: str,
    requested_by: str,
    sample_size: int,
    mode: str,
    top_k: int,
    candidate_limit: int,
    rerank_pool_size: int,
) -> tuple[dict[str, Any], bool]:
    async with get_db_connection() as conn:
        async with conn.cursor() as cur:
            await cur.execute(
                f"""
                INSERT INTO public.evaluation_runs (
                    tenant_id, requested_by, status, sample_size, mode, top_k,
                    candidate_limit, rerank_pool_size
                )
                VALUES (%s, %s, 'pending', %s, %s, %s, %s, %s)
                ON CONFLICT (tenant_id) WHERE status IN ('pending', 'running')
                DO NOTHING
                RETURNING {_RUN_COLUMNS};
                """,
                (
                    tenant_id,
                    requested_by,
                    sample_size,
                    mode,
                    top_k,
                    candidate_limit,
                    rerank_pool_size,
                ),
            )
            row = await cur.fetchone()
            created = row is not None
            if row is None:
                await cur.execute(
                    f"""
                    SELECT {_RUN_COLUMNS}
                    FROM public.evaluation_runs
                    WHERE tenant_id = %s AND status IN ('pending', 'running')
                    ORDER BY created_at DESC
                    LIMIT 1;
                    """,
                    (tenant_id,),
                )
                row = await cur.fetchone()
            if row is None:
                raise RuntimeError("Could not create or retrieve the active evaluation run.")
            await conn.commit()
            return _serialize_run(row), created


async def update_evaluation_run(
    *,
    run_id: str,
    tenant_id: str,
    status: str,
    report: dict[str, Any] | None = None,
    error_message: str | None = None,
) -> None:
    async with get_db_connection() as conn:
        async with conn.cursor() as cur:
            await cur.execute(
                """
                UPDATE public.evaluation_runs
                SET status = %s,
                    report = %s,
                    error_message = %s,
                    started_at = CASE
                        WHEN %s = 'running' THEN COALESCE(started_at, NOW())
                        ELSE started_at
                    END,
                    completed_at = CASE
                        WHEN %s IN ('completed', 'failed') THEN NOW()
                        ELSE completed_at
                    END
                WHERE id = %s AND tenant_id = %s;
                """,
                (
                    status,
                    Jsonb(report) if report is not None else None,
                    error_message,
                    status,
                    status,
                    run_id,
                    tenant_id,
                ),
            )
            if cur.rowcount != 1:
                raise RuntimeError("Evaluation run record was not found while updating its status.")
            await conn.commit()


async def get_evaluation_run(
    *,
    run_id: str,
    tenant_id: str,
    include_report: bool = False,
) -> dict[str, Any] | None:
    report_column = ", report" if include_report else ""
    async with get_db_connection() as conn:
        async with conn.cursor() as cur:
            await cur.execute(
                f"""
                SELECT {_RUN_COLUMNS}{report_column}
                FROM public.evaluation_runs
                WHERE id = %s AND tenant_id = %s;
                """,
                (run_id, tenant_id),
            )
            row = await cur.fetchone()
            if row is None:
                return None
            report = row[13] if include_report else None
            return _serialize_run(row[:13], report)


async def get_latest_evaluation_run(
    tenant_id: str,
) -> dict[str, Any] | None:
    async with get_db_connection() as conn:
        async with conn.cursor() as cur:
            await cur.execute(
                f"""
                SELECT {_RUN_COLUMNS}
                FROM public.evaluation_runs
                WHERE tenant_id = %s
                ORDER BY created_at DESC
                LIMIT 1;
                """,
                (tenant_id,),
            )
            row = await cur.fetchone()
            if row is None:
                return None
            return _serialize_run(row)


async def get_latest_completed_evaluation_run(
    tenant_id: str,
) -> dict[str, Any] | None:
    async with get_db_connection() as conn:
        async with conn.cursor() as cur:
            await cur.execute(
                f"""
                SELECT {_RUN_COLUMNS}, report
                FROM public.evaluation_runs
                WHERE tenant_id = %s AND status = 'completed'
                ORDER BY completed_at DESC
                LIMIT 1;
                """,
                (tenant_id,),
            )
            row = await cur.fetchone()
            if row is None:
                return None
            return _serialize_run(row[:13], row[13])


async def list_evaluation_runs(tenant_id: str, limit: int = 50) -> list[dict[str, Any]]:
    async with get_db_connection() as conn:
        async with conn.cursor() as cur:
            await cur.execute(
                f"""
                SELECT {_RUN_COLUMNS},
                    jsonb_build_object(
                        'retrieval', report->'retrieval',
                        'refusal', report->'refusal'
                    )
                FROM public.evaluation_runs
                WHERE tenant_id = %s AND status = 'completed'
                ORDER BY created_at DESC
                LIMIT %s;
                """,
                (tenant_id, limit),
            )
            rows = await cur.fetchall()
            return [_serialize_run(row[:13], row[13]) for row in rows]
