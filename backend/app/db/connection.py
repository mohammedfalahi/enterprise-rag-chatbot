from contextlib import asynccontextmanager
from psycopg_pool import AsyncConnectionPool
from app.core.config import settings

pool: AsyncConnectionPool | None = None


def _create_pool() -> AsyncConnectionPool:
    return AsyncConnectionPool(
        conninfo=settings.DATABASE_URL,
        open=False,
        max_size=10,
        min_size=2,
        timeout=30.0,
        max_waiting=50,
    )


pool = _create_pool()


@asynccontextmanager
async def get_db_connection():
    """Provides an async connection from the connection pool."""
    global pool
    if pool is None or (pool.closed and pool._opened):
        pool = _create_pool()
    if pool.closed and not pool._opened:
        await pool.open()
    async with pool.connection() as conn:
        yield conn


async def init_db_pool():
    """Initialize the connection pool."""
    global pool
    if pool is None or (pool.closed and pool._opened):
        pool = _create_pool()
    if pool.closed and not pool._opened:
        await pool.open()


async def close_db_pool():
    """Close the connection pool."""
    global pool
    if pool is not None and not pool.closed:
        await pool.close()
