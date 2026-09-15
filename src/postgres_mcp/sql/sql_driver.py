"""SQL driver adapter for PostgreSQL connections."""

import asyncio
import logging
import os
import re
from contextlib import aclosing
from dataclasses import dataclass
from typing import Any
from typing import AsyncGenerator
from typing import Dict
from typing import List
from typing import Optional
from urllib.parse import urlparse
from urllib.parse import urlunparse

import psycopg
from pglast.ast import SelectStmt
from pglast.parser import parse_sql
from psycopg.rows import dict_row
from psycopg_pool import AsyncConnectionPool
from typing_extensions import LiteralString

logger = logging.getLogger(__name__)

# Lines quoting caller SQL go to the SQL-audit logger: entries must land as
# single unwrapped lines, which the rich root handler breaks by wrapping at
# terminal width. The server attaches a plain single-line handler to this
# logger; until one is attached, records propagate to the root handler.
sql_audit_log = logging.getLogger("postgres_mcp.sql_audit")


def _env_int(name: str, default: int) -> int:
    """Read an integer from the environment, falling back to `default` on missing or invalid values."""
    value = os.environ.get(name)
    if value is None:
        return default
    try:
        return int(value)
    except ValueError:
        logger.warning(f"Invalid value {value!r} for {name}; using default {default}")
        return default


# Caps on the result set a single query may return, enforced while fetching so
# an oversized result cannot be materialized in this process's memory at all.
# Values <= 0 disable the corresponding cap.
MAX_RESULT_ROWS = _env_int("POSTGRES_MCP_MAX_RESULT_ROWS", 5000)
MAX_RESULT_BYTES = _env_int("POSTGRES_MCP_MAX_RESULT_BYTES", 5 * 1024 * 1024)

# Server-side statement timeout applied to read-only queries, so a pathological
# query is also canceled on the database server rather than only abandoned
# client-side. <= 0 disables it.
STATEMENT_TIMEOUT_SECONDS = _env_int("POSTGRES_MCP_STATEMENT_TIMEOUT_SECONDS", 30)

CLIENT_TIMEOUT_GRACE_SECONDS = 5


def client_query_timeout_seconds() -> Optional[int]:
    """Client-side query timeout paired with the server statement_timeout.

    Slightly above the server timeout so the server cancels first — a clean
    QueryCanceled that keeps the pooled connection healthy — leaving the client
    timeout to catch only what the server timeout cannot (e.g. network stalls).
    Returns None (no client timeout) when the server timeout is disabled.
    """
    if STATEMENT_TIMEOUT_SECONDS <= 0:
        return None
    return STATEMENT_TIMEOUT_SECONDS + CLIENT_TIMEOUT_GRACE_SECONDS


@dataclass
class ResultTruncation:
    """Details of a result set cut off by MAX_RESULT_ROWS / MAX_RESULT_BYTES."""

    rows_returned: int
    bytes_returned: int
    max_rows: int
    max_bytes: int


def _should_stream(query: str) -> bool:
    """Whether to fetch with cursor.stream(): a single plain SELECT.

    Streaming keeps an oversized result set out of the client-side libpq
    buffer (execute() would transfer it whole before the first fetch), but
    single-row mode supports neither multiple statements nor statements that
    return no rows, so anything else keeps the fetch-after-execute path.
    """
    try:
        statements = parse_sql(query)
    except Exception:
        return False
    if len(statements) != 1:
        return False
    statement = statements[0].stmt
    return isinstance(statement, SelectStmt) and statement.intoClause is None


def _approx_row_size(cells: Dict[str, Any]) -> int:
    """Approximate the serialized size of a row.

    Sums per-value sizes instead of len(str(cells)) so a huge string value is
    measured without making another full copy of it.
    """
    size = 0
    for key, value in cells.items():
        size += len(key) + 8
        size += len(value) if isinstance(value, str) else len(str(value))
    return size


def _sanitize_for_log(text: str) -> str:
    """Collapse caller SQL to one printable line for logging.

    Control characters are blanked so crafted SQL cannot forge or overwrite
    log entries, and password=... shapes (dblink/FDW connection strings) are
    redacted.
    """
    text = " ".join(text.split())
    text = "".join(ch if ch.isprintable() else " " for ch in text)
    return obfuscate_password(text) or ""


async def _rows_from_cursor(cursor: Any) -> AsyncGenerator[Any, None]:
    """Yield the cursor's remaining rows one at a time.

    fetchmany/fetchall materialize every fetched row as a Python object before
    the caller can apply any cap; one row per step bounds that spike to a
    single row.
    """
    while True:
        row = await cursor.fetchone()
        if row is None:
            return
        yield row


def _invalidates_pool(e: BaseException) -> bool:
    """Whether an error means the connection pool itself is unusable.

    Query-level failures (bad SQL, constraint violations, ...) and statement
    timeouts leave the underlying connections healthy, so the pool must
    survive them. Only connection-level errors (server gone away, broken
    socket, pool timeout/closed — all OperationalError or InterfaceError)
    warrant rebuilding the pool.
    """
    if isinstance(e, psycopg.errors.QueryCanceled):
        # Statement timeout / cancellation: the connection is fine.
        return False
    return isinstance(e, (psycopg.OperationalError, psycopg.InterfaceError))


def obfuscate_password(text: str | None) -> str | None:
    """
    Obfuscate password in any text containing connection information.
    Works on connection URLs, error messages, and other strings.
    """
    if text is None:
        return None

    if not text:
        return text

    # Try first as a proper URL
    try:
        parsed = urlparse(text)
        if parsed.scheme and parsed.netloc and parsed.password:
            # Replace password with asterisks in proper URL
            netloc = parsed.netloc.replace(parsed.password, "****")
            return urlunparse(parsed._replace(netloc=netloc))
    except Exception:
        pass

    # Handle strings that contain connection strings but aren't proper URLs
    # Match postgres://user:password@host:port/dbname pattern
    url_pattern = re.compile(r"(postgres(?:ql)?:\/\/[^:]+:)([^@]+)(@[^\/\s]+)")
    text = re.sub(url_pattern, r"\1****\3", text)

    # Match connection string parameters (password=xxx)
    # This simpler pattern captures password without quotes
    param_pattern = re.compile(r'(password=)([^\s&;"\']+)', re.IGNORECASE)
    text = re.sub(param_pattern, r"\1****", text)

    # Match password in DSN format with single quotes
    dsn_single_quote = re.compile(r"(password\s*=\s*')([^']+)(')", re.IGNORECASE)
    text = re.sub(dsn_single_quote, r"\1****\3", text)

    # Match password in DSN format with double quotes
    dsn_double_quote = re.compile(r'(password\s*=\s*")([^"]+)(")', re.IGNORECASE)
    text = re.sub(dsn_double_quote, r"\1****\3", text)

    return text


class DbConnPool:
    """Database connection manager using psycopg's connection pool."""

    def __init__(self, connection_url: Optional[str] = None):
        self.connection_url = connection_url
        self.pool: AsyncConnectionPool | None = None
        self._is_valid = False
        self._last_error = None
        self._connect_lock = asyncio.Lock()

    async def pool_connect(self, connection_url: Optional[str] = None) -> AsyncConnectionPool:
        """Initialize connection pool with retry logic."""
        # If we already have a valid pool, return it
        if self.pool and self._is_valid:
            return self.pool

        # Serialize pool (re)creation. Without the lock, concurrent callers
        # that each observed an invalid pool would each build an
        # AsyncConnectionPool and assign it to self.pool; every overwritten
        # pool is left open forever — leaking its connections and worker
        # tasks — and callers still holding a pool that a later winner
        # closed fail with "the pool 'pool-N' is closed".
        async with self._connect_lock:
            # Re-check under the lock: another task may have already rebuilt
            # the pool while we were waiting.
            if self.pool and self._is_valid:
                return self.pool

            url = connection_url or self.connection_url
            self.connection_url = url
            if not url:
                self._is_valid = False
                self._last_error = "Database connection URL not provided"
                raise ValueError(self._last_error)

            # Close any existing pool before creating a new one
            await self.close()

            try:
                # Configure connection pool with appropriate settings
                self.pool = AsyncConnectionPool(
                    conninfo=url,
                    min_size=1,
                    max_size=5,
                    open=False,  # Don't connect immediately, let's do it explicitly
                    # Validate connections on checkout so ones broken while
                    # idle (server restart, idle timeout) are discarded and
                    # replaced instead of surfacing as query errors.
                    check=AsyncConnectionPool.check_connection,
                )

                # Open the pool explicitly
                await self.pool.open()

                # Test the connection pool by executing a simple query
                async with self.pool.connection() as conn:
                    async with conn.cursor() as cursor:
                        await cursor.execute("SELECT 1")

                self._is_valid = True
                self._last_error = None
                return self.pool
            except Exception as e:
                self._is_valid = False
                self._last_error = str(e)

                # Clean up failed pool
                await self.close()

                raise ValueError(f"Connection attempt failed: {obfuscate_password(str(e))}") from e

    async def close(self) -> None:
        """Close the connection pool."""
        if self.pool:
            try:
                # Close the pool
                await self.pool.close()
            except Exception as e:
                logger.warning(f"Error closing connection pool: {e}")
            finally:
                self.pool = None
                self._is_valid = False

    @property
    def is_valid(self) -> bool:
        """Check if the connection pool is valid."""
        return self._is_valid

    @property
    def last_error(self) -> Optional[str]:
        """Get the last error message."""
        return self._last_error


class SqlDriver:
    """Adapter class that wraps a PostgreSQL connection with the interface expected by DTA."""

    @dataclass
    class RowResult:
        """Simple class to match the Griptape RowResult interface."""

        cells: Dict[str, Any]

    # Set when the most recent execute_query() hit a result-set cap.
    last_truncation: Optional[ResultTruncation] = None

    def __init__(
        self,
        conn: Any = None,
        engine_url: str | None = None,
    ):
        """
        Initialize with a PostgreSQL connection or pool.

        Args:
            conn: PostgreSQL connection object or pool
            engine_url: Connection URL string as an alternative to providing a connection
        """
        if conn:
            self.conn = conn
            # Check if this is a connection pool
            self.is_pool = isinstance(conn, DbConnPool)
        elif engine_url:
            # Don't connect here since we need async connection
            self.engine_url = engine_url
            self.conn = None
            self.is_pool = False
        else:
            raise ValueError("Either conn or engine_url must be provided")

    def connect(self):
        if self.conn is not None:
            return self.conn
        if self.engine_url:
            self.conn = DbConnPool(self.engine_url)
            self.is_pool = True
            return self.conn
        else:
            raise ValueError("Connection not established. Either conn or engine_url must be provided")

    async def execute_query(
        self,
        query: LiteralString,
        params: list[Any] | None = None,
        force_readonly: bool = False,
    ) -> Optional[List[RowResult]]:
        """
        Execute a query and return results.

        Args:
            query: SQL query to execute
            params: Query parameters
            force_readonly: Whether to enforce read-only mode

        Returns:
            List of RowResult objects or None on error. If the result set hit
            a cap, the returned rows are the prefix that fit and
            `self.last_truncation` describes the truncation.
        """
        self.last_truncation = None
        try:
            if self.conn is None:
                self.connect()
                if self.conn is None:
                    raise ValueError("Connection not established")

            # Handle connection pool vs direct connection
            if self.is_pool:
                # For pools, get a connection from the pool
                pool = await self.conn.pool_connect()
                async with pool.connection() as connection:
                    return await self._execute_with_connection(connection, query, params, force_readonly=force_readonly)
            else:
                # Direct connection approach
                return await self._execute_with_connection(self.conn, query, params, force_readonly=force_readonly)
        except Exception as e:
            # Mark pool as invalid only on connection-level errors. Doing it
            # for every exception meant any bad SQL statement poisoned the
            # shared pool, and the next call tore it down and built a new
            # one — the pool churn and "the pool 'pool-N' is closed" errors
            # of issue #98.
            if self.conn and self.is_pool:
                if _invalidates_pool(e):
                    self.conn._is_valid = False  # type: ignore
                    self.conn._last_error = str(e)  # type: ignore
            elif self.conn and not self.is_pool:
                self.conn = None

            raise e

    async def _execute_with_connection(self, connection, query, params, force_readonly) -> Optional[List[RowResult]]:
        """Execute query with the given connection."""
        transaction_started = False
        try:
            async with connection.cursor(row_factory=dict_row) as cursor:
                # Start read-only transaction
                if force_readonly:
                    await cursor.execute("BEGIN TRANSACTION READ ONLY")
                    transaction_started = True
                    if STATEMENT_TIMEOUT_SECONDS > 0:
                        # Cancel long queries on the server too: a client-side
                        # timeout (SafeSqlDriver's asyncio.timeout) only stops
                        # waiting and leaves the query running on the database.
                        # set_config(..., is_local=true) scopes it to this
                        # transaction, so pooled connections are not affected.
                        await cursor.execute(
                            "SELECT set_config('statement_timeout', %s, true)",
                            [f"{STATEMENT_TIMEOUT_SECONDS}s"],
                        )

                row_iter: AsyncGenerator[Any, None]
                if force_readonly and _should_stream(query):
                    # Stream single SELECT statements row-by-row so an
                    # oversized result set is never fully buffered in this
                    # process, neither as Python objects nor in the libpq
                    # result buffer. Terminating the iteration early (a cap
                    # hit) cancels the query server-side (psycopg >= 3.2).
                    # Read-only transactions only: a cancel aborts the
                    # transaction, which would silently roll back the write of
                    # a data-modifying CTE (WITH d AS (DELETE ...) SELECT ...)
                    # while still returning its truncated rows.
                    row_iter = cursor.stream(query, params)
                else:
                    if params:
                        await cursor.execute(query, params)
                    else:
                        await cursor.execute(query)

                    # For multiple statements, move to the last statement's results
                    while cursor.nextset():
                        pass

                    if cursor.description is None:  # No results (like DDL statements)
                        if not force_readonly:
                            await cursor.execute("COMMIT")
                        elif transaction_started:
                            await cursor.execute("ROLLBACK")
                            transaction_started = False
                        return None

                    row_iter = _rows_from_cursor(cursor)

                # Consume one row at a time and stop at the caps so a single
                # oversized result set cannot exhaust this process's memory:
                # at most one over-cap row is ever materialized, and the row
                # that would cross a cap is dropped rather than returned.
                max_rows = MAX_RESULT_ROWS
                max_bytes = MAX_RESULT_BYTES
                results: List[SqlDriver.RowResult] = []
                bytes_fetched = 0
                truncated = False
                async with aclosing(row_iter) as rows:
                    async for row in rows:
                        if max_rows > 0 and len(results) >= max_rows:
                            truncated = True
                            break
                        cells = dict(row)
                        if max_bytes > 0:
                            size = _approx_row_size(cells)
                            if bytes_fetched + size > max_bytes:
                                truncated = True
                                break
                            bytes_fetched += size
                        results.append(SqlDriver.RowResult(cells=cells))

                if truncated:
                    self.last_truncation = ResultTruncation(
                        rows_returned=len(results),
                        bytes_returned=bytes_fetched,
                        max_rows=max_rows,
                        max_bytes=max_bytes,
                    )
                    sql_audit_log.warning(
                        f"Query result truncated at {len(results)} rows / ~{bytes_fetched} bytes "
                        f"(caps: {max_rows} rows, {max_bytes} bytes): {_sanitize_for_log(query)[:100]}"
                    )

                # End the transaction appropriately
                if not force_readonly:
                    await cursor.execute("COMMIT")
                elif transaction_started:
                    await cursor.execute("ROLLBACK")
                    transaction_started = False

                return results

        except Exception as e:
            # Try to roll back the transaction if it's still active
            if transaction_started:
                try:
                    await connection.rollback()
                except Exception as rollback_error:
                    logger.error(f"Error rolling back transaction: {rollback_error}")

            logger.error(f"Error executing query ({query}): {e}")
            raise e
