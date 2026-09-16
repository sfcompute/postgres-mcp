"""Tests for the SQL attribution logging (log_tool_sql / sql_audit logger)."""

import io
import logging
from unittest.mock import AsyncMock
from unittest.mock import MagicMock
from unittest.mock import patch

import pytest

from postgres_mcp.server import LOGGED_SQL_MAX_LENGTH
from postgres_mcp.server import analyze_query_indexes
from postgres_mcp.server import execute_sql
from postgres_mcp.server import explain_query
from postgres_mcp.server import log_tool_sql
from postgres_mcp.server import sql_log
from postgres_mcp.server import sql_log_handler

LOGGER_NAME = "postgres_mcp.sql_audit"


@pytest.fixture(autouse=True)
def capture_sql_audit_log(caplog):
    # sql_log doesn't propagate to root (it bypasses the rich handler), so
    # caplog's root-attached handler never sees it — attach it directly.
    sql_log.addHandler(caplog.handler)
    yield
    sql_log.removeHandler(caplog.handler)


def test_log_tool_sql_logs_query_at_info(caplog):
    with caplog.at_level(logging.INFO, logger=LOGGER_NAME):
        log_tool_sql("execute_sql", "SELECT * FROM pg_stat_activity")
    assert "execute_sql request_id=- sql=SELECT * FROM pg_stat_activity" in caplog.text


def test_log_tool_sql_collapses_whitespace_to_one_line(caplog):
    with caplog.at_level(logging.INFO, logger=LOGGER_NAME):
        log_tool_sql("execute_sql", "SELECT *\n  FROM orders\n  WHERE id = 1")
    assert "sql=SELECT * FROM orders WHERE id = 1" in caplog.text
    assert "\n  FROM" not in caplog.text


def test_log_tool_sql_truncates_long_queries(caplog):
    sql = "SELECT '" + "x" * (LOGGED_SQL_MAX_LENGTH * 2) + "'"
    with caplog.at_level(logging.INFO, logger=LOGGER_NAME):
        log_tool_sql("execute_sql", sql)
    [record] = caplog.records
    assert f"[truncated, {len(sql)} chars total]" in record.message
    # The logged SQL itself is capped; the whole line stays a bounded size.
    assert len(record.message) < LOGGED_SQL_MAX_LENGTH + 100


def test_log_tool_sql_includes_request_id_header(caplog):
    request = MagicMock()
    request.headers = {"x-request-id": "abc123"}
    ctx = MagicMock()
    ctx.request_context.request = request
    with patch("postgres_mcp.server.mcp.get_context", return_value=ctx):
        with caplog.at_level(logging.INFO, logger=LOGGER_NAME):
            log_tool_sql("execute_sql", "SELECT 1")
    assert "execute_sql request_id=abc123 sql=SELECT 1" in caplog.text


def test_log_tool_sql_without_request_context_logs_dash(caplog):
    # stdio transport: get_context() raises outside an active request.
    with patch("postgres_mcp.server.mcp.get_context", side_effect=ValueError("no context")):
        with caplog.at_level(logging.INFO, logger=LOGGER_NAME):
            log_tool_sql("execute_sql", "SELECT 1")
    assert "execute_sql request_id=- sql=SELECT 1" in caplog.text


def test_sql_audit_logger_bypasses_rich_and_emits_one_unwrapped_line():
    # The attribution line must land in Loki as ONE entry: no rich handler
    # (which wraps at terminal width, splitting one message across entries),
    # no propagation to the rich root handler, and no wrapping at the maximum
    # realistic length — a full 2000-char truncated SQL stays on a single line.
    assert sql_log.propagate is False
    assert all(type(h).__name__ != "RichHandler" for h in sql_log.handlers)

    stream = io.StringIO()
    old_stream = sql_log_handler.setStream(stream)
    try:
        log_tool_sql("execute_sql", "SELECT '" + "x" * (LOGGED_SQL_MAX_LENGTH * 2) + "'")
    finally:
        if old_stream is not None:
            sql_log_handler.setStream(old_stream)

    out = stream.getvalue()
    assert out.endswith("\n") and out.count("\n") == 1, "line was wrapped or split"
    assert len(out) > LOGGED_SQL_MAX_LENGTH, "truncated SQL payload must survive on the one line"


def test_log_tool_sql_blanks_control_characters(caplog):
    # Raw ESC survives whitespace collapsing; it must not reach the log record,
    # or crafted SQL could overwrite output in a terminal tailing the logs.
    with caplog.at_level(logging.INFO, logger=LOGGER_NAME):
        log_tool_sql("execute_sql", "SELECT '\x1b[2K\x1b[1Aspoofed'")
    [record] = caplog.records
    assert "\x1b" not in record.message
    assert "spoofed" in record.message


def test_log_tool_sql_obfuscates_passwords(caplog):
    with caplog.at_level(logging.INFO, logger=LOGGER_NAME):
        log_tool_sql(
            "execute_sql",
            "SELECT * FROM dblink('host=db.internal password=hunter2', 'SELECT 1') AS t(x int)",
        )
    [record] = caplog.records
    assert "hunter2" not in record.message
    assert "password=****" in record.message


@pytest.mark.asyncio
async def test_explain_query_logs_sql_before_running_it(caplog):
    # analyze=True executes the caller's query, so the crash-attribution log
    # must land before the driver runs — even when the driver fails.
    with patch("postgres_mcp.server.get_sql_driver", AsyncMock(side_effect=RuntimeError("connection lost"))):
        with caplog.at_level(logging.INFO, logger=LOGGER_NAME):
            await explain_query(sql="SELECT pg_sleep(600)", analyze=True, hypothetical_indexes=[])
    assert "explain_query[analyze] request_id=- sql=SELECT pg_sleep(600)" in caplog.text


@pytest.mark.asyncio
async def test_analyze_query_indexes_logs_count_and_each_query(caplog):
    with patch("postgres_mcp.server.get_sql_driver", AsyncMock(side_effect=RuntimeError("connection lost"))):
        with caplog.at_level(logging.INFO, logger=LOGGER_NAME):
            await analyze_query_indexes(queries=["SELECT 1", "SELECT 2"], max_index_size_mb=10000, method="dta")
    assert "analyze_query_indexes: 2 queries" in caplog.text
    assert "analyze_query_indexes request_id=- sql=SELECT 1" in caplog.text
    assert "analyze_query_indexes request_id=- sql=SELECT 2" in caplog.text


@pytest.mark.asyncio
async def test_execute_sql_logs_query_before_running_it(caplog):
    """The SQL must be logged even when executing it fails — the whole point is
    that a crash-triggering query is identifiable from the last log lines."""
    driver = MagicMock()
    driver.execute_query = AsyncMock(side_effect=RuntimeError("connection lost"))
    with patch("postgres_mcp.server.get_sql_driver", AsyncMock(return_value=driver)):
        with caplog.at_level(logging.INFO, logger=LOGGER_NAME):
            await execute_sql(sql="SELECT pg_sleep(600)")
    assert "execute_sql request_id=- sql=SELECT pg_sleep(600)" in caplog.text
    assert "Error executing query: connection lost" in caplog.text
