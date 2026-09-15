"""Tests for the execute_sql query logging (log_tool_sql)."""

import logging
from unittest.mock import AsyncMock
from unittest.mock import MagicMock
from unittest.mock import patch

import pytest

from postgres_mcp.server import LOGGED_SQL_MAX_LENGTH
from postgres_mcp.server import execute_sql
from postgres_mcp.server import log_tool_sql

LOGGER_NAME = "postgres_mcp.server"


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
