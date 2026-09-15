"""Tests for the execute_sql tool's result-truncation notice."""

from unittest.mock import AsyncMock
from unittest.mock import MagicMock
from unittest.mock import patch

import pytest

import postgres_mcp.server as server
from postgres_mcp.sql import ResultTruncation
from postgres_mcp.sql import SqlDriver


def _driver_returning(rows, truncation=None):
    driver = MagicMock()
    driver.execute_query = AsyncMock(return_value=rows)
    driver.last_truncation = truncation
    return driver


@pytest.mark.asyncio
async def test_execute_sql_appends_truncation_notice():
    """A capped result must tell the caller it was truncated and how to narrow the query."""
    rows = [SqlDriver.RowResult(cells={"id": 1})]
    truncation = ResultTruncation(rows_returned=1, bytes_returned=12, max_rows=1, max_bytes=100)
    with patch("postgres_mcp.server.get_sql_driver", AsyncMock(return_value=_driver_returning(rows, truncation))):
        result = await server.execute_sql("SELECT * FROM big_table")

    text = result[0].text  # type: ignore[union-attr]
    assert "{'id': 1}" in text
    assert "truncated at 1 rows" in text
    assert "POSTGRES_MCP_MAX_RESULT_ROWS=1" in text
    assert "POSTGRES_MCP_MAX_RESULT_BYTES=100" in text
    assert "LIMIT" in text


@pytest.mark.asyncio
async def test_execute_sql_no_notice_when_not_truncated():
    """An uncapped result must come back without any truncation notice."""
    rows = [SqlDriver.RowResult(cells={"id": 1})]
    with patch("postgres_mcp.server.get_sql_driver", AsyncMock(return_value=_driver_returning(rows))):
        result = await server.execute_sql("SELECT * FROM small_table")

    text = result[0].text  # type: ignore[union-attr]
    assert "{'id': 1}" in text
    assert "truncated" not in text
