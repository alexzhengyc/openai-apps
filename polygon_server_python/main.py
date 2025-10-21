"""Polygon-style market data MCP server.

This module exposes three data-only tools that simulate a subset of Polygon's
stock APIs. Each handler returns structured JSON for use inside an MCP client
without embedding widgets or HTML resources.
"""

from __future__ import annotations

from copy import deepcopy
from datetime import UTC, datetime
from typing import Any, Dict, List

import mcp.types as types
from mcp.server.fastmcp import FastMCP
from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator


mcp = FastMCP(
    name="polygon-data-python",
    stateless_http=True,
)


def _ts_to_datetime(ms: int) -> datetime:
    """Convert millisecond timestamps into a timezone-aware datetime."""
    return datetime.fromtimestamp(ms / 1000, tz=UTC)


def _parse_iso_date(value: str) -> datetime:
    """Parse a YYYY-MM-DD (or ISO-8601) string into a UTC datetime at midnight."""
    if not value:
        raise ValueError("Date value is required.")
    try:
        dt = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as exc:
        raise ValueError(f"Invalid date format: {value}") from exc
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=UTC)
    return dt.astimezone(UTC)


AGGREGATE_DATA: Dict[str, Dict[str, List[Dict[str, Any]]]] = {
    "minute": {
        "AAPL": [
            {"t": 1714665600000, "o": 172.11, "h": 172.5, "l": 171.9, "c": 172.42, "v": 154321, "vw": 172.28},
            {"t": 1714665660000, "o": 172.42, "h": 172.7, "l": 172.1, "c": 172.33, "v": 98210, "vw": 172.38},
        ],
        "MSFT": [
            {"t": 1714665600000, "o": 403.55, "h": 404.0, "l": 403.2, "c": 403.88, "v": 112004, "vw": 403.64},
            {"t": 1714665660000, "o": 403.88, "h": 404.35, "l": 403.6, "c": 404.22, "v": 88954, "vw": 404.01},
        ],
    },
    "day": {
        "AAPL": [
            {"t": 1714608000000, "o": 170.1, "h": 173.3, "l": 169.7, "c": 172.62, "v": 51203450, "vw": 171.98},
            {"t": 1714521600000, "o": 168.5, "h": 170.9, "l": 167.8, "c": 170.15, "v": 43877621, "vw": 169.88},
        ],
        "MSFT": [
            {"t": 1714608000000, "o": 401.8, "h": 406.2, "l": 400.9, "c": 404.72, "v": 28765011, "vw": 404.05},
            {"t": 1714521600000, "o": 398.7, "h": 402.9, "l": 397.5, "c": 401.83, "v": 25450122, "vw": 400.96},
        ],
    },
}


SNAPSHOT_DATA: Dict[str, Dict[str, Any]] = {
    "AAPL": {
        "last_trade": {"price": 172.4, "size": 100, "exchange": "XNAS", "timestamp": 1714665660000},
        "last_quote": {"bid": 172.35, "ask": 172.39, "bid_size": 200, "ask_size": 180, "timestamp": 1714665660500},
        "minute": {"o": 172.11, "h": 172.5, "l": 171.9, "c": 172.33, "v": 252531},
        "day": {"o": 170.1, "h": 173.3, "l": 169.7, "c": 172.62, "v": 51203450},
        "prev_day": {"o": 168.5, "h": 170.9, "l": 167.8, "c": 170.15, "v": 43877621},
    },
    "MSFT": {
        "last_trade": {"price": 404.1, "size": 50, "exchange": "XNAS", "timestamp": 1714665660000},
        "last_quote": {"bid": 404.05, "ask": 404.12, "bid_size": 120, "ask_size": 130, "timestamp": 1714665660500},
        "minute": {"o": 403.55, "h": 404.0, "l": 403.2, "c": 404.22, "v": 200958},
        "day": {"o": 401.8, "h": 406.2, "l": 400.9, "c": 404.72, "v": 28765011},
        "prev_day": {"o": 398.7, "h": 402.9, "l": 397.5, "c": 401.83, "v": 25450122},
    },
}


DAILY_DATA: Dict[str, Dict[str, Dict[str, Any]]] = {
    "2024-05-02": {
        "AAPL": {"t": "2024-05-02", "o": 170.1, "h": 173.3, "l": 169.7, "c": 172.62, "v": 51203450},
        "MSFT": {"t": "2024-05-02", "o": 401.8, "h": 406.2, "l": 400.9, "c": 404.72, "v": 28765011},
    },
    "2024-05-01": {
        "AAPL": {"t": "2024-05-01", "o": 168.5, "h": 170.9, "l": 167.8, "c": 170.15, "v": 43877621},
        "MSFT": {"t": "2024-05-01", "o": 398.7, "h": 402.9, "l": 397.5, "c": 401.83, "v": 25450122},
    },
}


class AggregatesInput(BaseModel):
    ticker: str = Field(..., description="Ticker symbol to query, e.g., AAPL.")
    multiplier: int = Field(1, ge=1, description="Data aggregation multiplier.")
    timespan: str = Field(..., description="Aggregation window: minute or day.")
    from_: str = Field(..., alias="from", description="Start date/time ISO string.")
    to: str = Field(..., description="End date/time ISO string.")

    model_config = ConfigDict(populate_by_name=True, extra="forbid")

    @field_validator("timespan")
    @classmethod
    def _validate_timespan(cls, value: str) -> str:
        if value not in AGGREGATE_DATA:
            raise ValueError(f"Unsupported timespan '{value}'. Expected one of {list(AGGREGATE_DATA)}.")
        return value

    @field_validator("ticker")
    @classmethod
    def _normalize_ticker(cls, value: str) -> str:
        value = value.strip().upper()
        if not value:
            raise ValueError("Ticker is required.")
        return value


class SnapshotInput(BaseModel):
    ticker: str = Field(..., description="Ticker symbol to query.")

    model_config = ConfigDict(extra="forbid")

    @field_validator("ticker")
    @classmethod
    def _normalize_ticker(cls, value: str) -> str:
        value = value.strip().upper()
        if not value:
            raise ValueError("Ticker is required.")
        return value


class TickersDailyInput(BaseModel):
    tickers: List[str] = Field(..., description="Ticker symbols to query.")
    date: str = Field(..., description="Trading date (YYYY-MM-DD).")

    model_config = ConfigDict(extra="forbid")

    @field_validator("tickers")
    @classmethod
    def _normalize_tickers(cls, values: List[str]) -> List[str]:
        clean = []
        for value in values:
            symbol = value.strip().upper()
            if symbol:
                clean.append(symbol)
        if not clean:
            raise ValueError("At least one ticker is required.")
        return clean

    @field_validator("date")
    @classmethod
    def _validate_date(cls, value: str) -> str:
        _parse_iso_date(value)
        return value


TOOL_SCHEMAS: Dict[str, Dict[str, Any]] = {
    "get_aggregates": AggregatesInput.model_json_schema(),
    "get_snapshot": SnapshotInput.model_json_schema(),
    "get_tickers_daily": TickersDailyInput.model_json_schema(),
}


def _filter_aggregates(payload: AggregatesInput) -> List[Dict[str, Any]]:
    records = AGGREGATE_DATA.get(payload.timespan, {}).get(payload.ticker, [])
    if not records:
        return []

    start = _parse_iso_date(payload.from_)
    end = _parse_iso_date(payload.to)
    if end < start:
        start, end = end, start

    filtered: List[Dict[str, Any]] = []
    for row in records:
        ts = _ts_to_datetime(row["t"])
        if start <= ts <= end:
            filtered.append(deepcopy(row))
    return filtered


def _fetch_snapshot(ticker: str) -> Dict[str, Any] | None:
    snapshot = SNAPSHOT_DATA.get(ticker)
    return deepcopy(snapshot) if snapshot else None


def _fetch_daily(date: str, tickers: List[str]) -> List[Dict[str, Any]]:
    by_ticker = DAILY_DATA.get(date, {})
    results: List[Dict[str, Any]] = []
    for symbol in tickers:
        entry = by_ticker.get(symbol)
        if entry:
            results.append(deepcopy(entry))
    return results


@mcp._mcp_server.list_tools()
async def _list_tools() -> List[types.Tool]:
    descriptions = {
        "get_aggregates": "Retrieve aggregate bars for a ticker between two timestamps.",
        "get_snapshot": "Return the latest trade, quote, and session stats for a ticker.",
        "get_tickers_daily": "Fetch daily OHLCV bars for multiple tickers on a given date.",
    }

    return [
        types.Tool(
            name=tool_name,
            title=tool_name,
            description=descriptions[tool_name],
            inputSchema=TOOL_SCHEMAS[tool_name],
            _meta={
                "openai/resultCanProduceWidget": False,
            },
            annotations={
                "destructiveHint": False,
                "openWorldHint": False,
                "readOnlyHint": True,
            },
        )
        for tool_name in TOOL_SCHEMAS
    ]


async def _call_tool_request(req: types.CallToolRequest) -> types.ServerResult:
    tool = req.params.name
    arguments = req.params.arguments or {}

    try:
        if tool == "get_aggregates":
            payload = AggregatesInput.model_validate(arguments)
            data = _filter_aggregates(payload)
        elif tool == "get_snapshot":
            payload = SnapshotInput.model_validate(arguments)
            data = _fetch_snapshot(payload.ticker)
        elif tool == "get_tickers_daily":
            payload = TickersDailyInput.model_validate(arguments)
            data = _fetch_daily(payload.date, payload.tickers)
        else:
            raise ValueError(f"Unknown tool: {tool}")
    except ValidationError as exc:
        return types.ServerResult(
            types.CallToolResult(
                content=[
                    types.TextContent(
                        type="text",
                        text=f"Input validation error: {exc.errors()}",
                    )
                ],
                isError=True,
            )
        )
    except ValueError as exc:
        return types.ServerResult(
            types.CallToolResult(
                content=[
                    types.TextContent(
                        type="text",
                        text=str(exc),
                    )
                ],
                isError=True,
            )
        )

    if tool == "get_aggregates":
        structured_content: Dict[str, Any] = {"aggregates": data, "count": len(data)}
        content_text = f"{tool} returned {len(data)} aggregate bar(s)."
    elif tool == "get_snapshot":
        structured_content = data or {}
        ticker = arguments.get("ticker", "")
        if structured_content:
            content_text = f"{tool} returned snapshot data for {ticker}."
        else:
            content_text = f"{tool} had no data for {ticker}."
    else:  # get_tickers_daily
        structured_content = {"bars": data, "count": len(data)}
        content_text = f"{tool} returned {len(data)} daily bar(s)."

    return types.ServerResult(
        types.CallToolResult(
            content=[
                types.TextContent(
                    type="text",
                    text=content_text,
                )
            ],
            structuredContent=structured_content,
        )
    )


mcp._mcp_server.request_handlers[types.CallToolRequest] = _call_tool_request


app = mcp.streamable_http_app()

try:
    from starlette.middleware.cors import CORSMiddleware

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_methods=["*"],
        allow_headers=["*"],
        allow_credentials=False,
    )
except Exception:
    pass


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("main:app", host="0.0.0.0", port=8000)
