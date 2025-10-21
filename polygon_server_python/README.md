# Polygon MCP server (Python)

Prototype Model Context Protocol server that simulates a small slice of Polygon's market data APIs. It exposes three data-only tools and avoids widget markup so JSON responses can feed custom agents or experiments.

## Prerequisites

- Python 3.10+
- Recommended: create and activate a virtual environment

## Installation

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Run the server

```bash
python main.py
```

This launches a FastAPI app via uvicorn on `http://127.0.0.1:8000`. The MCP endpoints mirror the other demos in this repo (`GET /mcp` for SSE, `POST /mcp/messages` for follow-ups).

### Available tools

| Tool | Description | Sample invocation |
| --- | --- | --- |
| `get_aggregates` | Aggregated bars for a ticker between two timestamps. | `{"ticker": "AAPL", "multiplier": 1, "timespan": "day", "from": "2024-05-01", "to": "2024-05-02"}` |
| `get_snapshot` | Latest trade, quote, and session stats for a ticker. | `{"ticker": "MSFT"}` |
| `get_tickers_daily` | Daily OHLCV data for multiple tickers on a given date. | `{"tickers": ["AAPL", "MSFT"], "date": "2024-05-02"}` |

The data is static but structured to match Polygon's field names, making it easy to swap in real API calls later. All tools are flagged as read-only and return structured JSON in `structuredContent`.
