"""
Transcript Intelligence — MCP Server

Exposes 5 analytics tools over the MCP stdio transport so Claude Desktop
(or any MCP client) can call them directly.

Run via:
    python -m mcp_server.server          (inside Docker)
    scripts/run_mcp.sh                   (from Claude Desktop on Mac)
"""

from __future__ import annotations

from mcp.server.fastmcp import FastMCP

from mcp_server.tools.search import search_meetings
from mcp_server.tools.sentiment import get_sentiment_trends
from mcp_server.tools.churn import score_churn_risk
from mcp_server.tools.topics import find_recurring_topics
from mcp_server.tools.actions import get_action_items

# ---------------------------------------------------------------------------
# Server definition
# ---------------------------------------------------------------------------

mcp = FastMCP(
    name="transcript-intelligence",
    instructions=(
        "You are an analytics assistant for a B2B SaaS company called Aegis "
        "with access to 100 meeting transcripts (support, external, and internal calls). "
        "Use tools to answer questions. Always cite specific meeting titles or account names. "
        "For complex questions chain tools: use score_churn_risk first, then "
        "search_meetings for that account, then get_sentiment_trends to show the trend."
    ),
)

# ---------------------------------------------------------------------------
# Tool registration
# ---------------------------------------------------------------------------

mcp.add_tool(search_meetings)
mcp.add_tool(get_sentiment_trends)
mcp.add_tool(score_churn_risk)
mcp.add_tool(find_recurring_topics)
mcp.add_tool(get_action_items)

# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("[MCP] Starting transcript-intelligence server...", flush=True)
    # Pre-warm the DataStore so the first tool call isn't slow
    from mcp_server.data_store import get_df as _get_df
    _get_df()
    mcp.run(transport="stdio")
