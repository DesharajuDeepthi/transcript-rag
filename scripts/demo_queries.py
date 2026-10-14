"""
demo_queries.py — smoke-test all 5 MCP tools against real data.

Run from the project root:
    python scripts/demo_queries.py

No MCP client or running server needed — imports the tool functions directly.
"""

from __future__ import annotations

import pathlib
import sys

from mcp_server.tools.search import search_meetings
from mcp_server.tools.sentiment import get_sentiment_trends
from mcp_server.tools.churn import score_churn_risk
from mcp_server.tools.topics import find_recurring_topics
from mcp_server.tools.actions import get_action_items

SEPARATOR = "-" * 60


def check(result: dict, tool_name: str) -> None:
    """Assert the result is non-error and print a one-line summary."""
    if result.get("status") in ("failed", "denied"):
        print(f"  ✗ {tool_name}: {result}", file=sys.stderr)
        sys.exit(1)


def main() -> None:
    print(SEPARATOR)
    print("Transcript Intelligence — Tool Demo")
    print(SEPARATOR)

    # 1. search_meetings ──────────────────────────────────────────────────────
    r = search_meetings(query="outage", limit=3)
    check(r, "search_meetings")
    top = r["meetings"][0]["title"] if r["meetings"] else "(none)"
    print(f"1. search_meetings('outage')   → {r['total_found']} hits | top: '{top}'")

    # 2. get_sentiment_trends ─────────────────────────────────────────────────
    r = get_sentiment_trends(group_by="call_type")
    check(r, "get_sentiment_trends")
    groups = ", ".join(
        f"{k}={v['mean']}" for k, v in r["breakdown"].items()
    )
    print(
        f"2. get_sentiment_trends()      → mean={r['overall_mean']} | "
        f"trend={r['overall_trend']} | [{groups}]"
    )

    # 3. score_churn_risk ─────────────────────────────────────────────────────
    r = score_churn_risk(limit=5)
    check(r, "score_churn_risk")
    top_acct = r["accounts"][0]
    dist = r["risk_distribution"]
    print(
        f"3. score_churn_risk()          → {r['total_accounts_scored']} accounts | "
        f"Critical={dist.get('Critical', 0)} | "
        f"top='{top_acct['account']}' ({top_acct['risk_score']} pts)"
    )

    # 4. find_recurring_topics ────────────────────────────────────────────────
    r = find_recurring_topics(min_meetings=3)
    check(r, "find_recurring_topics")
    top_topic = r["topics"][0]
    neg = [t["topic"] for t in r["negative_drivers"]]
    print(
        f"4. find_recurring_topics()     → {r['total_recurring_topics']} recurring | "
        f"top='{top_topic['topic']}' ({top_topic['meeting_count']} mtgs) | "
        f"neg_drivers={neg}"
    )

    # 5. get_action_items ─────────────────────────────────────────────────────
    r = get_action_items(limit=5)
    check(r, "get_action_items")
    top_owner = r["workload_summary"][0]
    print(
        f"5. get_action_items()          → {r['total_found']} items | "
        f"top owner='{top_owner['owner']}' ({top_owner['action_count']} items)"
    )

    # Audit log check ─────────────────────────────────────────────────────────
    print(SEPARATOR)
    log_path = pathlib.Path("mcp_server/audit.log")
    if log_path.exists():
        n_lines = len(log_path.read_text().strip().splitlines())
        print(f"audit.log: {n_lines} entries written")
    else:
        print("audit.log: not found (no tools called yet?)")

    print(SEPARATOR)
    print("ALL 5 TOOLS WORKING")
    print(SEPARATOR)


if __name__ == "__main__":
    main()
