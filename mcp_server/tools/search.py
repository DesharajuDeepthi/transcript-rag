"""
Tool: search_meetings

Keyword search across title, summary, and topics columns.
"""

from __future__ import annotations

from typing import Any

from mcp_server.data_store import get_df
from mcp_server.governance import check_access, audit_log, timer


def search_meetings(
    query: str,
    call_type: str | None = None,
    limit: int = 10,
    role: str = "analyst",
) -> dict[str, Any]:
    """
    Search meetings by keyword across title, summary, and topics.

    Parameters
    ----------
    query     : keyword / phrase to search (case-insensitive, literal)
    call_type : optional pre-filter — "support" | "external" | "internal"
    limit     : max rows to return (default 10)
    role      : caller role for access control
    """
    allowed, reason = check_access(role, "search_meetings")
    if not allowed:
        return {"error": reason, "status": "denied"}

    with timer() as t:
        try:
            df = get_df()

            # Optional call_type pre-filter
            if call_type:
                df = df[df["call_type"] == call_type.lower()]

            # topics is a list column — join to string so str.contains works
            topics_str = df["topics"].apply(
                lambda lst: " ".join(lst) if isinstance(lst, list) else (lst or "")
            )
            combined = (
                df["title"].fillna("")
                + " "
                + df["summary"].fillna("")
                + " "
                + topics_str
            )

            mask = combined.str.contains(query, case=False, na=False, regex=False)
            hits = df[mask].copy()

            total_found = len(hits)
            page = hits.head(limit)

            meetings = []
            for _, row in page.iterrows():
                topics_list = row["topics"] if isinstance(row["topics"], list) else []
                meetings.append({
                    "meeting_id": row["meeting_id"],
                    "title": row["title"],
                    "call_type": row["call_type"],
                    "sub_theme": row["sub_theme"],
                    "sentiment_score": float(row["sentiment_score"]),
                    "date": str(row["start_time"].date()) if row["start_time"] is not None else None,
                    "topics": topics_list[:4],
                    "summary_snippet": (row["summary"] or "")[:200],
                })

            result: dict[str, Any] = {
                "query": query,
                "total_found": total_found,
                "returned": len(meetings),
                "call_type_filter": call_type,
                "meetings": meetings,
                "status": "ok",
            }
            summary = f"{total_found} hits for '{query}'"

        except Exception as exc:
            result = {"error": str(exc), "status": "failed"}
            summary = f"error: {exc}"

    audit_log(
        "search_meetings",
        {"query": query, "call_type": call_type, "limit": limit},
        summary, t.elapsed_ms, role, result["status"],
    )
    return result
