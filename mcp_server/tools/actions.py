"""
Tool: get_action_items

Extract, filter, and summarise action items across the meeting corpus.
"""

from __future__ import annotations

import re
from collections import Counter
from typing import Any

from mcp_server.data_store import get_df
from mcp_server.governance import check_access, audit_log, timer

# Matches "Owner Name: action description"
_ACTION_RE = re.compile(r"^([^:]+):\s*(.+)$")


def get_action_items(
    owner: str | None = None,
    keyword: str | None = None,
    call_type: str | None = None,
    limit: int = 20,
    role: str = "analyst",
) -> dict[str, Any]:
    """
    Return action items with optional filtering by owner, keyword, or call type.

    Parameters
    ----------
    owner     : partial owner name filter (case-insensitive)
    keyword   : substring filter on the action text (case-insensitive)
    call_type : optional pre-filter — "support" | "external" | "internal"
    limit     : max action items to return (default 20)
    role      : caller role for access control
    """
    allowed, reason = check_access(role, "get_action_items")
    if not allowed:
        return {"error": reason, "status": "denied"}

    with timer() as t:
        try:
            df = get_df()

            # Optional call_type pre-filter
            if call_type:
                df = df[df["call_type"] == call_type.lower()]

            # Parse every action item in the corpus
            parsed: list[dict[str, Any]] = []
            owner_counter: Counter = Counter()

            for _, row in df.iterrows():
                items = row["action_items"]
                if not isinstance(items, list):
                    continue
                for raw_item in items:
                    if not isinstance(raw_item, str):
                        continue
                    m = _ACTION_RE.match(raw_item.strip())
                    if m:
                        item_owner = m.group(1).strip()
                        item_action = m.group(2).strip()
                    else:
                        item_owner = "Unknown"
                        item_action = raw_item.strip()

                    owner_counter[item_owner] += 1

                    parsed.append({
                        "meeting_id": row["meeting_id"],
                        "title": row["title"],
                        "call_type": row["call_type"],
                        "owner": item_owner,
                        "action": item_action,
                        "raw": raw_item,
                    })

            # Apply filters
            filtered = parsed

            if owner:
                owner_lower = owner.lower()
                filtered = [a for a in filtered if owner_lower in a["owner"].lower()]

            if keyword:
                kw_lower = keyword.lower()
                filtered = [a for a in filtered if kw_lower in a["action"].lower()]

            total_found = len(filtered)

            # Workload summary — top 5 owners by total action count (pre-filter)
            workload_summary = [
                {"owner": o, "action_count": c}
                for o, c in owner_counter.most_common(5)
            ]

            result: dict[str, Any] = {
                "filters": {
                    "owner": owner,
                    "keyword": keyword,
                    "call_type": call_type,
                },
                "total_found": total_found,
                "returned": min(total_found, limit),
                "action_items": filtered[:limit],
                "workload_summary": workload_summary,
                "status": "ok",
            }
            summary = (
                f"{total_found} action items"
                + (f" matching owner='{owner}'" if owner else "")
                + (f" keyword='{keyword}'" if keyword else "")
            )

        except Exception as exc:
            result = {"error": str(exc), "status": "failed"}
            summary = f"error: {exc}"

    audit_log(
        "get_action_items",
        {"owner": owner, "keyword": keyword, "call_type": call_type, "limit": limit},
        summary, t.elapsed_ms, role, result["status"],
    )
    return result
