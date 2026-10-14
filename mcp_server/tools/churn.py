"""
Tool: score_churn_risk

Rank customer accounts by churn risk using the pre-built heuristic scorer.
"""

from __future__ import annotations

from collections import Counter
from typing import Any

from mcp_server.data_store import get_df
from mcp_server.governance import check_access, audit_log, timer
from src.churn_scorer import score_all_accounts


def score_churn_risk(
    min_score: float = 0,
    limit: int = 10,
    role: str = "analyst",
) -> dict[str, Any]:
    """
    Score all customer accounts for churn risk and return the top results.

    Parameters
    ----------
    min_score : only return accounts with risk_score >= this value (default 0)
    limit     : max accounts to return (default 10)
    role      : caller role for access control
    """
    allowed, reason = check_access(role, "score_churn_risk")
    if not allowed:
        return {"error": reason, "status": "denied"}

    with timer() as t:
        try:
            rankings = score_all_accounts(get_df(), min_meetings=1)

            # Apply min_score filter
            filtered = [r for r in rankings if r["risk_score"] >= min_score]

            # Risk distribution across ALL scored accounts (before limit)
            risk_dist = dict(Counter(r["risk_level"] for r in filtered))

            # Slice to requested limit
            page = filtered[:limit]

            accounts = []
            for r in page:
                accounts.append({
                    "account": r["account"],
                    "risk_score": r["risk_score"],
                    "risk_level": r["risk_level"],
                    "meeting_count": r["meeting_count"],
                    "avg_sentiment": r.get("avg_sentiment"),
                    "recommendation": r["recommendation"],
                    "top_evidence": r.get("evidence", [])[:2],
                    "score_components": r.get("components", {}),
                })

            result: dict[str, Any] = {
                "total_accounts_scored": len(filtered),
                "returned": len(accounts),
                "filter": {"min_score": min_score},
                "risk_distribution": risk_dist,
                "accounts": accounts,
                "status": "ok",
            }
            critical = risk_dist.get("Critical", 0)
            summary = (
                f"{len(filtered)} accounts scored; "
                f"{critical} Critical; top={page[0]['account']} ({page[0]['risk_score']})"
                if page else f"{len(filtered)} accounts scored; none above min_score={min_score}"
            )

        except Exception as exc:
            result = {"error": str(exc), "status": "failed"}
            summary = f"error: {exc}"

    audit_log(
        "score_churn_risk",
        {"min_score": min_score, "limit": limit},
        summary, t.elapsed_ms, role, result["status"],
    )
    return result
