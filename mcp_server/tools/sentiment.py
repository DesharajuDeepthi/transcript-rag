"""
Tool: get_sentiment_trends

Aggregate pre-computed sentiment scores by call_type, sub_theme, or week.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from mcp_server.data_store import get_df
from mcp_server.governance import check_access, audit_log, timer


def get_sentiment_trends(
    call_type: str | None = None,
    group_by: str = "call_type",
    role: str = "analyst",
) -> dict[str, Any]:
    """
    Aggregate sentiment scores across the corpus.

    Parameters
    ----------
    call_type : optional pre-filter — "support" | "external" | "internal"
    group_by  : "call_type" | "sub_theme" | "week"
    role      : caller role for access control
    """
    allowed, reason = check_access(role, "get_sentiment_trends")
    if not allowed:
        return {"error": reason, "status": "denied"}

    with timer() as t:
        try:
            df = get_df().copy()

            # Optional pre-filter
            if call_type:
                df = df[df["call_type"] == call_type.lower()]

            if len(df) == 0:
                result: dict[str, Any] = {
                    "filter": call_type,
                    "group_by": group_by,
                    "overall_mean": None,
                    "overall_trend": "no data",
                    "breakdown": {},
                    "most_negative_meetings": [],
                    "interpretation": "No meetings match the requested filter.",
                    "status": "ok",
                }
                audit_log("get_sentiment_trends",
                          {"call_type": call_type, "group_by": group_by},
                          "no data", t.elapsed_ms, role, "ok")
                return result

            # ── group column ──────────────────────────────────────────────
            if group_by == "week":
                df["week"] = df["start_time"].dt.to_period("W").astype(str)
                group_col = "week"
            elif group_by == "sub_theme":
                group_col = "sub_theme"
            else:
                group_col = "call_type"

            # ── per-group stats ───────────────────────────────────────────
            grouped = df.groupby(group_col)["sentiment_score"]
            breakdown: dict[str, Any] = {}
            for name, grp in grouped:
                breakdown[str(name)] = {
                    "mean": round(float(grp.mean()), 3),
                    "median": round(float(grp.median()), 3),
                    "std": round(float(grp.std()), 3) if len(grp) > 1 else 0.0,
                    "min": round(float(grp.min()), 3),
                    "max": round(float(grp.max()), 3),
                    "count": int(len(grp)),
                }

            # ── overall stats + linear trend ─────────────────────────────
            overall_mean = round(float(df["sentiment_score"].mean()), 3)

            valid_ts = df.dropna(subset=["start_time"]).sort_values("start_time")
            if len(valid_ts) >= 2:
                x = np.arange(len(valid_ts))
                slope = float(np.polyfit(x, valid_ts["sentiment_score"].values, 1)[0])
                if slope > 0.005:
                    overall_trend = "improving"
                elif slope < -0.005:
                    overall_trend = "declining"
                else:
                    overall_trend = "flat"
            else:
                slope = 0.0
                overall_trend = "insufficient data"

            # ── top-3 most negative meetings ─────────────────────────────
            neg_rows = df.nsmallest(3, "sentiment_score")
            most_negative = [
                {
                    "meeting_id": row["meeting_id"],
                    "title": row["title"],
                    "sentiment_score": float(row["sentiment_score"]),
                    "call_type": row["call_type"],
                    "sub_theme": row["sub_theme"],
                }
                for _, row in neg_rows.iterrows()
            ]

            # ── plain-English interpretation ──────────────────────────────
            if breakdown:
                worst_group = min(breakdown, key=lambda k: breakdown[k]["mean"])
                best_group = max(breakdown, key=lambda k: breakdown[k]["mean"])
                interpretation = (
                    f"Overall mean sentiment: {overall_mean}/5 ({overall_trend} trend). "
                    f"By {group_col}, '{worst_group}' is the most negative "
                    f"(mean {breakdown[worst_group]['mean']}) and '{best_group}' is the most positive "
                    f"(mean {breakdown[best_group]['mean']})."
                )
            else:
                interpretation = f"Overall mean sentiment: {overall_mean}/5 ({overall_trend} trend)."

            result = {
                "filter": call_type,
                "group_by": group_by,
                "overall_mean": overall_mean,
                "overall_trend": overall_trend,
                "breakdown": breakdown,
                "most_negative_meetings": most_negative,
                "interpretation": interpretation,
                "status": "ok",
            }
            summary = (
                f"mean={overall_mean}, trend={overall_trend}, "
                f"{len(breakdown)} groups by {group_by}"
            )

        except Exception as exc:
            result = {"error": str(exc), "status": "failed"}
            summary = f"error: {exc}"

    audit_log(
        "get_sentiment_trends",
        {"call_type": call_type, "group_by": group_by},
        summary, t.elapsed_ms, role, result["status"],
    )
    return result
