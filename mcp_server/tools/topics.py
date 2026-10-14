"""
Tool: find_recurring_topics

Surface topics that appear in multiple meetings and correlate them with sentiment.
"""

from __future__ import annotations

from collections import defaultdict
from typing import Any

from mcp_server.data_store import get_df
from mcp_server.governance import check_access, audit_log, timer


def find_recurring_topics(
    min_meetings: int = 3,
    call_type: str | None = None,
    role: str = "analyst",
) -> dict[str, Any]:
    """
    Find topics that appear in >= min_meetings meetings.

    Parameters
    ----------
    min_meetings : minimum number of meetings a topic must appear in (default 3)
    call_type    : optional pre-filter — "support" | "external" | "internal"
    role         : caller role for access control
    """
    allowed, reason = check_access(role, "find_recurring_topics")
    if not allowed:
        return {"error": reason, "status": "denied"}

    with timer() as t:
        try:
            df = get_df()

            # Optional call_type pre-filter
            if call_type:
                df = df[df["call_type"] == call_type.lower()]

            # Accumulate meeting IDs and sentiment scores per topic
            topic_meetings: dict[str, list[str]] = defaultdict(list)
            topic_scores: dict[str, list[float]] = defaultdict(list)

            for _, row in df.iterrows():
                topics = row["topics"]
                if not isinstance(topics, list):
                    continue
                sentiment = float(row["sentiment_score"])
                mid = row["meeting_id"]
                for topic in topics:
                    if topic:
                        topic_meetings[topic].append(mid)
                        topic_scores[topic].append(sentiment)

            # Filter to topics meeting the threshold
            recurring = []
            for topic, mids in topic_meetings.items():
                count = len(mids)
                if count < min_meetings:
                    continue
                scores = topic_scores[topic]
                avg_sent = round(sum(scores) / len(scores), 3)
                recurring.append({
                    "topic": topic,
                    "meeting_count": count,
                    "avg_sentiment": avg_sent,
                    "meeting_ids": mids,
                })

            # Sort by frequency descending
            recurring.sort(key=lambda x: x["meeting_count"], reverse=True)

            # Negative drivers: avg_sentiment < 3.0, top 3
            negative_drivers = sorted(
                [r for r in recurring if r["avg_sentiment"] < 3.0],
                key=lambda x: x["avg_sentiment"],
            )[:3]

            top_topics = recurring[:20]

            # Insight string
            if top_topics:
                top_3_names = ", ".join(f"'{t['topic']}'" for t in top_topics[:3])
                insight = (
                    f"{len(recurring)} topics appear in {min_meetings}+ meetings. "
                    f"Most frequent: {top_3_names}. "
                )
                if negative_drivers:
                    neg_names = ", ".join(f"'{t['topic']}'" for t in negative_drivers)
                    insight += f"Negative drivers (avg sentiment < 3.0): {neg_names}."
                else:
                    insight += "No strong negative-sentiment topics found."
            else:
                insight = f"No topics appear in {min_meetings}+ meetings with the current filter."

            result: dict[str, Any] = {
                "filter": {"min_meetings": min_meetings, "call_type": call_type},
                "total_recurring_topics": len(recurring),
                "topics": top_topics,
                "negative_drivers": negative_drivers,
                "insight": insight,
                "status": "ok",
            }
            summary = (
                f"{len(recurring)} recurring topics; "
                f"{len(negative_drivers)} negative drivers"
            )

        except Exception as exc:
            result = {"error": str(exc), "status": "failed"}
            summary = f"error: {exc}"

    audit_log(
        "find_recurring_topics",
        {"min_meetings": min_meetings, "call_type": call_type},
        summary, t.elapsed_ms, role, result["status"],
    )
    return result
