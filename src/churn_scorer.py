from __future__ import annotations
"""
Churn Risk Scorer for Transcript Intelligence.

Scores customer accounts by churn risk using transcript signals.
Heuristic weighted scorer — transparent and inspectable (not a black-box ML model).

Usage:
    from src.churn_scorer import score_all_accounts, add_account_column
    df = add_account_column(df)
    rankings = score_all_accounts(df)
"""

import re
import json
from pathlib import Path
from typing import Any

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go


OUTPUT_DIR = Path("outputs")
CHARTS_DIR = OUTPUT_DIR / "charts"

_TITLE_PATTERNS = [
    re.compile(r"^\w+\s*/\s*(.+?)\s*[-–]\s*", re.I),
    re.compile(r"support case\s*#?\d+\s*[-–]\s*(.+?)\s+(?:billing|inquiry|issue|case|access|integration|performance|error)", re.I),
]

# Trailing tokens that are product/tier/legal suffixes, not part of the account name.
# Stripped right-to-left so "Vanta Health Systems API Platform" → "Vanta Health Systems".
_SUFFIX_PATTERN = re.compile(
    r"\s+(?:API|Platform|Service|Services|Portal|Suite|Cloud|Hub|"
    r"Inc\.?|LLC\.?|Ltd\.?|Corp\.?|Co\.?|Group|Holdings|GmbH|SaaS|App|Apps)$",
    re.I,
)


def _normalize_account(name: str) -> str:
    """
    Strip trailing product/legal suffixes and normalise whitespace.

    Examples
    --------
    "Vanta Health Systems API"  -> "Vanta Health Systems"
    "Redwood Clinical Services" -> "Redwood Clinical Services"  (no change)
    "Cobalt Software, Inc."     -> "Cobalt Software"
    """
    prev = None
    while prev != name:          # iterate until stable (handles stacked suffixes)
        prev = name
        name = _SUFFIX_PATTERN.sub("", name).rstrip(" ,.")
    return name.strip()


def extract_account_name(title: str) -> str | None:
    """Extract and normalise the customer/account name from a meeting title."""
    if not title:
        return None
    for pattern in _TITLE_PATTERNS:
        match = pattern.search(title)
        if match:
            return _normalize_account(match.group(1).strip())
    return None


def add_account_column(df: pd.DataFrame) -> pd.DataFrame:
    """Add an 'account' column to the DataFrame by extracting from titles."""
    df = df.copy()
    df["account"] = df["title"].apply(extract_account_name)
    return df


# ============================================================================
# Scoring components
# ============================================================================

def _churn_signal_score(meetings: pd.DataFrame) -> float:
    """Each churn_signal key moment = 25 pts, capped at 50."""
    total = sum(
        1
        for moments in meetings["key_moments"]
        for km in (moments or [])
        if km.get("type") == "churn_signal"
    )
    return min(total * 25, 50)


def _sentiment_score_component(meetings: pd.DataFrame) -> float:
    """Low average sentiment → higher risk. Declining trend adds more."""
    if len(meetings) == 0:
        return 0.0
    avg = meetings["sentiment_score"].mean()
    points = 0.0
    if avg < 2.5:
        points += 30
    elif avg < 3.0:
        points += 15
    elif avg < 3.5:
        points += 5

    if len(meetings) >= 3:
        sorted_m = meetings.sort_values("start_time")
        mid = len(sorted_m) // 2
        early = sorted_m.iloc[:mid]["sentiment_score"].mean()
        late = sorted_m.iloc[mid:]["sentiment_score"].mean()
        if late < early - 0.5:
            points += 10

    return min(points, 40)


def _concern_score(meetings: pd.DataFrame) -> float:
    """Each concern key moment = 5 pts, capped at 20."""
    total = sum(
        1
        for moments in meetings["key_moments"]
        for km in (moments or [])
        if km.get("type") == "concern"
    )
    return min(total * 5, 20)


def _recent_negativity_score(meetings: pd.DataFrame) -> float:
    """Very recent negative meeting = extra risk signal."""
    if len(meetings) == 0:
        return 0.0
    most_recent = meetings.sort_values("start_time").iloc[-1]
    score = most_recent["sentiment_score"]
    if score < 2.5:
        return 15.0
    elif score < 3.0:
        return 5.0
    return 0.0


# ============================================================================
# Main scoring function
# ============================================================================

def score_account(df: pd.DataFrame, account: str) -> dict[str, Any]:
    """Score a single account for churn risk."""
    account_meetings = df[df["account"] == account].copy()

    if len(account_meetings) == 0:
        return {
            "account": account,
            "risk_score": 0,
            "risk_level": "Unknown",
            "meeting_count": 0,
            "components": {},
            "evidence": [],
            "recommendation": "No meetings found for this account.",
        }

    churn_pts = _churn_signal_score(account_meetings)
    sent_pts = _sentiment_score_component(account_meetings)
    concern_pts = _concern_score(account_meetings)
    recent_pts = _recent_negativity_score(account_meetings)
    total = churn_pts + sent_pts + concern_pts + recent_pts

    if total >= 70:
        risk_level = "Critical"
        recommendation = "Immediate executive outreach. High likelihood of churn."
    elif total >= 50:
        risk_level = "Alert"
        recommendation = "Schedule check-in within 1 week. Multiple warning signs."
    elif total >= 30:
        risk_level = "Watch"
        recommendation = "Monitor closely. Some negative signals present."
    else:
        risk_level = "Healthy"
        recommendation = "No immediate action needed."

    evidence = []
    for _, row in account_meetings.iterrows():
        for km in (row["key_moments"] or []):
            if km.get("type") in ("churn_signal", "concern"):
                evidence.append({
                    "meeting_id": row["meeting_id"],
                    "title": row["title"],
                    "type": km.get("type"),
                    "quote": km.get("text", "")[:200],
                    "speaker": km.get("speaker", ""),
                    "sentiment_score": float(row["sentiment_score"]),
                })

    evidence.sort(key=lambda e: (e["type"] != "churn_signal", e["sentiment_score"]))
    evidence = evidence[:5]

    most_recent_dt = account_meetings.sort_values("start_time").iloc[-1]["start_time"]
    most_recent_str = most_recent_dt.strftime("%Y-%m-%d") if pd.notna(most_recent_dt) else "N/A"

    return {
        "account": account,
        "risk_score": round(total, 1),
        "risk_level": risk_level,
        "meeting_count": int(len(account_meetings)),
        "avg_sentiment": round(float(account_meetings["sentiment_score"].mean()), 2),
        "most_recent_meeting": most_recent_str,
        "components": {
            "churn_signals": churn_pts,
            "sentiment": sent_pts,
            "concerns": concern_pts,
            "recent_negativity": recent_pts,
        },
        "evidence": evidence,
        "recommendation": recommendation,
    }


def score_all_accounts(df: pd.DataFrame, min_meetings: int = 1) -> list[dict]:
    """Score every customer account in the dataset, sorted by risk descending."""
    if "account" not in df.columns:
        df = add_account_column(df)

    customer_facing = df[df["call_type"].isin(["external", "support"])].copy()
    accounts = customer_facing["account"].dropna().unique()

    print(f"🔍 Scoring {len(accounts)} accounts for churn risk...")

    results = [
        score_account(customer_facing, account)
        for account in accounts
        if score_account(customer_facing, account)["meeting_count"] >= min_meetings
    ]
    results.sort(key=lambda x: x["risk_score"], reverse=True)

    OUTPUT_DIR.mkdir(exist_ok=True)
    with open(OUTPUT_DIR / "churn_rankings.json", "w") as f:
        json.dump(results, f, indent=2, default=str)
    print("✅ Saved rankings to outputs/churn_rankings.json")

    risk_dist = pd.Series([r["risk_level"] for r in results]).value_counts()
    print(f"\n📊 Risk distribution:")
    for level, count in risk_dist.items():
        print(f"   {level}: {count}")

    print(f"\n🚨 Top 5 highest-risk accounts:")
    for r in results[:5]:
        print(f"   [{r['risk_score']:.0f}] {r['account']} ({r['risk_level']}) — {r['meeting_count']} meetings")

    return results


# ============================================================================
# Charts
# ============================================================================

def chart_top_risk_accounts(rankings: list[dict], top_n: int = 15, save: bool = True) -> go.Figure:
    """Horizontal bar chart of top N highest-risk accounts."""
    top = rankings[:top_n]
    if not top:
        return go.Figure()

    df_chart = pd.DataFrame([
        {"account": r["account"], "risk_score": r["risk_score"],
         "risk_level": r["risk_level"], "meeting_count": r["meeting_count"]}
        for r in top
    ])

    color_map = {"Critical": "#d62728", "Alert": "#ff7f0e", "Watch": "#ffbb78", "Healthy": "#2ca02c"}

    fig = px.bar(
        df_chart,
        x="risk_score",
        y="account",
        orientation="h",
        color="risk_level",
        color_discrete_map=color_map,
        title=f"Top {top_n} Accounts by Churn Risk",
        labels={"risk_score": "Churn Risk Score (0–100)", "account": ""},
        hover_data=["meeting_count"],
    )
    fig.update_layout(
        height=max(400, 30 * top_n),
        yaxis={"categoryorder": "total ascending"},
    )

    if save:
        CHARTS_DIR.mkdir(parents=True, exist_ok=True)
        fig.write_html(CHARTS_DIR / "churn_risk_top.html")
        try:
            fig.write_image(CHARTS_DIR / "churn_risk_top.png")
        except Exception:
            pass
    return fig


def chart_risk_components(rankings: list[dict], top_n: int = 10, save: bool = True) -> go.Figure:
    """Stacked bar showing what drives each account's risk score."""
    top = rankings[:top_n]
    if not top:
        return go.Figure()

    component_data = [
        {"account": r["account"], "component": comp, "score": value}
        for r in top
        for comp, value in r["components"].items()
    ]

    fig = px.bar(
        pd.DataFrame(component_data),
        x="score",
        y="account",
        color="component",
        orientation="h",
        title=f"Risk Score Components — Top {top_n} Accounts",
        labels={"score": "Component Score", "account": ""},
    )
    fig.update_layout(height=max(400, 35 * top_n))

    if save:
        CHARTS_DIR.mkdir(parents=True, exist_ok=True)
        fig.write_html(CHARTS_DIR / "churn_risk_components.html")
        try:
            fig.write_image(CHARTS_DIR / "churn_risk_components.png")
        except Exception:
            pass
    return fig


if __name__ == "__main__":
    from src.loader import load_meetings
    from src.categorize import categorize_meetings
    df = load_meetings()
    df = categorize_meetings(df)
    df = add_account_column(df)
    rankings = score_all_accounts(df)
    chart_top_risk_accounts(rankings)
    chart_risk_components(rankings)
