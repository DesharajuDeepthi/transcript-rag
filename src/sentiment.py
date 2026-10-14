from __future__ import annotations
"""
Sentiment analysis module for Transcript Intelligence.

Aggregates pre-computed sentiment scores from summary.json across multiple
dimensions: call type, sub-theme, time, and meeting characteristics.
Does NOT re-run sentiment analysis — the data is already there.

Usage:
    from src.sentiment import generate_sentiment_report
    report = generate_sentiment_report(df)
"""

import json
from pathlib import Path
from typing import Any

import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go


OUTPUT_DIR = Path("outputs")
CHARTS_DIR = OUTPUT_DIR / "charts"


# ============================================================================
# Aggregation functions (each is "tool-shaped")
# ============================================================================

def sentiment_by_call_type(df: pd.DataFrame) -> dict[str, Any]:
    """Aggregate sentiment statistics by call type."""
    stats = df.groupby("call_type")["sentiment_score"].agg(
        ["mean", "median", "std", "count", "min", "max"]
    ).round(2)

    means = stats["mean"]
    most_negative = means.idxmin()
    most_positive = means.idxmax()
    spread = round(means.max() - means.min(), 2)

    insight = (
        f"{most_positive.title()} calls are the most positive (mean {means[most_positive]:.2f}); "
        f"{most_negative} calls are the most negative (mean {means[most_negative]:.2f}). "
        f"The gap of {spread} points is "
        f"{'large' if spread > 1.0 else 'moderate' if spread > 0.5 else 'small'}."
    )

    return {
        "by_call_type": stats.to_dict(orient="index"),
        "insight": insight,
    }


def sentiment_by_sub_theme(df: pd.DataFrame) -> dict[str, Any]:
    """Aggregate sentiment by sub-theme to find which topics drive negativity."""
    stats = df.groupby("sub_theme")["sentiment_score"].agg(
        ["mean", "count"]
    ).round(2).sort_values("mean")

    significant = stats[stats["count"] >= 2]
    top_negative = significant.head(3)

    examples: dict[str, list] = {}
    for theme in top_negative.index:
        examples[theme] = (
            df[df["sub_theme"] == theme]
            .nsmallest(2, "sentiment_score")[["meeting_id", "title", "sentiment_score"]]
            .to_dict(orient="records")
        )

    insight = (
        "The most negative sub-themes are: "
        + ", ".join([f"{t} ({stats.loc[t, 'mean']:.2f})" for t in top_negative.index])
        + ". These are leading indicators of business pain."
    ) if not top_negative.empty else "Insufficient data."

    return {
        "by_sub_theme": stats.to_dict(orient="index"),
        "top_negative_themes": top_negative.to_dict(orient="index"),
        "examples": examples,
        "insight": insight,
    }


def sentiment_over_time(df: pd.DataFrame, freq: str = "W") -> dict[str, Any]:
    """Compute sentiment trend over time (weekly or monthly)."""
    df_ts = df.dropna(subset=["start_time"]).copy()
    df_ts = df_ts.set_index("start_time")

    ts = (
        df_ts.groupby([pd.Grouper(freq=freq), "call_type"])["sentiment_score"]
        .agg(["mean", "count"])
        .reset_index()
    )

    overall = df_ts["sentiment_score"].resample(freq).mean().dropna()
    slope = 0.0
    trend_direction = "flat"
    if len(overall) >= 2:
        x = np.arange(len(overall))
        slope, _ = np.polyfit(x, overall.values, 1)
        trend_direction = "improving" if slope > 0.05 else "declining" if slope < -0.05 else "flat"

    return {
        "series": ts.to_dict(orient="records"),
        "trend_slope": round(float(slope), 3),
        "trend_direction": trend_direction,
        "insight": (
            f"Overall sentiment is {trend_direction} over the analyzed period "
            f"(slope: {slope:+.3f} per {freq.lower()})."
        ),
    }


def sentiment_vs_characteristics(df: pd.DataFrame) -> dict[str, Any]:
    """Correlate sentiment with meeting characteristics."""
    corrs = {
        "duration_min": df["sentiment_score"].corr(df["duration_min"]),
        "participant_count": df["sentiment_score"].corr(df["participant_count"]),
        "action_item_count": df["sentiment_score"].corr(df["action_item_count"]),
        "topic_count": df["sentiment_score"].corr(df["topic_count"]),
    }
    corrs = {k: round(float(v), 3) for k, v in corrs.items() if not np.isnan(v)}

    strongest = max(corrs.items(), key=lambda kv: abs(kv[1]))

    return {
        "correlations": corrs,
        "strongest_correlation": {"variable": strongest[0], "value": strongest[1]},
        "insight": (
            f"{strongest[0]} has the strongest correlation with sentiment ({strongest[1]:+.2f}). "
            f"More {strongest[0].replace('_', ' ')} = "
            f"{'higher' if strongest[1] > 0 else 'lower'} sentiment."
        ),
    }


def find_negative_outliers(df: pd.DataFrame, top_n: int = 10) -> list[dict]:
    """Find the most negative meetings with context."""
    negative = df.nsmallest(top_n, "sentiment_score")
    results = []
    for _, row in negative.iterrows():
        results.append({
            "meeting_id": row["meeting_id"],
            "title": row["title"],
            "call_type": row.get("call_type", ""),
            "sub_theme": row.get("sub_theme", ""),
            "sentiment_score": float(row["sentiment_score"]),
            "overall_sentiment": row["overall_sentiment"],
            "has_churn_signal": bool(row["has_churn_signal"]),
            "has_concern": bool(row["has_concern"]),
            "key_moment_count": int(row["key_moment_count"]),
            "summary_excerpt": (row["summary"] or "")[:200] + "...",
        })
    return results


def find_positive_outliers(df: pd.DataFrame, top_n: int = 10) -> list[dict]:
    """Find the most positive meetings."""
    positive = df.nlargest(top_n, "sentiment_score")
    results = []
    for _, row in positive.iterrows():
        results.append({
            "meeting_id": row["meeting_id"],
            "title": row["title"],
            "call_type": row.get("call_type", ""),
            "sub_theme": row.get("sub_theme", ""),
            "sentiment_score": float(row["sentiment_score"]),
            "has_positive_pivot": bool(row["has_positive_pivot"]),
            "summary_excerpt": (row["summary"] or "")[:200] + "...",
        })
    return results


# ============================================================================
# Charts
# ============================================================================

def chart_sentiment_by_call_type(df: pd.DataFrame, save: bool = True) -> go.Figure:
    """Box plot of sentiment by call type."""
    fig = px.box(
        df,
        x="call_type",
        y="sentiment_score",
        color="call_type",
        title="Sentiment Distribution by Call Type",
        labels={"sentiment_score": "Sentiment Score (1–5)", "call_type": "Call Type"},
        points="outliers",
    )
    fig.update_layout(showlegend=False, height=450)

    if save:
        CHARTS_DIR.mkdir(parents=True, exist_ok=True)
        fig.write_html(CHARTS_DIR / "sentiment_by_call_type.html")
        try:
            fig.write_image(CHARTS_DIR / "sentiment_by_call_type.png")
        except Exception:
            pass
    return fig


def chart_sentiment_over_time(df: pd.DataFrame, save: bool = True) -> go.Figure:
    """Line chart of weekly sentiment by call type."""
    df_ts = df.dropna(subset=["start_time"]).copy()
    ts = (
        df_ts.set_index("start_time")
        .groupby([pd.Grouper(freq="W"), "call_type"])["sentiment_score"]
        .mean()
        .reset_index()
    )

    fig = px.line(
        ts,
        x="start_time",
        y="sentiment_score",
        color="call_type",
        title="Sentiment Trend Over Time (Weekly Average)",
        labels={"sentiment_score": "Avg Sentiment", "start_time": "Week"},
        markers=True,
    )
    fig.update_layout(height=450)

    if save:
        CHARTS_DIR.mkdir(parents=True, exist_ok=True)
        fig.write_html(CHARTS_DIR / "sentiment_over_time.html")
        try:
            fig.write_image(CHARTS_DIR / "sentiment_over_time.png")
        except Exception:
            pass
    return fig


def chart_sentiment_heatmap(df: pd.DataFrame, save: bool = True) -> go.Figure:
    """Heatmap of sentiment by call_type × sub_theme."""
    if "sub_theme" not in df.columns or "call_type" not in df.columns:
        return go.Figure()

    pivot = df.pivot_table(
        index="sub_theme",
        columns="call_type",
        values="sentiment_score",
        aggfunc="mean",
    )

    fig = px.imshow(
        pivot,
        text_auto=".2f",
        color_continuous_scale="RdYlGn",
        range_color=[1, 5],
        title="Average Sentiment: Sub-Theme × Call Type",
        labels={"color": "Sentiment"},
        aspect="auto",
    )
    fig.update_layout(height=500)

    if save:
        CHARTS_DIR.mkdir(parents=True, exist_ok=True)
        fig.write_html(CHARTS_DIR / "sentiment_heatmap.html")
        try:
            fig.write_image(CHARTS_DIR / "sentiment_heatmap.png")
        except Exception:
            pass
    return fig


def chart_sentiment_distribution(df: pd.DataFrame, save: bool = True) -> go.Figure:
    """Histogram of overall sentiment labels."""
    counts = df["overall_sentiment"].value_counts().reset_index()
    counts.columns = ["sentiment", "count"]

    fig = px.bar(
        counts,
        x="sentiment",
        y="count",
        color="sentiment",
        title="Distribution of Overall Sentiment Labels",
        labels={"count": "Meeting Count", "sentiment": "Sentiment"},
    )
    fig.update_layout(showlegend=False, height=400)

    if save:
        CHARTS_DIR.mkdir(parents=True, exist_ok=True)
        fig.write_html(CHARTS_DIR / "sentiment_distribution.html")
        try:
            fig.write_image(CHARTS_DIR / "sentiment_distribution.png")
        except Exception:
            pass
    return fig


# ============================================================================
# Master report
# ============================================================================

def generate_sentiment_report(df: pd.DataFrame) -> dict:
    """Generate the complete sentiment analysis report and all charts."""
    print("📊 Generating sentiment report...")

    report = {
        "by_call_type": sentiment_by_call_type(df),
        "by_sub_theme": sentiment_by_sub_theme(df),
        "over_time": sentiment_over_time(df),
        "vs_characteristics": sentiment_vs_characteristics(df),
        "negative_outliers": find_negative_outliers(df),
        "positive_outliers": find_positive_outliers(df),
    }

    chart_sentiment_by_call_type(df)
    chart_sentiment_over_time(df)
    chart_sentiment_heatmap(df)
    chart_sentiment_distribution(df)

    OUTPUT_DIR.mkdir(exist_ok=True)
    with open(OUTPUT_DIR / "sentiment_report.json", "w") as f:
        json.dump(report, f, indent=2, default=str)

    print("✅ Sentiment report saved to outputs/sentiment_report.json")
    print("✅ Charts saved to outputs/charts/")
    print(f"\n🔑 Key findings:")
    print(f"  • {report['by_call_type']['insight']}")
    print(f"  • {report['by_sub_theme']['insight']}")
    print(f"  • {report['over_time']['insight']}")
    print(f"  • {report['vs_characteristics']['insight']}")

    return report


if __name__ == "__main__":
    from src.loader import load_meetings
    from src.categorize import categorize_meetings
    df = load_meetings()
    df = categorize_meetings(df)
    generate_sentiment_report(df)
