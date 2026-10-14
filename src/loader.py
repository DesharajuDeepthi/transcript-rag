from __future__ import annotations
"""
Data loader for Transcript Intelligence.

Loads 100 meeting transcripts from data/dataset/ and produces a flat pandas
DataFrame for downstream analysis. Uses pre-computed fields from summary.json
where available (summary text, topics, sentiment, key moments).

Usage:
    from src.loader import load_meetings
    df = load_meetings()
"""

import json
from pathlib import Path
from typing import Any

import pandas as pd
from tqdm import tqdm


DATA_DIR = Path("data/dataset")
OUTPUT_DIR = Path("outputs")


def _load_json(filepath: Path) -> dict | list:
    """Load a JSON file, returning empty dict on failure."""
    try:
        with open(filepath, "r", encoding="utf-8") as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError) as e:
        print(f"⚠️  Warning: Could not load {filepath}: {e}")
        return {}


def _parse_meeting(meeting_dir: Path) -> dict[str, Any]:
    """Parse a single meeting directory into a flat record."""
    meeting_id = meeting_dir.name

    info = _load_json(meeting_dir / "meeting-info.json")
    summary = _load_json(meeting_dir / "summary.json")
    speakers_meta = _load_json(meeting_dir / "speaker-meta.json")
    transcript = _load_json(meeting_dir / "transcript.json")

    key_moments = summary.get("keyMoments", []) or []
    key_moment_types = list({km.get("type", "") for km in key_moments if km.get("type")})

    speakers = list(speakers_meta.values()) if isinstance(speakers_meta, dict) else []

    transcript_data = transcript.get("data", []) if isinstance(transcript, dict) else []
    sentences = [s.get("sentence", "") for s in transcript_data if isinstance(s, dict)]
    full_text = " ".join(sentences)

    start_time = info.get("startTime")
    end_time = info.get("endTime")

    action_items = summary.get("actionItems", []) or []
    topics = summary.get("topics", []) or []

    return {
        "meeting_id": meeting_id,
        "title": info.get("title", ""),
        "organizer": info.get("organizerEmail", ""),
        "host": info.get("host", ""),
        "start_time": pd.to_datetime(start_time, utc=True) if start_time else None,
        "end_time": pd.to_datetime(end_time, utc=True) if end_time else None,
        "duration_min": float(info.get("duration", 0.0)),
        "participant_count": len(info.get("allEmails", [])),
        "participants": info.get("allEmails", []),
        "summary": summary.get("summary", ""),
        "action_items": action_items,
        "action_item_count": len(action_items),
        "topics": topics,
        "topic_count": len(topics),
        "overall_sentiment": summary.get("overallSentiment", "unknown"),
        "sentiment_score": float(summary.get("sentimentScore", 3.0)),
        "key_moments": key_moments,
        "key_moment_count": len(key_moments),
        "key_moment_types": key_moment_types,
        "has_churn_signal": any(km.get("type") == "churn_signal" for km in key_moments),
        "has_technical_issue": any(km.get("type") == "technical_issue" for km in key_moments),
        "has_concern": any(km.get("type") == "concern" for km in key_moments),
        "has_positive_pivot": any(km.get("type") == "positive_pivot" for km in key_moments),
        "speakers": speakers,
        "speaker_count": len(speakers),
        "transcript_sentence_count": len(sentences),
        "transcript_text": full_text,
    }


def load_meetings(data_dir: Path = DATA_DIR, save_csv: bool = True) -> pd.DataFrame:
    """
    Load all meetings from the dataset directory.

    Args:
        data_dir: Path to dataset directory (default: data/dataset/)
        save_csv: Whether to save the flat DataFrame as CSV

    Returns:
        pandas DataFrame with one row per meeting.
    """
    if not data_dir.exists():
        raise FileNotFoundError(f"Dataset directory not found: {data_dir}")

    meeting_dirs = sorted([d for d in data_dir.iterdir() if d.is_dir()])
    if not meeting_dirs:
        raise ValueError(f"No meeting directories found in {data_dir}")

    print(f"📂 Loading {len(meeting_dirs)} meetings from {data_dir}...")

    records = []
    for meeting_dir in tqdm(meeting_dirs, desc="Loading"):
        try:
            records.append(_parse_meeting(meeting_dir))
        except Exception as e:
            print(f"⚠️  Failed to parse {meeting_dir.name}: {e}")

    df = pd.DataFrame(records)
    df = df.sort_values("start_time").reset_index(drop=True)

    if save_csv:
        OUTPUT_DIR.mkdir(exist_ok=True)
        df_csv = df.copy()
        for col in ["participants", "action_items", "topics", "key_moments",
                    "key_moment_types", "speakers"]:
            df_csv[col] = df_csv[col].apply(json.dumps)
        df_csv.to_csv(OUTPUT_DIR / "meetings_flat.csv", index=False)
        print(f"✅ Saved {len(df)} meetings to {OUTPUT_DIR / 'meetings_flat.csv'}")

    return df


def quick_stats(df: pd.DataFrame) -> dict:
    """Print and return quick summary statistics about the dataset."""
    stats = {
        "total_meetings": len(df),
        "date_range": (
            df["start_time"].min().strftime("%Y-%m-%d") if df["start_time"].notna().any() else "N/A",
            df["start_time"].max().strftime("%Y-%m-%d") if df["start_time"].notna().any() else "N/A",
        ),
        "total_duration_hours": round(df["duration_min"].sum() / 60, 1),
        "avg_duration_min": round(df["duration_min"].mean(), 1),
        "avg_participants": round(df["participant_count"].mean(), 1),
        "total_action_items": int(df["action_item_count"].sum()),
        "total_unique_speakers": len(set().union(*df["speakers"])),
        "sentiment_distribution": df["overall_sentiment"].value_counts().to_dict(),
        "meetings_with_churn_signal": int(df["has_churn_signal"].sum()),
        "meetings_with_technical_issue": int(df["has_technical_issue"].sum()),
        "meetings_with_concern": int(df["has_concern"].sum()),
        "meetings_with_positive_pivot": int(df["has_positive_pivot"].sum()),
    }

    print("\n📊 Dataset Quick Stats:")
    print(f"  Total meetings:        {stats['total_meetings']}")
    print(f"  Date range:            {stats['date_range'][0]} → {stats['date_range'][1]}")
    print(f"  Total duration:        {stats['total_duration_hours']} hours")
    print(f"  Avg duration:          {stats['avg_duration_min']} min")
    print(f"  Avg participants:      {stats['avg_participants']}")
    print(f"  Total action items:    {stats['total_action_items']}")
    print(f"  Unique speakers:       {stats['total_unique_speakers']}")
    print(f"\n  Sentiment distribution:")
    for sent, count in stats["sentiment_distribution"].items():
        print(f"    {sent}: {count}")
    print(f"\n  Key moment flags:")
    print(f"    Churn signals:       {stats['meetings_with_churn_signal']}")
    print(f"    Technical issues:    {stats['meetings_with_technical_issue']}")
    print(f"    Concerns:            {stats['meetings_with_concern']}")
    print(f"    Positive pivots:     {stats['meetings_with_positive_pivot']}")

    return stats


if __name__ == "__main__":
    df = load_meetings()
    quick_stats(df)
