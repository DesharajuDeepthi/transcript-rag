"""
DataStore — singleton that loads and caches the meetings DataFrame.

Imported by all tool modules; never call load_meetings() directly from tools.
Thread-safe: first caller blocks until load completes, subsequent callers
return the cached DataFrame immediately.
"""

from __future__ import annotations

import threading
from pathlib import Path

import pandas as pd

# src.* is available via `pip install -e .` — no sys.path manipulation needed
from src.loader import load_meetings
from src.categorize import categorize_meetings
from src.churn_scorer import add_account_column

# ---------------------------------------------------------------------------
# Internal singleton state
# ---------------------------------------------------------------------------

_lock: threading.Lock = threading.Lock()
_df: pd.DataFrame | None = None

# Dataset directory: <project_root>/data/dataset
_DATA_DIR: Path = Path(__file__).parent.parent / "data" / "dataset"


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def get_df() -> pd.DataFrame:
    """
    Return the fully-prepared meetings DataFrame, loading it on first call.

    Thread-safe: concurrent first-callers block; only one load runs.
    Subsequent calls return the cached DataFrame with no I/O.
    """
    global _df

    if _df is not None:          # fast path — no lock needed after first load
        return _df

    with _lock:
        if _df is not None:      # another thread may have loaded while we waited
            return _df

        print("[DataStore] Loading...")

        raw = load_meetings(data_dir=_DATA_DIR, save_csv=False)
        categorized = categorize_meetings(raw, use_llm=False)
        ready = add_account_column(categorized)

        _df = ready
        print(f"[DataStore] Ready. {len(_df)} meetings loaded.")

    return _df


def reset() -> None:
    """
    Drop the cached DataFrame (tests only — not called in normal operation).
    """
    global _df
    with _lock:
        _df = None
