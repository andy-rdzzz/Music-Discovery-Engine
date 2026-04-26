from __future__ import annotations

import pandas as pd
from pathlib import Path
from typing import Iterator

TSV_COLS = ["user_id", "timestamp", "artist_id", "artist_name", "track_id", "track_name"]
SESSION_GAP_MINUTES = 30


def load_lastfm_events(tsv_path: str | Path, nrows: int | None = None) -> pd.DataFrame:
    """Parse Last.fm 1K TSV. Returns df sorted by (user_id, timestamp)."""
    path = Path(tsv_path)
    if not path.exists():
        raise FileNotFoundError(f"Last.fm TSV not found: {path}")

    df = pd.read_csv(
        path,
        sep="\t",
        header=None,
        names=TSV_COLS,
        nrows=nrows,
        on_bad_lines="skip",
    )
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True, errors="coerce")
    df = df.dropna(subset=["user_id", "timestamp", "artist_name", "track_name"])

    df["artist_norm"] = df["artist_name"].str.lower().str.strip()
    df["track_norm"] = df["track_name"].str.lower().str.strip()

    df = df.sort_values(["user_id", "timestamp"]).reset_index(drop=True)
    print(f"[lastfm] Loaded {len(df):,} events for {df['user_id'].nunique():,} users")
    return df


def segment_sessions(
    user_events: pd.DataFrame,
    gap_minutes: int = SESSION_GAP_MINUTES,
    min_session_length: int = 2,
) -> pd.DataFrame:
    """Add session_id column to user events (single user df, sorted by timestamp)."""
    events = user_events.copy()
    delta = events["timestamp"].diff()
    gap = pd.Timedelta(minutes=gap_minutes)
    
    new_session = (delta > gap) | (delta.isna()) #new session
    events["session_id"] = new_session.cumsum()

    # Drop sessions shorter than min
    session_sizes = events.groupby("session_id")["session_id"].transform("count")
    events = events[session_sizes >= min_session_length]
    return events.reset_index(drop=True)


def iter_user_sessions(
    df: pd.DataFrame,
    gap_minutes: int = SESSION_GAP_MINUTES,
    min_session_length: int = 2,
) -> Iterator[tuple[str, pd.DataFrame]]:
    """Yield (user_id, sessions_df) for each user, sessions already segmented."""
    for user_id, user_df in df.groupby("user_id"):
        segmented = segment_sessions(user_df, gap_minutes, min_session_length)
        if not segmented.empty:
            yield user_id, segmented
