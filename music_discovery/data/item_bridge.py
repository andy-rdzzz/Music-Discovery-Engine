from __future__ import annotations
import sqlite3
import pandas as pd


def _load_bad_song_ids(sid_mismatches_path: str) -> set[str]:
    bad: set[str] = set()
    
    with open(sid_mismatches_path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            if len(parts) > 2:
                bad.add(parts[1])
        
    return bad


def load_item_bridge(
    track_metadata_db: str,
    sid_mismatches_path: str,
) -> pd.DataFrame:
    
    bad_ids = _load_bad_song_ids(sid_mismatches_path)

    with sqlite3.connect(track_metadata_db) as conn:
        df = pd.read_sql_query(
            "SELECT song_id, track_id, artist_name, title FROM songs",
            conn,
        )

    before = len(df)
    df = df[~df["song_id"].isin(bad_ids)].copy()
    removed = before - len(df)
    if removed:
        print(f"[item_bridge] Removed {removed:,} songs matching sid_mismatches ({len(df):,} remain)")
    
    df = df.drop_duplicates(subset=["song_id"]).reset_index(drop=True)
    return df
    
    