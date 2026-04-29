from __future__ import annotations
import sqlite3
import pandas as pd


def _join_via_bridge(
    df: pd.DataFrame,
    bridge: pd.DataFrame,
    on: str = "track_id",
) -> pd.DataFrame:
    return df.merge(bridge[["song_id", "track_id"]], on=on, how="inner")


def _log_coverage(name: str, joined_ids: pd.Series, all_ids: pd.Series) -> None:
    pct = joined_ids.nunique() / max(all_ids.nunique(), 1) * 100
    print(f"[side_info] {name} coverage: {joined_ids.nunique():,} / {all_ids.nunique():,} songs ({pct:.1f}%)")
    if pct < 50:
        print(f"[side_info] WARNING: {name} coverage below 50% — check track_id bridge")


def load_song_tags(
    lastfm_tags_db: str,
    bridge: pd.DataFrame,
) -> pd.DataFrame:
    """Return (song_id, tag, weight) joined via song_id→track_id bridge.

    lastfm_tags.db uses a normalized schema:
      tids(tid TEXT)         — track IDs, rowid is the FK
      tags(tag TEXT)         — tag strings, rowid is the FK
      tid_tag(tid INT, tag INT, val FLOAT)  — integer FKs into tids/tags rowids
    """
    with sqlite3.connect(lastfm_tags_db) as conn:
        raw = pd.read_sql_query(
            """
            SELECT tids.tid AS track_id, tags.tag AS tag, tid_tag.val AS weight
            FROM tid_tag
            JOIN tids ON tid_tag.tid = tids.rowid
            JOIN tags ON tid_tag.tag = tags.rowid
            """,
            conn,
        )

    result = _join_via_bridge(raw, bridge, on="track_id")[["song_id", "tag", "weight"]]
    _log_coverage("tags", result["song_id"], bridge["song_id"])
    return result.reset_index(drop=True)


def _parse_similars_packed(df: pd.DataFrame) -> pd.DataFrame:

    df = df[df["target"].notna() & (df["target"] != "")].copy()

    # split each packed string into a list of tokens
    df["tokens"] = df["target"].str.split(",")
    exploded = df[["tid", "tokens"]].explode("tokens").reset_index(drop=True)
    exploded["tokens"] = exploded["tokens"].str.strip()

    # tokens alternate: even positions = track_id, odd positions = score
    exploded["pos"] = exploded.groupby(level=0).cumcount() if False else exploded.index % 2
    # re-index position within each source track
    exploded["_grp"] = (exploded["tid"] != exploded["tid"].shift()).cumsum()
    exploded["pos"] = exploded.groupby("_grp").cumcount() % 2

    even = exploded[exploded["pos"] == 0][["tid", "tokens"]].rename(columns={"tokens": "similar_track_id"})
    odd  = exploded[exploded["pos"] == 1]["tokens"].rename("score")

    result = even.reset_index(drop=True)
    result["score"] = pd.to_numeric(odd.reset_index(drop=True), errors="coerce")
    result = result.dropna(subset=["score"]).rename(columns={"tid": "track_id"})
    return result.reset_index(drop=True)


def load_song_similars(
    lastfm_similars_db: str,
    bridge: pd.DataFrame,
) -> pd.DataFrame:

    with sqlite3.connect(lastfm_similars_db) as conn:
        src = pd.read_sql_query("SELECT tid, target FROM similars_src", conn)

    unpacked = _parse_similars_packed(src)
    if unpacked.empty:
        return pd.DataFrame(columns=["song_id", "similar_song_id", "score"])

    track_to_song = bridge.set_index("track_id")["song_id"]

    unpacked = unpacked[unpacked["track_id"].isin(track_to_song.index)].copy()
    unpacked["song_id"] = unpacked["track_id"].map(track_to_song)

    unpacked = unpacked[unpacked["similar_track_id"].isin(track_to_song.index)].copy()
    unpacked["similar_song_id"] = unpacked["similar_track_id"].map(track_to_song)

    result = unpacked[["song_id", "similar_song_id", "score"]].drop_duplicates()
    _log_coverage("similars", result["song_id"], bridge["song_id"])
    return result.reset_index(drop=True)


def load_artist_similars(artist_similarity_db: str) -> pd.DataFrame:

    with sqlite3.connect(artist_similarity_db) as conn:
        df = pd.read_sql_query(
            "SELECT target AS artist_id, similar AS similar_artist_id FROM similarity",
            conn,
        )
    return df.reset_index(drop=True)
