from __future__ import annotations

import os
from concurrent.futures import ThreadPoolExecutor, as_completed

import duckdb
import pandas as pd

from music_discovery.data.item_bridge import load_item_bridge
from music_discovery.data.side_info import load_artist_similars, load_song_similars, load_song_tags


def _build_interactions(
    taste_profile_path: str,
    songs: pd.DataFrame,
    min_play_count: int,
    sample_n_users: int | None,
    holdout_k: int,
    seed: int = 42,
) -> pd.DataFrame:
    con = duckdb.connect()

    # register the songs bridge so DuckDB can join against it
    con.register("songs_bridge", songs[["song_id"]])

    con.execute(f"""
        CREATE TABLE raw_interactions AS
        SELECT column0 AS user_id,
               column1 AS song_id,
               column2::INTEGER AS play_count,
               ln(1 + column2::DOUBLE) AS weight
        FROM read_csv('{taste_profile_path}', delim='\t', header=false)
        WHERE column2 >= {min_play_count}
    """)

    if sample_n_users is not None:
        con.execute(f"""
            CREATE TABLE sampled AS
            SELECT r.* FROM raw_interactions r
            INNER JOIN (
                SELECT user_id FROM raw_interactions
                GROUP BY user_id
                ORDER BY random()
                LIMIT {sample_n_users}
            ) s USING (user_id)
        """)
        con.execute("DROP TABLE raw_interactions")
        con.execute("ALTER TABLE sampled RENAME TO raw_interactions")

    # inner join to drop interactions with no metadata
    con.execute("""
        CREATE TABLE interactions AS
        SELECT r.user_id, r.song_id, r.play_count, r.weight
        FROM raw_interactions r
        INNER JOIN songs_bridge s USING (song_id)
    """)

    before = con.execute("SELECT COUNT(*) FROM raw_interactions").fetchone()[0]
    after  = con.execute("SELECT COUNT(*) FROM interactions").fetchone()[0]
    print(f"[interactions] Dropped {before - after:,} interactions with no metadata ({after:,} remain)")

    # assign random holdout splits in SQL — no Python loop over users
    con.execute(f"""
        CREATE TABLE interactions_split AS
        WITH ranked AS (
            SELECT *,
                   setseed({seed / 2**31}),
                   ROW_NUMBER() OVER (
                       PARTITION BY user_id
                       ORDER BY random()
                   ) AS rn,
                   COUNT(*) OVER (PARTITION BY user_id) AS user_n
            FROM interactions
        )
        SELECT user_id, song_id, play_count, weight,
            CASE
                WHEN user_n > {holdout_k * 2} AND rn <= {holdout_k}
                    THEN 'val'
                WHEN user_n > {holdout_k * 2} AND rn <= {holdout_k * 2}
                    THEN 'test'
                ELSE 'train'
            END AS split
        FROM ranked
    """)

    df = con.execute("SELECT * FROM interactions_split").df()
    counts = df["split"].value_counts().to_dict()
    print(f"[interactions] Split counts: {counts}")
    con.close()
    return df


def build(
    taste_profile_path: str,
    track_metadata_db: str,
    sid_mismatches_path: str,
    lastfm_tags_db: str,
    lastfm_similars_db: str,
    artist_similarity_db: str,
    processed_dir: str,
    min_play_count: int = 1,
    sample_n_users: int | None = None,
    holdout_k: int = 10,
    skip_side_info: bool = False,
) -> None:

    os.makedirs(processed_dir, exist_ok=True)

    print("[interactions] Loading item bridge...")
    songs = load_item_bridge(track_metadata_db, sid_mismatches_path)

    print("[interactions] Building interactions via DuckDB (read + join + split)...")
    interactions = _build_interactions(
        taste_profile_path=taste_profile_path,
        songs=songs,
        min_play_count=min_play_count,
        sample_n_users=sample_n_users,
        holdout_k=holdout_k,
    )

    print("[interactions] Writing core parquets...")
    interactions.to_parquet(os.path.join(processed_dir, "interactions.parquet"), index=False)
    songs.to_parquet(os.path.join(processed_dir, "songs.parquet"), index=False)
    print(f"  interactions : {len(interactions):,} rows")
    print(f"  songs        : {len(songs):,} rows")

    if skip_side_info:
        print("[interactions] Skipping side-info (skip_side_info=True). Run again without flag to load tags/similars.")
        return

    print("[interactions] Loading side information in parallel (tags, similars, artist graph)...")
    tasks = {
        "song_tags":      lambda: load_song_tags(lastfm_tags_db, songs),
        "song_similars":  lambda: load_song_similars(lastfm_similars_db, songs),
        "artist_similars": lambda: load_artist_similars(artist_similarity_db),
    }
    results: dict[str, pd.DataFrame] = {}
    with ThreadPoolExecutor(max_workers=3) as executor:
        futures = {executor.submit(fn): name for name, fn in tasks.items()}
        for future in as_completed(futures):
            name = futures[future]
            results[name] = future.result()
            print(f"  [done] {name}: {len(results[name]):,} rows")

    results["song_tags"].to_parquet(os.path.join(processed_dir, "song_tags.parquet"), index=False)
    results["song_similars"].to_parquet(os.path.join(processed_dir, "song_similars.parquet"), index=False)
    results["artist_similars"].to_parquet(os.path.join(processed_dir, "artist_similars.parquet"), index=False)

    print("[interactions] Done.")


if __name__ == "__main__":
    import yaml

    with open("configs/default.yaml") as f:
        cfg = yaml.safe_load(f)

    d = cfg["data"]
    build(
        taste_profile_path=d["taste_profile"],
        track_metadata_db=d["track_metadata_db"],
        sid_mismatches_path=d["sid_mismatches"],
        lastfm_tags_db=d["lastfm_tags_db"],
        lastfm_similars_db=d["lastfm_similars_db"],
        artist_similarity_db=d["artist_similarity_db"],
        processed_dir=d["processed_dir"],
        min_play_count=d.get("min_play_count", 1),
        sample_n_users=d.get("sample_n_users"),
        holdout_k=d.get("holdout_k", 10),
    )
