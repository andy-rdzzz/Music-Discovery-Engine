from __future__ import annotations
import os
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from tqdm import tqdm

# Worker-global state — set once per process via initializer, never pickled per task
_W: dict = {}


def _init_worker(song_ids, emb, song_to_idx, song_to_artist, pop_idx_order, popularity_ranks, n_songs, p80_threshold, personas_dir, k):
    _W["song_ids"] = song_ids
    _W["emb"] = emb
    _W["song_to_idx"] = song_to_idx
    _W["song_to_artist"] = song_to_artist
    _W["pop_idx_order"] = pop_idx_order
    _W["popularity_ranks"] = popularity_ranks
    _W["n_songs"] = n_songs
    _W["p80_threshold"] = p80_threshold
    _W["personas_dir"] = personas_dir
    _W["k"] = k


def _recall_at_k(recommended: list, relevant: set, k: int) -> float:
    if not relevant:
        return 0.0
    return sum(1 for r in recommended[:k] if r in relevant) / len(relevant)


def _ndcg_at_k(recommended: list, relevant: set, k: int) -> float:
    if not relevant:
        return 0.0
    dcg = sum(
        1.0 / np.log2(rank + 2)
        for rank, r in enumerate(recommended[:k])
        if r in relevant
    )
    idcg = sum(1.0 / np.log2(rank + 2) for rank in range(min(len(relevant), k)))
    return dcg / idcg if idcg else 0.0


def _hit_rate_at_k(recommended: list, relevant: set, k: int) -> float:
    return float(any(r in relevant for r in recommended[:k]))


def _intra_list_diversity(rec_indices: list[int], emb: np.ndarray) -> float:
    if len(rec_indices) < 2:
        return 0.0
    vecs = emb[rec_indices]
    sims = cosine_similarity(vecs)
    n = len(rec_indices)
    upper = sims[np.triu_indices(n, k=1)]
    return float(1.0 - upper.mean())


def _build_popularity_ranks(interactions: pd.DataFrame) -> dict[str, int]:
    train = interactions[interactions["split"] == "train"]
    counts = train.groupby("song_id")["play_count"].sum().sort_values(ascending=False)
    return {sid: rank for rank, sid in enumerate(counts.index, 1)}


def _eval_user(uid: str, val_relevant: set, test_relevant: set, train_songs: set, u_vec: np.ndarray | None) -> list[dict]:
    from music_discovery.train.fit_personas import load_persona
    from music_discovery.models.scorer import recommend_diverse, optimize_weights, DEFAULT_WEIGHTS

    song_ids         = _W["song_ids"]
    emb              = _W["emb"]
    song_to_idx      = _W["song_to_idx"]
    song_to_artist   = _W["song_to_artist"]
    pop_idx_order    = _W["pop_idx_order"]
    popularity_ranks = _W["popularity_ranks"]
    n_songs          = _W["n_songs"]
    p80_threshold    = _W["p80_threshold"]
    personas_dir     = _W["personas_dir"]
    k                = _W["k"]

    # Fix 2: C-level set difference instead of 374k-iteration Python loop
    train_arr = np.array([song_to_idx[s] for s in train_songs if s in song_to_idx], dtype=np.intp)
    cand_idx = np.setdiff1d(np.arange(len(song_ids)), train_arr)
    candidates = song_ids[cand_idx]

    # Fix 1: real artist names instead of blank arrays
    candidate_artists = np.array([song_to_artist.get(s, "") for s in candidates])
    history_artists   = np.array([song_to_artist.get(song_ids[i], "") for i in train_arr])

    # Fix 1: early-exit integer scan — stops at k hits, not after 374k iterations
    train_idx_set = set(train_arr.tolist())
    pop_recs_idx = []
    for idx in pop_idx_order:
        if idx not in train_idx_set:
            pop_recs_idx.append(idx)
            if len(pop_recs_idx) == k:
                break
    pop_recs = song_ids[pop_recs_idx].tolist()

    als_scores = None
    if u_vec is not None and u_vec.any():
        als_scores = emb[cand_idx] @ u_vec
        top_pos = np.argsort(als_scores)[::-1][:k]
        als_recs = candidates[top_pos].tolist()
    else:
        als_recs = candidates[:k].tolist()

    try:
        persona = load_persona(os.path.join(personas_dir, uid))
    except FileNotFoundError:
        persona = None

    if persona is not None:
        try:
            # Fix 3: map val songs → positions in candidates, then optimize weights
            cand_pos_map = {int(idx): pos for pos, idx in enumerate(cand_idx)}
            val_indices = np.array([
                cand_pos_map[song_to_idx[s]]
                for s in val_relevant
                if s in song_to_idx and song_to_idx[s] in cand_pos_map
            ], dtype=np.intp)

            weights = optimize_weights(
                candidate_emb=emb[cand_idx],
                candidate_artists=candidate_artists,
                persona=persona,
                user_history_emb=emb[train_arr],
                user_history_artists=history_artists,
                held_out_indices=val_indices,
            )

            top_n = recommend_diverse(
                persona=persona,
                candidate_emb=emb[cand_idx],
                candidate_artists=candidate_artists,
                history_emb=emb[train_arr],
                history_artists=history_artists,
                history_indices=np.empty(0, dtype=np.intp),
                n_recs=k,
                weights=weights,
                relevance_scores=als_scores,
            )
            persona_recs = candidates[top_n].tolist()
        except Exception as e:
            import warnings
            warnings.warn(f"[holdout_eval] persona reranker failed for {uid}: {e}")
            persona_recs = als_recs
    else:
        persona_recs = als_recs

    # Fix 3: evaluate all models on test split (val used only for weight optimization)
    def _metrics(recs: list, label: str) -> dict:
        rec_idx = [song_to_idx[s] for s in recs if s in song_to_idx]
        pop_r = [popularity_ranks.get(s, n_songs + 1) for s in recs]
        return {
            "user_id": uid,
            "model": label,
            f"Recall@{k}": _recall_at_k(recs, test_relevant, k),
            f"NDCG@{k}": _ndcg_at_k(recs, test_relevant, k),
            f"HitRate@{k}": _hit_rate_at_k(recs, test_relevant, k),
            "ILD": _intra_list_diversity(rec_idx, emb),
            "mean_popularity_rank": float(np.mean(pop_r)),
            "long_tail_pct": float(np.mean([r > p80_threshold for r in pop_r])),
        }

    return [
        _metrics(pop_recs, "popularity"),
        _metrics(als_recs, "als"),
        _metrics(persona_recs, "persona_reranker"),
    ]


def run(
    interactions_path: str,
    song_embeddings_path: str,
    personas_dir: str,
    output_dir: str,
    k: int = 10,
    n_users: int | None = None,
    n_jobs: int | None = None,
) -> pd.DataFrame:
    os.makedirs(output_dir, exist_ok=True)

    print("[holdout_eval] Loading interactions...")
    interactions = pd.read_parquet(interactions_path)
    print(f"  {len(interactions):,} rows")

    # Fix 1: load artist metadata
    processed_dir = os.path.dirname(interactions_path)
    songs_path = os.path.join(processed_dir, "songs.parquet")
    if os.path.exists(songs_path):
        songs_df = pd.read_parquet(songs_path, columns=["song_id", "artist_name"])
        song_to_artist: dict[str, str] = dict(zip(songs_df["song_id"], songs_df["artist_name"].fillna("")))
        print(f"  {len(song_to_artist):,} song→artist entries loaded")
    else:
        song_to_artist = {}
        print("  [warn] songs.parquet not found — artist novelty will be zero")

    print("[holdout_eval] Loading song embeddings...")
    emb_df = pd.read_parquet(song_embeddings_path)
    song_ids = emb_df["song_id"].to_numpy()
    emb_cols = [c for c in emb_df.columns if c.startswith("emb_")]
    emb = emb_df[emb_cols].to_numpy(dtype=np.float32)
    song_to_idx = {s: i for i, s in enumerate(song_ids)}
    print(f"  {len(song_ids):,} songs × {emb.shape[1]} dims")

    print("[holdout_eval] Building popularity ranks...")
    popularity_ranks = _build_popularity_ranks(interactions)
    n_songs = len(song_ids)
    p80_threshold = int(np.percentile(list(popularity_ranks.values()), 80))
    pop_idx_order = np.argsort([popularity_ranks.get(s, n_songs + 1) for s in song_ids])

    train_df = interactions[interactions["split"] == "train"]
    val_df   = interactions[interactions["split"] == "val"]
    test_df  = interactions[interactions["split"] == "test"]

    users = val_df["user_id"].unique()
    if n_users:
        users = users[:n_users]
    user_set = set(users)
    print(f"[holdout_eval] Target users: {len(users)}")

    print("[holdout_eval] Loading user embeddings...")
    user_emb_path = song_embeddings_path.replace("song_embeddings", "user_embeddings")
    user_to_emb: dict[str, np.ndarray] = {}
    if os.path.exists(user_emb_path):
        udf = pd.read_parquet(user_emb_path)
        udf = udf[udf["user_id"].isin(user_set)]
        emb_cols_u = [c for c in udf.columns if c.startswith("emb_")]
        emb_matrix_u = udf[emb_cols_u].to_numpy(dtype=np.float32)
        user_to_emb = dict(zip(udf["user_id"].to_numpy(), emb_matrix_u))
        print(f"  {len(user_to_emb)} user embeddings loaded")

    print("[holdout_eval] Building per-user song sets...")
    train_filtered = train_df[train_df["user_id"].isin(user_set)]
    val_filtered   = val_df[val_df["user_id"].isin(user_set)]
    test_filtered  = test_df[test_df["user_id"].isin(user_set)]

    train_songs_by_user: dict[str, set] = defaultdict(set)
    for uid, grp in train_filtered.groupby("user_id"):
        train_songs_by_user[uid] = set(grp["song_id"])

    val_songs_by_user: dict[str, set] = defaultdict(set)
    for uid, grp in val_filtered.groupby("user_id"):
        val_songs_by_user[uid] = set(grp["song_id"])

    test_songs_by_user: dict[str, set] = defaultdict(set)
    for uid, grp in test_filtered.groupby("user_id"):
        test_songs_by_user[uid] = set(grp["song_id"])

    workers = n_jobs or min(os.cpu_count() or 1, 8)
    print(f"[holdout_eval] Evaluating {len(users)} users at k={k} ({workers} workers)...")
    print("[holdout_eval] Note: weights optimized on val, metrics reported on test split")

    init_args = (song_ids, emb, song_to_idx, song_to_artist, pop_idx_order, popularity_ranks, n_songs, p80_threshold, personas_dir, k)

    rows_out = []
    futures = {}
    with ProcessPoolExecutor(max_workers=workers, initializer=_init_worker, initargs=init_args) as pool:
        for uid in users:
            val_relevant  = val_songs_by_user.get(uid, set())
            test_relevant = test_songs_by_user.get(uid, set())
            if not val_relevant or not test_relevant:
                continue
            f = pool.submit(
                _eval_user,
                uid,
                val_relevant,
                test_relevant,
                train_songs_by_user.get(uid, set()),
                user_to_emb.get(uid),
            )
            futures[f] = uid

        with tqdm(total=len(futures), desc="Evaluating users") as bar:
            for future in as_completed(futures):
                rows_out.extend(future.result())
                bar.update(1)

    results = pd.DataFrame(rows_out)
    results.to_csv(os.path.join(output_dir, "holdout_metrics.csv"), index=False)

    summary = results.groupby("model").mean(numeric_only=True).drop(columns=["user_id"], errors="ignore")
    summary.to_csv(os.path.join(output_dir, "holdout_summary.csv"))

    print("[holdout_eval] Summary:")
    print(summary.to_string())
    print(f"\n[holdout_eval] Results → {output_dir}/holdout_metrics.csv")
    return results


if __name__ == "__main__":
    import yaml

    with open("configs/default.yaml") as f:
        cfg = yaml.safe_load(f)

    d = cfg.get("data", {})
    run(
        interactions_path=f"{d.get('processed_dir', 'data/processed')}/interactions.parquet",
        song_embeddings_path="models/embeddings/song_embeddings.parquet",
        personas_dir="models/personas",
        output_dir="results/holdout",
        k=10,
    )
