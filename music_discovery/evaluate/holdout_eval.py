from __future__ import annotations
import os
from collections import defaultdict

import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

def _recall_at_k(recommended: list, relevant: set, k: int) -> float:
    if not relevant:
        return 0.0
    hits = sum(1 for r in recommended[:k] if r in relevant)
    return hits / len(relevant)

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



def run(
    interactions_path: str,
    song_embeddings_path: str,
    personas_dir: str,
    output_dir: str,
    k: int = 10,
    n_users: int | None = None,
) -> pd.DataFrame:
    """Evaluate taste-alignment on val holdout split.

    Compares three models — all ranked by taste, not popularity:
      - Popularity baseline (diagnostic only: proves our model isn't defaulting to charts)
      - ALS cosine (user embedding × song embeddings)
      - ALS + persona reranker (diversified via GMM personas)

    Metrics: Recall@K, NDCG@K, HitRate@K, ILD, mean_popularity_rank, long_tail_pct.
    """
    os.makedirs(output_dir, exist_ok=True)

    print("[holdout_eval] Loading data...")
    interactions = pd.read_parquet(interactions_path)
    emb_df = pd.read_parquet(song_embeddings_path)

    song_ids = emb_df["song_id"].to_numpy()
    emb_cols = [c for c in emb_df.columns if c.startswith("emb_")]
    emb = emb_df[emb_cols].to_numpy(dtype=np.float32)
    song_to_idx = {s: i for i, s in enumerate(song_ids)}

    popularity_ranks = _build_popularity_ranks(interactions)
    n_songs = len(song_ids)
    p80_threshold = int(np.percentile(list(popularity_ranks.values()), 80))
    pop_order = sorted(song_ids, key=lambda s: popularity_ranks.get(s, n_songs + 1))

    train_df = interactions[interactions["split"] == "train"]
    val_df = interactions[interactions["split"] == "val"]

    users = val_df["user_id"].unique()
    if n_users:
        users = users[:n_users]

    user_emb_path = song_embeddings_path.replace("song_embeddings", "user_embeddings")
    user_to_emb: dict[str, np.ndarray] = {}
    if os.path.exists(user_emb_path):
        udf = pd.read_parquet(user_emb_path)
        user_to_emb = {
            row["user_id"]: row[emb_cols].to_numpy(dtype=np.float32)
            for _, row in udf.iterrows()
        }

    from music_discovery.train.fit_personas import load_persona
    from music_discovery.models.scorer import recommend_diverse, DEFAULT_WEIGHTS

    rows_out = []
    print(f"[holdout_eval] Evaluating {len(users)} users at k={k}...")

    for uid in users:
        relevant = set(val_df[val_df["user_id"] == uid]["song_id"])
        if not relevant:
            continue

        train_songs = set(train_df[train_df["user_id"] == uid]["song_id"])
        candidates = [s for s in song_ids if s not in train_songs]
        cand_idx = [song_to_idx[s] for s in candidates]

        # popularity baseline (diagnostic — not used as a training signal)
        pop_recs = [s for s in pop_order if s in set(candidates)][:k]

        # ALS cosine ranking
        u_vec = user_to_emb.get(uid)
        if u_vec is not None and u_vec.any():
            scores = emb[cand_idx] @ u_vec
            top_idx = np.argsort(scores)[::-1][:k]
            als_recs = [candidates[i] for i in top_idx]
        else:
            als_recs = candidates[:k]

        # persona reranker
        try:
            persona = load_persona(os.path.join(personas_dir, uid))
            history_idx = np.array(
                [song_to_idx[s] for s in train_songs if s in song_to_idx], dtype=np.intp
            )
            top_n = recommend_diverse(
                persona=persona,
                candidate_emb=emb[cand_idx],
                candidate_artists=np.array([""] * len(cand_idx)),
                history_emb=emb[history_idx],
                history_artists=np.array([""] * len(history_idx)),
                history_indices=history_idx,
                n_recs=k,
                weights=DEFAULT_WEIGHTS,
            )
            persona_recs = [candidates[i] for i in top_n]
        except Exception:
            persona_recs = als_recs

        def _metrics(recs: list, label: str) -> dict:
            rec_idx = [song_to_idx[s] for s in recs if s in song_to_idx]
            pop_r = [popularity_ranks.get(s, n_songs + 1) for s in recs]
            return {
                "user_id": uid,
                "model": label,
                f"Recall@{k}": _recall_at_k(recs, relevant, k),
                f"NDCG@{k}": _ndcg_at_k(recs, relevant, k),
                f"HitRate@{k}": _hit_rate_at_k(recs, relevant, k),
                "ILD": _intra_list_diversity(rec_idx, emb),
                "mean_popularity_rank": float(np.mean(pop_r)),
                "long_tail_pct": float(np.mean([r > p80_threshold for r in pop_r])),
            }

        rows_out.extend([
            _metrics(pop_recs, "popularity"),
            _metrics(als_recs, "als"),
            _metrics(persona_recs, "persona_reranker"),
        ])

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
