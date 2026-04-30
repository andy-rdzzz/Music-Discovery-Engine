from __future__ import annotations
import os
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics.pairwise import cosine_similarity
from tqdm import tqdm

# Worker-global state — set once per process via initializer, never pickled per task
_W: dict = {}


def _init_worker(song_ids, emb, song_to_idx, song_to_artist, pop_idx_order, popularity_ranks, n_songs, p80_threshold, personas_dir, k, rerank_top_m, song_catalog):
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
    _W["rerank_top_m"] = rerank_top_m
    _W["song_catalog"] = song_catalog


def _recall_at_k(recommended: list, relevant: set, k: int, covered_relevant: set | None = None) -> tuple[float, float]:
    if not relevant:
        return 0.0, 0.0
    hits = sum(1 for r in recommended[:k] if r in relevant)
    raw = hits / len(relevant)
    if covered_relevant is None:
        covered_relevant = relevant
    adj = hits / len(covered_relevant) if covered_relevant else 0.0
    return raw, adj


def _ndcg_at_k(recommended: list, relevant: set, k: int, covered_relevant: set | None = None) -> tuple[float, float]:
    if not relevant:
        return 0.0, 0.0
    dcg = sum(
        1.0 / np.log2(rank + 2)
        for rank, r in enumerate(recommended[:k])
        if r in relevant
    )
    idcg_raw = sum(1.0 / np.log2(rank + 2) for rank in range(min(len(relevant), k)))
    raw = dcg / idcg_raw if idcg_raw else 0.0
    if covered_relevant is None:
        covered_relevant = relevant
    idcg_adj = sum(1.0 / np.log2(rank + 2) for rank in range(min(len(covered_relevant), k)))
    adj = dcg / idcg_adj if idcg_adj else 0.0
    return raw, adj


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


def _write_summary_plots(summary: pd.DataFrame, output_dir: str, k: int) -> None:
    if summary.empty:
        return

    summary_plot = summary.copy()
    summary_plot.index = summary_plot.index.astype(str)
    models = summary_plot.index.tolist()

    raw_adj_pairs = [
        (f"Recall@{k}", f"Recall_adj@{k}", "Recall"),
        (f"NDCG@{k}", f"NDCG_adj@{k}", "NDCG"),
    ]
    available_pairs = [(raw, adj, label) for raw, adj, label in raw_adj_pairs if raw in summary_plot.columns and adj in summary_plot.columns]

    if available_pairs:
        fig, axes = plt.subplots(1, len(available_pairs), figsize=(6 * len(available_pairs), 4), squeeze=False)
        x = np.arange(len(models))
        width = 0.35

        for ax, (raw_col, adj_col, label) in zip(axes[0], available_pairs):
            raw_vals = summary_plot[raw_col].to_numpy(dtype=float)
            adj_vals = summary_plot[adj_col].to_numpy(dtype=float)
            ax.bar(x - width / 2, raw_vals, width=width, label="Raw", color="#4C72B0")
            ax.bar(x + width / 2, adj_vals, width=width, label="Adjusted", color="#55A868")
            ax.set_xticks(x)
            ax.set_xticklabels(models, rotation=15, ha="right")
            ax.set_ylim(0, max(1.0, float(np.nanmax(np.r_[raw_vals, adj_vals])) * 1.15))
            ax.set_title(f"{label}@{k}")
            ax.grid(axis="y", alpha=0.25)
            ax.legend()

        fig.suptitle("Holdout Relevance Metrics: Raw vs Coverage-Adjusted")
        fig.tight_layout()
        fig.savefig(os.path.join(output_dir, "holdout_adjusted_metrics.png"), dpi=150, bbox_inches="tight")
        plt.close(fig)

    tradeoff_cols = [
        ("ILD", "Intra-list Diversity", "#C44E52"),
        ("long_tail_pct", "Long-tail Share", "#8172B2"),
        ("mean_popularity_rank", "Mean Popularity Rank", "#CCB974"),
    ]
    available_tradeoffs = [(col, title, color) for col, title, color in tradeoff_cols if col in summary_plot.columns]
    if not available_tradeoffs:
        return

    fig, axes = plt.subplots(1, len(available_tradeoffs), figsize=(5 * len(available_tradeoffs), 4), squeeze=False)
    x = np.arange(len(models))
    for ax, (col, title, color) in zip(axes[0], available_tradeoffs):
        vals = summary_plot[col].to_numpy(dtype=float)
        ax.bar(x, vals, color=color, alpha=0.9)
        ax.set_xticks(x)
        ax.set_xticklabels(models, rotation=15, ha="right")
        ax.set_title(title)
        ax.grid(axis="y", alpha=0.25)
    fig.suptitle("Holdout Trade-offs: Diversity and Popularity Bias")
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, "holdout_tradeoffs.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)


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
    rerank_top_m     = _W["rerank_top_m"]
    song_catalog     = _W["song_catalog"]

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

    used_als_fallback = u_vec is None or not u_vec.any()
    als_scores = None
    if not used_als_fallback:
        als_scores = emb[cand_idx] @ u_vec
        top_pos = np.argsort(als_scores)[::-1][:k]
        als_recs = candidates[top_pos].tolist()
    else:
        als_recs = candidates[:k].tolist()

    covered_val  = val_relevant  & song_catalog
    covered_test = test_relevant & song_catalog
    coverage_val  = len(covered_val)  / len(val_relevant)  if val_relevant  else 0.0
    coverage_test = len(covered_test) / len(test_relevant) if test_relevant else 0.0

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
                rerank_top_m=rerank_top_m,
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
        recall_raw, recall_adj = _recall_at_k(recs, test_relevant, k, covered_test)
        ndcg_raw, ndcg_adj = _ndcg_at_k(recs, test_relevant, k, covered_test)
        return {
            "user_id": uid,
            "model": label,
            f"Recall@{k}": recall_raw,
            f"Recall_adj@{k}": recall_adj,
            f"NDCG@{k}": ndcg_raw,
            f"NDCG_adj@{k}": ndcg_adj,
            f"HitRate@{k}": _hit_rate_at_k(recs, test_relevant, k),
            "ILD": _intra_list_diversity(rec_idx, emb),
            "mean_popularity_rank": float(np.mean(pop_r)),
            "long_tail_pct": float(np.mean([r > p80_threshold for r in pop_r])),
            "val_coverage": coverage_val,
            "test_coverage": coverage_test,
            "als_fallback": float(used_als_fallback),
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
    rerank_top_m: int | None = 500,
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
        n_emb_users_total = len(udf)
        udf = udf[udf["user_id"].isin(user_set)]
        emb_cols_u = [c for c in udf.columns if c.startswith("emb_")]
        emb_matrix_u = udf[emb_cols_u].to_numpy(dtype=np.float32)
        user_to_emb = dict(zip(udf["user_id"].to_numpy(), emb_matrix_u))
        print(f"  {len(user_to_emb)} user embeddings loaded (total in file: {n_emb_users_total})")

        n_interactions_users_total = interactions["user_id"].nunique()
        if abs(n_interactions_users_total - n_emb_users_total) / max(n_interactions_users_total, 1) > 0.10:
            print(
                f"  [warn] Total interaction users ({n_interactions_users_total}) vs total embedding users "
                f"({n_emb_users_total}) differ by >10%. Artifacts may be from different processing runs."
            )

        overlap = len(user_to_emb)
        overlap_ratio = overlap / len(user_set) if user_set else 0.0
        print(f"  User embedding overlap: {overlap}/{len(user_set)} ({overlap_ratio:.1%})")
        _MIN_OVERLAP = 0.80
        if overlap_ratio < _MIN_OVERLAP:
            raise RuntimeError(
                f"Artifact mismatch: only {overlap_ratio:.1%} of eval users have embeddings "
                f"({overlap}/{len(user_set)}). "
                "Re-run `train als` before evaluating, or check artifact paths."
            )

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

    song_catalog = set(song_to_idx.keys())
    init_args = (song_ids, emb, song_to_idx, song_to_artist, pop_idx_order, popularity_ranks, n_songs, p80_threshold, personas_dir, k, rerank_top_m, song_catalog)

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

    if "als_fallback" in results.columns:
        fallback_rate = results["als_fallback"].mean()
        print(f"[holdout_eval] ALS fallback rate (missing u_vec): {fallback_rate:.1%}")

    if "test_coverage" in results.columns:
        avg_coverage = results["test_coverage"].mean()
        print(f"[holdout_eval] Avg test-set song catalog coverage: {avg_coverage:.1%}")

    _write_summary_plots(summary, output_dir, k)
    print(f"[holdout_eval] Plots → {output_dir}/holdout_adjusted_metrics.png, {output_dir}/holdout_tradeoffs.png")

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
