"""Recommendation quality metrics: Discovery@K, NDCG@K, artist repetition, persona coverage."""

from __future__ import annotations

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from collections import Counter
from pathlib import Path
from tqdm import tqdm

from music_discovery.train.fit_personas import load_persona
from music_discovery.models.scorer import recommend_diverse, optimize_weights, DEFAULT_WEIGHTS


def _ndcg_at_k(top_n: np.ndarray, relevant_set: set, k: int) -> float:
    rel = np.array([1.0 if idx in relevant_set else 0.0 for idx in top_n[:k]])
    if rel.sum() == 0:
        return 0.0
    dcg = (rel / np.log2(np.arange(2, len(rel) + 2))).sum()
    n_ideal = min(len(relevant_set), k)
    ideal = np.zeros(k)
    ideal[:n_ideal] = 1.0
    idcg = (ideal / np.log2(np.arange(2, k + 2))).sum()
    return float(dcg / idcg) if idcg > 0 else 0.0


def _discovery_at_k(top_n: np.ndarray, discovery_set: set, k: int) -> float:
    hits = sum(1 for idx in top_n[:k] if idx in discovery_set)
    return hits / min(k, len(top_n))


def _artist_repetition_rate(top_n: np.ndarray, artists: np.ndarray, k: int) -> float:
    rec_artists = [artists[idx] for idx in top_n[:k]]
    counts = Counter(rec_artists)
    repeat = sum(1 for a in rec_artists if counts[a] > 1)
    return repeat / len(rec_artists) if rec_artists else 0.0


def _persona_coverage(top_n: np.ndarray, candidate_emb: np.ndarray, persona, k: int) -> float:
    top_k = top_n[:k]
    if len(top_k) == 0 or persona.k <= 1:
        return 1.0
    dominant = persona.dominant_persona(candidate_emb[top_k])
    return len(set(dominant.tolist())) / persona.k


def run(
    interactions_path: str,
    song_embeddings_path: str,
    personas_dir: str,
    output_dir: str,
    n_users: int = 20,
    k: int = 10,
    n_recs: int = 20,
):
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    interactions = pd.read_parquet(interactions_path)
    emb_df = pd.read_parquet(song_embeddings_path)

    songs_path = Path(interactions_path).parent / "songs.parquet"
    songs_df = pd.read_parquet(songs_path) if songs_path.exists() else pd.DataFrame(columns=["song_id", "artist_name"])

    emb_cols = [c for c in emb_df.columns if c.startswith("emb_")]
    candidate_emb = emb_df[emb_cols].to_numpy(dtype=np.float32)
    song_to_idx = {s: i for i, s in enumerate(emb_df["song_id"])}

    merged = emb_df[["song_id"]].merge(songs_df[["song_id", "artist_name"]], on="song_id", how="left")
    candidate_artists = merged["artist_name"].fillna("").to_numpy()

    train_df = interactions[interactions["split"] == "train"]
    val_df = interactions[interactions["split"] == "val"]

    personas_path = Path(personas_dir)
    available_users = [d.name for d in personas_path.iterdir() if (d / "persona.pkl").exists()]
    users = available_users[:n_users]
    print(f"[rec_quality] Evaluating {len(users)} users at K={k}, n_recs={n_recs}")

    records: list[dict] = []
    all_top_n: list[np.ndarray] = []
    all_train_sets: list[set] = []
    all_discovery_sets: list[set] = []

    for user_id in tqdm(users, desc="Users"):
        persona = load_persona(personas_path / user_id)

        train_songs = set(train_df[train_df["user_id"] == user_id]["song_id"])
        val_songs = set(val_df[val_df["user_id"] == user_id]["song_id"])

        train_idx = np.array([song_to_idx[s] for s in train_songs if s in song_to_idx], dtype=np.intp)
        val_idx = np.array([song_to_idx[s] for s in val_songs if s in song_to_idx], dtype=np.intp)

        if len(train_idx) == 0:
            continue

        history_emb = candidate_emb[train_idx]
        history_artists = candidate_artists[train_idx]
        train_set = set(train_idx.tolist())
        discovery_set = set(val_idx.tolist())

        weights = optimize_weights(
            candidate_emb, candidate_artists, persona,
            history_emb, history_artists, val_idx,
        )

        top_n = recommend_diverse(
            persona=persona,
            candidate_emb=candidate_emb,
            candidate_artists=candidate_artists,
            history_emb=history_emb,
            history_artists=history_artists,
            history_indices=train_idx,
            n_recs=n_recs,
            weights=weights,
        )

        records.append({
            "user_id": user_id,
            "discovery_at_k": _discovery_at_k(top_n, discovery_set, k),
            "ndcg_at_k": _ndcg_at_k(top_n, discovery_set, k),
            "artist_repetition_rate": _artist_repetition_rate(top_n, candidate_artists, k),
            "persona_coverage": _persona_coverage(top_n, candidate_emb, persona, k),
            "n_personas": persona.k,
        })
        all_top_n.append(top_n)
        all_train_sets.append(train_set)
        all_discovery_sets.append(discovery_set)

    if not records:
        print("[rec_quality] No results generated.")
        return

    results_df = pd.DataFrame(records)
    results_df.to_csv(out / "recommendation_quality.csv", index=False)

    means = results_df[["discovery_at_k", "ndcg_at_k", "artist_repetition_rate", "persona_coverage"]].mean()
    print(f"\n[rec_quality] Results at K={k} (mean over {len(results_df)} users):")
    print(f"  Discovery@{k}:        {means['discovery_at_k']:.3f}")
    print(f"  NDCG@{k}:             {means['ndcg_at_k']:.3f}")
    print(f"  Artist repetition:    {means['artist_repetition_rate']:.3f}  (target <0.20)")
    print(f"  Persona coverage:     {means['persona_coverage']:.3f}  (target →1.0)")

    # ── Graph 1: Discovery@K at K=5,10,20 ──
    ks = [ki for ki in [5, 10, 20] if ki <= n_recs]
    disc_vals = [np.mean([_discovery_at_k(t, s, ki) for t, s in zip(all_top_n, all_discovery_sets)]) for ki in ks]

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.bar([f"K={ki}" for ki in ks], disc_vals, color="#4C72B0", alpha=0.85, edgecolor="white")
    ax.set_ylabel("Discovery Rate")
    ax.set_title("Discovery Rate at K (val holdout)")
    ax.set_ylim(0, 1)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out / "discovery_at_k.png", dpi=150)
    plt.close(fig)

    # ── Graph 2: Persona coverage box plot ──
    fig, ax = plt.subplots(figsize=(5, 4))
    ax.boxplot(results_df["persona_coverage"], patch_artist=True,
               boxprops=dict(facecolor="#4C72B0", alpha=0.7))
    ax.set_ylabel("Persona Coverage")
    ax.set_title(f"Fraction of Personas Represented in Top-{k}")
    ax.set_xticks([1])
    ax.set_xticklabels([f"K={k}"])
    fig.tight_layout()
    fig.savefig(out / "persona_coverage.png", dpi=150)
    plt.close(fig)

    print(f"[rec_quality] Plots saved to {out}")
