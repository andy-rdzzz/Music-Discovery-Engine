"""Recommendation quality metrics: Replay@K, Discovery@K, NDCG@K, artist repetition, persona coverage."""

from __future__ import annotations

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
from tqdm import tqdm

from music_discovery.train.fit_personas import load_persona
from music_discovery.models.scorer import recommend_diverse, optimize_weights, DEFAULT_WEIGHTS


# ---------------------------------------------------------------------------
# Metric helpers
# ---------------------------------------------------------------------------

def _ndcg_at_k(top_n: np.ndarray, relevant_set: set, k: int) -> float:
    rel = np.array([1.0 if idx in relevant_set else 0.0 for idx in top_n[:k]])
    if rel.sum() == 0:
        return 0.0
    dcg = (rel / np.log2(np.arange(2, len(rel) + 2))).sum()
    ideal = np.sort(rel)[::-1]
    idcg = (ideal / np.log2(np.arange(2, len(ideal) + 2))).sum()
    return float(dcg / idcg) if idcg > 0 else 0.0


def _replay_at_k(top_n: np.ndarray, train_set: set, k: int) -> float:
    hits = sum(1 for idx in top_n[:k] if idx in train_set)
    return hits / min(k, len(top_n))


def _discovery_at_k(top_n: np.ndarray, discovery_set: set, k: int) -> float:
    hits = sum(1 for idx in top_n[:k] if idx in discovery_set)
    return hits / min(k, len(top_n))


def _artist_repetition_rate(top_n: np.ndarray, artists: np.ndarray, k: int) -> float:
    """Fraction of top-K recs whose artist appears more than once in the list."""
    from collections import Counter
    rec_artists = [artists[idx] for idx in top_n[:k]]
    counts = Counter(rec_artists)
    repeat = sum(1 for a in rec_artists if counts[a] > 1)
    return repeat / len(rec_artists) if rec_artists else 0.0


def _persona_coverage(top_n: np.ndarray, candidate_emb: np.ndarray, persona, k: int) -> float:
    """Fraction of user's personas with at least 1 track in top-K."""
    top_k = top_n[:k]
    if len(top_k) == 0 or persona.k <= 1:
        return 1.0
    dominant = persona.dominant_persona(candidate_emb[top_k])
    return len(set(dominant.tolist())) / persona.k


# ---------------------------------------------------------------------------
# Main runner
# ---------------------------------------------------------------------------

def run(
    processed_dir: str,
    embeddings_path: str,
    personas_dir: str,
    output_dir: str,
    n_users: int = 20,
    k: int = 10,
    n_recs: int = 20,
):
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    history_df = pd.read_parquet(Path(processed_dir) / "user_history.parquet")
    emb_df = pd.read_parquet(embeddings_path)
    emb_cols = [c for c in emb_df.columns if c.startswith("emb_")]
    candidate_emb = emb_df[emb_cols].to_numpy(dtype=np.float32)
    candidate_artists = emb_df["artist_norm"].to_numpy()

    personas_path = Path(personas_dir)
    available_users = [d.name for d in personas_path.iterdir() if (d / "persona.pkl").exists()]
    users = available_users[:n_users]
    print(f"[rec_quality] Evaluating {len(users)} users at K={k}, n_recs={n_recs}")

    has_discovery_col = "is_discovery" in history_df.columns
    records: list[dict] = []
    all_top_n: list[np.ndarray] = []
    all_train_sets: list[set] = []
    all_discovery_sets: list[set] = []

    for user_id in tqdm(users, desc="Users"):
        persona = load_persona(personas_path / user_id)

        train_rows = history_df[(history_df["user_id"] == user_id) & (history_df["split"] == "train")]
        train_indices = train_rows["track_idx"].to_numpy()
        valid_train = train_indices[train_indices < len(candidate_emb)]

        if len(valid_train) == 0:
            continue

        history_emb = candidate_emb[valid_train]
        history_artists = candidate_artists[valid_train]
        train_set = set(valid_train.tolist())

        eval_rows = history_df[(history_df["user_id"] == user_id) & (history_df["split"] == "eval")]
        disc_rows = eval_rows[eval_rows["is_discovery"]] if has_discovery_col else eval_rows
        disc_indices = disc_rows["track_idx"].to_numpy()
        discovery_set = set(disc_indices[disc_indices < len(candidate_emb)].tolist())

        held_out = np.array(list(discovery_set))
        weights = optimize_weights(
            candidate_emb, candidate_artists, persona,
            history_emb, history_artists, held_out,
        )

        top_n = recommend_diverse(
            persona=persona,
            candidate_emb=candidate_emb,
            candidate_artists=candidate_artists,
            history_emb=history_emb,
            history_artists=history_artists,
            history_indices=valid_train,
            n_recs=n_recs,
            weights=weights,
        )

        records.append({
            "user_id": user_id,
            "replay_at_k": _replay_at_k(top_n, train_set, k),
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

    means = results_df[["replay_at_k", "discovery_at_k", "ndcg_at_k",
                         "artist_repetition_rate", "persona_coverage"]].mean()
    print(f"\n[rec_quality] Results at K={k} (mean over {len(results_df)} users):")
    print(f"  Replay@{k}:           {means['replay_at_k']:.3f}")
    print(f"  Discovery@{k}:        {means['discovery_at_k']:.3f}")
    print(f"  NDCG@{k}:             {means['ndcg_at_k']:.3f}")
    print(f"  Artist repetition:    {means['artist_repetition_rate']:.3f}  (target <0.20)")
    print(f"  Persona coverage:     {means['persona_coverage']:.3f}  (target →1.0)")

    # ── Graph 1: Replay vs Discovery at K=5,10,20 ──
    # Use stored top_n arrays — just slice at different K values
    ks = [ki for ki in [5, 10, 20] if ki <= n_recs]
    replay_vals = [np.mean([_replay_at_k(t, s, ki) for t, s in zip(all_top_n, all_train_sets)]) for ki in ks]
    disc_vals = [np.mean([_discovery_at_k(t, s, ki) for t, s in zip(all_top_n, all_discovery_sets)]) for ki in ks]

    fig, ax = plt.subplots(figsize=(7, 4))
    x = np.arange(len(ks))
    width = 0.35
    ax.bar(x - width / 2, replay_vals, width, label="Replay@K", color="#DD8452", alpha=0.85)
    ax.bar(x + width / 2, disc_vals, width, label="Discovery@K", color="#4C72B0", alpha=0.85)
    ax.set_xticks(x)
    ax.set_xticklabels([f"K={ki}" for ki in ks])
    ax.set_ylabel("Rate")
    ax.set_title("Replay vs Discovery Rate at K")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out / "replay_vs_discovery.png", dpi=150)
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
