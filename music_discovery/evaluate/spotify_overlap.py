"""Experiment 3 — Popularity bias comparison vs popularity-ranked baseline."""

from __future__ import annotations

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
from tqdm import tqdm

from music_discovery.train.fit_personas import load_persona
from music_discovery.models.scorer import score_candidates, DEFAULT_WEIGHTS
from music_discovery.models.scorer import sonic_fit, emotional_fit, surprise_score


def _popularity_baseline(history_df: pd.DataFrame, candidate_artists: np.ndarray, n: int) -> np.ndarray:
    """Return indices of top-N tracks by global play count (popularity proxy)."""
    play_counts = history_df["track_idx"].value_counts()
    top_n = play_counts.head(n).index.to_numpy()
    return top_n[top_n < len(candidate_artists)]


def _popularity_percentile(track_indices: np.ndarray, global_counts: pd.Series) -> np.ndarray:
    """Return popularity percentile (0=most obscure, 1=most popular) for each track."""
    counts = global_counts.reindex(track_indices, fill_value=0).to_numpy()
    ranks = pd.Series(counts).rank(pct=True).to_numpy()
    return ranks


def _jaccard(set_a: set, set_b: set) -> float:
    if not set_a and not set_b:
        return 1.0
    return len(set_a & set_b) / len(set_a | set_b)


def _intra_list_diversity(emb: np.ndarray) -> float:
    """Mean pairwise cosine distance in embedding space."""
    if len(emb) < 2:
        return 0.0
    sims = emb @ emb.T
    n = len(emb)
    mask = np.triu(np.ones((n, n), dtype=bool), k=1)
    distances = 1.0 - sims[mask]
    return float(distances.mean())


def run(
    processed_dir: str,
    embeddings_path: str,
    personas_dir: str,
    output_dir: str,
    n_users: int = 10,
    n_recs: int = 20,
):
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    history_df = pd.read_parquet(Path(processed_dir) / "user_history.parquet")
    emb_df = pd.read_parquet(embeddings_path)
    emb_cols = [c for c in emb_df.columns if c.startswith("emb_")]
    candidate_emb = emb_df[emb_cols].to_numpy(dtype=np.float32)
    candidate_artists = emb_df["artist_norm"].to_numpy()

    global_counts = history_df["track_idx"].value_counts()

    from music_discovery.data.features import FEATURE_VECTOR_COLS
    track_features_df = pd.read_parquet(Path(processed_dir) / "track_features.parquet")
    feat_matrix = track_features_df[FEATURE_VECTOR_COLS].to_numpy(dtype=np.float32)

    def feat_kl(our_idx: np.ndarray, base_idx: np.ndarray, feat_col: str) -> float:
        col_idx = FEATURE_VECTOR_COLS.index(feat_col)
        our_vals = feat_matrix[our_idx, col_idx]
        base_vals = feat_matrix[base_idx, col_idx]
        if len(our_vals) == 0 or len(base_vals) == 0:
            return np.nan
        bins = np.linspace(0, 1, 20)
        our_hist, _ = np.histogram(our_vals, bins=bins, density=True)
        base_hist, _ = np.histogram(base_vals, bins=bins, density=True)
        eps = 1e-10
        our_hist = (our_hist + eps) / (our_hist + eps).sum()
        base_hist = (base_hist + eps) / (base_hist + eps).sum()
        return float(np.sum(our_hist * np.log(our_hist / base_hist)))

    personas_path = Path(personas_dir)
    available_users = [d.name for d in personas_path.iterdir() if (d / "persona.pkl").exists()]
    users = available_users[:n_users]
    print(f"[exp3] Running on {len(users)} users, {n_recs} recs each")

    baseline_top_n = _popularity_baseline(history_df, candidate_artists, n_recs)

    records: list[dict] = []
    our_diversities: list[float] = []
    baseline_diversities: list[float] = []

    for user_id in tqdm(users, desc="Users"):
        persona = load_persona(personas_path / user_id)
        train = history_df[(history_df["user_id"] == user_id) & (history_df["split"] == "train")]
        train_idx = train["track_idx"].to_numpy()
        valid_train = train_idx[train_idx < len(candidate_emb)]

        if len(valid_train) == 0:
            continue

        history_emb = candidate_emb[valid_train]
        history_artists = candidate_artists[valid_train]

        scores = score_candidates(candidate_emb, candidate_artists, persona, history_emb, history_artists)
        scores[valid_train] = -np.inf
        our_top_n = np.argsort(-scores)[:n_recs]

        # Popularity percentiles
        our_pctile = _popularity_percentile(our_top_n, global_counts)
        base_pctile = _popularity_percentile(baseline_top_n, global_counts)

        # Jaccard overlap
        jaccard = _jaccard(set(our_top_n.tolist()), set(baseline_top_n.tolist()))

        # KL divergence per audio feature (12D feature space, not embeddings)
        kl_scores = {f: feat_kl(our_top_n, baseline_top_n, f) for f in FEATURE_VECTOR_COLS}

        # Intra-list diversity
        our_div = _intra_list_diversity(candidate_emb[our_top_n])
        base_div = _intra_list_diversity(candidate_emb[baseline_top_n])
        our_diversities.append(our_div)
        baseline_diversities.append(base_div)

        records.append({
            "user_id": user_id,
            "our_mean_popularity": float(our_pctile.mean()),
            "base_mean_popularity": float(base_pctile.mean()),
            "jaccard": jaccard,
            "our_diversity": our_div,
            "base_diversity": base_div,
            **{f"kl_{f}": v for f, v in kl_scores.items()},
            "our_top_n": our_top_n.tolist(),
            "base_top_n": baseline_top_n.tolist(),
            "our_pctile": our_pctile.tolist(),
            "base_pctile": base_pctile.tolist(),
        })

    if not records:
        print("[exp3] No results generated.")
        return

    results_df = pd.DataFrame([{k: v for k, v in r.items() if not isinstance(v, list)} for r in records])
    results_df.to_csv(out / "popularity_bias_results.csv", index=False)

    # ── Graph 1: CDF of popularity percentiles ──
    all_our_pctile = np.concatenate([r["our_pctile"] for r in records])
    all_base_pctile = np.concatenate([r["base_pctile"] for r in records])

    fig, ax = plt.subplots(figsize=(7, 5))
    for vals, label, color in [
        (all_our_pctile, "Our Model", "#4C72B0"),
        (all_base_pctile, "Popularity Baseline", "#DD8452"),
    ]:
        sorted_vals = np.sort(vals)
        cdf = np.arange(1, len(sorted_vals) + 1) / len(sorted_vals)
        ax.plot(sorted_vals, cdf, label=label, color=color, linewidth=2)
    ax.set_xlabel("Popularity Percentile (0=obscure, 1=mainstream)")
    ax.set_ylabel("Cumulative Fraction")
    ax.set_title("CDF of Recommendation Popularity\nOur Model vs Popularity Baseline")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out / "cdf_popularity.png", dpi=150)
    plt.close(fig)

    # ── Graph 2: KL divergence per feature ──
    from music_discovery.data.features import FEATURE_VECTOR_COLS
    kl_means = {f: results_df[f"kl_{f}"].mean() for f in FEATURE_VECTOR_COLS if f"kl_{f}" in results_df.columns}

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.bar(list(kl_means.keys()), list(kl_means.values()), color="#4C72B0", edgecolor="white")
    ax.set_title("Feature KL Divergence: Our Recs vs Popularity Baseline")
    ax.set_ylabel("KL Divergence")
    ax.set_xticklabels(list(kl_means.keys()), rotation=30, ha="right")
    fig.tight_layout()
    fig.savefig(out / "bar_kl_divergence.png", dpi=150)
    plt.close(fig)

    # ── Graph 3: Intra-list diversity box plot ──
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.boxplot(
        [our_diversities, baseline_diversities],
        labels=["Our Model", "Popularity Baseline"],
        patch_artist=True,
        boxprops=dict(facecolor="#4C72B0", alpha=0.7),
    )
    ax.set_ylabel("Mean Pairwise Cosine Distance")
    ax.set_title("Intra-List Diversity")
    fig.tight_layout()
    fig.savefig(out / "boxplot_diversity.png", dpi=150)
    plt.close(fig)

    mean_jaccard = results_df["jaccard"].mean()
    mean_our_pop = results_df["our_mean_popularity"].mean()
    mean_base_pop = results_df["base_mean_popularity"].mean()
    print(f"[exp3] Mean Jaccard overlap: {mean_jaccard:.3f}")
    print(f"[exp3] Mean popularity — ours: {mean_our_pop:.3f} | baseline: {mean_base_pop:.3f}")
    print(f"[exp3] Results saved to {out}")
