"""Experiment 1 — scoring weight sensitivity for the current 4-component scorer."""

from __future__ import annotations

from itertools import combinations
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from tqdm import tqdm

from music_discovery.models.scorer import (
    DEFAULT_WEIGHTS,
    familiarity_score,
    novelty_score,
    persona_specificity,
    sonic_fit,
)
from music_discovery.train.fit_personas import load_persona


COMPONENT_NAMES = [
    "sonic_fit",
    "novelty",
    "persona_specificity",
    "familiarity",
]
COMPONENT_DISPLAY_NAMES = {
    "sonic_fit": "Sonic fit",
    "novelty": "Novelty",
    "persona_specificity": "Persona specificity",
    "familiarity": "Familiarity",
}
ABLATION_LABELS = [
    "sonic_fit_only",
    "novelty_only",
    "persona_specificity_only",
    "familiarity_only",
    "default_weights",
]
ABLATION_DISPLAY_NAMES = {
    "sonic_fit_only": "Sonic fit only",
    "novelty_only": "Novelty only",
    "persona_specificity_only": "Persona specificity only",
    "familiarity_only": "Familiarity only",
    "default_weights": "Default weights",
}


def _ndcg_at_k(scores: np.ndarray, relevant_mask: np.ndarray, k: int = 10) -> float:
    order = np.argsort(-scores)
    rel = relevant_mask[order].astype(float)
    k = min(k, len(rel))
    dcg = (rel[:k] / np.log2(np.arange(2, k + 2))).sum()
    ideal = np.sort(relevant_mask.astype(float))[::-1]
    idcg = (ideal[:k] / np.log2(np.arange(2, k + 2))).sum()
    return float(dcg / idcg) if idcg > 0 else 0.0


def _compute_components(
    candidate_emb: np.ndarray,
    candidate_artists: np.ndarray,
    history_emb: np.ndarray,
    history_artists: np.ndarray,
    persona,
) -> np.ndarray:
    s1 = (sonic_fit(candidate_emb, persona) + 1.0) / 2.0
    s2 = novelty_score(candidate_emb, candidate_artists, history_emb, history_artists)
    s3 = persona_specificity(candidate_emb, persona)
    s4 = familiarity_score(candidate_emb, persona)
    return np.stack([s1, s2, s3, s4], axis=1)


def _pairwise_weight_grid(default_weights: np.ndarray, step: float):
    default_weights = np.asarray(default_weights, dtype=float)
    if default_weights.shape != (4,):
        raise ValueError("Expected 4 default weights for the scorer")

    values = np.arange(0.0, 1.0 + step / 2.0, step)
    pair_grids: dict[tuple[str, str], list[np.ndarray]] = {}

    for i, j in combinations(range(len(COMPONENT_NAMES)), 2):
        others = [idx for idx in range(len(COMPONENT_NAMES)) if idx not in (i, j)]
        base_other_sum = float(default_weights[others].sum())
        grid_weights: list[np.ndarray] = []
        for wi in values:
            for wj in values:
                remaining = 1.0 - wi - wj
                if remaining < -1e-9:
                    continue
                remaining = max(0.0, remaining)
                w = np.zeros(4, dtype=float)
                w[i] = wi
                w[j] = wj
                if remaining > 0 and base_other_sum > 0:
                    scaled = default_weights[others] / base_other_sum * remaining
                    for idx, val in zip(others, scaled):
                        w[idx] = val
                elif remaining > 0:
                    for idx in others:
                        w[idx] = remaining / len(others)
                w = np.clip(w, 0.0, 1.0)
                total = w.sum()
                if total <= 0:
                    continue
                grid_weights.append(w / total)
        pair_grids[(COMPONENT_NAMES[i], COMPONENT_NAMES[j])] = grid_weights

    return pair_grids


def _ablation_weight_sets(default_weights: np.ndarray) -> list[tuple[str, np.ndarray]]:
    singletons = [
        ("sonic_fit_only", np.array([1.0, 0.0, 0.0, 0.0])),
        ("novelty_only", np.array([0.0, 1.0, 0.0, 0.0])),
        ("persona_specificity_only", np.array([0.0, 0.0, 1.0, 0.0])),
        ("familiarity_only", np.array([0.0, 0.0, 0.0, 1.0])),
        ("default_weights", np.asarray(default_weights, dtype=float)),
    ]
    return singletons


def _plot_pairwise_heatmaps(results_df: pd.DataFrame, out: Path, k: int):
    grid = results_df[results_df["analysis"] == "pairwise_heatmap"].copy()
    if grid.empty:
        return

    pairs = list(combinations(COMPONENT_NAMES, 2))
    fig, axes = plt.subplots(2, 3, figsize=(18, 10), sharex=False, sharey=False)
    axes = axes.flatten()
    heatmap_artist = None

    for ax, (x_name, y_name) in zip(axes, pairs):
        pair_df = grid[(grid["sweep_x"] == x_name) & (grid["sweep_y"] == y_name)]
        if pair_df.empty:
            ax.set_axis_off()
            continue

        heat = (
            pair_df.groupby([f"w_{y_name}", f"w_{x_name}"])["ndcg"]
            .mean()
            .reset_index()
            .pivot(index=f"w_{y_name}", columns=f"w_{x_name}", values="ndcg")
            .sort_index()
        )
        if heat.empty:
            ax.set_axis_off()
            continue

        data = heat.to_numpy(dtype=float)
        heatmap_artist = ax.imshow(
            data,
            cmap="YlOrRd",
            vmin=0.0,
            vmax=1.0,
            origin="lower",
            aspect="auto",
        )
        ax.set_title(
            f"{COMPONENT_DISPLAY_NAMES[x_name]} vs {COMPONENT_DISPLAY_NAMES[y_name]}"
        )
        ax.set_xlabel(f"Weight: {COMPONENT_DISPLAY_NAMES[x_name]}")
        ax.set_ylabel(f"Weight: {COMPONENT_DISPLAY_NAMES[y_name]}")
        ax.set_xticks(np.arange(len(heat.columns)))
        ax.set_xticklabels([f"{val:.1f}" for val in heat.columns], rotation=45, ha="right")
        ax.set_yticks(np.arange(len(heat.index)))
        ax.set_yticklabels([f"{val:.1f}" for val in heat.index])

    fig.suptitle(
        f"Weight sensitivity across scorer component pairs — mean NDCG@{k}\n"
        "Other component weights are rescaled from the default configuration",
        fontsize=14,
    )
    if heatmap_artist is not None:
        fig.colorbar(heatmap_artist, ax=axes.tolist(), shrink=0.85, label=f"NDCG@{k}")
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(out / "heatmap_weight_sensitivity.png", dpi=150)
    plt.close(fig)


def _plot_ablation(results_df: pd.DataFrame, out: Path, k: int):
    ablation = results_df[results_df["analysis"] == "ablation"].copy()
    if ablation.empty:
        return

    ablation_mean = ablation.groupby("label")["ndcg"].mean().reindex(ABLATION_LABELS)
    colors = ["#4C72B0", "#DD8452", "#55A868", "#C44E52", "#8172B2"]

    fig, ax = plt.subplots(figsize=(9, 4.5))
    ablation_mean.plot(kind="bar", ax=ax, color=colors, edgecolor="white")
    ax.set_title(f"4-component scorer ablation — mean NDCG@{k}")
    ax.set_ylabel(f"NDCG@{k}")
    ax.set_xlabel("Scoring configuration")
    ax.set_xticklabels(
        [ABLATION_DISPLAY_NAMES[label] for label in ablation_mean.index],
        rotation=20,
        ha="right",
    )
    ax.set_ylim(0, 1)
    fig.tight_layout()
    fig.savefig(out / "bar_ablation.png", dpi=150)
    plt.close(fig)


def run(
    interactions_path: str,
    song_embeddings_path: str,
    personas_dir: str,
    output_dir: str,
    n_users: int = 10,
    k: int = 10,
    weight_step: float = 0.1,
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
    test_df = interactions[interactions["split"] == "test"]

    interaction_users = set(interactions["user_id"].unique())
    personas_path = Path(personas_dir)
    available_users = [
        d.name for d in personas_path.iterdir() if (d / "persona.pkl").exists() and d.name in interaction_users
    ]
    users = available_users[:n_users]
    print(f"[exp1] Running on {len(users)} users")

    default_weights = np.asarray(DEFAULT_WEIGHTS, dtype=float)
    pairwise_grids = _pairwise_weight_grid(default_weights, weight_step)
    ablation_sets = _ablation_weight_sets(default_weights)

    all_results: list[dict] = []

    for user_id in tqdm(users, desc="Users"):
        persona = load_persona(personas_path / user_id)

        train_songs = set(train_df[train_df["user_id"] == user_id]["song_id"])
        val_songs = set(val_df[val_df["user_id"] == user_id]["song_id"])
        test_songs = set(test_df[test_df["user_id"] == user_id]["song_id"])

        train_idx = np.array([song_to_idx[s] for s in train_songs if s in song_to_idx], dtype=np.intp)
        val_idx = np.array([song_to_idx[s] for s in val_songs if s in song_to_idx], dtype=np.intp)
        test_idx = np.array([song_to_idx[s] for s in test_songs if s in song_to_idx], dtype=np.intp)

        if len(train_idx) == 0 or len(val_idx) == 0 or len(test_idx) == 0:
            continue

        history_emb = candidate_emb[train_idx]
        history_artists = candidate_artists[train_idx]
        test_mask = np.zeros(len(candidate_emb), dtype=bool)
        test_mask[test_idx] = True

        components = _compute_components(
            candidate_emb=candidate_emb,
            candidate_artists=candidate_artists,
            history_emb=history_emb,
            history_artists=history_artists,
            persona=persona,
        )

        for (x_name, y_name), weight_grid in pairwise_grids.items():
            for weights in weight_grid:
                scores = components @ weights
                ndcg = _ndcg_at_k(scores, test_mask, k)
                row = {
                    "user_id": user_id,
                    "analysis": "pairwise_heatmap",
                    "label": None,
                    "sweep_x": x_name,
                    "sweep_y": y_name,
                    "ndcg": ndcg,
                    "k": k,
                    "w1": float(weights[0]),
                    "w2": float(weights[1]),
                    "w3": float(weights[2]),
                    "w4": float(weights[3]),
                }
                for component_name, weight in zip(COMPONENT_NAMES, weights):
                    row[f"w_{component_name}"] = float(weight)
                all_results.append(row)

        for label, weights in ablation_sets:
            weights = np.asarray(weights, dtype=float)
            scores = components @ weights
            ndcg = _ndcg_at_k(scores, test_mask, k)
            row = {
                "user_id": user_id,
                "analysis": "ablation",
                "label": label,
                "sweep_x": None,
                "sweep_y": None,
                "ndcg": ndcg,
                "k": k,
                "w1": float(weights[0]),
                "w2": float(weights[1]),
                "w3": float(weights[2]),
                "w4": float(weights[3]),
            }
            for component_name, weight in zip(COMPONENT_NAMES, weights):
                row[f"w_{component_name}"] = float(weight)
            all_results.append(row)

    results_df = pd.DataFrame(all_results)
    results_df.to_csv(out / "weight_sensitivity_results.csv", index=False)

    _plot_pairwise_heatmaps(results_df, out, k)
    _plot_ablation(results_df, out, k)

    print(f"[exp1] Results saved to {out}")
