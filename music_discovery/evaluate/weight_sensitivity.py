"""Experiment 1 — Scoring weight sensitivity: NDCG heatmap + ablation bar chart."""

from __future__ import annotations

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from tqdm import tqdm

from music_discovery.train.fit_personas import load_persona
from music_discovery.models.scorer import sonic_fit, emotional_fit, novelty_score, familiarity_score, DEFAULT_WEIGHTS


def _ndcg_at_k(scores: np.ndarray, relevant_mask: np.ndarray, k: int = 10) -> float:
    order = np.argsort(-scores)
    rel = relevant_mask[order].astype(float)
    k = min(k, len(rel))
    dcg = (rel[:k] / np.log2(np.arange(2, k + 2))).sum()
    ideal = np.sort(relevant_mask.astype(float))[::-1]
    idcg = (ideal[:k] / np.log2(np.arange(2, k + 2))).sum()
    return float(dcg / idcg) if idcg > 0 else 0.0


def _load_user_data(
    user_id: str,
    history_df: pd.DataFrame,
    emb_df: pd.DataFrame,
    candidate_emb: np.ndarray,
    candidate_artists: np.ndarray,
):
    train = history_df[(history_df["user_id"] == user_id) & (history_df["split"] == "train")]
    eval_ = history_df[(history_df["user_id"] == user_id) & (history_df["split"] == "eval")]

    # Filter eval to discovery-only rows when the column is available
    if "is_discovery" in eval_.columns:
        eval_ = eval_[eval_["is_discovery"]]

    train_idx = train["track_idx"].to_numpy()
    eval_idx = eval_["track_idx"].to_numpy()
    valid_train = train_idx[train_idx < len(candidate_emb)]
    valid_eval = eval_idx[eval_idx < len(candidate_emb)]

    history_emb = candidate_emb[valid_train]
    history_artists = candidate_artists[valid_train]

    held_out_mask = np.zeros(len(candidate_emb), dtype=bool)
    held_out_mask[valid_eval] = True

    return history_emb, history_artists, held_out_mask


def run(
    processed_dir: str,
    embeddings_path: str,
    personas_dir: str,
    output_dir: str,
    n_users: int = 10,
    k: int = 10,
    weight_step: float = 0.1,
):
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    history_df = pd.read_parquet(Path(processed_dir) / "user_history.parquet")
    emb_df = pd.read_parquet(embeddings_path)
    emb_cols = [c for c in emb_df.columns if c.startswith("emb_")]
    candidate_emb = emb_df[emb_cols].to_numpy(dtype=np.float32)
    candidate_artists = emb_df["artist_norm"].to_numpy()

    # Select users that have persona files
    personas_path = Path(personas_dir)
    available_users = [d.name for d in personas_path.iterdir() if (d / "persona.pkl").exists()]
    users = available_users[:n_users]
    print(f"[exp1] Running on {len(users)} users")

    # Pre-compute component scores per user
    all_results: list[dict] = []

    for user_id in tqdm(users, desc="Users"):
        persona = load_persona(personas_path / user_id)
        history_emb, history_artists, held_out_mask = _load_user_data(
            user_id, history_df, emb_df, candidate_emb, candidate_artists
        )
        if history_emb.shape[0] == 0 or held_out_mask.sum() == 0:
            continue

        s1 = (sonic_fit(candidate_emb, persona) + 1.0) / 2.0
        s2 = novelty_score(candidate_emb, candidate_artists, history_emb, history_artists)
        s3 = emotional_fit(candidate_emb, persona)
        s4 = familiarity_score(candidate_emb, persona)
        components = np.stack([s1, s2, s3, s4], axis=1)  # (N, 4)

        # Grid sweep over (w_persona_fit, w_novelty); w_emotional fixed at 0.15, w_familiarity at remainder
        step = weight_step
        for w1 in np.arange(0, 1 + step, step):
            for w2 in np.arange(0, 1 - w1 + step, step):
                w3 = 1.0 - w1 - w2
                if w3 < -1e-6:
                    continue
                w3 = max(0.0, w3)
                w = np.array([w1, w2, w3, 0.0])
                scores = components @ w
                ndcg = _ndcg_at_k(scores, held_out_mask, k)
                all_results.append({"user_id": user_id, "w1": round(w1, 2), "w2": round(w2, 2), "w3": round(w3, 2), "ndcg": ndcg})

        # Ablation: single component only + default 4-component weights
        for label, w in [
            ("persona_fit_only",  [1, 0, 0, 0]),
            ("novelty_only",      [0, 1, 0, 0]),
            ("emotional_only",    [0, 0, 1, 0]),
            ("familiarity_only",  [0, 0, 0, 1]),
            ("default_weights",   DEFAULT_WEIGHTS.tolist()),
        ]:
            scores = components @ np.array(w)
            ndcg = _ndcg_at_k(scores, held_out_mask, k)
            all_results.append({"user_id": user_id, "w1": w[0], "w2": w[1], "w3": w[2], "ndcg": ndcg, "label": label})

    results_df = pd.DataFrame(all_results)
    results_df.to_csv(out / "weight_sensitivity_results.csv", index=False)

    # ── Graph 1: NDCG heatmap (w1 vs w2, w3=1-w1-w2) ──
    grid = results_df.dropna(subset=["w1", "w2"]).copy()
    grid = grid[~grid.get("label", pd.Series(dtype=str)).notna()] if "label" in grid.columns else grid
    pivot = grid.groupby(["w1", "w2"])["ndcg"].mean().reset_index()
    heat = pivot.pivot(index="w2", columns="w1", values="ndcg")

    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(heat, ax=ax, cmap="YlOrRd", annot=False, cbar_kws={"label": "NDCG@10"})
    ax.set_title("Scoring Weight Sensitivity — NDCG@10\n(w_surprise = 1 − w_sonic − w_emotional)")
    ax.set_xlabel("w_sonic")
    ax.set_ylabel("w_emotional")
    fig.tight_layout()
    fig.savefig(out / "heatmap_weight_sensitivity.png", dpi=150)
    plt.close(fig)

    # ── Graph 2: Ablation bar chart ──
    ablation_labels = ["persona_fit_only", "novelty_only", "emotional_only", "familiarity_only", "default_weights"]
    if "label" in results_df.columns:
        ablation = results_df[results_df["label"].isin(ablation_labels)]
        ablation_mean = ablation.groupby("label")["ndcg"].mean().reindex(ablation_labels)

        fig, ax = plt.subplots(figsize=(8, 4))
        ablation_mean.plot(kind="bar", ax=ax, color=["#4C72B0","#DD8452","#55A868","#C44E52","#8172B2"], edgecolor="white")
        ax.set_title("Component Ablation — Mean NDCG@10")
        ax.set_ylabel("NDCG@10")
        ax.set_xticklabels(ablation_labels, rotation=30, ha="right")
        ax.set_ylim(0, 1)
        fig.tight_layout()
        fig.savefig(out / "bar_ablation.png", dpi=150)
        plt.close(fig)

    print(f"[exp1] Results saved to {out}")
