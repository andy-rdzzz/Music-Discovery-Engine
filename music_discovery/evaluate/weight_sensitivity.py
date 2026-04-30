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

    personas_path = Path(personas_dir)
    available_users = [d.name for d in personas_path.iterdir() if (d / "persona.pkl").exists()]
    users = available_users[:n_users]
    print(f"[exp1] Running on {len(users)} users")

    all_results: list[dict] = []

    for user_id in tqdm(users, desc="Users"):
        persona = load_persona(personas_path / user_id)

        train_songs = set(train_df[train_df["user_id"] == user_id]["song_id"])
        val_songs = set(val_df[val_df["user_id"] == user_id]["song_id"])

        train_idx = np.array([song_to_idx[s] for s in train_songs if s in song_to_idx], dtype=np.intp)
        val_idx = np.array([song_to_idx[s] for s in val_songs if s in song_to_idx], dtype=np.intp)

        if len(train_idx) == 0 or len(val_idx) == 0:
            continue

        history_emb = candidate_emb[train_idx]
        history_artists = candidate_artists[train_idx]

        held_out_mask = np.zeros(len(candidate_emb), dtype=bool)
        held_out_mask[val_idx] = True

        s1 = (sonic_fit(candidate_emb, persona) + 1.0) / 2.0
        s2 = novelty_score(candidate_emb, candidate_artists, history_emb, history_artists)
        s3 = emotional_fit(candidate_emb, persona)
        s4 = familiarity_score(candidate_emb, persona)
        components = np.stack([s1, s2, s3, s4], axis=1)  # (N, 4)

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
    grid = results_df[~results_df.get("label", pd.Series(dtype=str)).notna()] if "label" in results_df.columns else results_df
    pivot = grid.groupby(["w1", "w2"])["ndcg"].mean().reset_index()
    heat = pivot.pivot(index="w2", columns="w1", values="ndcg")

    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(heat, ax=ax, cmap="YlOrRd", annot=False, cbar_kws={"label": "NDCG@10"})
    ax.set_title("Scoring Weight Sensitivity — NDCG@10\n(w_familiarity = 1 − w_persona − w_novelty − w_emotional)")
    ax.set_xlabel("w_persona_fit")
    ax.set_ylabel("w_novelty")
    fig.tight_layout()
    fig.savefig(out / "heatmap_weight_sensitivity.png", dpi=150)
    plt.close(fig)

    # ── Graph 2: Ablation bar chart ──
    ablation_labels = ["persona_fit_only", "novelty_only", "emotional_only", "familiarity_only", "default_weights"]
    if "label" in results_df.columns:
        ablation = results_df[results_df["label"].isin(ablation_labels)]
        ablation_mean = ablation.groupby("label")["ndcg"].mean().reindex(ablation_labels)

        fig, ax = plt.subplots(figsize=(8, 4))
        ablation_mean.plot(kind="bar", ax=ax, color=["#4C72B0", "#DD8452", "#55A868", "#C44E52", "#8172B2"], edgecolor="white")
        ax.set_title("Component Ablation — Mean NDCG@10")
        ax.set_ylabel("NDCG@10")
        ax.set_xticklabels(ablation_labels, rotation=30, ha="right")
        ax.set_ylim(0, 1)
        fig.tight_layout()
        fig.savefig(out / "bar_ablation.png", dpi=150)
        plt.close(fig)

    print(f"[exp1] Results saved to {out}")
