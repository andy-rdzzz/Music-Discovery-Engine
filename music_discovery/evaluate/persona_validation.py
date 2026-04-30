"""Experiment 2 — Persona count validation: K vs NDCG, BIC curve, UMAP scatter."""

from __future__ import annotations

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
from tqdm import tqdm

from sklearn.mixture import GaussianMixture
from music_discovery.models.gmm import fit_user_persona, bic_curve, UserPersonaModel
from music_discovery.models.scorer import sonic_fit, emotional_fit, novelty_score, DEFAULT_WEIGHTS


def _ndcg_at_k(scores: np.ndarray, relevant_mask: np.ndarray, k: int = 10) -> float:
    order = np.argsort(-scores)
    rel = relevant_mask[order].astype(float)
    k = min(k, len(rel))
    dcg = (rel[:k] / np.log2(np.arange(2, k + 2))).sum()
    ideal = np.sort(relevant_mask.astype(float))[::-1]
    idcg = (ideal[:k] / np.log2(np.arange(2, k + 2))).sum()
    return float(dcg / idcg) if idcg > 0 else 0.0


def _genre_entropy(history_emb: np.ndarray) -> float:
    return float(np.var(history_emb))


def _classify_user(entropy: float, low_q: float, high_q: float) -> str:
    if entropy < low_q:
        return "niche"
    elif entropy > high_q:
        return "diverse"
    return "mainstream"


def run(
    interactions_path: str,
    song_embeddings_path: str,
    personas_dir: str,
    output_dir: str,
    n_users: int = 10,
    k_range: list[int] | None = None,
):
    if k_range is None:
        k_range = [1, 2, 3, 4, 5]

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

    entropies: dict[str, float] = {}
    for user_id in users:
        train_songs = set(train_df[train_df["user_id"] == user_id]["song_id"])
        train_idx = np.array([song_to_idx[s] for s in train_songs if s in song_to_idx], dtype=np.intp)
        if len(train_idx) > 0:
            entropies[user_id] = _genre_entropy(candidate_emb[train_idx])

    if not entropies:
        print("[exp2] No users with embeddings found.")
        return

    vals = list(entropies.values())
    low_q, high_q = np.percentile(vals, 33), np.percentile(vals, 66)

    results: list[dict] = []
    bic_records: list[dict] = []

    for user_id in tqdm(users, desc="Users"):
        train_songs = set(train_df[train_df["user_id"] == user_id]["song_id"])
        val_songs = set(val_df[val_df["user_id"] == user_id]["song_id"])

        train_idx = np.array([song_to_idx[s] for s in train_songs if s in song_to_idx], dtype=np.intp)
        val_idx = np.array([song_to_idx[s] for s in val_songs if s in song_to_idx], dtype=np.intp)

        if len(train_idx) < 10 or len(val_idx) == 0 or user_id not in entropies:
            continue

        history_emb = candidate_emb[train_idx]
        history_artists = candidate_artists[train_idx]

        held_out_mask = np.zeros(len(candidate_emb), dtype=bool)
        held_out_mask[val_idx] = True

        user_type = _classify_user(entropies[user_id], low_q, high_q)

        bic_model = fit_user_persona(user_id, history_emb, k_min=2, k_max=max(k_range))
        bic_k = bic_model.k

        bic_scores = bic_curve(history_emb, k_min=1, k_max=max(k_range))
        for k_val, bic_val in bic_scores.items():
            bic_records.append({"user_id": user_id, "k": k_val, "bic": bic_val, "user_type": user_type})

        for k in k_range:
            k_actual = max(1, min(k, len(train_idx) // 2))
            try:
                gmm = GaussianMixture(n_components=k_actual, covariance_type="full", random_state=42, n_init=3)
                gmm.fit(history_emb)

                persona = UserPersonaModel(
                    user_id=user_id, k=k_actual,
                    means=gmm.means_, covariances=gmm.covariances_,
                    weights=gmm.weights_, gmm=gmm,
                )
                s1 = (sonic_fit(candidate_emb, persona) + 1.0) / 2.0
                s2 = emotional_fit(candidate_emb, persona)
                s3 = novelty_score(candidate_emb, candidate_artists, history_emb, history_artists)
                scores = DEFAULT_WEIGHTS[0] * s1 + DEFAULT_WEIGHTS[1] * s2 + DEFAULT_WEIGHTS[2] * s3
                ndcg = _ndcg_at_k(scores, held_out_mask)
                results.append({"user_id": user_id, "k": k, "ndcg": ndcg, "user_type": user_type, "bic_k": bic_k, "is_bic": k == bic_k})
            except Exception:
                continue

    if not results:
        print("[exp2] No results generated.")
        return

    results_df = pd.DataFrame(results)
    bic_df = pd.DataFrame(bic_records)
    results_df.to_csv(out / "persona_validation_results.csv", index=False)
    bic_df.to_csv(out / "bic_curves.csv", index=False)

    # ── Graph 1: NDCG@10 vs K by user type ──
    colors = {"niche": "#4C72B0", "mainstream": "#DD8452", "diverse": "#55A868"}
    fig, ax = plt.subplots(figsize=(8, 5))
    for user_type, grp in results_df.groupby("user_type"):
        mean_ndcg = grp.groupby("k")["ndcg"].mean()
        ax.plot(mean_ndcg.index, mean_ndcg.values, marker="o", label=user_type, color=colors.get(user_type))
        bic_k_mode = grp.groupby("k")["is_bic"].sum().idxmax() if grp["is_bic"].any() else None
        if bic_k_mode and bic_k_mode in mean_ndcg.index:
            ax.scatter([bic_k_mode], [mean_ndcg[bic_k_mode]], marker="*", s=200, color=colors.get(user_type), zorder=5)

    ax.set_xlabel("K (number of personas)")
    ax.set_ylabel("NDCG@10")
    ax.set_title("Persona Count vs Recommendation Quality by User Type\n(★ = BIC-selected K)")
    ax.legend()
    ax.set_xticks(k_range)
    fig.tight_layout()
    fig.savefig(out / "line_ndcg_vs_k.png", dpi=150)
    plt.close(fig)

    # ── Graph 2: BIC curves for 3 representative users ──
    rep_users = []
    for utype in ["niche", "mainstream", "diverse"]:
        candidates = [u for u, e in entropies.items() if _classify_user(e, low_q, high_q) == utype]
        if candidates:
            rep_users.append(candidates[0])

    if rep_users and not bic_df.empty:
        fig, ax = plt.subplots(figsize=(7, 4))
        for user_id in rep_users:
            u_bic = bic_df[bic_df["user_id"] == user_id]
            if not u_bic.empty:
                utype = u_bic["user_type"].iloc[0]
                ax.plot(u_bic["k"], u_bic["bic"], marker="o", label=f"{utype} ({user_id[:8]})", color=colors.get(utype))
        ax.set_xlabel("K")
        ax.set_ylabel("BIC Score")
        ax.set_title("BIC Score vs K for Representative Users")
        ax.legend()
        fig.tight_layout()
        fig.savefig(out / "bic_curves.png", dpi=150)
        plt.close(fig)

    # ── Graph 3: UMAP scatter with persona assignments ──
    try:
        import umap
        if rep_users:
            user_id = rep_users[0]
            train_songs = set(train_df[train_df["user_id"] == user_id]["song_id"])
            train_idx = np.array([song_to_idx[s] for s in train_songs if s in song_to_idx], dtype=np.intp)
            if len(train_idx) >= 20:
                h_emb = candidate_emb[train_idx]
                reducer = umap.UMAP(n_components=2, random_state=42, n_neighbors=min(15, len(h_emb) - 1))
                emb_2d = reducer.fit_transform(h_emb)

                persona = fit_user_persona(user_id, h_emb, k_min=2, k_max=5)
                assignments = persona.dominant_persona(h_emb)

                fig, ax = plt.subplots(figsize=(7, 6))
                for k in range(persona.k):
                    mask = assignments == k
                    ax.scatter(emb_2d[mask, 0], emb_2d[mask, 1], s=15, alpha=0.6, label=f"Persona {k + 1}")
                ax.set_title(f"UMAP: Song Embeddings + Personas\nUser: {user_id[:12]}")
                ax.legend(markerscale=2)
                fig.tight_layout()
                fig.savefig(out / "umap_personas.png", dpi=150)
                plt.close(fig)
    except ImportError:
        print("[exp2] umap-learn not installed — skipping UMAP plot")

    print(f"[exp2] Results saved to {out}")
