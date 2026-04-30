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
from music_discovery.models.scorer import DEFAULT_WEIGHTS, optimize_weights, score_candidates


def _ndcg_at_k(scores: np.ndarray, relevant_mask: np.ndarray, k: int = 10) -> float:
    finite_mask = np.isfinite(scores)
    if not finite_mask.any():
        return 0.0

    order = np.argsort(-scores[finite_mask])
    rel = relevant_mask[finite_mask][order].astype(float)
    k = min(k, len(rel))
    if k == 0:
        return 0.0

    dcg = (rel[:k] / np.log2(np.arange(2, k + 2))).sum()
    ideal = np.sort(relevant_mask[finite_mask].astype(float))[::-1]
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


def _fit_persona_with_k(user_id: str, history_emb: np.ndarray, k: int) -> UserPersonaModel | None:
    if len(history_emb) < 2:
        return None

    n_unique = len(np.unique(history_emb, axis=0))
    k_actual = max(1, min(k, len(history_emb) - 1, n_unique))
    if k_actual < 1:
        return None

    try:
        gmm = GaussianMixture(
            n_components=k_actual,
            covariance_type="diag",
            random_state=42,
            n_init=1,
            max_iter=200,
        )
        gmm.fit(history_emb)
        return UserPersonaModel(
            user_id=user_id,
            k=k_actual,
            means=gmm.means_,
            covariances=gmm.covariances_,
            weights=gmm.weights_,
            gmm=gmm,
        )
    except Exception:
        return None


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
    test_df = interactions[interactions["split"] == "test"]

    interaction_users = set(interactions["user_id"].unique())
    personas_path = Path(personas_dir)
    available_users = [
        d.name for d in personas_path.iterdir()
        if (d / "persona.pkl").exists() and d.name in interaction_users
    ]
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
        test_songs = set(test_df[test_df["user_id"] == user_id]["song_id"])

        train_idx = np.array([song_to_idx[s] for s in train_songs if s in song_to_idx], dtype=np.intp)
        val_idx = np.array([song_to_idx[s] for s in val_songs if s in song_to_idx], dtype=np.intp)
        test_idx = np.array([song_to_idx[s] for s in test_songs if s in song_to_idx], dtype=np.intp)

        if len(train_idx) < 10 or len(test_idx) == 0 or user_id not in entropies:
            continue

        history_emb = candidate_emb[train_idx]
        history_artists = candidate_artists[train_idx]

        eval_mask = np.ones(len(candidate_emb), dtype=bool)
        eval_mask[train_idx] = False
        if not eval_mask.any():
            continue

        val_mask = np.zeros(len(candidate_emb), dtype=bool)
        val_mask[val_idx] = True
        val_eval_idx = np.flatnonzero(val_mask & eval_mask)
        eval_pos_map = {idx: pos for pos, idx in enumerate(np.flatnonzero(eval_mask))}
        val_positions = np.array([eval_pos_map[idx] for idx in val_eval_idx], dtype=np.intp)

        test_mask = np.zeros(len(candidate_emb), dtype=bool)
        test_mask[test_idx] = True
        test_eval_mask = test_mask & eval_mask
        if not test_eval_mask.any():
            continue

        user_type = _classify_user(entropies[user_id], low_q, high_q)

        bic_model = fit_user_persona(user_id, history_emb, k_min=2, k_max=max(k_range))
        bic_k = bic_model.k

        bic_scores = bic_curve(history_emb, k_min=1, k_max=max(k_range))
        for k_val, bic_val in bic_scores.items():
            bic_records.append({"user_id": user_id, "k": k_val, "bic": bic_val, "user_type": user_type})

        candidate_emb_eval = candidate_emb[eval_mask]
        candidate_artists_eval = candidate_artists[eval_mask]
        test_mask_eval = test_eval_mask[eval_mask]

        for k in k_range:
            persona = _fit_persona_with_k(user_id, history_emb, k)
            if persona is None:
                continue

            weights = DEFAULT_WEIGHTS.copy()
            if len(val_positions) > 0:
                try:
                    weights = optimize_weights(
                        candidate_emb=candidate_emb_eval,
                        candidate_artists=candidate_artists_eval,
                        persona=persona,
                        user_history_emb=history_emb,
                        user_history_artists=history_artists,
                        held_out_indices=val_positions,
                    )
                except Exception:
                    weights = DEFAULT_WEIGHTS.copy()

            try:
                scores = score_candidates(
                    candidate_emb=candidate_emb_eval,
                    candidate_artists=candidate_artists_eval,
                    persona=persona,
                    user_history_emb=history_emb,
                    user_history_artists=history_artists,
                    weights=weights,
                )
                ndcg = _ndcg_at_k(scores, test_mask_eval)
                results.append(
                    {
                        "user_id": user_id,
                        "k": k,
                        "ndcg": ndcg,
                        "user_type": user_type,
                        "bic_k": bic_k,
                        "is_bic": k == bic_k,
                    }
                )
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
