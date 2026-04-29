from __future__ import annotations

import os

import implicit
import numpy as np
import pandas as pd
import scipy.sparse as sp
import yaml


def _build_sparse_matrix(
    interactions: pd.DataFrame,
) -> tuple[sp.csr_matrix, np.ndarray, np.ndarray]:
    train = interactions[interactions["split"] == "train"].copy()
    user_ids = np.array(sorted(train["user_id"].unique()))
    song_ids = np.array(sorted(train["song_id"].unique()))

    user_idx = {u: i for i, u in enumerate(user_ids)}
    song_idx = {s: i for i, s in enumerate(song_ids)}

    rows = train["user_id"].map(user_idx).to_numpy(dtype=np.int32)
    cols = train["song_id"].map(song_idx).to_numpy(dtype=np.int32)
    data = np.log1p(train["play_count"].to_numpy(dtype=np.float32))

    matrix = sp.csr_matrix((data, (rows, cols)), shape=(len(user_ids), len(song_ids)))
    return matrix, user_ids, song_ids


def train_als(
    interactions_path: str,
    output_dir: str,
    factors: int = 128,
    iterations: int = 20,
    regularization: float = 0.01,
    use_gpu: bool = False,
) -> None:
    os.makedirs(output_dir, exist_ok=True)

    print("[train_als] Loading interactions...")
    interactions = pd.read_parquet(interactions_path)

    print("[train_als] Building sparse user×song matrix...")
    matrix, user_ids, song_ids = _build_sparse_matrix(interactions)
    print(f"  shape: {matrix.shape}, nnz: {matrix.nnz:,}")

    print("[train_als] Training ALS model...")
    model = implicit.als.AlternatingLeastSquares(
        factors=factors,
        iterations=iterations,
        regularization=regularization,
        use_gpu=use_gpu,
    )
    model.fit(matrix)

    print("[train_als] Saving embeddings...")
    emb_cols = [f"emb_{i}" for i in range(factors)]

    song_emb_df = pd.DataFrame(model.item_factors, columns=emb_cols)
    song_emb_df.insert(0, "song_id", song_ids)
    song_emb_df.to_parquet(os.path.join(output_dir, "song_embeddings.parquet"), index=False)

    user_emb_df = pd.DataFrame(model.user_factors, columns=emb_cols)
    user_emb_df.insert(0, "user_id", user_ids)
    user_emb_df.to_parquet(os.path.join(output_dir, "user_embeddings.parquet"), index=False)

    print("[train_als] Done.")
    print(f"  song_embeddings: {len(song_emb_df):,} songs × {factors} dims")
    print(f"  user_embeddings: {len(user_emb_df):,} users × {factors} dims")


if __name__ == "__main__":
    with open("configs/default.yaml") as f:
        cfg = yaml.safe_load(f)

    d = cfg.get("data", {})
    a = cfg.get("als", {})
    train_als(
        interactions_path=f"{d.get('processed_dir', 'data/processed')}/interactions.parquet",
        output_dir="models/embeddings",
        factors=a.get("factors", 128),
        iterations=a.get("iterations", 20),
        regularization=a.get("regularization", 0.01),
        use_gpu=a.get("use_gpu", False),
    )
