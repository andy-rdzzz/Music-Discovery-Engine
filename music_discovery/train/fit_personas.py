from __future__ import annotations
import pickle
import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm

from music_discovery.models.gmm import UserPersonaModel, fit_user_persona


def load_user_embeddings(
    user_history_path: str | Path,
    track_embeddings_path: str | Path,
    split: str = "train",
) -> dict[str, np.ndarray]:
    """
    Join user history (train split) with track embeddings.
    Returns {user_id: (N, D) embedding array}.
    """
    history_df = pd.read_parquet(user_history_path)
    emb_df = pd.read_parquet(track_embeddings_path)

    emb_cols = [c for c in emb_df.columns if c.startswith("emb_")]
    emb_matrix = emb_df[emb_cols].to_numpy(dtype=np.float32)

    if split:
        history_df = history_df[history_df["split"] == split]

    user_embeddings: dict[str, np.ndarray] = {}
    for user_id, group in history_df.groupby("user_id"):
        # Fix 5: deduplicate to unique tracks so GMM fits taste breadth, not replay frequency
        indices = group["track_idx"].drop_duplicates().to_numpy()
        valid = indices[indices < len(emb_matrix)]
        if len(valid) > 0:
            user_embeddings[str(user_id)] = emb_matrix[valid]

    return user_embeddings


def fit_all_personas(
    user_history_path: str | Path,
    track_embeddings_path: str | Path,
    output_dir: str | Path,
    k_min: int = 2,
    k_max: int = 10,
    covariance_type: str = "full",
    min_samples: int = 20,
) -> dict[str, UserPersonaModel]:
    """
    Fit a UserPersonaModel for every user. Saves models as pickle files.
    Returns {user_id: UserPersonaModel}.
    """
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    user_embeddings = load_user_embeddings(user_history_path, track_embeddings_path)
    print(f"[personas] Fitting personas for {len(user_embeddings)} users")

    models: dict[str, UserPersonaModel] = {}
    for user_id, embeddings in tqdm(user_embeddings.items(), desc="Fitting GMMs"):
        model = fit_user_persona(
            user_id=user_id,
            embeddings=embeddings,
            k_min=k_min,
            k_max=k_max,
            covariance_type=covariance_type,
            min_samples=min_samples,
        )
        models[user_id] = model

        user_dir = out / user_id
        user_dir.mkdir(exist_ok=True)
        with open(user_dir / "persona.pkl", "wb") as f:
            pickle.dump(model, f)

    k_values = [m.k for m in models.values()]
    if k_values:
        print(f"[personas] Done. K stats — mean={np.mean(k_values):.1f}, min={min(k_values)}, max={max(k_values)}")
    else:
        print("[personas] Done. No users fitted.")
    return models


def load_persona(user_dir: str | Path) -> UserPersonaModel:
    with open(Path(user_dir) / "persona.pkl", "rb") as f:
        return pickle.load(f)
