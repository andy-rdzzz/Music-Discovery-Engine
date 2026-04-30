from __future__ import annotations
import os
import pickle
import numpy as np
import pandas as pd
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from tqdm import tqdm

from music_discovery.models.gmm import UserPersonaModel, fit_user_persona


def load_user_embeddings(
    user_history_path: str | Path,
    track_embeddings_path: str | Path,
    split: str = "train",
) -> dict[str, np.ndarray]:
    """
    Join user history (train split) with song embeddings by song_id.
    Returns {user_id: (N, D) embedding array}, repeating each song
    embedding by play_count so persona fitting reflects listen frequency.
    """
    history_df = pd.read_parquet(user_history_path)
    emb_df = pd.read_parquet(track_embeddings_path)

    emb_cols = [c for c in emb_df.columns if c.startswith("emb_")]
    song_to_emb = emb_df.set_index("song_id")[emb_cols]

    if split:
        history_df = history_df[history_df["split"] == split]

    user_embeddings: dict[str, np.ndarray] = {}
    for user_id, group in history_df.groupby("user_id"):
        valid_rows = group[group["song_id"].isin(song_to_emb.index)]
        if len(valid_rows) == 0:
            continue

        base_embeddings = song_to_emb.loc[valid_rows["song_id"]].to_numpy(dtype=np.float32)
        if "play_count" in valid_rows.columns:
            repeat_counts = (
                valid_rows["play_count"]
                .fillna(1)
                .astype(np.int32)
                .clip(lower=1)
                .to_numpy()
            )
        else:
            repeat_counts = np.ones(len(valid_rows), dtype=np.int32)

        user_embeddings[str(user_id)] = np.repeat(base_embeddings, repeat_counts, axis=0)

    return user_embeddings


def _fit_and_save(
    user_id: str,
    embeddings: np.ndarray,
    output_dir: Path,
    k_min: int,
    k_max: int,
    covariance_type: str,
    min_samples: int,
    n_init: int,
    resume: bool,
) -> tuple[str, int, bool]:
    user_dir = output_dir / user_id
    pkl_path = user_dir / "persona.pkl"

    if resume and pkl_path.exists():
        return user_id, -1, True  # skipped

    model = fit_user_persona(
        user_id=user_id,
        embeddings=embeddings,
        k_min=k_min,
        k_max=k_max,
        covariance_type=covariance_type,
        min_samples=min_samples,
        n_init=n_init,
    )
    user_dir.mkdir(exist_ok=True)
    with open(pkl_path, "wb") as f:
        pickle.dump(model, f)
    return user_id, model.k, False


def fit_all_personas(
    user_history_path: str | Path,
    track_embeddings_path: str | Path,
    output_dir: str | Path,
    k_min: int = 2,
    k_max: int = 5,
    covariance_type: str = "diag",
    min_samples: int = 20,
    n_init: int = 1,
    n_jobs: int | None = None,
    resume: bool = True,
) -> dict[str, int]:
    """
    Fit a UserPersonaModel for every user in parallel. Saves models as pickle files.
    resume=True skips users that already have a persona.pkl on disk.
    Returns {user_id: k} for newly fitted users.
    """
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    user_embeddings = load_user_embeddings(user_history_path, track_embeddings_path)
    n_users = len(user_embeddings)
    print(f"[personas] {n_users} users loaded  (resume={resume})")

    workers = n_jobs or (os.cpu_count() or 1)

    k_values: dict[str, int] = {}
    n_skipped = 0
    futures_map = {}

    with ProcessPoolExecutor(max_workers=workers) as pool:
        for user_id, embeddings in user_embeddings.items():
            f = pool.submit(
                _fit_and_save,
                user_id, embeddings, out,
                k_min, k_max, covariance_type, min_samples, n_init, resume,
            )
            futures_map[f] = user_id

        with tqdm(total=n_users, desc="Fitting GMMs") as bar:
            for future in as_completed(futures_map):
                uid, k, skipped = future.result()
                if skipped:
                    n_skipped += 1
                else:
                    k_values[uid] = k
                bar.update(1)

    vals = list(k_values.values())
    if vals:
        print(f"[personas] Fitted {len(vals)} users. K stats — mean={np.mean(vals):.1f}, min={min(vals)}, max={max(vals)}")
    if n_skipped:
        print(f"[personas] Skipped {n_skipped} already-fitted users (resume=True)")
    if not vals and not n_skipped:
        print("[personas] Done. No users fitted.")
    return k_values


def load_persona(user_dir: str | Path) -> UserPersonaModel:
    with open(Path(user_dir) / "persona.pkl", "rb") as f:
        return pickle.load(f)
