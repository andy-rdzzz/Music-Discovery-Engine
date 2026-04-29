from __future__ import annotations
import numpy as np
import pandas as pd

def load_taste_profile(
    path: str, 
    min_play_count: int = 1,
    sample_n_users: int | None = None, 
    chunksize: int = 500_000
) -> pd.DataFrame:
    
    
    chunks = []
    reader = pd.read_csv(path,
                         sep="\t",
                         header = None,
                         names = ["user_id", "song_id", "play_count"],
                         dtype={"user_id": str, "song_id": str, "play_count": np.int32},
                         chunksize=chunksize
    )

    for chunk in reader:
        if min_play_count > 1:
            chunk = chunk[chunk["play_count"] >= min_play_count]
        chunks.append(chunk)
        
    df = pd.concat(chunks, ignore_index=True)
    
    if sample_n_users is not None:
        users = df["user_id"].unique()
        rng = np.random.default_rng(42)
        sampled = rng.choice(users, size=min(sample_n_users, len(users)), replace=False)
        df = df[df["user_id"].isin(sampled)].copy()

    df["weight"] = np.log1p(df["play_count"].astype(float))
    return df.reset_index(drop=True)