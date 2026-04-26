from __future__ import annotations
import yaml
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split

from music_discovery.models.mlp import EmbeddingMLP
from music_discovery.models.triplet import TripletDataset, mine_semi_hard_negatives
from music_discovery.data.features import build_feature_matrix


def load_config(config_path: str | Path) -> dict[str, Any]:
    with open(config_path) as f:
        return yaml.safe_load(f)


def train_embeddings(
    track_features_path: str | Path,
    triplets_path: str | Path,
    output_dir: str | Path,
    config_path: str | Path | None = None,
    device: str | None = None,
) -> EmbeddingMLP:
    """
    Train EmbeddingMLP and save weights to output_dir/embedding_model.pt.

    Returns the trained model.
    """
    cfg = load_config(config_path) if config_path else {}
    model_cfg = cfg.get("model", {})
    train_cfg = cfg.get("training", {})

    input_dim = model_cfg.get("input_dim", 12)
    hidden_dims = model_cfg.get("hidden_dims", [64, 48])
    embedding_dim = model_cfg.get("embedding_dim", 24)
    dropout = float(model_cfg.get("dropout", 0.2))

    batch_size = train_cfg.get("batch_size", 512)
    epochs = train_cfg.get("epochs", 50)
    lr = float(train_cfg.get("lr", 1e-3))
    weight_decay = float(train_cfg.get("weight_decay", 2e-3))
    margin = float(train_cfg.get("margin", 0.4))
    patience = train_cfg.get("patience", 10)
    uniformity_weight = float(train_cfg.get("uniformity_weight", 0.05))

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    dev = torch.device(device)
    print(f"[train] Device: {dev}")

    # Load data
    track_df = pd.read_parquet(track_features_path)
    triplets_df = pd.read_parquet(triplets_path)

    from music_discovery.data.features import FEATURE_VECTOR_COLS
    feature_matrix = build_feature_matrix(track_df)

    dataset = TripletDataset(triplets_df, feature_matrix)

    val_size = max(1, int(0.2 * len(dataset)))
    train_size = len(dataset) - val_size
    train_ds, val_ds = random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=False)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=0)

    model = EmbeddingMLP(input_dim, hidden_dims, embedding_dim, dropout).to(dev)
    criterion = nn.TripletMarginLoss(margin=margin, p=2)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    best_val_loss = float("inf")
    epochs_no_improve = 0
    history: list[dict] = []

    for epoch in range(1, epochs + 1):
        # Train
        model.train()
        train_loss = 0.0
        for anchor, positive, negative in train_loader:
            anchor = anchor.to(dev)
            positive = positive.to(dev)
            negative = negative.to(dev)

            optimizer.zero_grad()
            emb_a = model(anchor)
            emb_p = model(positive)
            emb_n = model(negative)

            # Online semi-hard negative mining within the batch.
            # all_emb stays in the compute graph so gradients flow through
            # the selected negatives. Only the index selection is no_grad.
            B = emb_a.size(0)
            all_emb = torch.cat([emb_a, emb_p, emb_n], dim=0)  # (3B, D)
            batch_indices = torch.stack([
                torch.arange(B, device=dev),
                torch.arange(B, 2 * B, device=dev),
                torch.arange(2 * B, 3 * B, device=dev),
            ], dim=1)
            with torch.no_grad():
                mined = mine_semi_hard_negatives(all_emb.detach(), batch_indices, margin)
            emb_n = all_emb[mined[:, 2]]  # replaced negatives, still in graph

            loss = criterion(emb_a, emb_p, emb_n)
            threshold = 0.5 / embedding_dim
            var_penalty = torch.mean(torch.relu(threshold - emb_a.var(dim=0)))
            loss = loss + uniformity_weight * var_penalty
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.5)
            optimizer.step()
            train_loss += loss.item() * anchor.size(0)

        train_loss /= train_size
        scheduler.step()

        # Validate
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for anchor, positive, negative in val_loader:
                anchor = anchor.to(dev)
                positive = positive.to(dev)
                negative = negative.to(dev)
                emb_a = model(anchor)
                emb_p = model(positive)
                emb_n = model(negative)
                loss = criterion(emb_a, emb_p, emb_n)
                val_loss += loss.item() * anchor.size(0)
        val_loss /= val_size

        history.append({"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss})
        print(f"[train] Epoch {epoch:3d}/{epochs} | train={train_loss:.4f} | val={val_loss:.4f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            epochs_no_improve = 0
            torch.save(model.state_dict(), out_dir / "embedding_model.pt")
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                print(f"[train] Early stopping at epoch {epoch} (patience={patience})")
                break

    # Load best weights
    model.load_state_dict(torch.load(out_dir / "embedding_model.pt", map_location=dev))

    # Save training history
    pd.DataFrame(history).to_csv(out_dir / "training_history.csv", index=False)
    print(f"[train] Best val loss: {best_val_loss:.4f} — model saved to {out_dir}/embedding_model.pt")

    return model


def embed_all_tracks(
    model: EmbeddingMLP,
    track_features_path: str | Path,
    output_path: str | Path,
    device: str | None = None,
    batch_size: int = 2048,
) -> np.ndarray:
    """
    Run all tracks through the trained model and save embeddings as parquet.

    Returns (N, 24) embedding array.
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    dev = torch.device(device)

    track_df = pd.read_parquet(track_features_path)
    feature_matrix = build_feature_matrix(track_df)
    features_tensor = torch.from_numpy(feature_matrix).to(dev)

    model.eval().to(dev)
    all_embeddings = []
    with torch.no_grad():
        for start in range(0, len(features_tensor), batch_size):
            batch = features_tensor[start:start + batch_size]
            all_embeddings.append(model(batch).cpu().numpy())

    embeddings = np.vstack(all_embeddings)

    emb_cols = [f"emb_{i}" for i in range(embeddings.shape[1])]
    emb_df = track_df[["artist_norm", "track_norm"]].copy()
    for i, col in enumerate(emb_cols):
        emb_df[col] = embeddings[:, i]

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    emb_df.to_parquet(output_path, index=False)
    print(f"[embed] Saved {len(emb_df):,} track embeddings to {output_path}")
    return embeddings
