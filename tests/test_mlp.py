import numpy as np
import pandas as pd
import pytest
import torch

from music_discovery.models.mlp import EmbeddingMLP
from music_discovery.models.triplet import TripletDataset, mine_semi_hard_negatives


# ---------------------------------------------------------------------------
# EmbeddingMLP
# ---------------------------------------------------------------------------

def make_model(**kwargs) -> EmbeddingMLP:
    return EmbeddingMLP(**kwargs)


def test_output_shape_default():
    model = make_model()
    x = torch.randn(8, 12)
    out = model(x)
    assert out.shape == (8, 24)


def test_output_shape_custom_dims():
    model = make_model(input_dim=12, hidden_dims=[128, 64], embedding_dim=32)
    x = torch.randn(4, 12)
    out = model(x)
    assert out.shape == (4, 32)


def test_output_is_l2_normalised():
    model = make_model()
    model.eval()
    x = torch.randn(16, 12)
    with torch.no_grad():
        out = model(x)
    norms = out.norm(p=2, dim=1)
    torch.testing.assert_close(norms, torch.ones(16), atol=1e-5, rtol=0)


def test_output_l2_normalised_training_mode():
    """L2 normalisation must hold even in train mode (BatchNorm active)."""
    model = make_model()
    model.train()
    x = torch.randn(32, 12)
    out = model(x)
    norms = out.norm(p=2, dim=1)
    torch.testing.assert_close(norms, torch.ones(32), atol=1e-5, rtol=0)


def test_forward_no_nan():
    model = make_model()
    model.eval()
    x = torch.zeros(4, 12)
    with torch.no_grad():
        out = model(x)
    assert not torch.isnan(out).any()


def test_gradients_flow():
    model = make_model()
    model.train()
    x = torch.randn(8, 12)
    out = model(x)
    loss = out.sum()
    loss.backward()
    for name, param in model.named_parameters():
        assert param.grad is not None, f"No grad for {name}"


def test_different_inputs_give_different_outputs():
    model = make_model()
    model.eval()
    x1 = torch.randn(4, 12)
    x2 = torch.randn(4, 12)
    with torch.no_grad():
        o1 = model(x1)
        o2 = model(x2)
    assert not torch.allclose(o1, o2)


# ---------------------------------------------------------------------------
# TripletDataset
# ---------------------------------------------------------------------------

def make_triplet_dataset(n_tracks: int = 20, n_triplets: int = 10):
    rng = np.random.default_rng(0)
    features = rng.random((n_tracks, 12)).astype(np.float32)
    # Normalise rows
    norms = np.linalg.norm(features, axis=1, keepdims=True)
    features = features / norms

    indices = rng.integers(0, n_tracks, size=(n_triplets, 3))
    triplets_df = pd.DataFrame(indices, columns=["anchor_idx", "pos_idx", "neg_idx"])
    return TripletDataset(triplets_df, features), features


def test_dataset_length():
    ds, _ = make_triplet_dataset(n_triplets=15)
    assert len(ds) == 15


def test_dataset_item_shapes():
    ds, _ = make_triplet_dataset()
    a, p, n = ds[0]
    assert a.shape == (12,)
    assert p.shape == (12,)
    assert n.shape == (12,)


def test_dataset_item_dtype():
    ds, _ = make_triplet_dataset()
    a, p, n = ds[0]
    assert a.dtype == torch.float32


def test_dataset_values_match_features():
    ds, features = make_triplet_dataset()
    anchor_idx = ds.triplets[0, 0]
    a, _, _ = ds[0]
    torch.testing.assert_close(a, torch.from_numpy(features[anchor_idx]))


# ---------------------------------------------------------------------------
# Semi-hard negative mining
# ---------------------------------------------------------------------------

def test_semi_hard_output_shape():
    B, D = 8, 24
    embeddings = torch.randn(B, D)
    embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=1)
    indices = torch.randint(0, B, (B, 3))
    out = mine_semi_hard_negatives(embeddings, indices, margin=0.3)
    assert out.shape == (B, 3)


def test_semi_hard_anchors_positives_unchanged():
    B, D = 8, 24
    embeddings = torch.nn.functional.normalize(torch.randn(B, D), p=2, dim=1)
    indices = torch.randint(0, B, (B, 3))
    out = mine_semi_hard_negatives(embeddings, indices, margin=0.3)
    torch.testing.assert_close(out[:, 0], indices[:, 0])
    torch.testing.assert_close(out[:, 1], indices[:, 1])


def test_semi_hard_negative_in_valid_range():
    """Mined negatives must be valid batch indices."""
    B, D = 16, 24
    embeddings = torch.nn.functional.normalize(torch.randn(B, D), p=2, dim=1)
    indices = torch.randint(0, B, (B, 3))
    out = mine_semi_hard_negatives(embeddings, indices, margin=0.3)
    assert (out[:, 2] >= 0).all()
    assert (out[:, 2] < B).all()
