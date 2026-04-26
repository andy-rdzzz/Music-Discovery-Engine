import numpy as np
import pytest

from music_discovery.models.gmm import UserPersonaModel, fit_user_persona, bic_curve


def make_embeddings(n: int = 100, d: int = 24, n_clusters: int = 3, seed: int = 0) -> np.ndarray:
    """Synthetic embeddings with clear cluster structure, always returns exactly n rows."""
    rng = np.random.default_rng(seed)
    centers = rng.standard_normal((n_clusters, d)) * 3
    # Distribute n rows across clusters, handling remainder
    counts = [n // n_clusters + (1 if i < n % n_clusters else 0) for i in range(n_clusters)]
    parts = [rng.standard_normal((counts[k], d)) * 0.3 + centers[k] for k in range(n_clusters)]
    emb = np.vstack(parts)
    # L2 normalise
    emb = emb / np.linalg.norm(emb, axis=1, keepdims=True)
    return emb.astype(np.float32)


# ---------------------------------------------------------------------------
# fit_user_persona
# ---------------------------------------------------------------------------

def test_fit_returns_persona_model():
    emb = make_embeddings()
    model = fit_user_persona("user_1", emb)
    assert isinstance(model, UserPersonaModel)


def test_persona_k_in_range():
    emb = make_embeddings(n=120)
    model = fit_user_persona("user_1", emb, k_min=2, k_max=6)
    assert 2 <= model.k <= 6


def test_persona_means_shape():
    emb = make_embeddings(n=100, d=24)
    model = fit_user_persona("user_1", emb)
    assert model.means.shape[1] == 24
    assert model.means.shape[0] == model.k


def test_persona_weights_sum_to_one():
    emb = make_embeddings()
    model = fit_user_persona("user_1", emb)
    assert abs(model.weights.sum() - 1.0) < 1e-6


def test_persona_fallback_single_gaussian_small_data():
    """Fewer than min_samples → K=1 fallback."""
    rng = np.random.default_rng(1)
    emb = rng.standard_normal((10, 24)).astype(np.float32)
    model = fit_user_persona("user_1", emb, min_samples=20)
    assert model.k == 1


def test_persona_stores_user_id():
    emb = make_embeddings()
    model = fit_user_persona("user_42", emb)
    assert model.user_id == "user_42"


# ---------------------------------------------------------------------------
# posterior / dominant_persona / log_prob
# ---------------------------------------------------------------------------

def test_posterior_shape():
    emb = make_embeddings(n=60)
    model = fit_user_persona("u", emb)
    query = make_embeddings(n=10, seed=99)
    post = model.posterior(query)
    assert post.shape == (10, model.k)


def test_posterior_rows_sum_to_one():
    emb = make_embeddings(n=60)
    model = fit_user_persona("u", emb)
    query = make_embeddings(n=10, seed=99)
    post = model.posterior(query)
    np.testing.assert_allclose(post.sum(axis=1), 1.0, atol=1e-6)


def test_dominant_persona_shape():
    emb = make_embeddings(n=60)
    model = fit_user_persona("u", emb)
    query = make_embeddings(n=8, seed=5)
    dom = model.dominant_persona(query)
    assert dom.shape == (8,)
    assert dom.max() < model.k


def test_log_prob_shape():
    emb = make_embeddings(n=60)
    model = fit_user_persona("u", emb)
    query = make_embeddings(n=5, seed=7)
    lp = model.log_prob(query)
    assert lp.shape == (5,)


# ---------------------------------------------------------------------------
# bic_curve
# ---------------------------------------------------------------------------

def test_bic_curve_returns_dict():
    emb = make_embeddings(n=80, n_clusters=3)
    scores = bic_curve(emb, k_min=1, k_max=5)
    assert isinstance(scores, dict)
    assert len(scores) > 0


def test_bic_curve_keys_in_range():
    emb = make_embeddings(n=80)
    scores = bic_curve(emb, k_min=1, k_max=5)
    for k in scores:
        assert 1 <= k <= 5


def test_bic_minimum_at_true_k():
    """BIC should favour K=3 for clearly 3-cluster data."""
    emb = make_embeddings(n=150, n_clusters=3, seed=42)
    scores = bic_curve(emb, k_min=1, k_max=6)
    best_k = min(scores, key=scores.__getitem__)
    # Allow ±1 tolerance since BIC is not guaranteed exact
    assert abs(best_k - 3) <= 1, f"Expected BIC min near K=3, got K={best_k}"
