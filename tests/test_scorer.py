"""Tests for scoring function — sonic, emotional, surprise, full score, weight optimizer."""

import numpy as np
import pytest

from music_discovery.models.gmm import fit_user_persona
from music_discovery.models.scorer import (
    sonic_fit,
    emotional_fit,
    surprise_score,
    score_candidates,
    optimize_weights,
    DEFAULT_WEIGHTS,
)


def make_norm_emb(n: int, d: int = 24, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    x = rng.standard_normal((n, d)).astype(np.float32)
    return x / np.linalg.norm(x, axis=1, keepdims=True)


def make_persona(n: int = 80, d: int = 24, seed: int = 0):
    emb = make_norm_emb(n, d, seed)
    return fit_user_persona("user_test", emb, k_min=2, k_max=4)


# ---------------------------------------------------------------------------
# sonic_fit
# ---------------------------------------------------------------------------

def test_sonic_fit_shape():
    persona = make_persona()
    candidates = make_norm_emb(10)
    scores = sonic_fit(candidates, persona)
    assert scores.shape == (10,)


def test_sonic_fit_range():
    """Cosine sim in [-1,1]."""
    persona = make_persona()
    candidates = make_norm_emb(20)
    scores = sonic_fit(candidates, persona)
    assert scores.min() >= -1.0 - 1e-5
    assert scores.max() <= 1.0 + 1e-5


def test_sonic_fit_identical_to_centroid_is_max():
    """A candidate identical to a persona centroid should score near 1."""
    persona = make_persona()
    centroid = persona.means[0:1]  # (1, D)
    centroid = centroid / np.linalg.norm(centroid, axis=1, keepdims=True)
    score = sonic_fit(centroid.astype(np.float32), persona)
    assert score[0] > 0.95


# ---------------------------------------------------------------------------
# emotional_fit
# ---------------------------------------------------------------------------

def test_emotional_fit_shape():
    persona = make_persona()
    candidates = make_norm_emb(15)
    scores = emotional_fit(candidates, persona)
    assert scores.shape == (15,)


def test_emotional_fit_in_unit_range():
    persona = make_persona()
    candidates = make_norm_emb(30)
    scores = emotional_fit(candidates, persona)
    assert scores.min() >= -1e-6
    assert scores.max() <= 1.0 + 1e-6


# ---------------------------------------------------------------------------
# surprise_score
# ---------------------------------------------------------------------------

def make_artists(n: int, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    all_artists = [f"artist_{i}" for i in range(20)]
    return np.array([rng.choice(all_artists) for _ in range(n)])


def test_surprise_score_shape():
    candidates = make_norm_emb(10)
    history = make_norm_emb(20, seed=1)
    c_artists = make_artists(10)
    h_artists = make_artists(20, seed=1)
    scores = surprise_score(candidates, c_artists, history, h_artists)
    assert scores.shape == (10,)


def test_surprise_score_range():
    candidates = make_norm_emb(20)
    history = make_norm_emb(15, seed=1)
    c_artists = make_artists(20)
    h_artists = make_artists(15, seed=1)
    scores = surprise_score(candidates, c_artists, history, h_artists)
    assert scores.min() >= 0.0 - 1e-6
    assert scores.max() <= 1.0 + 1e-6


def test_surprise_known_artist_lower_novelty():
    """Known artist should have artist_novelty=0, unknown=1."""
    candidates = make_norm_emb(2)
    history = make_norm_emb(5, seed=1)
    # artist_0 is known, artist_99 is unknown
    c_artists = np.array(["artist_0", "artist_99"])
    h_artists = np.array(["artist_0", "artist_1", "artist_2", "artist_3", "artist_4"])
    scores = surprise_score(candidates, c_artists, history, h_artists)
    # Unknown artist score >= known artist score (novelty component)
    assert scores[1] >= scores[0]


# ---------------------------------------------------------------------------
# score_candidates
# ---------------------------------------------------------------------------

def test_score_candidates_shape():
    persona = make_persona()
    candidates = make_norm_emb(25)
    history = make_norm_emb(30, seed=1)
    c_artists = make_artists(25)
    h_artists = make_artists(30, seed=1)
    scores = score_candidates(candidates, c_artists, persona, history, h_artists)
    assert scores.shape == (25,)


def test_score_candidates_non_uniform():
    """Scores should not all be equal — there must be variation."""
    persona = make_persona()
    candidates = make_norm_emb(50)
    history = make_norm_emb(40, seed=1)
    c_artists = make_artists(50)
    h_artists = make_artists(40, seed=1)
    scores = score_candidates(candidates, c_artists, persona, history, h_artists)
    assert scores.std() > 1e-4


def test_score_candidates_custom_weights():
    persona = make_persona()
    candidates = make_norm_emb(10)
    history = make_norm_emb(10, seed=1)
    c_artists = make_artists(10)
    h_artists = make_artists(10, seed=1)
    w = np.array([1.0, 0.0, 0.0])
    scores = score_candidates(candidates, c_artists, persona, history, h_artists, weights=w)
    assert scores.shape == (10,)


# ---------------------------------------------------------------------------
# optimize_weights
# ---------------------------------------------------------------------------

def test_optimize_weights_fallback_small_history():
    """< min_songs → returns DEFAULT_WEIGHTS."""
    persona = make_persona()
    candidates = make_norm_emb(20)
    history = make_norm_emb(10, seed=1)
    c_artists = make_artists(20)
    h_artists = make_artists(10, seed=1)
    held_out = np.array([0, 1, 2])
    w = optimize_weights(candidates, c_artists, persona, history, h_artists, held_out, min_songs=50)
    np.testing.assert_array_equal(w, DEFAULT_WEIGHTS)


def test_optimize_weights_sums_to_one():
    persona = make_persona(n=120)
    candidates = make_norm_emb(50)
    history = make_norm_emb(60, seed=1)
    c_artists = make_artists(50)
    h_artists = make_artists(60, seed=1)
    held_out = np.arange(10)
    w = optimize_weights(candidates, c_artists, persona, history, h_artists, held_out, min_songs=50)
    assert abs(w.sum() - 1.0) < 1e-5


def test_optimize_weights_non_negative():
    persona = make_persona(n=120)
    candidates = make_norm_emb(50)
    history = make_norm_emb(60, seed=1)
    c_artists = make_artists(50)
    h_artists = make_artists(60, seed=1)
    held_out = np.arange(10)
    w = optimize_weights(candidates, c_artists, persona, history, h_artists, held_out, min_songs=50)
    assert (w >= 0).all()
