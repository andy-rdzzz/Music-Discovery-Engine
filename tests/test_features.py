import numpy as np
import pandas as pd
import pytest

from music_discovery.data.features import (
    engineer_features,
    build_feature_matrix,
    FEATURE_VECTOR_COLS,
    LOUDNESS_MIN,
    LOUDNESS_MAX,
    TEMPO_MIN,
    TEMPO_MAX,
)


def make_tracks(**overrides) -> pd.DataFrame:
    """Minimal valid track dataframe with sensible defaults."""
    base = dict(
        danceability=0.7,
        energy=0.8,
        loudness=-5.0,
        speechiness=0.05,
        acousticness=0.1,
        instrumentalness=0.0,
        liveness=0.1,
        valence=0.6,
        tempo=120.0,
    )
    base.update(overrides)
    return pd.DataFrame([base])


# ---------------------------------------------------------------------------
# Shape / columns
# ---------------------------------------------------------------------------

def test_engineer_features_adds_expected_columns():
    df = engineer_features(make_tracks())
    for col in ["loudness_norm", "tempo_norm", "arousal", "chill_factor", "vocal_presence"]:
        assert col in df.columns, f"Missing column: {col}"


def test_feature_vector_cols_length():
    assert len(FEATURE_VECTOR_COLS) == 12


def test_build_feature_matrix_shape():
    df = engineer_features(make_tracks())
    mat = build_feature_matrix(df)
    assert mat.shape == (1, 12)
    assert mat.dtype == np.float32


# ---------------------------------------------------------------------------
# Normalisation ranges
# ---------------------------------------------------------------------------

def test_loudness_norm_clipped_to_unit():
    quiet = engineer_features(make_tracks(loudness=LOUDNESS_MIN))
    loud = engineer_features(make_tracks(loudness=LOUDNESS_MAX))
    assert quiet["loudness_norm"].iloc[0] == pytest.approx(0.0)
    assert loud["loudness_norm"].iloc[0] == pytest.approx(1.0)


def test_loudness_norm_out_of_range_clipped():
    df = engineer_features(make_tracks(loudness=10.0))  # above max
    assert df["loudness_norm"].iloc[0] == pytest.approx(1.0)


def test_tempo_norm_boundaries():
    slow = engineer_features(make_tracks(tempo=TEMPO_MIN))
    fast = engineer_features(make_tracks(tempo=TEMPO_MAX))
    assert slow["tempo_norm"].iloc[0] == pytest.approx(0.0)
    assert fast["tempo_norm"].iloc[0] == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# Semantic / circumplex validity
# ---------------------------------------------------------------------------

def test_arousal_high_energy_dance():
    """High energy + high danceability → arousal near 1."""
    df = engineer_features(make_tracks(energy=1.0, danceability=1.0))
    assert df["arousal"].iloc[0] == pytest.approx(1.0)


def test_arousal_low_energy_dance():
    """Low energy + low danceability → arousal near 0."""
    df = engineer_features(make_tracks(energy=0.0, danceability=0.0))
    assert df["arousal"].iloc[0] == pytest.approx(0.0)


def test_chill_factor_acoustic_low_energy():
    """Pure acoustic, zero energy → max chill."""
    df = engineer_features(make_tracks(acousticness=1.0, energy=0.0))
    assert df["chill_factor"].iloc[0] == pytest.approx(1.0)


def test_chill_factor_loud_track_is_not_chill():
    """High energy acoustic song is less chill than low energy one."""
    chill = engineer_features(make_tracks(acousticness=0.8, energy=0.1))
    loud = engineer_features(make_tracks(acousticness=0.8, energy=0.9))
    assert chill["chill_factor"].iloc[0] > loud["chill_factor"].iloc[0]


def test_vocal_presence_pure_instrumental():
    """instrumentalness=1 → vocal_presence=0 (no vocals)."""
    df = engineer_features(make_tracks(instrumentalness=1.0, speechiness=0.0))
    assert df["vocal_presence"].iloc[0] == pytest.approx(0.0)


def test_vocal_presence_pure_speech():
    """speechiness=1 → vocal_presence=0 (spoken word, not sung)."""
    df = engineer_features(make_tracks(instrumentalness=0.0, speechiness=1.0))
    assert df["vocal_presence"].iloc[0] == pytest.approx(0.0)


def test_vocal_presence_sung_vocal():
    """Low instrumentalness + low speechiness → high vocal_presence (sung melody)."""
    df = engineer_features(make_tracks(instrumentalness=0.0, speechiness=0.05))
    assert df["vocal_presence"].iloc[0] > 0.9


# ---------------------------------------------------------------------------
# L2 normalisation
# ---------------------------------------------------------------------------

def test_feature_matrix_l2_normalised():
    """Each row in build_feature_matrix must have unit L2 norm."""
    rows = [make_tracks(energy=e, danceability=d) for e, d in [(0.2, 0.3), (0.9, 0.8), (0.5, 0.5)]]
    df = pd.concat([engineer_features(r) for r in rows], ignore_index=True)
    mat = build_feature_matrix(df)
    norms = np.linalg.norm(mat, axis=1)
    np.testing.assert_allclose(norms, 1.0, atol=1e-6)


def test_feature_matrix_zero_vector_no_nan():
    """All-zero row should not produce NaN (division protected)."""
    df = engineer_features(make_tracks(
        danceability=0, energy=0, loudness=LOUDNESS_MIN,
        speechiness=0, acousticness=0, instrumentalness=0,
        liveness=0, valence=0, tempo=TEMPO_MIN,
    ))
    mat = build_feature_matrix(df)
    assert not np.isnan(mat).any()


# ---------------------------------------------------------------------------
# Multi-row consistency
# ---------------------------------------------------------------------------

def test_engineer_features_raises_on_zero_loudness_range():
    with pytest.raises(ValueError, match="loudness_max must differ"):
        engineer_features(make_tracks(), loudness_min=0.0, loudness_max=0.0)


def test_engineer_features_raises_on_zero_tempo_range():
    with pytest.raises(ValueError, match="tempo_max must differ"):
        engineer_features(make_tracks(), tempo_min=120.0, tempo_max=120.0)


def test_engineer_features_preserves_row_count():
    df = pd.concat([make_tracks(energy=i / 10) for i in range(10)], ignore_index=True)
    out = engineer_features(df)
    assert len(out) == 10


def test_build_feature_matrix_multi_row_shape():
    df = pd.concat([engineer_features(make_tracks(energy=i / 10)) for i in range(5)], ignore_index=True)
    mat = build_feature_matrix(df)
    assert mat.shape == (5, 12)
