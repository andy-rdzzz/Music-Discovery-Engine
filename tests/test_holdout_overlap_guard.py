from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from music_discovery.evaluate import holdout_eval


class _FakeFuture:
    def __init__(self, value):
        self._value = value

    def result(self):
        return self._value


class _FakePool:
    def __init__(self, max_workers=None, initializer=None, initargs=()):
        self._initializer = initializer
        self._initargs = initargs

    def __enter__(self):
        if self._initializer:
            self._initializer(*self._initargs)
        return self

    def __exit__(self, exc_type, exc, tb):
        return False

    def submit(self, _fn, uid, *_args):
        row = {
            "user_id": uid,
            "model": "als",
            "Recall@10": 0.1,
            "Recall_adj@10": 0.2,
            "NDCG@10": 0.1,
            "NDCG_adj@10": 0.2,
            "HitRate@10": 1.0,
            "ILD": 0.5,
            "mean_popularity_rank": 2.0,
            "long_tail_pct": 0.25,
            "val_coverage": 1.0,
            "test_coverage": 1.0,
            "als_fallback": 0.0,
        }
        return _FakeFuture([row])


def _make_interactions() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {"user_id": "u1", "song_id": "s1", "play_count": 5, "split": "train"},
            {"user_id": "u1", "song_id": "s2", "play_count": 1, "split": "val"},
            {"user_id": "u1", "song_id": "s3", "play_count": 1, "split": "test"},
            {"user_id": "u2", "song_id": "s2", "play_count": 3, "split": "train"},
            {"user_id": "u2", "song_id": "s3", "play_count": 1, "split": "val"},
            {"user_id": "u2", "song_id": "s4", "play_count": 1, "split": "test"},
            {"user_id": "u3", "song_id": "s3", "play_count": 4, "split": "train"},
            {"user_id": "u3", "song_id": "s4", "play_count": 1, "split": "val"},
            {"user_id": "u3", "song_id": "s1", "play_count": 1, "split": "test"},
        ]
    )


def _make_song_embeddings() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "song_id": ["s1", "s2", "s3", "s4"],
            "emb_0": [1.0, 0.9, 0.1, 0.0],
            "emb_1": [0.0, 0.1, 0.9, 1.0],
        }
    )


def _make_songs() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "song_id": ["s1", "s2", "s3", "s4"],
            "artist_name": ["a1", "a2", "a3", "a4"],
        }
    )


def _run_with_mocks(monkeypatch: pytest.MonkeyPatch, tmp_path: Path, user_embeddings: pd.DataFrame, n_users: int | None = None):
    processed_dir = tmp_path / "processed"
    processed_dir.mkdir()
    output_dir = tmp_path / "out"

    interactions_path = processed_dir / "interactions.parquet"
    songs_path = processed_dir / "songs.parquet"
    song_embeddings_path = tmp_path / "song_embeddings.parquet"
    user_embeddings_path = tmp_path / "user_embeddings.parquet"

    tables = {
        str(interactions_path): _make_interactions(),
        str(songs_path): _make_songs(),
        str(song_embeddings_path): _make_song_embeddings(),
        str(user_embeddings_path): user_embeddings,
    }

    monkeypatch.setattr(holdout_eval.os.path, "exists", lambda path: str(path) in tables)
    monkeypatch.setattr(
        holdout_eval.pd,
        "read_parquet",
        lambda path, columns=None: tables[str(path)][columns] if columns else tables[str(path)],
    )
    monkeypatch.setattr(holdout_eval, "ProcessPoolExecutor", _FakePool)
    monkeypatch.setattr(holdout_eval, "as_completed", lambda futures: list(futures))

    return holdout_eval.run(
        interactions_path=str(interactions_path),
        song_embeddings_path=str(song_embeddings_path),
        personas_dir=str(tmp_path / "personas"),
        output_dir=str(output_dir),
        k=10,
        n_users=n_users,
        n_jobs=1,
        rerank_top_m=50,
    )


def test_holdout_run_succeeds_with_healthy_overlap_and_writes_plots(monkeypatch, tmp_path):
    user_embeddings = pd.DataFrame(
        {
            "user_id": ["u1", "u2", "u3"],
            "emb_0": [1.0, 0.5, 0.2],
            "emb_1": [0.0, 0.5, 0.8],
        }
    )

    results = _run_with_mocks(monkeypatch, tmp_path, user_embeddings)

    assert set(results["user_id"]) == {"u1", "u2", "u3"}
    out_files = {path.name for path in (tmp_path / "out").iterdir()}
    assert "holdout_adjusted_metrics.png" in out_files
    assert "holdout_tradeoffs.png" in out_files


def test_holdout_run_raises_on_low_overlap(monkeypatch, tmp_path):
    user_embeddings = pd.DataFrame(
        {
            "user_id": ["u1"],
            "emb_0": [1.0],
            "emb_1": [0.0],
        }
    )

    with pytest.raises(RuntimeError, match="Artifact mismatch"):
        _run_with_mocks(monkeypatch, tmp_path, user_embeddings)


def test_holdout_subset_eval_allows_large_embedding_file_when_overlap_is_good(monkeypatch, tmp_path):
    user_embeddings = pd.DataFrame(
        {
            "user_id": ["u1", "u2", "u3", "u4", "u5"],
            "emb_0": [1.0, 0.5, 0.2, 0.4, 0.9],
            "emb_1": [0.0, 0.5, 0.8, 0.6, 0.1],
        }
    )

    results = _run_with_mocks(monkeypatch, tmp_path, user_embeddings, n_users=1)

    assert set(results["user_id"]) == {"u1"}
