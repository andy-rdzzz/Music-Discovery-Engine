from __future__ import annotations

import numpy as np
import pytest

from music_discovery.evaluate import holdout_eval
from music_discovery.models import scorer
from music_discovery.train import fit_personas


@pytest.fixture(autouse=True)
def seed_worker_state():
    holdout_eval._W.clear()
    song_ids = np.array(["s1", "s2", "s3", "s4"], dtype=object)
    emb = np.array(
        [
            [1.0, 0.0],
            [0.8, 0.2],
            [0.2, 0.8],
            [0.0, 1.0],
        ],
        dtype=np.float32,
    )
    holdout_eval._W.update(
        {
            "song_ids": song_ids,
            "emb": emb,
            "song_to_idx": {song_id: idx for idx, song_id in enumerate(song_ids)},
            "song_to_artist": {"s1": "a1", "s2": "a2", "s3": "a3", "s4": "a4"},
            "pop_idx_order": np.array([0, 1, 2, 3], dtype=np.intp),
            "popularity_ranks": {"s1": 1, "s2": 2, "s3": 3, "s4": 4},
            "n_songs": 4,
            "p80_threshold": 3,
            "personas_dir": "/tmp/personas",
            "k": 2,
            "rerank_top_m": 7,
            "song_catalog": {"s1", "s2", "s3"},
        }
    )
    yield
    holdout_eval._W.clear()


def _rows_by_model(rows: list[dict]) -> dict[str, dict]:
    return {row["model"]: row for row in rows}


def test_eval_user_marks_als_fallback_when_user_embedding_missing(monkeypatch):
    monkeypatch.setattr(fit_personas, "load_persona", lambda _path: (_ for _ in ()).throw(FileNotFoundError()))

    rows = holdout_eval._eval_user(
        uid="u1",
        val_relevant={"s2"},
        test_relevant={"s3", "s4"},
        train_songs={"s1"},
        u_vec=None,
    )

    for row in rows:
        assert row["als_fallback"] == 1.0
        assert row["val_coverage"] == 1.0
        assert row["test_coverage"] == 0.5


def test_eval_user_uses_als_scores_when_embedding_present(monkeypatch):
    monkeypatch.setattr(fit_personas, "load_persona", lambda _path: (_ for _ in ()).throw(FileNotFoundError()))

    rows = holdout_eval._eval_user(
        uid="u1",
        val_relevant={"s2"},
        test_relevant={"s3"},
        train_songs={"s1"},
        u_vec=np.array([0.1, 1.0], dtype=np.float32),
    )

    for row in rows:
        assert row["als_fallback"] == 0.0

    by_model = _rows_by_model(rows)
    assert by_model["als"]["HitRate@2"] == 1.0


def test_eval_user_missing_persona_falls_back_to_als(monkeypatch):
    monkeypatch.setattr(fit_personas, "load_persona", lambda _path: (_ for _ in ()).throw(FileNotFoundError()))

    rows = holdout_eval._eval_user(
        uid="u1",
        val_relevant={"s2"},
        test_relevant={"s3"},
        train_songs={"s1"},
        u_vec=np.array([0.1, 1.0], dtype=np.float32),
    )

    by_model = _rows_by_model(rows)
    assert by_model["persona_reranker"]["Recall@2"] == by_model["als"]["Recall@2"]
    assert by_model["persona_reranker"]["NDCG@2"] == by_model["als"]["NDCG@2"]


def test_eval_user_passes_rerank_top_m_and_relevance_scores(monkeypatch):
    calls: dict[str, object] = {}

    monkeypatch.setattr(fit_personas, "load_persona", lambda _path: object())
    monkeypatch.setattr(scorer, "optimize_weights", lambda **_kwargs: np.array([1.0, 0.0, 0.0, 0.0]))

    def fake_recommend_diverse(**kwargs):
        calls.update(kwargs)
        return np.array([1, 0], dtype=np.intp)

    monkeypatch.setattr(scorer, "recommend_diverse", fake_recommend_diverse)

    rows = holdout_eval._eval_user(
        uid="u1",
        val_relevant={"s2"},
        test_relevant={"s3"},
        train_songs={"s1"},
        u_vec=np.array([0.1, 1.0], dtype=np.float32),
    )

    by_model = _rows_by_model(rows)
    assert calls["rerank_top_m"] == 7
    assert calls["relevance_scores"] is not None
    assert by_model["persona_reranker"]["HitRate@2"] == 1.0


def test_eval_user_persona_reranker_exception_falls_back_to_als(monkeypatch):
    monkeypatch.setattr(fit_personas, "load_persona", lambda _path: object())
    monkeypatch.setattr(scorer, "optimize_weights", lambda **_kwargs: np.array([1.0, 0.0, 0.0, 0.0]))
    monkeypatch.setattr(
        scorer,
        "recommend_diverse",
        lambda **_kwargs: (_ for _ in ()).throw(RuntimeError("boom")),
    )

    rows = holdout_eval._eval_user(
        uid="u1",
        val_relevant={"s2"},
        test_relevant={"s3"},
        train_songs={"s1"},
        u_vec=np.array([0.1, 1.0], dtype=np.float32),
    )

    by_model = _rows_by_model(rows)
    assert by_model["persona_reranker"]["Recall@2"] == by_model["als"]["Recall@2"]
    assert by_model["persona_reranker"]["NDCG@2"] == by_model["als"]["NDCG@2"]
