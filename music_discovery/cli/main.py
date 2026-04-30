from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import typer

app = typer.Typer(help="Music Discovery Engine — popularity-free recommendations.")


def _load_config(config_path: str | None) -> dict:
    if config_path is None:
        default = Path(__file__).parents[2] / "configs" / "default.yaml"
        config_path = str(default) if default.exists() else None
    if config_path and Path(config_path).exists():
        try:
            import yaml
        except ImportError as exc:
            raise RuntimeError("PyYAML required. Run: pip install pyyaml") from exc
        with open(config_path) as f:
            return yaml.safe_load(f)
    return {}


# ---------------------------------------------------------------------------
# process  — MSD data pipeline
# ---------------------------------------------------------------------------

@app.command()
def process(
    config: Optional[str] = typer.Option(None, help="Path to default.yaml"),
    skip_side_info: bool = typer.Option(False, help="Skip tags/similars/artist-graph (faster dev iteration)"),
):
    """Parse MSD Taste Profile + metadata → canonical processed parquets."""
    from music_discovery.data.interactions import build

    cfg = _load_config(config)
    d = cfg.get("data", {})

    build(
        taste_profile_path=d.get("taste_profile", "data/raw/train_triplets.txt"),
        track_metadata_db=d.get("track_metadata_db", "data/raw/track_metadata.db"),
        sid_mismatches_path=d.get("sid_mismatches", "data/raw/sid_mismatches.txt"),
        lastfm_tags_db=d.get("lastfm_tags_db", "data/raw/lastfm_tags.db"),
        lastfm_similars_db=d.get("lastfm_similars_db", "data/raw/lastfm_similars.db"),
        artist_similarity_db=d.get("artist_similarity_db", "data/raw/artist_similarity.db"),
        processed_dir=d.get("processed_dir", "data/processed"),
        min_play_count=d.get("min_play_count", 1),
        sample_n_users=d.get("sample_n_users"),
        holdout_k=d.get("holdout_k", 10),
        skip_side_info=skip_side_info or d.get("skip_side_info", False),
    )


# ---------------------------------------------------------------------------
# train  — ALS matrix factorization
# ---------------------------------------------------------------------------

@app.command()
def train(
    interactions: str = typer.Option("data/processed/interactions.parquet"),
    output_dir: str = typer.Option("models/embeddings"),
    config: Optional[str] = typer.Option(None, help="Path to default.yaml"),
):
    """Train ALS model on Taste Profile interactions → song + user embeddings."""
    from music_discovery.train.train_als import train_als

    cfg = _load_config(config)
    als_cfg = cfg.get("als", {})

    typer.echo("[train] Training ALS model...")
    train_als(
        interactions_path=interactions,
        output_dir=output_dir,
        factors=als_cfg.get("factors", 128),
        iterations=als_cfg.get("iterations", 20),
        regularization=als_cfg.get("regularization", 0.01),
        use_gpu=als_cfg.get("use_gpu", False),
    )
    typer.echo(f"[train] Done. Embeddings saved to {output_dir}")


# ---------------------------------------------------------------------------
# profile  — fit GMM personas per user
# ---------------------------------------------------------------------------

@app.command()
def profile(
    interactions: str = typer.Option("data/processed/interactions.parquet"),
    song_embeddings: str = typer.Option("models/embeddings/song_embeddings.parquet"),
    output_dir: str = typer.Option("models/personas"),
    n_jobs: int = typer.Option(0, help="Parallel workers (0 = auto)"),
    resume: bool = typer.Option(True, help="Skip users that already have persona.pkl"),
    n_init: int = typer.Option(0, help="GMM random restarts per k (0 = use config)"),
    config: Optional[str] = typer.Option(None, help="Path to default.yaml"),
):
    """Fit Gaussian persona models for all users over ALS song embeddings."""
    from music_discovery.train.fit_personas import fit_all_personas

    cfg = _load_config(config)
    gmm_cfg = cfg.get("gmm", {})

    typer.echo("[profile] Fitting user personas...")
    fit_all_personas(
        user_history_path=interactions,
        track_embeddings_path=song_embeddings,
        output_dir=output_dir,
        k_min=gmm_cfg.get("k_min", 2),
        k_max=gmm_cfg.get("k_max", 5),
        covariance_type=gmm_cfg.get("covariance_type", "diag"),
        n_init=n_init or gmm_cfg.get("n_init", 1),
        n_jobs=n_jobs or None,
        resume=resume,
    )
    typer.echo(f"[profile] Done. Persona models saved to {output_dir}")


# ---------------------------------------------------------------------------
# recommend
# ---------------------------------------------------------------------------

@app.command()
def recommend(
    user_id: str = typer.Option(..., help="User ID to generate recommendations for"),
    song_embeddings: str = typer.Option("models/embeddings/song_embeddings.parquet"),
    songs: str = typer.Option("data/processed/songs.parquet"),
    interactions: str = typer.Option("data/processed/interactions.parquet"),
    personas_dir: str = typer.Option("models/personas"),
    n: int = typer.Option(20, help="Number of recommendations"),
    output: str = typer.Option("recommendations.json"),
    config: Optional[str] = typer.Option(None, help="Path to default.yaml"),
):
    """Generate top-N recommendations for a user."""
    from music_discovery.train.fit_personas import load_persona
    from music_discovery.models.scorer import optimize_weights, recommend_diverse, DEFAULT_WEIGHTS

    cfg = _load_config(config)
    scoring_cfg = cfg.get("scoring", {})
    rec_cfg = cfg.get("recommend", {})
    default_w = np.array(scoring_cfg.get("default_weights", DEFAULT_WEIGHTS.tolist()))

    typer.echo(f"[recommend] Loading persona for {user_id}...")
    persona = load_persona(str(Path(personas_dir) / user_id))

    emb_df = pd.read_parquet(song_embeddings)
    songs_df = pd.read_parquet(songs)
    emb_cols = [c for c in emb_df.columns if c.startswith("emb_")]
    candidate_emb = emb_df[emb_cols].to_numpy(dtype=np.float32)

    # artist names aligned to embedding rows via song_id
    merged = emb_df[["song_id"]].merge(songs_df[["song_id", "artist_name"]], on="song_id", how="left")
    candidate_artists = merged["artist_name"].fillna("").to_numpy()

    idf = pd.read_parquet(interactions)
    user_train = idf[(idf["user_id"] == user_id) & (idf["split"] == "train")]
    song_to_idx = {sid: i for i, sid in enumerate(emb_df["song_id"])}
    history_indices = np.array(
        [song_to_idx[s] for s in user_train["song_id"] if s in song_to_idx], dtype=np.intp
    )
    user_history_emb = candidate_emb[history_indices]
    user_history_artists = candidate_artists[history_indices]

    user_val = idf[(idf["user_id"] == user_id) & (idf["split"] == "val")]
    held_out = np.array(
        [song_to_idx[s] for s in user_val["song_id"] if s in song_to_idx], dtype=np.intp
    )

    weights = optimize_weights(
        candidate_emb, candidate_artists, persona,
        user_history_emb, user_history_artists, held_out,
    ) if len(held_out) else default_w

    typer.echo(f"[recommend] Running diverse pipeline (weights={np.round(weights, 3)})...")
    top_n_indices = recommend_diverse(
        persona=persona,
        candidate_emb=candidate_emb,
        candidate_artists=candidate_artists,
        history_emb=user_history_emb,
        history_artists=user_history_artists,
        history_indices=history_indices,
        n_recs=n,
        weights=weights,
        tau=rec_cfg.get("persona_temp", 0.7),
        min_slots=rec_cfg.get("min_slots_per_persona", 1),
        max_fraction=rec_cfg.get("max_slots_fraction", 0.6),
        exploration_fraction=rec_cfg.get("exploration_fraction", 0.15),
        mmr_lambda=rec_cfg.get("mmr_lambda", 0.7),
        max_per_artist=rec_cfg.get("max_per_artist", 2),
    )

    results = []
    for rank, idx in enumerate(top_n_indices, 1):
        row = merged.iloc[idx]
        results.append({"rank": rank, "song_id": emb_df["song_id"].iloc[idx], "artist": row["artist_name"]})

    Path(output).parent.mkdir(parents=True, exist_ok=True)
    with open(output, "w") as f:
        json.dump({"user_id": user_id, "recommendations": results}, f, indent=2)

    typer.echo(f"[recommend] Top {n} saved to {output}")
    for r in results[:5]:
        typer.echo(f"  #{r['rank']:2d}  {r['artist']} — {r['song_id']}")


# ---------------------------------------------------------------------------
# evaluate (sub-commands)
# ---------------------------------------------------------------------------

evaluate_app = typer.Typer(help="Run evaluation experiments.")
app.add_typer(evaluate_app, name="evaluate")


@evaluate_app.command("holdout")
def evaluate_holdout(
    interactions: str = typer.Option("data/processed/interactions.parquet"),
    song_embeddings: str = typer.Option("models/embeddings/song_embeddings.parquet"),
    personas_dir: str = typer.Option("models/personas"),
    output_dir: str = typer.Option("results/holdout"),
    k: int = typer.Option(10, help="Cut-off for @K metrics"),
    n_users: int = typer.Option(0, help="Limit evaluation to N users (0 = all)"),
    n_jobs: int = typer.Option(0, help="Parallel workers (0 = auto)"),
):
    """Leave-k-out holdout evaluation: Recall/NDCG/HitRate + diversity metrics."""
    from music_discovery.evaluate.holdout_eval import run as run_holdout
    run_holdout(interactions, song_embeddings, personas_dir, output_dir, k, n_users or None, n_jobs or None)


@evaluate_app.command("weight-sensitivity")
def evaluate_weight_sensitivity(
    interactions: str = typer.Option("data/processed/interactions.parquet"),
    song_embeddings: str = typer.Option("models/embeddings/song_embeddings.parquet"),
    personas_dir: str = typer.Option("models/personas"),
    output_dir: str = typer.Option("results/exp1"),
    n_users: int = typer.Option(10),
):
    """Experiment 1: scoring weight sensitivity (NDCG heatmap + ablation)."""
    from music_discovery.evaluate.weight_sensitivity import run as run_exp1
    run_exp1(interactions, song_embeddings, personas_dir, output_dir, n_users)


@evaluate_app.command("persona-validation")
def evaluate_persona_validation(
    interactions: str = typer.Option("data/processed/interactions.parquet"),
    song_embeddings: str = typer.Option("models/embeddings/song_embeddings.parquet"),
    personas_dir: str = typer.Option("models/personas"),
    output_dir: str = typer.Option("results/exp2"),
    n_users: int = typer.Option(10),
):
    """Experiment 2: BIC persona count validation (K vs NDCG + UMAP)."""
    from music_discovery.evaluate.persona_validation import run as run_exp2
    run_exp2(interactions, song_embeddings, personas_dir, output_dir, n_users)


@evaluate_app.command("popularity-bias")
def evaluate_popularity_bias(
    interactions: str = typer.Option("data/processed/interactions.parquet"),
    song_embeddings: str = typer.Option("models/embeddings/song_embeddings.parquet"),
    personas_dir: str = typer.Option("models/personas"),
    output_dir: str = typer.Option("results/exp3"),
    n_users: int = typer.Option(10),
):
    """Experiment 3: popularity bias comparison vs baseline."""
    from music_discovery.evaluate.recommendation_quality import run as run_exp3
    run_exp3(interactions, song_embeddings, personas_dir, output_dir, n_users)


if __name__ == "__main__":
    app()
