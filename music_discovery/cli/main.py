from __future__ import annotations
import json
import pickle
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Optional

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
            raise RuntimeError(
                "PyYAML is required to read config files. Install pyyaml or run without a config file."
            ) from exc
        with open(config_path) as f:
            return yaml.safe_load(f)
    return {}


# ---------------------------------------------------------------------------
# train
# ---------------------------------------------------------------------------

@app.command()
def train(
    data_dir: str = typer.Option("data/processed", help="Directory with track_features.parquet and triplets.parquet"),
    output_dir: str = typer.Option("models/embeddings", help="Where to save embedding_model.pt"),
    config: Optional[str] = typer.Option(None, help="Path to default.yaml"),
    epochs: Optional[int] = typer.Option(None, help="Override epochs from config"),
    device: Optional[str] = typer.Option(None, help="cpu | cuda | mps"),
):
    """Train the metric learning MLP on Last.fm triplets."""
    from music_discovery.train.train_embeddings import train_embeddings, embed_all_tracks

    data = Path(data_dir)
    # Resolve config path so train_embeddings always reads the file
    resolved_config = config or str(Path(__file__).parents[2] / "configs" / "default.yaml")
    cfg = _load_config(resolved_config)
    if epochs is not None:
        cfg.setdefault("training", {})["epochs"] = epochs

    typer.echo(f"[train] Loading data from {data}")
    model = train_embeddings(
        track_features_path=data / "track_features.parquet",
        triplets_path=data / "triplets.parquet",
        output_dir=output_dir,
        config_path=resolved_config,
        device=device,
    )

    typer.echo("[train] Embedding all tracks...")
    embed_all_tracks(
        model=model,
        track_features_path=data / "track_features.parquet",
        output_path=Path(output_dir) / "track_embeddings.parquet",
        device=device,
    )
    typer.echo(f"[train] Done. Model + embeddings saved to {output_dir}")


# ---------------------------------------------------------------------------
# process  (data pipeline)
# ---------------------------------------------------------------------------

@app.command()
def process(
    config: Optional[str] = typer.Option(None, help="Path to default.yaml"),
):
    """Run the full data pipeline: Kaggle + Last.fm → processed parquet files."""
    from music_discovery.data.kaggle_spotify import load_kaggle_tracks
    from music_discovery.data.lastfm import load_lastfm_events
    from music_discovery.data.triplets import build_track_features, build_triplets, save_processed

    cfg = _load_config(config)
    data_cfg = cfg.get("data", {})
    feat_cfg = cfg.get("features", {})

    kaggle_path = data_cfg.get("kaggle_csv", "data/raw/spotify_tracks.csv")
    lastfm_path = data_cfg.get("lastfm_tsv", "data/raw/lastfm-dataset-1K/userid-timestamp-artid-artname-traid-traname.tsv")
    out_dir = data_cfg.get("processed_dir", "data/processed")
    gap = data_cfg.get("session_gap_minutes", 30)
    min_sess = data_cfg.get("min_session_length", 2)

    typer.echo("[process] Loading Kaggle tracks...")
    kaggle_df = load_kaggle_tracks(kaggle_path)

    typer.echo("[process] Engineering features...")
    track_features_df = build_track_features(kaggle_df, feat_cfg)

    typer.echo("[process] Loading Last.fm events...")
    lastfm_df = load_lastfm_events(lastfm_path)

    typer.echo("[process] Building triplets...")
    miss_log = Path(out_dir) / "unmatched_events.parquet"
    triplets_df, user_history_df = build_triplets(
        lastfm_df, track_features_df,
        gap_minutes=gap,
        min_session_length=min_sess,
        miss_log_path=miss_log,
    )

    save_processed(track_features_df, triplets_df, user_history_df, out_dir)
    typer.echo(f"[process] Done. Processed data saved to {out_dir}")


# ---------------------------------------------------------------------------
# profile
# ---------------------------------------------------------------------------

@app.command()
def profile(
    user_history: str = typer.Option(..., help="Path to user_history.parquet"),
    track_embeddings: str = typer.Option(..., help="Path to track_embeddings.parquet"),
    output_dir: str = typer.Option("models/personas", help="Directory to save persona models"),
    config: Optional[str] = typer.Option(None, help="Path to default.yaml"),
):
    """Fit Gaussian persona models for all users."""
    from music_discovery.train.fit_personas import fit_all_personas

    cfg = _load_config(config)
    gmm_cfg = cfg.get("gmm", {})

    typer.echo("[profile] Fitting user personas...")
    fit_all_personas(
        user_history_path=user_history,
        track_embeddings_path=track_embeddings,
        output_dir=output_dir,
        k_min=gmm_cfg.get("k_min", 2),
        k_max=gmm_cfg.get("k_max", 10),
        covariance_type=gmm_cfg.get("covariance_type", "full"),
    )
    typer.echo(f"[profile] Done. Persona models saved to {output_dir}")


# ---------------------------------------------------------------------------
# recommend
# ---------------------------------------------------------------------------

@app.command()
def recommend(
    user_model: str = typer.Option(..., help="Directory containing persona.pkl"),
    candidate_pool: str = typer.Option(..., help="Path to track_embeddings.parquet"),
    user_history: str = typer.Option(..., help="Path to user_history.parquet"),
    user_id: str = typer.Option(..., help="User ID to generate recommendations for"),
    n: int = typer.Option(20, help="Number of recommendations"),
    output: str = typer.Option("recommendations.json", help="Output JSON path"),
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
    persona = load_persona(user_model)

    emb_df = pd.read_parquet(candidate_pool)
    emb_cols = [c for c in emb_df.columns if c.startswith("emb_")]
    candidate_emb = emb_df[emb_cols].to_numpy(dtype=np.float32)
    candidate_artists = emb_df["artist_norm"].to_numpy()

    history_df = pd.read_parquet(user_history)
    user_train = history_df[(history_df["user_id"] == user_id) & (history_df["split"] == "train")]
    history_indices = user_train["track_idx"].to_numpy()
    valid_idx = history_indices[history_indices < len(candidate_emb)]
    user_history_emb = candidate_emb[valid_idx]
    user_history_artists = candidate_artists[valid_idx]

    # Try to optimise weights on discovery eval rows (fallback to defaults)
    eval_rows = history_df[(history_df["user_id"] == user_id) & (history_df["split"] == "eval")]
    if "is_discovery" in eval_rows.columns:
        eval_rows = eval_rows[eval_rows["is_discovery"]]
    held_out_raw = eval_rows["track_idx"].to_numpy()
    held_out = np.unique(held_out_raw[(held_out_raw >= 0) & (held_out_raw < len(candidate_emb))])
    weights = optimize_weights(
        candidate_emb, candidate_artists, persona,
        user_history_emb, user_history_artists, held_out,
    )

    typer.echo(f"[recommend] Running diverse pipeline (weights={np.round(weights, 3)})...")
    top_n_indices = recommend_diverse(
        persona=persona,
        candidate_emb=candidate_emb,
        candidate_artists=candidate_artists,
        history_emb=user_history_emb,
        history_artists=user_history_artists,
        history_indices=valid_idx,
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
        results.append({
            "rank": rank,
            "artist": emb_df["artist_norm"].iloc[idx],
            "track": emb_df["track_norm"].iloc[idx],
        })

    Path(output).parent.mkdir(parents=True, exist_ok=True)
    with open(output, "w") as f:
        json.dump({"user_id": user_id, "recommendations": results}, f, indent=2)

    typer.echo(f"[recommend] Top {n} recommendations saved to {output}")
    for r in results[:5]:
        typer.echo(f"  #{r['rank']:2d}  {r['artist']} — {r['track']}")



# ---------------------------------------------------------------------------
# Analyze-coverage
# ---------------------------------------------------------------------------
@app.command("analyze-coverage")
def analyze_coverage(
    config: Optional[str] = typer.Option(None, help="Path to default.yaml"),
    processed_dir: str = typer.Option("data/processed", help="Directory with processed parquet files"),
    output: str = typer.Option("data/processed/coverage_report.json", help="Output path for coverage report"),
    near_miss_threshold: int = typer.Option(70, help="Minimum fuzzy score to consider a near-miss")
):
    """ Analyze unmmatched events and surface candidate normalization rules"""
    from music_discovery.data.kaggle_spotify import load_kaggle_tracks
    from music_discovery.data.coverage_analysis import analyze_coverage as run_analysis
    
    cfg = _load_config(config)
    data_cfg = cfg.get("data", {})
    kaggle_path = data_cfg.get("kaggle_csv", "data/raw/spotify_tracks.csv")
    unmatched_path = Path(processed_dir) / "unmatched_events.parquet"
    
    if not unmatched_path.exists():
        typer.echo(f"[coverage] No unmatched_events.parquet found at {unmatched_path}. Run 'process' first")
        raise typer.Exit(1)
    
    typer.echo("[coverage] Loading Kaggle catalog...")
    catalog_df = load_kaggle_tracks(kaggle_path)
    
    typer.echo("[coverage] Analyzing unmatched events...")
    run_analysis(
        unmatched_events_path=unmatched_path,
        catalog_df=catalog_df,
        output_path=output,
        near_miss_threshold=near_miss_threshold,
    )
    
# ---------------------------------------------------------------------------
# evaluate (sub-commands)
# ---------------------------------------------------------------------------

evaluate_app = typer.Typer(help="Run evaluation experiments.")
app.add_typer(evaluate_app, name="evaluate")


@evaluate_app.command("weight-sensitivity")
def evaluate_weight_sensitivity(
    processed_dir: str = typer.Option("data/processed", help="Directory with processed parquet files"),
    embeddings_path: str = typer.Option("models/embeddings/track_embeddings.parquet"),
    personas_dir: str = typer.Option("models/personas"),
    output_dir: str = typer.Option("results/exp1"),
    n_users: int = typer.Option(10),
):
    """Experiment 1: scoring weight sensitivity (NDCG heatmap + ablation)."""
    from music_discovery.evaluate.weight_sensitivity import run as run_exp1
    run_exp1(processed_dir, embeddings_path, personas_dir, output_dir, n_users)


@evaluate_app.command("persona-validation")
def evaluate_persona_validation(
    processed_dir: str = typer.Option("data/processed"),
    embeddings_path: str = typer.Option("models/embeddings/track_embeddings.parquet"),
    personas_dir: str = typer.Option("models/personas"),
    output_dir: str = typer.Option("results/exp2"),
    n_users: int = typer.Option(10),
):
    """Experiment 2: BIC persona count validation (K vs NDCG + UMAP)."""
    from music_discovery.evaluate.persona_validation import run as run_exp2
    run_exp2(processed_dir, embeddings_path, personas_dir, output_dir, n_users)


@evaluate_app.command("popularity-bias")
def evaluate_popularity_bias(
    processed_dir: str = typer.Option("data/processed"),
    embeddings_path: str = typer.Option("models/embeddings/track_embeddings.parquet"),
    personas_dir: str = typer.Option("models/personas"),
    output_dir: str = typer.Option("results/exp3"),
    n_users: int = typer.Option(10),
):
    """Experiment 3: popularity bias comparison vs baseline."""
    from music_discovery.evaluate.spotify_overlap import run as run_exp3
    run_exp3(processed_dir, embeddings_path, personas_dir, output_dir, n_users)


@evaluate_app.command("recommendation-quality")
def evaluate_recommendation_quality(
    processed_dir: str = typer.Option("data/processed"),
    embeddings_path: str = typer.Option("models/embeddings/track_embeddings.parquet"),
    personas_dir: str = typer.Option("models/personas"),
    output_dir: str = typer.Option("results/rec_quality"),
    n_users: int = typer.Option(20),
    k: int = typer.Option(10, help="Cut-off for @K metrics"),
):
    """Recommendation quality: Replay@K, Discovery@K, NDCG@K, artist repetition, persona coverage."""
    from music_discovery.evaluate.recommendation_quality import run as run_rq
    run_rq(processed_dir, embeddings_path, personas_dir, output_dir, n_users, k)


@evaluate_app.command("matching-quality")
def evaluate_matching_quality(
    kaggle_csv: str = typer.Option("data/raw/spotify_tracks.csv"),
    lastfm_tsv: str = typer.Option("data/raw/lastfm-dataset-1K/userid-timestamp-artid-artname-traid-traname.tsv"),
    benchmark_csv: str = typer.Option("data/eval/matching_benchmark.csv"),
    output_dir: str = typer.Option("results/matching_quality"),
    n_records: int = typer.Option(200, help="Benchmark size; must be between 100 and 300."),
    seed: int = typer.Option(42),
    predictions_csv: Optional[str] = typer.Option(
        None,
        help="Optional CSV with benchmark_id,predicted_track_idx|predicted_artist_norm,predicted_track_norm,score,accepted",
    ),
    regenerate: bool = typer.Option(False, help="Rebuild the benchmark CSV before scoring."),
):
    """Generate/score a 100-300 row benchmark for matching Last.fm queries to Spotify tracks."""
    from music_discovery.evaluate.matching_quality import run as run_mq

    run_mq(
        kaggle_csv=kaggle_csv,
        lastfm_tsv=lastfm_tsv,
        benchmark_csv=benchmark_csv,
        output_dir=output_dir,
        n_records=n_records,
        seed=seed,
        predictions_csv=predictions_csv,
        regenerate=regenerate,
    )


if __name__ == "__main__":
    app()
