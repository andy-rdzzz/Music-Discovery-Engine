# Music Discovery Engine

Popularity-free recommendation engine that learns musical similarity from listening behavior and models each listener as multiple taste personas.

This repository is built as a research-grade Python package with a CLI. The core objective is to improve discovery quality for users with diverse, non-mainstream taste without using popularity in ranking.

## Why this exists

Most production recommenders optimize engagement and tend to amplify mainstream tracks. This project intentionally removes popularity from the ranking function and instead combines:

1. Metric learning over session-level co-listening.
2. Persona modeling with Gaussian Mixture Models (GMMs).
3. A weighted ranking function balancing fit and novelty.

## What is being built

The system has three major stages:

1. Data pipeline
	 1. Load pre-computed Spotify-style audio features.
	 2. Parse Last.fm listening events.
	 3. Segment listening sessions and build triplets.
2. Representation + persona modeling
	 1. Train an MLP to embed tracks into a similarity space.
	 2. Fit user-specific GMM personas with BIC-selected K.
3. Recommendation + evaluation
	 1. Score candidates with sonic fit, emotional fit, and surprise.
	 2. Run three experiments to validate design claims.

## Project status

Implemented today:

1. End-to-end CLI pipeline: process -> train -> profile -> recommend.
2. Experiment runners for:
	 1. Weight sensitivity.
	 2. Persona validation.
	 3. Popularity-bias comparison.
3. Unit tests for features, MLP, GMM, and scorer behavior.

## Repository map

```text
music_discovery/
	cli/
		main.py                 # Typer CLI commands
	data/
		kaggle_spotify.py       # Feature CSV loading + validation
		lastfm.py               # Last.fm event parsing
		features.py             # 12D feature engineering
		triplets.py             # Session-based triplet creation
	models/
		mlp.py                  # Embedding model
		triplet.py              # Dataset + semi-hard negative mining
		gmm.py                  # Persona model wrappers
		scorer.py               # Scoring + weight optimization
	train/
		train_embeddings.py     # Metric learning loop + embedding export
		fit_personas.py         # Per-user persona fitting + persistence
	evaluate/
		weight_sensitivity.py   # Experiment 1
		persona_validation.py   # Experiment 2
		spotify_overlap.py      # Experiment 3 (popularity baseline comparison)

configs/default.yaml        # Primary config
tests/                      # Unit tests
files/PRD.md                # Product requirements
files/PROPOSAL.md           # Method proposal
```

## Data sources and constraints

This project is designed around public datasets with pre-computed features.

1. Kaggle Spotify Tracks CSV for track-level features.
2. Last.fm 1K users dataset for timestamped listening events.

Important context:

1. Live Spotify audio feature access is no longer reliable for new apps.
2. The pipeline assumes local files under paths configured in configs/default.yaml.

Default paths:

1. data/raw/spotify_tracks.csv
2. data/raw/lastfm-dataset-1K/userid-timestamp-artid-artname-traid-traname.tsv
3. data/processed/

## Feature schema

Input starts with 9 normalized Spotify-style attributes and adds 3 engineered interaction features:

1. arousal
2. chill_factor
3. vocal_presence

Final vector: 12D, L2-normalized before model input.

## Model and ranking overview

### Embedding model

1. MLP trained with triplet margin loss.
2. Online semi-hard negatives mined in-batch.
3. Uniformity regularization term to reduce representation collapse.

### Persona model

1. Per-user GMM fit on training-history embeddings.
2. K selected by BIC in configured range.
3. Fallback behavior for low-sample users handled by model logic.

### Scoring model

Candidate score is a weighted combination of:

1. sonic_fit
2. emotional_fit
3. surprise

Default scoring weights are loaded from config and can be optimized per user when held-out signal is available.

## Quickstart

### 1. Environment setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pip install -e .
```

### 2. Prepare raw data

1. Place Kaggle CSV at data/raw/spotify_tracks.csv.
2. Place Last.fm TSV at data/raw/lastfm-dataset-1K/userid-timestamp-artid-artname-traid-traname.tsv.

Adjust paths in configs/default.yaml if needed.

### 3. Run full pipeline

```bash
# Build processed parquet artifacts
music-discovery process --config configs/default.yaml

# Train embedding model and export track embeddings
music-discovery train --data-dir data/processed --output-dir models/embeddings --config configs/default.yaml

# Fit personas for all users
music-discovery profile \
	--user-history data/processed/user_history.parquet \
	--track-embeddings models/embeddings/track_embeddings.parquet \
	--output-dir models/personas \
	--config configs/default.yaml
```

### 4. Generate recommendations

```bash
music-discovery recommend \
	--user-model models/personas/<user_id> \
	--candidate-pool models/embeddings/track_embeddings.parquet \
	--user-history data/processed/user_history.parquet \
	--user-id <user_id> \
	--n 20 \
	--output results/recommendations_<user_id>.json
```

## Evaluation experiments

```bash
# Experiment 1: weight sensitivity
music-discovery evaluate weight-sensitivity \
	--processed-dir data/processed \
	--embeddings-path models/embeddings/track_embeddings.parquet \
	--personas-dir models/personas \
	--output-dir results/exp1

# Experiment 2: persona count validation
music-discovery evaluate persona-validation \
	--processed-dir data/processed \
	--embeddings-path models/embeddings/track_embeddings.parquet \
	--personas-dir models/personas \
	--output-dir results/exp2

# Experiment 3: popularity bias comparison
music-discovery evaluate popularity-bias \
	--processed-dir data/processed \
	--embeddings-path models/embeddings/track_embeddings.parquet \
	--personas-dir models/personas \
	--output-dir results/exp3
```

## Expected artifacts

After a successful run, contributors should see:

1. data/processed/track_features.parquet
2. data/processed/triplets.parquet
3. data/processed/user_history.parquet
4. models/embeddings/embedding_model.pt
5. models/embeddings/track_embeddings.parquet
6. models/embeddings/training_history.csv
7. models/personas/<user_id>/persona.pkl
8. results/exp*/ charts and csv summaries

## Testing

```bash
pytest -q
```

Current tests focus on:

1. Feature correctness and normalization invariants.
2. MLP output shape and L2 constraints.
3. GMM persona behavior.
4. Scoring function consistency.

## Contributor guide

### Recommended workflow

1. Start by reading files/PRD.md and files/PROPOSAL.md for intent and evaluation goals.
2. Run the baseline pipeline once before changing model logic.
3. Keep changes config-driven where possible.
4. Add or update tests in tests/ for every behavioral change.
5. Preserve the core design principle: no popularity term in ranking.

### High-impact contribution areas

1. Better triplet mining or curriculum strategies.
2. Persona robustness for sparse users and long-tail histories.
3. Weight optimization stability and diagnostics.
4. More rigorous offline metrics and confidence intervals.
5. Reproducibility improvements (seed control, deterministic runs, artifact versioning).

### Code quality expectations

1. Python 3.10+.
2. Type hints where practical.
3. Clear module boundaries between data, models, train, and evaluate.
4. Minimal coupling between CLI and model internals.

## Research notes and assumptions

1. This is not a real-time serving system.
2. It is intentionally offline-first and experiment-oriented.
3. Candidate generation currently assumes full-pool scoring from track embeddings.
4. Evaluation targets ranking quality, diversity, and popularity-bias behavior.

## References in this repo

1. Product requirements: files/PRD.md
2. Method proposal: files/PROPOSAL.md

If you are joining as a contributor engineer, start with the Quickstart, run process/train/profile once, and then inspect experiment outputs before proposing model changes.