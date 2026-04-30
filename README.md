# Music Discovery Engine

Popularity-free recommendation engine that learns musical similarity from collaborative listening behavior and models each listener as multiple taste personas.

Built as a research-grade Python package with a CLI. Core objective: improve discovery quality for users with diverse, non-mainstream taste without using popularity in ranking.

## Why this exists

Most production recommenders optimize engagement and amplify mainstream tracks. This project removes popularity from the ranking function entirely and instead combines:

1. ALS collaborative filtering over user–song interaction history.
2. Persona modeling with per-user Gaussian Mixture Models (GMMs).
3. A four-component weighted ranking function balancing fit, novelty, and familiarity.

## Pipeline overview

```
Million Song Dataset (MSD)
  └─ Taste Profile interactions  →  process  →  interactions.parquet
  └─ Track metadata (SQLite)     →  process  →  songs.parquet

interactions.parquet  →  train als  →  song_embeddings.parquet
                                    →  user_embeddings.parquet

song_embeddings.parquet + interactions.parquet  →  profile  →  models/personas/<user_id>/persona.pkl

song_embeddings.parquet + persona.pkl  →  recommend  →  ranked candidate list
```

Three evaluation experiments validate the design claims (see `music_discovery/evaluate/`).

## What changed from earlier approaches

The model previously described in this README (triplet-loss MLP, Last.fm + Kaggle Spotify CSV) has been replaced. The current approach:

| Aspect | Old | Current |
|--------|-----|---------|
| Embeddings | Triplet-loss MLP (12D features) | ALS collaborative filtering (128D) |
| Data | Last.fm + Kaggle Spotify CSV | MSD Taste Profile + track metadata |
| Scoring components | sonic_fit, emotional_fit, surprise | sonic_fit, novelty, persona_specificity, familiarity |
| Artifact name | `track_embeddings.parquet` | `song_embeddings.parquet` |

## Repository map

```text
music_discovery/
  cli/
    main.py                   # Typer CLI entry point
  data/
    interactions.py           # MSD Taste Profile parsing + train/val/test split
    songs.py                  # Track metadata loading from SQLite
  models/
    als.py                    # ALS training wrapper (implicit-ALS)
    gmm.py                    # UserPersonaModel + BIC selection
    scorer.py                 # Four-component scorer + weight optimization
  train/
    train_als.py              # ALS fit loop + embedding export
    fit_personas.py           # Per-user GMM fit + persona persistence
  evaluate/
    holdout_eval.py           # Main holdout evaluation (coverage-adjusted metrics)
    weight_sensitivity.py     # Experiment 1: weight grid + ablation
    persona_validation.py     # Experiment 2: K vs NDCG + BIC curves
    recommendation_quality.py # Experiment 3: popularity baseline comparison

configs/default.yaml          # Primary config (all paths + hyperparameters)
notebooks/
  03_embedding_analysis.ipynb # ALS embedding quality + UMAP visualizations
  04_persona_analysis.ipynb   # GMM persona fleet analysis + case studies
  05_experiment_results.ipynb # Experiment results dashboard
tests/                        # Unit tests
files/PRD.md                  # Product requirements
files/PROPOSAL.md             # Method proposal + design rationale
```

## Data sources

Built around the [Million Song Dataset](http://millionsongdataset.com/):

1. **MSD Taste Profile** — 48M user–song play-count interactions, subsampled to 2,000 users.
2. **MSD track metadata** — `track_metadata.db` SQLite file for song/artist names.

Default paths (configure in `configs/default.yaml`):

```yaml
data:
  msd_taste_profile_dir: data/raw/msd/taste_profile/
  msd_metadata_db: data/raw/msd/track_metadata.db
  processed_dir: data/processed/
```

User subsampling is deterministic (`setseed(42)` in DuckDB) — reruns produce the same 2,000-user subset.

## Model details

### ALS embeddings

- Library: `implicit` (CPU-optimized ALS).
- Embedding dimension: 128.
- Trained on binarized play-count signal (played ≥ 1 → positive implicit feedback).
- Exports two artifacts: `song_embeddings.parquet` (per-song 128D vectors) and `user_embeddings.parquet` (per-user 128D vectors).

### Persona model

- Per-user GMM fit on the user's training-history song embeddings.
- K selected by BIC over range 2–8 (default; configurable).
- Diagonal covariance for tractability on sparse histories.
- Persona files serialized to `models/personas/<user_id>/persona.pkl`.

### Scoring function

Candidate score is a weighted sum of four normalized components:

| Component | Default weight | Description |
|-----------|---------------|-------------|
| `sonic_fit` | 0.45 | Cosine similarity to persona centroids |
| `novelty` | 0.30 | Distance from listened-to embeddings + artist diversity |
| `persona_specificity` | 0.15 | Log-likelihood under the user's GMM |
| `familiarity` | 0.10 | Soft-match to known artists in history |

Weights are optimized per user on the val split when val signal is available; default weights used as fallback.

Popularity is **never** a scoring input.

## Holdout results

Evaluation over 500 held-out users, test split, NDCG@10:

| System | Recall_adj@10 | NDCG_adj@10 | HR_adj@10 |
|--------|--------------|-------------|----------|
| Popularity baseline | 0.0206 | 0.0241 | 0.0847 |
| ALS (no persona) | 0.0477 | 0.0521 | 0.1603 |
| ALS + Persona scorer | 0.0417 | 0.0461 | 0.1512 |

`_adj` = coverage-adjusted: denominator is `relevant ∩ song_catalog` to avoid penalizing missing catalog entries (~33% of held-out songs have no embedding).

ALS achieves **+131% lift** over popularity on Recall_adj@10.

## Quickstart

### 1. Environment setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pip install -e .
```

### 2. Prepare raw data

Download and place MSD files:

- Taste Profile interactions → `data/raw/msd/taste_profile/`
- Track metadata SQLite → `data/raw/msd/track_metadata.db`

Adjust paths in `configs/default.yaml` if needed.

### 3. Run full pipeline

```bash
# Parse + split interactions, extract song metadata
music-discovery process --config configs/default.yaml

# Train ALS and export 128D embeddings
music-discovery train --config configs/default.yaml

# Fit GMM personas for all users
music-discovery profile --config configs/default.yaml

# Remove stale persona dirs from previous runs (optional)
music-discovery profile --config configs/default.yaml --clean-stale
```

### 4. Generate recommendations

```bash
music-discovery recommend \
  --user-id <user_id> \
  --n 20 \
  --config configs/default.yaml
```

### 5. Run holdout evaluation

```bash
music-discovery evaluate \
  --config configs/default.yaml \
  --output-dir results/holdout_top500
```

## Evaluation experiments

```bash
# Experiment 1: scoring weight sensitivity + ablation
music-discovery experiment weight-sensitivity \
  --interactions data/processed/interactions.parquet \
  --embeddings models/embeddings/song_embeddings.parquet \
  --personas-dir models/personas \
  --output-dir results/exp1

# Experiment 2: persona count (K) vs NDCG + BIC curves
music-discovery experiment persona-validation \
  --interactions data/processed/interactions.parquet \
  --embeddings models/embeddings/song_embeddings.parquet \
  --personas-dir models/personas \
  --output-dir results/exp2

# Experiment 3: recommendation quality vs popularity baseline
music-discovery experiment recommendation-quality \
  --interactions data/processed/interactions.parquet \
  --embeddings models/embeddings/song_embeddings.parquet \
  --personas-dir models/personas \
  --output-dir results/exp3
```

## Expected artifacts

After a successful end-to-end run:

```
data/processed/
  interactions.parquet          # user–song splits (train/val/test)
  songs.parquet                 # song + artist metadata

models/embeddings/
  song_embeddings.parquet       # 128D ALS song vectors
  user_embeddings.parquet       # 128D ALS user vectors
  als_model.npz                 # serialized ALS model

models/personas/
  <user_id>/
    persona.pkl                 # UserPersonaModel (GMM)

results/
  holdout_top500/               # per-user metrics CSV + summary
  exp1/                         # heatmap_weight_sensitivity.png + bar_ablation.png
  exp2/                         # line_ndcg_vs_k.png + bic_curves.png + umap_personas.png
  exp3/                         # popularity comparison charts
```

## Evaluation metric notes

The holdout pipeline reports both raw and coverage-adjusted metrics:

- **Raw** (`Recall@k`, `NDCG@k`, `HR@k`): denominator = all held-out relevant songs.
- **Adjusted** (`Recall_adj@k`, `NDCG_adj@k`, `HR_adj@k`): denominator = relevant songs that exist in the embedding catalog.

Use adjusted metrics when comparing model quality. Raw metrics reflect catalog gaps, not ranking ability.

The pipeline also reports `fallback_rate` (fraction of users where ALS user vector was missing) and `test_coverage` (fraction of held-out relevant songs covered by the catalog).

## Testing

```bash
pytest -q
```

Tests cover:

1. Scorer component behavior and weight normalization.
2. GMM persona fitting and BIC selection.
3. ALS training output shape.
4. Holdout metric correctness (raw vs adjusted).

## Contributor guide

Start with `files/PRD.md` and `files/PROPOSAL.md` for intent and evaluation goals.

### Recommended workflow

1. Read `files/PRD.md` and `files/PROPOSAL.md`.
2. Run `process → train → profile` once before changing model logic.
3. Inspect `results/holdout_top500/` to establish a baseline.
4. Keep changes config-driven; no popularity term in any ranking path.
5. Add tests for every behavioral change.

### High-impact areas

1. ALS hyperparameter tuning (factors, regularization, iterations).
2. Persona robustness for sparse users.
3. Weight optimization stability and per-user diagnostics.
4. Confidence intervals and statistical tests on holdout metrics.
5. Artifact versioning and reproducibility.

## Design constraints

- **No popularity in ranking**: popularity is never a scoring component, not even as a tiebreaker.
- **Offline-first**: no live API calls required; all artifacts are local parquet files.
- **Artifact lineage**: process → train → profile → evaluate must run in order; the overlap guard in `holdout_eval.py` raises `RuntimeError` if artifacts are mismatched.
- **Deterministic sampling**: DuckDB `setseed(42)` ensures the same 2,000 users are selected on every run.

## References

- Product requirements: `files/PRD.md`
- Method proposal: `files/PROPOSAL.md`
- Embedding analysis: `notebooks/03_embedding_analysis.ipynb`
- Persona analysis: `notebooks/04_persona_analysis.ipynb`
- Experiment results: `notebooks/05_experiment_results.ipynb`
