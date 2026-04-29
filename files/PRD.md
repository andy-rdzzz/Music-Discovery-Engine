# Product Requirements Document — Music Discovery Engine

**Type:** Academic / Research  
**Stack:** Python · implicit · scikit-learn  
**Deliverable:** Python package (`music_discovery`) + CLI  
**Date:** 2026-04-29 (MSD pivot)

---

## 1. Problem Statement

Mainstream platforms (Spotify, Apple Music) optimize for engagement via popularity signals, creating feedback loops that over-serve trending content. Users with diverse, non-mainstream taste receive recommendations that overfit to recently-played genres, ignoring the full breadth of their musical identity. This project builds a discovery engine that learns from large-scale implicit listening behavior, models each user's taste as multiple simultaneous personas, and ranks candidates by fit + novelty + diversity — with no popularity signal in the ranking function.

---

## 2. Goals

| # | Goal |
|---|------|
| G1 | Learn robust song and user representations from large-scale implicit listening feedback (MSD Taste Profile) |
| G2 | Model each user as K Gaussian personas (BIC-selected) over learned item embeddings |
| G3 | Rank candidates by fit + novelty + diversity — zero popularity weight in the scoring function |
| G4 | Incorporate side information from Last.fm tags, track similarity, and artist similarity graphs |
| G5 | Ship as a clean Python package + CLI |
| G6 | Run reproducible offline evaluations against strong baselines (popularity, item-item, ALS) |

## 3. Non-Goals

- Real-time serving / production latency requirements
- Web UI or mobile interface
- Timestamp / session modeling (Taste Profile has no timestamps; not a requirement)
- Raw audio signal processing (no waveforms; interaction-learned embeddings only)
- Live API dependence (no Spotify OAuth, no live Last.fm calls)
- Streaming history ingestion from live APIs

---

## 4. Data Strategy

### 4.1 Core Dataset: MSD Taste Profile

**Primary source:** Million Song Dataset Taste Profile subset  
**File:** `data/raw/train_triplets.txt`  
**Format:** tab-separated `(user_id, song_id, play_count)`, ~48M rows, 2.9 GB  
**Content:** Implicit listening behavior — how many times each user played each song

This is the interaction signal. ALS learns user and song embeddings directly from these counts, weighted by `log1p(play_count)`.

**Scale config:** `data.sample_n_users` (null = full dataset) and `data.min_play_count` allow dev-time subsampling without code changes.

### 4.2 Item Bridge: track_metadata.db

**File:** `data/raw/track_metadata.db` (SQLite)  
**Purpose:** Map `song_id` (Taste Profile key) → `track_id`, `artist_name`, `title`  
**Critical:** This is the join layer between Taste Profile song IDs and all other MSD assets.

### 4.3 Data Quality Filter: sid_mismatches.txt

**File:** `data/raw/sid_mismatches.txt`  
**Purpose:** Set of `song_id` values with known bad mappings in the Taste Profile  
**Applied:** Before any join — mismatched song IDs are excluded from interactions and songs tables

### 4.4 Side Information

| File | Key | Content |
|------|-----|---------|
| `lastfm_tags.db` | `track_id` | User-assigned semantic tags per track |
| `lastfm_similars.db` | `track_id` | Track similarity scores |
| `artist_similarity.db` | `artist_id` | Artist similarity graph |
| `msd_summary_file.h5` | `song_id` | Optional metadata (loudness, tempo, key, etc.) |

**ID join note:** `lastfm_tags.db` and `lastfm_similars.db` are keyed by `track_id`, not `song_id`. The `track_metadata.db` bridge (song_id → track_id) must be applied before joining side info. Join coverage is validated after mismatch filtering — log warning if < 50% of songs have tags or similars.

### 4.5 Data Pipeline Architecture

```
[train_triplets.txt]          [track_metadata.db]   [sid_mismatches.txt]
         |                           |                       |
         v                           v                       v
[taste_profile.py]            [item_bridge.py]  ←  filter bad song_ids
 Parse (user, song, count)     song_id → track_id,
 log1p weight                  artist_name, title
 sample_n_users / min_plays
         |                           |
         +───────────────────────────+
                        |
                        v
              [interactions.py]
               merge + random holdout split (holdout_k per user)
                        |
         ┌──────────────┼──────────────┐
         v              v              v
interactions.parquet  songs.parquet  (side info via side_info.py)
                                     song_tags.parquet
                                     song_similars.parquet
```

### 4.6 Train/Val/Test Split

Taste Profile has no timestamps, so temporal ordering is meaningless.

**Strategy:** Per-user random holdout  
- Sample `holdout_k` (default 10) interactions per user → **val**  
- Sample another `holdout_k` interactions per user → **test**  
- Remainder → **train**  
- Stored in `interactions.parquet` as a `split` column: `"train"` / `"val"` / `"test"`

---

## 5. Representation Strategy

### 5.1 Primary: Implicit-Feedback Learned Embeddings

**Model:** `implicit.als.AlternatingLeastSquares`  
**Input:** Sparse user×song matrix with `log1p(play_count)` weights  
**Output:** `factors`-dimensional embedding vectors for every song and user  

This replaces the MLP + triplet loss approach. ALS is the correct native fit for implicit count data at MSD scale.

### 5.2 Optional: Side-Information Enrichment (Phase 4)

| Source | Use |
|--------|-----|
| Last.fm tags | TF-IDF weighted tag features for semantic enrichment of songs |
| Track similarity | Item-item fallback and cold-start-ish reranking |
| Artist similarity | Exploration smoothing across similar artists |

Side info supplements learned embeddings — it does not replace them.

---

## 6. System Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                     music_discovery                          │
│                                                             │
│  ┌──────────────┐   ┌──────────────┐   ┌─────────────────┐ │
│  │    data/     │   │   models/    │   │     cli/        │ │
│  │              │   │              │   │                 │ │
│  │taste_profile │   │  gmm.py      │   │ recommend train │ │
│  │item_bridge   │   │  scorer.py   │   │ profile evaluate│ │
│  │side_info     │   │              │   │                 │ │
│  │interactions  │   │              │   │                 │ │
│  └──────────────┘   └──────────────┘   └─────────────────┘ │
│                                                             │
│  ┌──────────────┐   ┌──────────────────────────────────┐   │
│  │   train/     │   │         evaluate/                │   │
│  │              │   │                                  │   │
│  │ train_als    │   │ holdout_eval  weight_sensitivity  │   │
│  │ fit_personas │   │ persona_validation               │   │
│  └──────────────┘   └──────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────┘
```

**Pipeline stages:**
```
ingest + clean interactions
        ↓
build item metadata bridge
        ↓
split train / val / test (random holdout per user)
        ↓
train ALS baseline → song_embeddings + user_embeddings
        ↓
fit GMM personas over consumed song embeddings per user
        ↓
rerank with novelty / diversity / side info
        ↓
evaluate vs. baselines (popularity, item-item, ALS)
```

---

## 7. Component Specifications

### 7.1 ALS Recommender (Primary Model)

**Library:** `implicit.als.AlternatingLeastSquares`  
**Input:** Sparse user×song CSR matrix, values = `log1p(play_count)`  
**Output:** `factors`-dim embeddings for all users and songs

**Training config:**
```yaml
als:
  factors: 128
  iterations: 20
  regularization: 0.01
  use_gpu: false
```

**Popularity baseline:** Sort songs by `sum(play_count)` over training interactions. Used as the weakest baseline to beat.

### 7.2 Gaussian Mixture Model (User Personas)

**Goal:** Represent each user's taste as K soft-assignment Gaussian clusters in ALS embedding space.

**Process:**
1. Collect user's song embeddings from `song_embeddings.parquet`, indexed by the songs they played in train split — **deduplicated to unique songs**
2. Fit GMM with K components, full covariance
3. Select K via **BIC** — range K ∈ [2, 10], pick argmin(BIC)
4. Each persona k = (mean μ_k, covariance Σ_k, weight π_k)

**Implementation:** `sklearn.mixture.GaussianMixture` (unchanged from prior version)  
**Minimum songs per user:** 20 (below this: single Gaussian fallback)

### 7.3 Scoring Function

**Goal:** Rank candidate songs for a user beyond raw ALS score, emphasizing novelty and diversity.

**Score formula:**
```
score(song, user) = w₁ · sonic_fit(song, user)
                  + w₂ · novelty(song, user)
                  + w₃ · emotional_fit(song, user)
                  + w₄ · familiarity(song, user)

where w₁ + w₂ + w₃ + w₄ = 1
```

| Component | Weight | Formula |
|-----------|--------|---------|
| `sonic_fit` | 0.45 | Max cosine similarity to any persona centroid |
| `novelty` | 0.30 | 0.5 × artist_novelty + 0.5 × embedding distance from known songs |
| `emotional_fit` | 0.15 | Tag-based proxy (Phase 4) or GMM log-prob fallback |
| `familiarity` | 0.10 | GMM log-probability (Mahalanobis) |

**Diversity:** MMR reranking (`mmr_lambda=0.7`) + per-persona slot allocation + artist cap (max 2 per artist).

---

## 8. Package Structure

```
music-discovery-engine/
├── music_discovery/
│   ├── __init__.py
│   ├── data/
│   │   ├── __init__.py
│   │   ├── taste_profile.py    # Parse train_triplets.txt, log1p weights, sampling
│   │   ├── item_bridge.py      # SQLite join song_id → track_id, sid mismatch filter
│   │   ├── side_info.py        # Tags/similars/artist DBs via track_id bridge
│   │   └── interactions.py     # Orchestrate → canonical parquets + random holdout split
│   ├── models/
│   │   ├── __init__.py
│   │   ├── gmm.py              # UserPersonaModel (sklearn GMM wrapper)
│   │   └── scorer.py           # ScoringFunction + MMR + slot allocation
│   ├── train/
│   │   ├── __init__.py
│   │   ├── train_als.py        # implicit ALS training + popularity baseline
│   │   └── fit_personas.py     # GMM fitting per user on ALS song embeddings
│   ├── evaluate/
│   │   ├── __init__.py
│   │   ├── holdout_eval.py     # Leave-k-out harness + all metrics
│   │   ├── weight_sensitivity.py
│   │   └── persona_validation.py
│   └── cli/
│       ├── __init__.py
│       └── main.py
├── data/
│   ├── raw/                    # train_triplets.txt, track_metadata.db, *.db, *.h5
│   └── processed/              # interactions.parquet, songs.parquet, song_tags.parquet, ...
├── models/
│   ├── embeddings/             # song_embeddings.parquet, user_embeddings.parquet
│   └── personas/               # per-user GMM artifacts
├── results/                    # baseline_metrics.csv, experiment outputs
├── notebooks/
├── configs/
│   └── default.yaml
├── tests/
└── README.md
```

---

## 9. CLI Design

```bash
# Process raw MSD data → canonical parquets
music-discovery process --config configs/default.yaml

# Train ALS baseline
music-discovery train --config configs/default.yaml

# Fit personas for all users
music-discovery profile --config configs/default.yaml

# Get recommendations for a user
music-discovery recommend \
  --user-id <user_id> \
  --n 20 \
  --config configs/default.yaml

# Run experiments
music-discovery evaluate holdout --config configs/default.yaml
music-discovery evaluate weight-sensitivity --config configs/default.yaml
music-discovery evaluate persona-validation --config configs/default.yaml
```

---

## 10. Experiments

### Experiment 1 — Scoring Weight Sensitivity

**Question:** Does each scoring component (sonic_fit, novelty, emotional_fit) independently contribute?

**Method:**
- Ablation: run scoring with each single component alone (w=1.0, others=0)
- Grid sweep over weight combinations, measure NDCG@10

**Graphs:** heatmap (w_sonic vs w_novelty, color=NDCG@10) + grouped bar chart (single-component vs learned weights)

---

### Experiment 2 — Persona Count Validation

**Question:** Does BIC-selected K produce better-personalized, more diverse recommendations than fixed K?

**Method:**
- Classify test users by catalog diversity (genre entropy or unique artist count)
- Fit GMM at K ∈ {1,2,3,4,5} and BIC-selected K per user
- Measure NDCG@10 and intra-list diversity per K

**Graphs:** K vs NDCG line chart per user type + BIC curve for representative users + UMAP with persona ellipses

---

### Experiment 3 — Popularity Bias Comparison

**Question:** Do recommendations avoid mainstream content vs. a popularity baseline?

**Method:**
- Compare: popularity-ranked baseline vs. ALS baseline vs. persona reranker
- Measure: popularity percentile of recommended songs, long-tail exposure, intra-list diversity

**Graphs:** CDF of popularity percentile (our model vs baselines) + ILD box plot

---

## 11. Evaluation Metrics

### 11.1 Ranking Quality

| Metric | K | What It Proves |
|--------|---|---------------|
| Recall@K | 10, 20 | Fraction of held-out songs recovered |
| NDCG@K | 10, 20 | Ranking quality (position-weighted) |
| HitRate@K | 10, 20 | Did at least one held-out song appear in top K |
| MAP@K | 10 | Mean average precision |

### 11.2 Discovery & Diversity

| Metric | What It Measures |
|--------|-----------------|
| Catalog coverage | % of song catalog recommended at least once across all users |
| Artist coverage | % unique artists across recommendations |
| Intra-list diversity (ILD) | Mean pairwise cosine distance within a user's recommendation list |
| Long-tail exposure | % recommended songs below 80th popularity percentile |
| Novelty / popularity-bias | Mean popularity rank of recommended songs |

### 11.3 Baselines

| Baseline | Description |
|----------|-------------|
| Popularity | Rank by total `play_count` across all training users |
| Item-item | Cosine similarity in ALS song embedding space |
| ALS (no reranker) | Raw ALS scores, no persona reranking |
| ALS + persona reranker | Full pipeline — the target system |

### 11.4 Ground Truth

Per-user random holdout: `holdout_k=10` interactions held out for val, another 10 for test. A recommended song is a **hit** if it appears in the user's holdout set. NDCG uses binary relevance (1 = in holdout, 0 = not).

---

## 12. Implementation Phases

| Phase | Goal | Key Deliverables |
|-------|------|-----------------|
| **P0: Docs** | Align documentation | Updated PRD, PROPOSAL, README |
| **P1: Data** | Canonical parquets from MSD | interactions.parquet, songs.parquet, song_tags.parquet |
| **P2: Baseline** | First trustworthy model | song_embeddings.parquet, baseline_metrics.csv |
| **P3: Personas** | Differentiating layer | user persona artifacts, persona vs baseline ablation |
| **P4: Side info** | Semantic enrichment | tag/similarity enriched reranker, ablation table |
| **P5: Alignment** | Repo reflects reality | Final PRD, README, CLI docs |

---

## 13. Dependencies

```toml
[project]
dependencies = [
  "implicit>=0.7",      # ALS matrix factorization
  "scikit-learn>=1.3",  # GMM
  "pandas>=2.0",
  "numpy>=1.24",
  "scipy>=1.10",        # sparse matrices, weight optimization
  "pyarrow",            # parquet I/O
  "typer",              # CLI
  "pyyaml",             # configs
  "matplotlib",
  "seaborn",
  "umap-learn",
  "tqdm",
  "h5py",               # msd_summary_file.h5
]
```

---

## 14. Verification Checklist

**Data pipeline (Phase 1):**
- [ ] `python -m music_discovery.data.interactions` completes without error
- [ ] `interactions.parquet` row count matches expected after `min_play_count` filter
- [ ] `songs.parquet` excludes all song_ids from `sid_mismatches.txt`
- [ ] `side_info.py` logs tag coverage ≥ 50% (or warns with count if not)
- [ ] `side_info.py` logs similar-track coverage ≥ 50% (or warns)

**ALS training (Phase 2):**
- [ ] `python -m music_discovery.train.train_als` completes without error
- [ ] `baseline_metrics.csv` shows ALS Recall@10 > popularity baseline Recall@10
- [ ] Song embeddings shape: `(n_songs, factors)`
- [ ] User embeddings shape: `(n_users, factors)`

**Personas (Phase 3):**
- [ ] GMM inputs use ALS song embeddings (not MLP), deduplicated per user
- [ ] BIC selects K > 1 for ≥ 70% of diverse users
- [ ] Persona reranker shows ILD improvement vs. raw ALS

**General:**
- [ ] All tests pass: `pytest tests/`
- [ ] CLI commands work end-to-end: `process → train → profile → recommend`
- [ ] Experiment 3: our recs show lower mean popularity percentile than popularity baseline
