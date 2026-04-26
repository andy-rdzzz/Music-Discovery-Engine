# Product Requirements Document — Music Discovery Engine

**Type:** Academic / Research  
**Stack:** Python · PyTorch · scikit-learn  
**Deliverable:** Python package (`music_discovery`) + CLI  
**Date:** 2026-04-25

---

## 1. Problem Statement

Mainstream platforms (Spotify, Apple Music) optimize for engagement via popularity signals, creating feedback loops that over-serve trending content. Users with diverse, non-mainstream taste receive recommendations that overfit to recently-played genres, ignoring the full breadth of their musical identity. This project builds a discovery engine that removes popularity from ranking entirely and models each user's taste as multiple simultaneous personas.

---

## 2. Goals

| # | Goal |
|---|------|
| G1 | Train a metric learning model that embeds songs into a space where playlist co-occurrence = musical similarity |
| G2 | Model each user as K Gaussian personas (BIC-selected), capturing multi-genre taste |
| G3 | Score candidate songs by sonic fit, emotional fit, and novelty — zero popularity weight |
| G4 | Learn optimal scoring weights per user via optimization |
| G5 | Expose everything through a clean Python package + CLI |
| G6 | Run three reproducible experiments validating each methodological claim |

## 3. Non-Goals

- Real-time serving / production latency requirements
- Web UI or mobile interface
- Collaborative filtering or social features
- Audio signal processing (no raw audio, features-only)
- Streaming history ingestion from live Spotify OAuth

---

## 4. Data Strategy

### 4.1 Critical Context: Spotify API Deprecation

> **Warning:** Spotify deprecated the `/audio-features` endpoint for all new developer apps on **November 27, 2024** with no migration path. New apps cannot access audio features via the live API. The Spotify Million Playlist Dataset (MPD) also requires direct approval from Spotify Research and is not reliably accessible.

This project uses two fully public, pre-computed datasets instead.

### 4.2 Chosen Data Sources

| Dataset | Purpose | Access | Size |
|---------|---------|--------|------|
| **Kaggle Spotify Tracks Dataset** | All 9 audio features pre-computed as CSV | Kaggle API (free account) | ~50 MB |
| **Last.fm 1K Users Dataset** | User listening history → session-based triplet construction | [zenodo.org/records/6090214](https://zenodo.org/records/6090214) | ~1 GB TSV |

**Why this combination:**
- Kaggle dataset contains the exact Spotify feature set (danceability, energy, valence, etc.) for ~114K tracks — zero HDF5 parsing, immediate use
- Last.fm 1K has 19M timestamped play events across 992 users — timestamps enable session segmentation without needing explicit playlists
- Last.fm is the official companion dataset to MSD; artist+track name join is well-documented

### 4.3 Triplet Construction from Last.fm Sessions

No explicit playlists required. Sessions are inferred from timestamps:

```
session boundary = gap > 30 minutes between consecutive plays

For each user session:
  anchor   = track A (played at time t)
  positive = track B (played in same session, t' within 30 min of t)
  negative = track C (random track from a different user's session)
```

Triplets are only formed for tracks that exist in the Kaggle features dataset (join on `artist_name + track_name`, lowercased + stripped).

### 4.4 Data Pipeline Architecture

```
[Kaggle Spotify CSV]             [Last.fm 1K TSV]
        |                               |
        v                               v
[kaggle_spotify.py]            [lastfm.py]
 Load + validate 9 features     Parse events, segment sessions
        |                               |
        v                               v
[features.py]                  [triplets.py]
 Engineer 3 interaction         Join to Kaggle features
 features → 12D vector          Build (anchor, pos, neg) triplets
        |                               |
        +---------------+---------------+
                        v
              [track_features.parquet]   ← 12D feature vectors per track
              [triplets.parquet]         ← triplet index file
              [user_history.parquet]     ← per-user track list for GMM
```

### 4.5 Audio Feature Reference

All 9 base features sourced directly from Kaggle CSV (no derivation needed):

| Feature | Type | Range |
|---------|------|-------|
| `danceability` | float | [0, 1] |
| `energy` | float | [0, 1] |
| `loudness` | float | [-60, 0] dB → normalized |
| `speechiness` | float | [0, 1] |
| `acousticness` | float | [0, 1] |
| `instrumentalness` | float | [0, 1] |
| `liveness` | float | [0, 1] |
| `valence` | float | [0, 1] |
| `tempo` | float | [40, 220] BPM → normalized |

---

## 5. Feature Engineering

### 5.1 Base Vector (9D)

`[danceability, energy, loudness_norm, acousticness, instrumentalness, liveness, valence, tempo_norm, speechiness]`

- `loudness_norm`: min-max scale from [-60, 0] dB → [0, 1]
- `tempo_norm`: min-max scale from [40, 220] BPM → [0, 1]

### 5.2 Engineered Interaction Features (3D)

Grounded in **Russell's Valence-Arousal Circumplex Model** — captures latent musical dimensions not expressed by any single base feature. Designed to be universally applicable regardless of user taste profile (user-specificity is handled downstream by the GMM, not the features).

| Feature | Formula | Captures | Music Psychology Basis |
|---------|---------|---------|----------------------|
| `arousal` | `(energy + danceability) / 2` | How activating/energizing the music feels | Russell's arousal axis — orthogonal to valence |
| `chill_factor` | `acousticness × (1 − energy)` | Mellow, organic, low-intensity music | Calm quadrant in circumplex (low arousal, acoustic) |
| `vocal_presence` | `(1 − instrumentalness) × (1 − speechiness)` | Sung vocal music vs. spoken word vs. pure instrumental | Distinguishes sung melody from rap/speech and from fully instrumental |

**Why not user-relative features:** Raw features feed a *shared* embedding space trained across all users. User-specificity enters at the GMM layer — persona centroids naturally sit in the region of embedding space where the user actually listens. A non-danceable-music listener's centroids will be in the low-danceability region regardless of whether danceability appears as a feature.

**Final input vector: 12D** = [9 base + 3 engineered], L2-normalized before model input.

---

## 6. System Architecture

```
┌─────────────────────────────────────────────────────────┐
│                    music_discovery                       │
│                                                         │
│  ┌──────────┐   ┌──────────┐   ┌──────────────────────┐│
│  │  data/   │   │ models/  │   │     cli/             ││
│  │          │   │          │   │                      ││
│  │kaggle.py │   │ mlp.py   │   │ recommend  train     ││
│  │lastfm.py │   │ gmm.py   │   │ profile    evaluate  ││
│  │ features │   │ scorer.py│   │                      ││
│  └──────────┘   └──────────┘   └──────────────────────┘│
└─────────────────────────────────────────────────────────┘
```

---

## 7. Component Specifications

### 7.1 Metric Learning MLP

**Goal:** Map 12D audio features → 24D L2-normalized embedding space where playlist co-occurrence ≈ proximity.

**Architecture:**
```
Input(12) → Linear(12→64) → BatchNorm → ReLU
          → Linear(64→48) → BatchNorm → ReLU  → Dropout(0.2)
          → Linear(48→24) → L2Normalize
```

**Training:**
- Loss: `TripletMarginLoss(margin=0.3, p=2)`
- Triplet mining: semi-hard negatives from Last.fm session pairs
  - Anchor: song A from user session S
  - Positive: song B from same session S (within 30-min window)
  - Negative: song C from a different user's session
- Optimizer: Adam, lr=1e-3, weight decay=1e-4
- Scheduler: CosineAnnealingLR
- Batch size: 512 triplets
- Epochs: 50 (early stopping on validation triplet loss)

**Hyperparameters to tune:**
- Margin: {0.2, 0.3, 0.5}
- Hidden dims: {[64,48], [128,64], [256,128]}
- Dropout: {0.1, 0.2, 0.3}

### 7.2 Gaussian Mixture Model (User Personas)

**Goal:** Represent each user's taste as K soft-assignment Gaussian clusters (personas) in embedding space.

**Process:**
1. Collect user's song embeddings (24D) from their listening history
2. Fit GMM with K components, full covariance
3. Select K via **BIC** — range K ∈ [1, 8], pick argmin(BIC)
4. Each persona k = (mean μ_k, covariance Σ_k, weight π_k)

**Implementation:** `sklearn.mixture.GaussianMixture`

**Minimum songs per user:** 20 (below this: single Gaussian fallback)

**Outputs per user:**
- K personas with parameters
- Soft assignment probability per song per persona
- Dominant persona per song (argmax posterior)

### 7.3 Scoring Function

**Goal:** Rank candidate songs for a user given seed context.

**Score formula:**
```
score(song, user) = w₁ · sonic_fit(song, user)
                  + w₂ · emotional_fit(song, user)
                  + w₃ · surprise(song, user)

where w₁ + w₂ + w₃ = 1
```

**Components:**

| Component | Weight | Formula |
|-----------|--------|---------|
| `sonic_fit` | 0.45 (init) | Max cosine similarity between song embedding and each persona centroid |
| `emotional_fit` | 0.35 (init) | Gaussian log-probability of song's valence under the closest persona's valence marginal |
| `surprise` | 0.20 (init) | 0.5 × artist_novelty + 0.5 × embedding_distance_from_known |

**artist_novelty:** 1 if artist not in user history, else 0  
**embedding_distance_from_known:** min cosine distance from song to all songs in user history (normalized to [0,1])

**Weight Learning:**
- Optimize weights per user via `scipy.optimize.minimize` (L-BFGS-B with simplex constraint)
- Objective: maximize intra-playlist ranking (songs from user's playlists ranked above random)
- Fallback to default weights if user has < 50 songs in history

---

## 8. Package Structure

```
music-discovery-engine/
├── music_discovery/
│   ├── __init__.py
│   ├── data/
│   │   ├── __init__.py
│   │   ├── kaggle_spotify.py   # Kaggle CSV loader + validation
│   │   ├── lastfm.py           # Last.fm 1K TSV parser + session segmentation
│   │   ├── triplets.py         # Triplet construction from Last.fm sessions
│   │   └── features.py         # Feature engineering (12D vector builder)
│   ├── models/
│   │   ├── __init__.py
│   │   ├── mlp.py          # EmbeddingMLP (PyTorch nn.Module)
│   │   ├── triplet.py      # Triplet dataset + semi-hard mining
│   │   ├── gmm.py          # UserPersonaModel (sklearn GMM wrapper)
│   │   └── scorer.py       # ScoringFunction + weight optimizer
│   ├── train/
│   │   ├── __init__.py
│   │   ├── train_embeddings.py   # MLP training loop
│   │   └── fit_personas.py       # GMM fitting per user
│   ├── evaluate/
│   │   ├── __init__.py
│   │   ├── weight_sensitivity.py # Experiment 1
│   │   ├── persona_validation.py # Experiment 2
│   │   └── spotify_overlap.py    # Experiment 3
│   └── cli/
│       ├── __init__.py
│       └── main.py         # Typer-based CLI
├── notebooks/
│   ├── 01_data_exploration.ipynb
│   ├── 02_feature_engineering.ipynb
│   ├── 03_embedding_analysis.ipynb
│   ├── 04_persona_exploration.ipynb
│   └── 05_experiment_results.ipynb
├── scripts/
│   ├── download_kaggle.sh      # kaggle datasets download command
│   └── download_lastfm.sh      # wget from Zenodo
├── tests/
│   ├── test_features.py
│   ├── test_mlp.py
│   ├── test_gmm.py
│   └── test_scorer.py
├── configs/
│   └── default.yaml        # All hyperparameters in one place
├── pyproject.toml
└── README.md
```

---

## 9. CLI Design

```bash
# Train the embedding model
music-discovery train --data ./data/processed --epochs 50 --config configs/default.yaml

# Fit personas for a user
music-discovery profile --user-history ./data/users/user_001.json --output ./models/user_001/

# Get recommendations
music-discovery recommend \
  --user-model ./models/user_001/ \
  --candidate-pool ./data/processed/track_embeddings.parquet \
  --n 20 \
  --output recommendations.json

# Run experiments
music-discovery evaluate weight-sensitivity --user-model ./models/user_001/
music-discovery evaluate persona-validation --users ./data/users/
music-discovery evaluate spotify-overlap --users ./data/users/ --spotify-recs ./data/spotify_baseline/
```

---

## 10. Experiments

### Experiment 1 — Scoring Weight Sensitivity

**Question:** Does each scoring component (sonic, emotional, surprise) independently contribute, or does one dominate?

**Method:**
- Fix a test user set (5–10 diverse users from Last.fm 1K, held-out final sessions)
- Ablation: run scoring with each single component alone (w=1.0, others=0)
- Grid sweep: w₁ ∈ [0,1], w₂ ∈ [0,1−w₁], w₃ = 1−w₁−w₂, step=0.05
- Measure NDCG@10 at each weight combination

**Graphs for report:**
1. **Heatmap** — x=w_sonic, y=w_emotional (w_surprise=1−x−y), color=NDCG@10. Shows optimal weight region.
2. **Grouped bar chart** — NDCG@10 for: sonic only / emotional only / surprise only / learned weights / default weights. Shows each component's independent contribution.

**Expected finding:** Learned weights outperform any single component; no single component dominates.

---

### Experiment 2 — Persona Count Validation

**Question:** Does BIC-selected K produce better-personalized recommendations than fixed K?

**Method:**
- Classify 10 test users into 3 types by genre entropy: niche (low), diverse (high), mainstream (mid)
- For each user: fit GMM at K ∈ {1, 2, 3, 4, 5} and BIC-selected K
- Evaluate NDCG@10 and Hit Rate@20 on held-out listening sessions per K

**Graphs for report:**
1. **Line chart** — x=K, y=NDCG@10, one line per user type (niche / diverse / mainstream). BIC-selected K marked with a star on each line.
2. **BIC curve** — x=K, y=BIC score for 2–3 representative users. Shows the elbow that BIC selects.
3. **UMAP scatter** — user's song embeddings in 2D with Gaussian ellipses overlaid for each persona. One plot per representative user.

**Expected finding:** BIC-selected K matches or outperforms best fixed K; diverse users benefit most from K>2.

---

### Experiment 3 — Popularity Bias Comparison

**Question:** Do our recommendations avoid mainstream content compared to a popularity-ranked baseline?

**Method:**
- Baseline: top-N tracks by total Last.fm play count (pure popularity ranking)
- Our model: recommendations for same users and candidate pool
- Measure per recommendation set:
  - Popularity percentile (track's global play count rank, 0=obscure, 1=mainstream)
  - Jaccard similarity between our recs and baseline recs
  - KL divergence of each audio feature distribution (our recs vs. baseline) — computed in **12D feature space** (`track_features.parquet`, `FEATURE_VECTOR_COLS`) not in 24D embedding space. Embedding dims are latent and uninterpretable per-dimension; feature space gives human-readable divergence per audio attribute.
  - Intra-list diversity: mean pairwise cosine distance in **24D embedding space** (captures overall sonic distance between recommended tracks)

**Implementation note:** KL divergence and intra-list diversity intentionally use different spaces. KL uses the 12D interpretable feature space for report readability. Diversity uses the 24D embedding space because that is the model's learned similarity metric.

**Graphs for report:**
1. **CDF plot** — x=popularity percentile, y=cumulative fraction of recommended tracks. Two curves: our model vs. popularity baseline. Our model's curve should shift left.
2. **Feature KL divergence bar chart** — one bar per audio feature (12D feature space) showing how much our recs diverge from popularity baseline in each dimension.
3. **Intra-list diversity box plot** — distribution of pairwise cosine distances in 24D embedding space for our recs vs. baseline across all test users.

**Expected finding:** Our recommendations skew toward lower popularity percentile and show higher intra-list diversity.

---

## 11. Evaluation Metrics

### 11.1 Embedding Quality (MLP)

| Metric | Graph | What It Proves |
|--------|-------|---------------|
| Validation triplet loss vs. epoch | Line chart | Model converged, no overfitting |
| Mean cosine sim: session-mates vs. random pairs (+ t-test, p<0.05) | Bar chart with error bars | Embedding space encodes musical similarity |
| Silhouette score on full embedding set | Reported in table | Songs cluster by sonic similarity |
| t-SNE / UMAP of embeddings colored by genre | 2D scatter plot | Visual proof of structure in embedding space |

### 11.2 Persona Quality (GMM)

| Metric | Graph | What It Proves |
|--------|-------|---------------|
| BIC score vs. K per user | Line chart (2–3 users) | BIC selection is principled, not arbitrary |
| Persona coverage (% songs with posterior p > 0.2) | Bar chart per user | Soft assignment captures multi-genre taste |
| UMAP with Gaussian ellipses per persona | 2D scatter + ellipses | Personas are spatially coherent |

### 11.3 Recommendation Quality

| Metric | K | Graph | What It Proves |
|--------|---|-------|---------------|
| NDCG@K | 5, 10, 20 | Bar chart (component ablation) | Ranking quality vs. held-out sessions |
| Hit Rate@K | 10, 20 | Bar chart | Absolute accuracy of top-N recs |
| Mean Reciprocal Rank (MRR) | — | Reported in table | How high the first hit appears |
| Intra-list diversity | — | Box plot across users | Recs are sonically varied |
| Novelty (mean popularity percentile) | — | CDF comparison | Engine avoids mainstream bias |

### 11.4 Ground Truth Construction

Since no explicit ratings exist, held-out sessions from Last.fm 1K serve as implicit ground truth:
- **80/20 split per user**: first 80% of listening history → train GMM + optimize weights; last 20% → evaluation
- A recommended track is a **hit** if it appears in the user's held-out sessions
- NDCG uses binary relevance (1 = in held-out, 0 = not)

---

## 12. Implementation Phases

| Phase | Deliverable | Key Files |
|-------|-------------|-----------|
| **P1: Data** | Kaggle loader, Last.fm parser, session triplets, feature engineer | `data/kaggle_spotify.py`, `data/lastfm.py`, `data/triplets.py`, `data/features.py` |
| **P2: Embeddings** | MLP + Triplet training loop, saved model | `models/mlp.py`, `models/triplet.py`, `train/train_embeddings.py` |
| **P3: Personas** | GMM fit + BIC selection per user | `models/gmm.py`, `train/fit_personas.py` |
| **P4: Scoring** | Scoring function + weight optimizer | `models/scorer.py` |
| **P5: CLI** | All CLI commands working end-to-end | `cli/main.py` |
| **P6: Experiments** | 3 experiment scripts + notebooks | `evaluate/`, `notebooks/` |

---

## 13. Dependencies

```toml
[project]
dependencies = [
  "torch>=2.0",
  "scikit-learn>=1.3",
  "pandas>=2.0",
  "numpy>=1.24",
  "pyarrow",          # parquet I/O
  "typer",            # CLI
  "pyyaml",           # configs
  "scipy",            # weight optimization
  "matplotlib",       # experiment plots
  "seaborn",
  "umap-learn",       # UMAP embeddings visualization
  "tqdm",
]
```

---

## 14. Verification Checklist

- [ ] `music-discovery train` completes without error, val triplet loss decreasing
- [ ] Embedding space: cosine similarity of session-mates > random pairs (t-test p<0.05)
- [ ] GMM BIC selects K>1 for at least 70% of diverse users
- [ ] Scoring function produces non-uniform rankings (not all scores equal)
- [ ] Experiment 1 shows at least 1 component is non-redundant (ablation gap > 0.05 NDCG)
- [ ] Experiment 3 Jaccard overlap < 0.3 (our recs differ meaningfully from Spotify baseline)
- [ ] All tests pass: `pytest tests/`
- [ ] CLI commands documented in README with examples
