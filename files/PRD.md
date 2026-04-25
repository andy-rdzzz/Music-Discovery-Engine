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

> **Warning:** Spotify deprecated the `/audio-features` endpoint for all new developer apps on **November 27, 2024** with no migration path. New apps cannot access audio features via the live API.

This eliminates a live-API pipeline. Pre-computed datasets are required.

### 4.2 Recommended Data Sources

| Dataset | Purpose | Access | Size |
|---------|---------|--------|------|
| **Million Song Dataset (MSD)** | Audio features (Echo Nest, Spotify-compatible) | Free, public | ~280 GB full / 1 GB subset |
| **Spotify Million Playlist Dataset (MPD)** | Playlist co-occurrence for triplet loss | Request from [Spotify Research](https://research.atspotify.com/2020/09/the-million-playlist-dataset-remastered) | ~5 GB compressed |
| **MSD + Last.fm subset** | Simulated user listening history | Free, paired with MSD | ~2 GB |
| **FMA small/medium** | Fallback if MPD access denied | Free, GitHub | 7 GB / 25 GB |

### 4.3 Data Pipeline Architecture

```
[MSD HDF5 files]          [MPD JSON files]         [Last.fm / simulated users]
      |                         |                              |
      v                         v                              v
[Feature Extraction]    [Playlist Parser]            [User History Parser]
      |                         |                              |
      v                         v                              v
[song_features.parquet] [playlists.parquet]        [user_history.parquet]
      |                         |                              |
      +------------+------------+                              |
                   v                                           |
           [Feature Engineer]                                  |
                   |                                           |
                   v                                           v
           [track_embeddings]                         [user_embeddings]
```

### 4.4 MSD Feature Availability

The following 9 Echo Nest features (Spotify-compatible) are available in MSD:

| Feature | Type | MSD Field |
|---------|------|-----------|
| `danceability` | float [0,1] | `songs.danceability` |
| `energy` | float [0,1] | `songs.energy` |
| `loudness` | float dB | `songs.loudness` |
| `speechiness` | float [0,1] | (derive from hotttnesss proxy or FMA) |
| `acousticness` | float [0,1] | `songs.acousticness` |
| `instrumentalness` | float [0,1] | `songs.instrumentalness` |
| `liveness` | float [0,1] | `songs.liveness` |
| `valence` | float [0,1] | `songs.valence` |
| `tempo` | float BPM | `songs.tempo` |

> **Note:** `speechiness` coverage in MSD is partial. Substitute with FMA's librosa-derived feature or drop and replace with a third engineered feature.

---

## 5. Feature Engineering

### 5.1 Base Vector (9D)

`[danceability, energy, loudness_norm, acousticness, instrumentalness, liveness, valence, tempo_norm, speechiness]`

- `loudness_norm`: min-max scale from [-60, 0] dB → [0, 1]
- `tempo_norm`: min-max scale from [40, 220] BPM → [0, 1]

### 5.2 Engineered Interaction Features (3D)

| Feature | Formula | Captures |
|---------|---------|---------|
| `energy_dance` | `energy × danceability` | High-intensity dance music dimension |
| `valence_energy` | `valence × energy` | Emotional arousal (happy-energetic vs sad-mellow) |
| `acoustic_instrumental` | `acousticness × instrumentalness` | Organic, non-vocal instrumental dimension |

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
│  │ msd.py   │   │ mlp.py   │   │ recommend  train     ││
│  │ mpd.py   │   │ gmm.py   │   │ profile    evaluate  ││
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
- Triplet mining: semi-hard negatives from MPD playlists
  - Anchor: song A from playlist P
  - Positive: song B from same playlist P
  - Negative: song C from a different playlist P'
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
│   │   ├── msd.py          # MSD HDF5 loader
│   │   ├── mpd.py          # MPD JSON parser + triplet generator
│   │   ├── lastfm.py       # Last.fm / simulated user history
│   │   └── features.py     # Feature extraction + engineering
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
│   ├── download_msd_subset.sh
│   └── prepare_mpd.py
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
- Fix a test user set (5–10 diverse users)
- Sweep weight combinations: grid over w₁ ∈ [0,1], w₂ ∈ [0,1−w₁], w₃ = 1−w₁−w₂, step=0.05
- Measure: NDCG@10 on held-out playlist songs
- Ablate: score with each single component alone vs. all three

**Output:** Heatmap of NDCG by weight combination. Component contribution table.

### Experiment 2 — Persona Count Validation

**Question:** Does BIC-selected K produce better recommendations than fixed K?

**Method:**
- For 10 diverse users: fit GMM at K ∈ {1, 2, 3, 4, 5, BIC-selected}
- Compare recommendation NDCG@10 per K
- Measure: intra-playlist hit rate at fixed N=20 recommendations

**Output:** Line chart of NDCG vs K per user type (niche, diverse, mainstream).

### Experiment 3 — Spotify Overlap Analysis

**Question:** How different are our recommendations from Spotify's own?

**Method:**
- Collect Spotify recommendations via API (if accessible) or use a held-out MPD baseline
- Measure Jaccard similarity: |our_recs ∩ spotify_recs| / |our_recs ∪ spotify_recs|
- Measure popularity percentile of our recommendations vs. Spotify's
- Measure audio feature distribution divergence (KL divergence on each feature)

**Hypothesis:** Our recommendations have lower mean popularity rank and higher feature diversity.

**Output:** Overlap table, popularity CDF comparison, feature KL table.

---

## 11. Evaluation Metrics

| Metric | Used In | Description |
|--------|---------|-------------|
| NDCG@10 | Exp 1, 2 | Ranking quality using playlist membership as ground truth |
| Hit Rate @N | Exp 2 | % of recommended songs that appear in held-out playlists |
| Jaccard Similarity | Exp 3 | Set overlap with Spotify baseline |
| Popularity Percentile | Exp 3 | Average chart-position rank of recommendations |
| Triplet Loss (val) | Training | Embedding model convergence |
| BIC Score | GMM | Persona count selection quality |

---

## 12. Implementation Phases

| Phase | Deliverable | Key Files |
|-------|-------------|-----------|
| **P1: Data** | MSD loader, MPD parser, feature engineer | `data/msd.py`, `data/mpd.py`, `data/features.py` |
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
  "h5py",             # MSD HDF5
  "typer",            # CLI
  "pyyaml",           # configs
  "scipy",            # weight optimization
  "matplotlib",       # experiment plots
  "seaborn",
  "tqdm",
]
```

---

## 14. Verification Checklist

- [ ] `music-discovery train` completes without error, val triplet loss decreasing
- [ ] Embedding space: cosine similarity of playlist-mates > random pairs (t-test p<0.05)
- [ ] GMM BIC selects K>1 for at least 70% of diverse users
- [ ] Scoring function produces non-uniform rankings (not all scores equal)
- [ ] Experiment 1 shows at least 1 component is non-redundant (ablation gap > 0.05 NDCG)
- [ ] Experiment 3 Jaccard overlap < 0.3 (our recs differ meaningfully from Spotify baseline)
- [ ] All tests pass: `pytest tests/`
- [ ] CLI commands documented in README with examples
