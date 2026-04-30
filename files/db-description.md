# Data and Database Description for the Music Discovery Engine

This document explains every important data file used by the project, where it comes from, why it is needed, and how it contributes to the recommendation pipeline.

The short version is:

- **`train_triplets.txt`** provides the **user–song interaction signal**.
- **`track_metadata.db`** provides the **ID bridge and human-readable metadata** needed to connect interactions to tracks, artists, and titles.
- **`sid_mismatches.txt`** provides a **data-quality filter** that removes known bad song mappings.
- **`lastfm_tags.db`** and **`lastfm_similars.db`** provide **semantic side information** about tracks.
- **`artist_similarity.db`** provides an **artist-level graph** that can support future diversity and exploration logic.
- **`msd_summary_file.h5`** is an **optional metadata/audio-feature source** for future extensions; it is not the main driver of the current model.
- The project then builds **processed parquet tables**, **ALS embeddings**, and **persona models** from those raw files.

---

## 1. External Raw Data Files

These are the files you need to obtain from the Million Song Dataset ecosystem before running the pipeline.

### 1.1 `data/raw/train_triplets.txt`

- **What it is:** The Echo Nest Taste Profile Subset in tab-separated triplet form.
- **Official source:** https://millionsongdataset.com/tasteprofile/
- **Format:**  
  `user_id<TAB>song_id<TAB>play_count`
- **Primary key(s):**
  - `user_id`
  - `song_id`
- **Why this file exists in the project:**  
  This is the core behavioral dataset. It tells us which users listened to which songs, and how often.
- **How the project uses it:**  
  In `music_discovery/data/interactions.py`, this file is parsed into a user–song interaction table. The code also:
  - filters out low play counts,
  - optionally samples users for faster experimentation,
  - applies a train/validation/test split.
- **How it contributes to the model:**  
  This is the **single most important file** for the recommender. The ALS model is trained directly on this signal. The project transforms `play_count` into a weighted implicit-feedback signal using `log1p(play_count)`, which becomes the training input for matrix factorization.
- **Why it matters for the report:**  
  This file is the reason the project is a recommender system rather than just a music similarity engine. Without it, there would be no personalized user preference learning.

---

### 1.2 `data/raw/track_metadata.db`

- **What it is:** An MSD SQLite metadata database containing track-level metadata.
- **Official source:** https://millionsongdataset.com/pages/getting-dataset  
  See the **“Additional Files”** section for the SQLite databases.
- **Supporting documentation:** https://millionsongdataset.com/faq
- **Format:** SQLite database
- **Main table used by this project:** `songs`
- **Important fields used by the code:**
  - `song_id`
  - `track_id`
  - `artist_name`
  - `title`
- **Why this file exists in the project:**  
  The interaction data is keyed by **`song_id`**, while some side-information datasets are keyed by **`track_id`**. This file is the bridge between those ID systems.
- **How the project uses it:**  
  In `music_discovery/data/item_bridge.py`, the project reads this database and constructs a canonical `songs` bridge table. That table is later used to:
  - keep only valid songs,
  - map `song_id -> track_id`,
  - attach artist/title metadata,
  - join in Last.fm side-information.
- **How it contributes to the model:**  
  It does **not** provide the collaborative signal itself, but it is structurally essential. Without it:
  - the interactions could not be reliably joined to Last.fm tags/similarities,
  - the recommender could not produce human-readable outputs,
  - artist-level constraints and reporting would be much harder.
- **Why it matters for the report:**  
  This file is the **entity-resolution layer** of the project. It turns raw recommendation IDs into coherent items and enables multi-source fusion.

---

### 1.3 `data/raw/sid_mismatches.txt`

- **What it is:** A list of known bad song–track mappings associated with the Taste Profile / MSD match process.
- **Official source:** https://millionsongdataset.com/tasteprofile/
- **Related warning page:** https://millionsongdataset.com/lastfm/
- **Format:** plain text
- **Why this file exists in the project:**  
  The MSD/Taste Profile ecosystem has known mapping issues. Some `song_id` values should not be trusted.
- **How the project uses it:**  
  In `music_discovery/data/item_bridge.py`, the project parses this file and removes bad `song_id` values **before** building the canonical song bridge.
- **How it contributes to the model:**  
  This file improves **data integrity**. It prevents corrupted or mismatched items from contaminating:
  - the interaction matrix,
  - metadata joins,
  - side-information joins,
  - evaluation outputs.
- **Why it matters for the report:**  
  This file is a strong example of **data-quality control**. It shows that the pipeline is not just training on raw data blindly; it explicitly removes known bad mappings.

---

### 1.4 `data/raw/lastfm_tags.db`

- **What it is:** The Last.fm tag dataset distributed as an SQLite database.
- **Official source:** https://millionsongdataset.com/lastfm/
- **Format:** SQLite database
- **Relevant schema used by the project:**
  - `tids(tid)`
  - `tags(tag)`
  - `tid_tag(tid, tag, val)`
- **Join key:** `track_id`
- **Why this file exists in the project:**  
  Tags provide a semantic description of songs (for example: genre, mood, style, decade labels, etc.).
- **How the project uses it:**  
  In `music_discovery/data/side_info.py`, the project reads the normalized SQLite schema, reconstructs `(track_id, tag, weight)` records, and joins them back to `song_id` through the metadata bridge.
- **How it contributes to the model:**  
  In the current codebase, tags are mainly used as **side-information assets for analysis and future extensions**, not as the primary training signal. Their main value is that they provide interpretable metadata that can support:
  - explanation,
  - semantic profiling,
  - future hybrid recommendations,
  - qualitative validation.
- **Why it matters for the report:**  
  This file strengthens the project’s story by showing that the recommender is capable of going beyond pure collaborative filtering and can be extended with richer semantic context.

---

### 1.5 `data/raw/lastfm_similars.db`

- **What it is:** The Last.fm song-similarity dataset distributed as an SQLite database.
- **Official source:** https://millionsongdataset.com/lastfm/
- **Format:** SQLite database
- **Relevant table used by the project:** `similars_src`
- **Join key:** `track_id`
- **Why this file exists in the project:**  
  This file contains track-to-track similarity relationships computed from Last.fm behavior and metadata.
- **How the project uses it:**  
  In `music_discovery/data/side_info.py`, the project unpacks the similarity representation, converts `track_id` pairs into `song_id` pairs, and writes the result as `song_similars.parquet`.
- **How it contributes to the model:**  
  Similar-track data is valuable because it captures a form of **item-item structure** independent of the ALS factorization. In the current project, it primarily serves as a side-information resource and analysis aid. Conceptually, it can support:
  - explanation of recommendations,
  - future graph-based reranking,
  - evaluation of embedding neighborhoods,
  - discovery-oriented recommendation logic.
- **Why it matters for the report:**  
  It shows that the project has access to meaningful **content/behavioral side structure** beyond the user–item matrix, even if the current final model does not fully exploit it.

---

### 1.6 `data/raw/artist_similarity.db`

- **What it is:** An MSD SQLite database containing artist-to-artist similarity links.
- **Official source:** https://millionsongdataset.com/pages/getting-dataset  
  See the **“Additional Files”** section.
- **Supporting documentation:** https://millionsongdataset.com/faq
- **Format:** SQLite database
- **Relevant table used by the project:** `similarity`
- **Fields used:**
  - `target AS artist_id`
  - `similar AS similar_artist_id`
- **Why this file exists in the project:**  
  Artist similarity is useful for exploration, diversity control, and future artist-graph methods.
- **How the project uses it:**  
  In `music_discovery/data/side_info.py`, it is loaded and exported as `artist_similars.parquet`.
- **How it contributes to the model:**  
  In the current system, it is a **supporting data asset** rather than a core training input. Its main contribution is architectural: it gives the project a clean way to later add:
  - artist-based diversification,
  - graph expansion,
  - “similar artist” exploration logic.
- **Why it matters for the report:**  
  It supports the argument that the system was built with extensibility in mind and is not restricted to a single collaborative signal.

---

### 1.7 `data/raw/msd_summary_file.h5` (optional)

- **What it is:** The Million Song Dataset summary HDF5 file for the full corpus.
- **Official source:** https://millionsongdataset.com/pages/getting-dataset
- **Supporting documentation:**  
  - https://millionsongdataset.com/faq  
  - https://millionsongdataset.com/pages/find-song-specific-name-or-feature
- **Format:** HDF5
- **Why this file exists in the project:**  
  It is an optional fast-access metadata container for the full MSD.
- **How the project currently uses it:**  
  It is listed in the config as optional, but the current production pipeline does **not** depend on it.
- **How it could contribute to the model:**  
  It can be used for future extensions such as:
  - audio-descriptor analysis,
  - metadata augmentation,
  - year/tempo/key/loudness-based features,
  - embedding diagnostics.
- **Why it matters for the report:**  
  This file is important to mention because it shows the project can grow from a collaborative+persona recommender into a richer hybrid system that uses audio and metadata features directly.

---

## 2. Project-Generated Processed Data

These files are **not downloaded externally**. They are produced by the project’s `process` pipeline.

### 2.1 `data/processed/interactions.parquet`

- **Generated from:** `train_triplets.txt` + `track_metadata.db` + `sid_mismatches.txt`
- **How it is created:** `music_discovery/data/interactions.py`
- **Main fields:**
  - `user_id`
  - `song_id`
  - `play_count`
  - `weight`
  - `split` (`train`, `val`, `test`)
- **Why it exists:**  
  This is the canonical cleaned interaction table used across training and evaluation.
- **How it contributes to the model:**  
  It is the direct training input for ALS and the basis for all offline evaluation.

---

### 2.2 `data/processed/songs.parquet`

- **Generated from:** `track_metadata.db` filtered by `sid_mismatches.txt`
- **Main fields:**
  - `song_id`
  - `track_id`
  - `artist_name`
  - `title`
- **Why it exists:**  
  This is the project’s canonical item table.
- **How it contributes to the model:**  
  It supports joins, evaluation readability, and artist-aware recommendation constraints.

---

### 2.3 `data/processed/song_tags.parquet`

- **Generated from:** `lastfm_tags.db` joined through `songs.parquet`
- **Main fields:**
  - `song_id`
  - `tag`
  - `weight`
- **Why it exists:**  
  Makes semantic tags easy to use from pandas/parquet without re-querying SQLite.
- **How it contributes to the model:**  
  Side-information for interpretation and future hybridization.

---

### 2.4 `data/processed/song_similars.parquet`

- **Generated from:** `lastfm_similars.db` joined through `songs.parquet`
- **Main fields:**
  - `song_id`
  - `similar_song_id`
  - `score`
- **Why it exists:**  
  Stores a clean song-level similarity graph.
- **How it contributes to the model:**  
  Useful for graph-based or explanation-based extensions to the recommender.

---

### 2.5 `data/processed/artist_similars.parquet`

- **Generated from:** `artist_similarity.db`
- **Main fields:**
  - `artist_id`
  - `similar_artist_id`
- **Why it exists:**  
  Stores artist-graph information in a fast analytical format.
- **How it contributes to the model:**  
  Supports future artist-diversification and exploration logic.

---

## 3. Model Artifacts Built from the Data

These are learned artifacts created by the training pipeline.

### 3.1 `models/embeddings/song_embeddings.parquet`

- **Generated from:** `data/processed/interactions.parquet`
- **How it is created:** `music_discovery/train/train_als.py`
- **Model:** implicit-feedback ALS
- **Main fields:**
  - `song_id`
  - `emb_0 ... emb_(factors-1)`
- **Why it exists:**  
  This stores the learned latent representation of each song.
- **How it contributes to the model:**  
  This is the main item representation used throughout the project:
  - ALS relevance scoring,
  - user-persona fitting,
  - candidate scoring,
  - diversity computations.
- **Why it matters for the report:**  
  This file is the core “embedding approach” artifact. It is the representation layer that turns raw interactions into a structured music space.

---

### 3.2 `models/embeddings/user_embeddings.parquet`

- **Generated from:** `data/processed/interactions.parquet`
- **How it is created:** `music_discovery/train/train_als.py`
- **Main fields:**
  - `user_id`
  - `emb_0 ... emb_(factors-1)`
- **Why it exists:**  
  Stores the learned latent representation of each user.
- **How it contributes to the model:**  
  Used for:
  - ALS recommendation scoring,
  - relevance-guided shortlist generation,
  - holdout evaluation,
  - blending collaborative relevance with persona reranking.
- **Why it matters for the report:**  
  This file is what makes the recommender personalized at the user level before the persona layer is even applied.

---

### 3.3 `models/personas/<user_id>/persona.pkl`

- **Generated from:** user histories mapped into `song_embeddings.parquet`
- **How it is created:** `music_discovery/train/fit_personas.py`
- **Model:** Gaussian Mixture Model (GMM)
- **What it stores conceptually:**
  - persona centroids,
  - covariance structure,
  - mixture weights,
  - fitted sklearn GMM object.
- **Why it exists:**  
  The ALS model gives one user vector per user, but real listeners often have multiple taste modes. Persona models approximate that multimodality.
- **How it contributes to the model:**  
  This is the key artifact behind the project’s “persona-aware” layer:
  - identifies multiple taste clusters,
  - supports persona-specific scoring,
  - enables more diverse and discovery-oriented reranking.
- **Why it matters for the report:**  
  This file is the central object that distinguishes the project from a plain collaborative-filtering baseline.

---

## 4. Why Each Dataset Matters to the Final System

If you want one concise report paragraph, this is the most important interpretation:

- **Taste Profile (`train_triplets.txt`)** gives the project its **personalization signal**.
- **Track metadata (`track_metadata.db`)** gives the project its **join logic and readable items**.
- **Mismatch filtering (`sid_mismatches.txt`)** gives the project **data quality control**.
- **Last.fm tags/similars** give the project **semantic and relational side-information**.
- **Artist similarity** gives the project a path toward **graph-based diversification**.
- **ALS embeddings** convert sparse listening data into a usable **latent music space**.
- **Persona GMMs** split that space into **multiple user taste modes**, which is what supports discovery-oriented reranking.

In other words:

> The model is fundamentally powered by the Taste Profile interaction data, structurally enabled by the metadata bridge, cleaned by mismatch filtering, enriched by optional side-information, represented through ALS embeddings, and behaviorally diversified through persona mixtures.

---

## 5. Suggested Citation / Source Links for the Report

Use the following official pages in the report:

- **Million Song Dataset main site:** https://millionsongdataset.com/
- **Taste Profile subset:** https://millionsongdataset.com/tasteprofile/
- **Last.fm dataset:** https://millionsongdataset.com/lastfm/
- **Getting the dataset / additional files:** https://millionsongdataset.com/pages/getting-dataset
- **MSD FAQ (summary file + SQLite DB descriptions):** https://millionsongdataset.com/faq

These are the best high-level source links for documenting where the files come from and what they contain.
