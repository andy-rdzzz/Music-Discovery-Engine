**Motivation:**

Current music applications like Spotify are fundamentally optimized to retain engagement rather than foster discovery. These platforms rely heavily on stream counts and popularity scores, which creates a feedback loop that continuously favors mainstream content. Users with a more diverse and non-mainstream taste are underserved, and listeners with widely varied musical identities receive recommendations that overfit to the genre they streamed most recently, ignoring the richness and breadth of their listening history.

**Proposed Methods:**

**Implicit Matrix Factorization (ALS)**
We train an Alternating Least Squares (ALS) model on the Million Song Dataset Taste Profile — roughly 48 million (user, song, play_count) interactions. Play counts are transformed with log1p and used as confidence weights in the implicit feedback model. ALS learns dense embedding vectors for every user and song in a shared latent space, where proximity encodes listening affinity learned from actual large-scale behavior rather than hand-crafted audio features.

**Gaussian Mixture Models for Personal Discovery**
Once song embeddings are learned, we fit a Gaussian Mixture Model (GMM) to the embeddings of each user's consumed songs. The number of personas K is selected per user using the Bayesian Information Criterion (BIC). GMM uses soft cluster assignment, which is ideal for capturing users whose taste spans multiple genres — a pop listener who also explores ambient and jazz receives distinct persona components for each mode rather than a single blended centroid.

**Scoring Function**
Each candidate song is scored as a weighted combination of: sonic fit (max cosine similarity to any persona centroid), novelty (artist novelty + embedding distance from known songs), emotional fit (tag-based proxy or GMM log-probability), and familiarity (Mahalanobis distance under the closest persona). Popularity plays no role in ranking — it appears only as an evaluation diagnostic to confirm the engine avoids mainstream bias.

**Diversity and Reranking**
Recommendations are allocated across persona slots proportionally to persona weight (with temperature flattening), then reranked within each slot using Maximal Marginal Relevance (MMR) to reduce redundancy. A hard per-artist cap ensures no artist dominates a recommendation list.

**Planned Experiments:**

**Scoring Weight Sensitivity:** Vary the weights across scoring components and measure how recommendations change. The goal is to validate whether each component independently contributes or whether one component dominates.

**Persona Count Validation:** For diverse test users, compare recommendations produced with fixed K values against BIC-selected K, measuring NDCG@K and intra-list diversity.

**Popularity Bias Comparison:** For the same test users, compare our model's recommendations against a popularity-ranked baseline and a raw ALS baseline, measuring popularity percentile distribution, long-tail exposure, and intra-list diversity.
