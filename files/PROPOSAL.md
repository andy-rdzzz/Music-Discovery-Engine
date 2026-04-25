**Motivation:**

Current music applications like Spotify are fundamentally optimized to primarily retain engagement rather than discovery. These platforms rely heavily on stream counts and popularity scores, which creates a feedback loop that continuously favors mainstream content. Users with a more diverse and non-mainstream taste are underserved, while listeners with widely diversified music tastes will receive recommendations that overfit to the genre they have streamed most recently, ignoring the richness and breadth of their musical identity.

**Proposed Methods:**

**Metric Learning**
We will train an MLP (Multi-layer Perceptron) to map 12-dimensional audio feature vectors (9 from the Spotify API and 3 from engineered interaction features) into a 24-dimensional normalized embedding space. The training approach will use a Triplet Margin Loss over playlist concurrence from curated Spotify playlists, learning musical similarity from actual user behavior rather than hardcoded distances.

**Gaussian Mixture Models for personal discovery**
We will fit a Gaussian Mixture Model (GMM) to the user song embeddings. The number of personas K is selected per user using the Bayesian Information Criterion (BIC). GMMuses a soft cluster assignment, which is ideal for correctly capturing songs that bridge
multiple genres.

**Scoring Function**
Each candidate's song is scored as a weighted combination of sonic fit, emotional fit (35%, Gaussian probability under the persona’s valence distribution), and user-relative surprise (20%, artist novelty, and embedding distance from known songs). Popularity plays no role in ranking its just mere metadata.

**Planned Experiments:**

**Scoring Weight Sensitivity:** We will vary the weights and measure how recommendations alter. The main goal is to validate whether each component actually contributes or identify if there is a dominating weight.

**Persona Count Validation:** For diverse test users, compare recommendations produced with fixed K values against the BIC-selected K.

**Spotify Overlap Analysis:** For the same test users and seed tracks, we will collect Spotify’s recommendations via the API and measure the overlap with our model’s output.