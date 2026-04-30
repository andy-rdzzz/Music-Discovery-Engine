from __future__ import annotations

import numpy as np
from scipy.optimize import minimize

from music_discovery.models.gmm import UserPersonaModel

DEFAULT_WEIGHTS = np.array([0.45, 0.30, 0.15, 0.10])  # sonic_fit, novelty, persona_specificity, familiarity
DEFAULT_RELEVANCE_BLEND = 0.80


# ---------------------------------------------------------------------------
# Component scorers
# ---------------------------------------------------------------------------

def sonic_fit(
    candidate_emb: np.ndarray,
    persona: UserPersonaModel,
) -> np.ndarray:
    """
    Max cosine similarity between each candidate and any persona centroid.
    candidate_emb: (N, D) → returns (N,) in [-1, 1].
    """
    means_norm = persona.means / np.linalg.norm(persona.means, axis=1, keepdims=True)
    sims = candidate_emb @ means_norm.T   # (N, K)
    return sims.max(axis=1)               # (N,)


def persona_specificity(
    candidate_emb: np.ndarray,
    persona: UserPersonaModel,
) -> np.ndarray:
    """
    1 - normalized Shannon entropy of the GMM posterior.
    High = candidate maps clearly to one persona (good); low = spread across all (bad).
    Returns (N,) in [0, 1].
    """
    posteriors = persona.posterior(candidate_emb)  # (N, K)
    eps = 1e-9
    entropy = -np.sum(posteriors * np.log(posteriors + eps), axis=1)
    max_entropy = np.log(persona.k)
    if max_entropy > 0:
        return 1.0 - (entropy / max_entropy)
    return np.ones(len(candidate_emb))


def novelty_score(
    candidate_emb: np.ndarray,
    candidate_artists: np.ndarray,
    user_history_emb: np.ndarray,
    user_history_artists: np.ndarray,
) -> np.ndarray:
    """
    0.5 * artist_novelty + 0.5 * embedding distance from known tracks.
    Returns (N,) in [0, 1].
    """
    known_artists = set(user_history_artists)
    artist_novelty = np.array([1.0 if a not in known_artists else 0.0 for a in candidate_artists])

    sims = candidate_emb @ user_history_emb.T   # (N, M)
    max_sims = sims.max(axis=1)
    emb_distance = np.clip(1.0 - max_sims, 0.0, 1.0)

    return 0.5 * artist_novelty + 0.5 * emb_distance


def familiarity_score(
    candidate_emb: np.ndarray,
    persona: UserPersonaModel,
) -> np.ndarray:
    """
    GMM log-probability for each candidate, normalized to [0, 1] across the batch.
    Measures confidence that the candidate falls inside the user's taste distribution.
    Uses full covariance (Mahalanobis geometry), not cosine similarity — avoids
    overlap with sonic_fit.
    Returns (N,).
    """
    log_p = persona.log_prob(candidate_emb)  # (N,)
    lp_min, lp_max = log_p.min(), log_p.max()
    if lp_max > lp_min:
        return (log_p - lp_min) / (lp_max - lp_min)
    return np.zeros(len(candidate_emb))


def _minmax_normalize(values: np.ndarray) -> np.ndarray:
    """Normalize finite scores to [0, 1] while preserving array shape."""
    finite = np.isfinite(values)
    if not finite.any():
        return np.zeros(len(values), dtype=np.float32)

    out = np.zeros(len(values), dtype=np.float32)
    finite_vals = values[finite]
    v_min = finite_vals.min()
    v_max = finite_vals.max()
    if v_max > v_min:
        out[finite] = (finite_vals - v_min) / (v_max - v_min)
    elif finite_vals.size:
        out[finite] = 1.0
    return out



# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------

def score_candidates(
    candidate_emb: np.ndarray,
    candidate_artists: np.ndarray,
    persona: UserPersonaModel,
    user_history_emb: np.ndarray,
    user_history_artists: np.ndarray,
    weights: np.ndarray | None = None,
    relevance_scores: np.ndarray | None = None,
    relevance_blend: float = DEFAULT_RELEVANCE_BLEND,
) -> np.ndarray:
    """
    4-component scoring:
      w1*persona_fit + w2*novelty + w3*emotional_fit + w4*familiarity
    Returns (N,) scores in [0, 1].
    """
    if weights is None:
        weights = DEFAULT_WEIGHTS

    w1, w2, w3, w4 = weights
    s1 = (sonic_fit(candidate_emb, persona) + 1.0) / 2.0  # [-1,1] → [0,1]
    s2 = novelty_score(candidate_emb, candidate_artists, user_history_emb, user_history_artists)
    s3 = persona_specificity(candidate_emb, persona)
    s4 = familiarity_score(candidate_emb, persona)

    persona_scores = w1 * s1 + w2 * s2 + w3 * s3 + w4 * s4
    if relevance_scores is None:
        return persona_scores

    alpha = float(np.clip(relevance_blend, 0.0, 1.0))
    if alpha <= 0.0:
        return persona_scores

    base_scores = _minmax_normalize(relevance_scores)
    return alpha * base_scores + (1.0 - alpha) * persona_scores


def optimize_weights(
    candidate_emb: np.ndarray,
    candidate_artists: np.ndarray,
    persona: UserPersonaModel,
    user_history_emb: np.ndarray,
    user_history_artists: np.ndarray,
    held_out_indices: np.ndarray,
    min_songs: int = 50,
) -> np.ndarray:
    """
    Learn per-user weights via L-BFGS-B. Objective: maximise NDCG@10 on held-out tracks.
    Falls back to DEFAULT_WEIGHTS if user has fewer than min_songs tracks.
    Returns (4,) weight array summing to 1.
    """
    if len(user_history_emb) < min_songs:
        return DEFAULT_WEIGHTS.copy()

    s1 = (sonic_fit(candidate_emb, persona) + 1.0) / 2.0
    s2 = novelty_score(candidate_emb, candidate_artists, user_history_emb, user_history_artists)
    s3 = persona_specificity(candidate_emb, persona)
    s4 = familiarity_score(candidate_emb, persona)

    components = np.stack([s1, s2, s3, s4], axis=1)  # (N, 4)
    n_candidates = len(candidate_emb)
    held_out_mask = np.zeros(n_candidates, dtype=bool)
    valid_held = held_out_indices[held_out_indices < n_candidates]
    held_out_mask[valid_held] = True

    def neg_ndcg(w: np.ndarray) -> float:
        scores = components @ w
        order = np.argsort(-scores)
        relevance = held_out_mask[order].astype(float)
        k = min(10, n_candidates)
        dcg = (relevance[:k] / np.log2(np.arange(2, k + 2))).sum()
        ideal = np.sort(relevance)[::-1]
        idcg = (ideal[:k] / np.log2(np.arange(2, k + 2))).sum()
        return -(dcg / idcg) if idcg > 0 else 0.0

    constraints = {"type": "eq", "fun": lambda w: w.sum() - 1.0}
    bounds = [(0.0, 1.0)] * 4
    result = minimize(neg_ndcg, DEFAULT_WEIGHTS.copy(), method="SLSQP", bounds=bounds, constraints=constraints)
    if result.success:
        w = np.clip(result.x, 0, 1)
        return w / w.sum()
    return DEFAULT_WEIGHTS.copy()


# ---------------------------------------------------------------------------
# Persona slot allocation
# ---------------------------------------------------------------------------

def allocate_persona_slots(
    pi: np.ndarray,
    n_recs: int,
    tau: float = 0.7,
    min_slots: int = 1,
    max_fraction: float = 0.6,
    exploration_fraction: float = 0.15,
) -> tuple[np.ndarray, int]:
    """
    Tempered-proportional persona slot allocation with min/max constraints.

    Parameters
    ----------
    pi : GMM mixture weights (K,), sum to 1
    tau : temperature — <1 flattens distribution, =1 raw proportional
    min_slots : guaranteed minimum slots per persona
    max_fraction : max fraction of persona slots one persona can claim
    exploration_fraction : fraction of n_recs reserved for wildcard exploration

    Returns
    -------
    slots : (K,) integer slots per persona
    n_exploration : integer slots for exploration pool
    """
    K = len(pi)
    n_exploration = max(1, int(round(exploration_fraction * n_recs)))
    n_persona = n_recs - n_exploration

    # Temper
    w = pi ** tau
    w = w / w.sum()

    # Initial proportional allocation
    raw = w * n_persona
    slots = np.floor(raw).astype(int)

    # Enforce min_slots
    slots = np.maximum(slots, min_slots)

    # Enforce max cap
    max_slots = max(min_slots, int(np.floor(max_fraction * n_persona)))
    slots = np.minimum(slots, max_slots)

    # Adjust total to n_persona
    diff = n_persona - slots.sum()
    order_desc = np.argsort(-w)
    order_asc = np.argsort(w)

    if diff > 0:
        for k in order_desc:
            if diff <= 0:
                break
            add = min(max_slots - slots[k], diff)
            slots[k] += add
            diff -= add
    elif diff < 0:
        for k in order_asc:
            if diff >= 0:
                break
            remove = min(slots[k] - min_slots, -diff)
            slots[k] -= remove
            diff += remove

    return slots, n_exploration


# ---------------------------------------------------------------------------
# MMR reranking + artist cap
# ---------------------------------------------------------------------------

def mmr_rerank(
    indices: np.ndarray,
    scores: np.ndarray,
    embeddings: np.ndarray,
    n: int,
    lambda_: float = 0.7,
) -> np.ndarray:
    """
    Greedy Maximal Marginal Relevance selection.
    mmr_i = lambda_ * relevance_i - (1 - lambda_) * max_{j in selected} cosine(i, j)
    Embeddings assumed L2-normalized (cosine sim = dot product).

    Parameters
    ----------
    indices : candidate track indices into `embeddings`
    scores  : relevance score for each candidate (parallel to indices)
    n       : number of items to select
    Returns selected indices (up to n).
    """
    if len(indices) == 0:
        return np.array([], dtype=int)
    if len(indices) <= n:
        return indices

    remaining = list(range(len(indices)))
    selected_positions: list[int] = []

    while len(selected_positions) < n and remaining:
        if not selected_positions:
            best_pos = max(remaining, key=lambda p: scores[p])
        else:
            sel_emb = embeddings[indices[selected_positions]]  # (|sel|, D)
            best_mmr = -np.inf
            best_pos = remaining[0]
            for p in remaining:
                emb = embeddings[indices[p]]
                sim = float((sel_emb @ emb).max())
                mmr = lambda_ * scores[p] - (1.0 - lambda_) * sim
                if mmr > best_mmr:
                    best_mmr = mmr
                    best_pos = p

        selected_positions.append(best_pos)
        remaining.remove(best_pos)

    return indices[np.array(selected_positions)]


def _apply_artist_cap(
    indices: np.ndarray,
    artists: np.ndarray,
    max_per_artist: int = 2,
) -> np.ndarray:
    """Keep at most max_per_artist tracks per artist. Preserves input order."""
    artist_counts: dict[str, int] = {}
    kept: list[int] = []
    for idx in indices:
        artist = artists[idx]
        count = artist_counts.get(artist, 0)
        if count < max_per_artist:
            kept.append(int(idx))
            artist_counts[artist] = count + 1
    return np.array(kept, dtype=int)


# ---------------------------------------------------------------------------
# Full diverse recommendation pipeline
# ---------------------------------------------------------------------------

def recommend_diverse(
    persona: UserPersonaModel,
    candidate_emb: np.ndarray,
    candidate_artists: np.ndarray,
    history_emb: np.ndarray,
    history_artists: np.ndarray,
    history_indices: np.ndarray,
    n_recs: int = 20,
    weights: np.ndarray | None = None,
    tau: float = 0.7,
    min_slots: int = 1,
    max_fraction: float = 0.6,
    exploration_fraction: float = 0.15,
    mmr_lambda: float = 0.7,
    max_per_artist: int = 2,
    relevance_scores: np.ndarray | None = None,
    relevance_blend: float = DEFAULT_RELEVANCE_BLEND,
    rerank_top_m: int | None = None,
) -> np.ndarray:
    """
    Full diverse recommendation pipeline:
      1. (Optional) Restrict to ALS top-M shortlist via rerank_top_m
      2. Score candidates (4-component formula)
      3. Allocate slots per persona (tempered proportional + constraints)
      4. Per persona bucket: MMR rerank → select allocated slots
      5. Fill exploration pool from remaining candidates
      6. Apply artist hard cap
      7. Pad back to n_recs if artist cap reduced count
    Returns array of n_recs track indices.
    """
    scores = score_candidates(
        candidate_emb, candidate_artists, persona,
        history_emb, history_artists, weights,
        relevance_scores=relevance_scores,
        relevance_blend=relevance_blend,
    )

    # Restrict persona reranking to ALS top-M shortlist
    if rerank_top_m is not None and relevance_scores is not None and rerank_top_m < len(scores):
        top_m_mask = np.zeros(len(scores), dtype=bool)
        top_m_mask[np.argsort(-relevance_scores)[:rerank_top_m]] = True
        scores[~top_m_mask] = -np.inf

    # Exclude heard tracks
    valid_history = history_indices[history_indices < len(candidate_emb)]
    scores[valid_history] = -np.inf

    # Dominant persona per candidate
    dominant = persona.dominant_persona(candidate_emb)  # (N,)

    # Slot allocation
    slots, n_exploration = allocate_persona_slots(
        persona.weights, n_recs, tau, min_slots, max_fraction, exploration_fraction
    )

    used_mask = np.zeros(len(candidate_emb), dtype=bool)
    used_mask[valid_history] = True

    selected: list[int] = []

    for k in range(persona.k):
        n_select = int(slots[k]) if k < len(slots) else 0
        if n_select <= 0:
            continue

        bucket_mask = (dominant == k) & ~used_mask
        bucket_indices = np.where(bucket_mask)[0]
        if len(bucket_indices) == 0:
            continue

        bucket_scores = scores[bucket_indices]
        # Oversample for MMR: take 3x the slots before reranking
        oversample = min(len(bucket_indices), n_select * 3)
        top_bucket = bucket_indices[np.argsort(-bucket_scores)[:oversample]]
        top_scores = scores[top_bucket]

        chosen = mmr_rerank(top_bucket, top_scores, candidate_emb, n_select, mmr_lambda)
        selected.extend(chosen.tolist())
        used_mask[chosen] = True

    # Exploration pool from remaining candidates
    remaining_indices = np.where(~used_mask & (scores > -np.inf))[0]
    if len(remaining_indices) > 0:
        top_remaining = remaining_indices[np.argsort(-scores[remaining_indices])][:n_exploration]
        selected.extend(top_remaining.tolist())

    selected_arr = np.array(selected, dtype=int)
    if len(selected_arr) == 0:
        # Fallback: plain top-N
        return np.argsort(-scores)[:n_recs]

    # Order by score before artist cap so cap keeps best-scored tracks per artist
    order = np.argsort(-scores[selected_arr])
    selected_arr = _apply_artist_cap(selected_arr[order], candidate_artists, max_per_artist)

    # Pad back to n_recs if artist cap reduced the count
    if len(selected_arr) < n_recs:
        selected_set = set(selected_arr.tolist()) | set(valid_history.tolist())
        fill_pool = np.array([i for i in np.argsort(-scores) if i not in selected_set])
        needed = n_recs - len(selected_arr)
        selected_arr = np.concatenate([selected_arr, fill_pool[:needed]])

    return selected_arr[:n_recs]
