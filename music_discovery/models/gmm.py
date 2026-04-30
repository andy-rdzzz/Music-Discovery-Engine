from __future__ import annotations
import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from sklearn.mixture import GaussianMixture


@dataclass
class UserPersonaModel:
    user_id: str
    k: int
    means: np.ndarray        # (K, D)
    covariances: np.ndarray  # (K, D, D)
    weights: np.ndarray      # (K,)
    gmm: GaussianMixture = field(repr=False)

    def posterior(self, embeddings: np.ndarray) -> np.ndarray:
        """Soft assignment probabilities. Returns (N, K)."""
        return self.gmm.predict_proba(embeddings)

    def dominant_persona(self, embeddings: np.ndarray) -> np.ndarray:
        """Argmax persona index per embedding. Returns (N,)."""
        return self.gmm.predict(embeddings)

    def log_prob(self, embeddings: np.ndarray) -> np.ndarray:
        """Log-likelihood of each embedding under the mixture. Returns (N,)."""
        return self.gmm.score_samples(embeddings)


def fit_user_persona(
    user_id: str,
    embeddings: np.ndarray,
    k_min: int = 2,
    k_max: int = 5,
    covariance_type: str = "diag",
    min_samples: int = 20,
    n_init: int = 1,
    random_state: int = 42,
) -> UserPersonaModel:
    """
    Fit GMM to user embeddings. K selected by BIC over [k_min, k_max].
    Falls back to K=1  < than min_samples embeddings.
    """
    n = len(embeddings)

    # sklearn GMM requires >= 2 samples; single-sample users skip GMM entirely
    if n < 2:
        gmm = GaussianMixture(n_components=1, covariance_type="diag", random_state=random_state)
        # Manually set attributes so the model is usable without fitting
        gmm.means_ = embeddings if n == 1 else np.zeros((1, embeddings.shape[1]))
        gmm.covariances_ = np.ones((1, embeddings.shape[1]))
        gmm.weights_ = np.array([1.0])
        gmm.precisions_cholesky_ = np.ones((1, embeddings.shape[1]))
        gmm.converged_ = True
        gmm.n_iter_ = 0
        return UserPersonaModel(user_id=user_id, k=1, means=gmm.means_,
                                covariances=gmm.covariances_, weights=gmm.weights_, gmm=gmm)

    # Cap k by unique rows to avoid ConvergenceWarning from duplicate embeddings
    n_unique = len(np.unique(embeddings, axis=0))
    k_max_eff = min(k_max, n_unique - 1, n - 1)

    if n < min_samples or k_max_eff < k_min:
        k_range = [1]
    else:
        k_range = list(range(k_min, k_max_eff + 1))
        if not k_range:
            k_range = [1]

    best_bic = np.inf
    best_gmm = None
    best_k = k_range[0]

    for k in k_range:
        try:
            gmm = GaussianMixture(
                n_components=k,
                covariance_type=covariance_type,
                random_state=random_state,
                n_init=n_init,
                max_iter=200,
            )
            gmm.fit(embeddings)
            bic = gmm.bic(embeddings)
            if bic < best_bic:
                best_bic = bic
                best_gmm = gmm
                best_k = k
        except Exception:
            continue

    if best_gmm is None:
        best_gmm = GaussianMixture(n_components=1, covariance_type=covariance_type, random_state=random_state)
        best_gmm.fit(embeddings)
        best_k = 1

    return UserPersonaModel(
        user_id=user_id,
        k=best_k,
        means=best_gmm.means_,
        covariances=best_gmm.covariances_,
        weights=best_gmm.weights_,
        gmm=best_gmm,
    )


def bic_curve(
    embeddings: np.ndarray,
    k_min: int = 1,
    k_max: int = 5,
    covariance_type: str = "diag",
    n_init: int = 1,
    random_state: int = 42,
) -> dict[int, float]:
    """Return BIC scores for each K"""
    scores = {}
    for k in range(k_min, min(k_max, len(embeddings) // 2) + 1):
        try:
            gmm = GaussianMixture(
                n_components=k,
                covariance_type=covariance_type,
                random_state=random_state,
                n_init=n_init,
            )
            gmm.fit(embeddings)
            scores[k] = gmm.bic(embeddings)
        except Exception:
            pass
    return scores
