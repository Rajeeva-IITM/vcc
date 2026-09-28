"""Kernel-ridge regression from a perturbation embedding to a gene-space response.

Shared by the model-free baselines (`scripts/baseline_poincare_ridge*.py`). Two fit paths,
one contract (`(embeddings, targets) -> predictions`):

* :func:`fit_kernel_ridge` -- small `n` (a few hundred). Forms the full inverse `(K+lambda
  I)^{-1}` and scores every `(lambda, gamma)` by closed-form leave-one-out, then returns
  `alpha` for the winner.
* :func:`fit_predict_large` -- large `n` (10^4). The full `alpha = (K+lambda I)^{-1} Y` is an
  `n x n x G` matmul (~10^15 flops at `n=11468, G=18080`) and never formed. Predictions for a
  small test set factor as `hat(Y) = [K_test (K+lambda I)^{-1}] Y`, a `(n_test x n)` solve
  followed by one `(n_test x n) @ (n x G)` product. Hyperparameters are chosen by the same LOO
  rule on a random subsample, where the full inverse is cheap.

The kernel is smooth in the embedding (RBF by default), so the map interpolates held-out
perturbations from their embedding neighbours -- the property the whole VCC transfer story
turns on.
"""

from __future__ import annotations

import numpy as np
from scipy.linalg import cho_factor, cho_solve


def sq_dists(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    """Pairwise squared Euclidean distances, `(len(A), len(B))`."""
    a2 = (A * A).sum(1)[:, None]
    b2 = (B * B).sum(1)[None, :]
    return np.maximum(a2 + b2 - 2.0 * (A @ B.T), 0.0)


def kernel(A: np.ndarray, B: np.ndarray, kind: str, gamma: float) -> np.ndarray:
    """Kernel matrix `k(A_i, B_j)` for the chosen family."""
    if kind == "rbf":
        return np.exp(-gamma * sq_dists(A, B))
    if kind == "cosine":
        An = A / np.clip(np.linalg.norm(A, axis=1, keepdims=True), 1e-12, None)
        Bn = B / np.clip(np.linalg.norm(B, axis=1, keepdims=True), 1e-12, None)
        return An @ Bn.T
    if kind == "linear":
        return A @ B.T
    raise ValueError(f"unknown kernel {kind!r}")


def specific_cosine(pred: np.ndarray, true: np.ndarray, own_col: np.ndarray) -> float:
    """Mean over rows of cos(pred_i, true_i) with each row's own-gene column removed.

    The own gene carries the knockdown itself; excluding it isolates the downstream response
    direction, which is what the DE metrics score. `own_col[i] < 0` keeps the whole row.
    """
    dot = np.einsum("ng,ng->n", pred, true)
    pn2 = np.einsum("ng,ng->n", pred, pred)
    tn2 = np.einsum("ng,ng->n", true, true)
    rows = np.flatnonzero(own_col >= 0)
    cols = own_col[rows]
    dot[rows] -= pred[rows, cols] * true[rows, cols]
    pn2[rows] -= pred[rows, cols] ** 2
    tn2[rows] -= true[rows, cols] ** 2
    denom = np.sqrt(np.clip(pn2 * tn2, 1e-24, None))
    return float(np.mean(dot / denom))


def median_gamma0(E: np.ndarray, rng: np.random.Generator, n_sub: int = 3000) -> float:
    """Median-heuristic RBF bandwidth `1 / (2 * median pairwise sq-dist)`, on a subsample."""
    if len(E) > n_sub:
        E = E[rng.choice(len(E), n_sub, replace=False)]
    d2 = sq_dists(E, E)
    med = np.median(d2[np.triu_indices(len(E), k=1)])
    return 1.0 / (2.0 * max(med, 1e-12))


def _loo_specific(K: np.ndarray, Y: np.ndarray, lam: float, own: np.ndarray) -> float:
    """Closed-form LOO specific-cosine for one `(K, lambda)`.

    With `G = (K + lambda I)^{-1}` and `alpha = G Y`, the LOO prediction for row `i` is
    `Y_i - alpha_i / G_ii`. No refitting.
    """
    n = len(K)
    G = np.linalg.inv(K + lam * np.eye(n))
    alpha = G @ Y
    loo = Y - alpha / np.clip(np.diag(G)[:, None], 1e-12, None)
    return specific_cosine(loo, Y, own)


def fit_kernel_ridge(
    E_train: np.ndarray,
    Y: np.ndarray,
    own_col: np.ndarray,
    kind: str,
    lambdas: list[float],
    gamma_scales: list[float],
    seed: int = 0,
) -> tuple[np.ndarray, float, dict]:
    """Small-`n` fit: pick `(lambda, gamma)` by full LOO, return `alpha`, `gamma`, report."""
    rng = np.random.default_rng(seed)
    gamma0 = median_gamma0(E_train, rng)
    scales = gamma_scales if kind == "rbf" else [1.0]

    best = (-2.0, None, None, None)
    grid = []
    for scale in scales:
        gamma = gamma0 * scale
        K = kernel(E_train, E_train, kind, gamma)
        for lam in lambdas:
            score = _loo_specific(K, Y, lam, own_col)
            grid.append((kind, scale, lam, score))
            if score > best[0]:
                G = np.linalg.inv(K + lam * np.eye(len(K)))
                best = (score, G @ Y, gamma, (lam, scale))
    score, alpha, gamma, (lam, scale) = best
    return (
        alpha,
        gamma,
        {
            "kernel": kind,
            "gamma": gamma,
            "gamma_scale": scale,
            "lambda": lam,
            "loo_specific_cosine": score,
            "grid": grid,
        },
    )


def fit_predict_large(
    E_train: np.ndarray,
    Y: np.ndarray,
    E_test: np.ndarray,
    own_col: np.ndarray,
    kind: str,
    lambdas: list[float],
    gamma_scales: list[float],
    seed: int = 0,
    n_sub: int = 3000,
    log=None,
) -> tuple[np.ndarray, dict]:
    """Large-`n` fit: LOO on a subsample, then predict the test set without forming `alpha`.

    Returns `(Y_hat_test, report)` where `Y_hat_test` is `(len(E_test), Y.shape[1])`.
    """

    def _say(msg: str) -> None:
        if log is not None:
            log(msg)

    n = len(E_train)
    rng = np.random.default_rng(seed)
    gamma0 = median_gamma0(E_train, rng)
    scales = gamma_scales if kind == "rbf" else [1.0]

    # ---- hyperparameters by LOO on a random subsample (full inverse is cheap there) ----
    sub = rng.choice(n, min(n_sub, n), replace=False)
    Es, Ys, owns = E_train[sub], Y[sub], own_col[sub]
    best, grid = (-2.0, None, None), []
    for scale in scales:
        Ksub = kernel(Es, Es, kind, gamma0 * scale)
        for lam in lambdas:
            score = _loo_specific(Ksub, Ys, lam, owns)
            grid.append((kind, scale, lam, score))
            if score > best[0]:
                best = (score, lam, scale)
    score, lam, scale = best
    gamma = gamma0 * scale
    _say(
        f"HP by LOO on {len(sub):,}-pt subsample: {kind} lambda={lam:g} "
        f"scale={scale:g} -> specific-cos {score:+.4f}"
    )

    # ---- final: factor (K + lambda I) once, solve for the test rows, then W @ Y ----
    _say(f"building {n:,}x{n:,} kernel and factoring")
    K = kernel(E_train, E_train, kind, gamma)
    K[np.diag_indices_from(K)] += lam
    c = cho_factor(K, lower=True, overwrite_a=True)
    Wt = cho_solve(c, kernel(E_test, E_train, kind, gamma).T)  # (n, n_test)
    _say(f"projecting to {len(E_test)} targets x {Y.shape[1]:,} genes")
    Y_hat = (Wt.T @ Y).astype(np.float32)  # (n_test, G)
    return Y_hat, {
        "kernel": kind,
        "gamma": gamma,
        "gamma_scale": scale,
        "lambda": lam,
        "loo_specific_cosine": score,
        "grid": grid,
        "n_train": n,
    }
