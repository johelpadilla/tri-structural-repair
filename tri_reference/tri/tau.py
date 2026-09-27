"""Systemic Tau (tau_s) on a common discrete event index.

Each module i contributes a series z_i(k), k = 1..W, sampled on the SAME
event index (e.g. spike counts per bin, or a state variable).  For a pair
(i, j) we use Kendall's tau_b, which handles ties:

    tau_b = (n_c - n_d) / sqrt((n_0 - n_1)(n_0 - n_2))

Equivalently tau_b is the cosine between the "sign vectors"
s_i[k,l] = sign(z_i(l) - z_i(k)), k < l.  Hence the matrix of pairwise tau_b
is a Gram matrix (positive semidefinite), so the mean pairwise value
tau_s >= -1/(N-1).

Convention: if a series is constant in the window (all pairs tied), tau_b is
undefined; we set it to 0 and report the fraction of undefined pairs.
"""
from __future__ import annotations
import numpy as np
from functools import lru_cache


@lru_cache(maxsize=32)
def _pairs(W: int):
    k, l = np.triu_indices(W, 1)
    return k, l


def sign_vectors(Z: np.ndarray) -> np.ndarray:
    Z = np.asarray(Z, dtype=float)
    k, l = _pairs(Z.shape[1])
    return np.sign(Z[:, l] - Z[:, k])


def tau_b_matrix(Z: np.ndarray):
    """Return (T, defined): pairwise Kendall tau_b matrix and definedness mask.

    Z has shape (N, W): N modules, W points of the common event index.
    """
    Z = np.asarray(Z, dtype=float)
    if Z.size and np.all((Z == 0) | (Z == 1)):
        return _tau_b_binary(Z)
    S = sign_vectors(Z)
    G = S @ S.T                      # n_c - n_d off-diagonal; untied-pair counts on diagonal
    nrm = np.sqrt(np.clip(np.diag(G), 0, None))
    den = np.outer(nrm, nrm)
    defined = den > 0
    T = np.zeros_like(G)
    np.divide(G, den, out=T, where=defined)
    np.fill_diagonal(T, 1.0)
    return T, defined


def _tau_b_binary(Z):
    """Exact fast path: for 0/1 series Kendall tau_b equals the phi coefficient
    (Pearson correlation of the indicators); cost O(N^2 W) instead of O(N^2 W^2)."""
    Zc = Z - Z.mean(axis=1, keepdims=True)
    G = Zc @ Zc.T
    nrm = np.sqrt(np.clip(np.diag(G), 0, None))
    den = np.outer(nrm, nrm)
    defined = den > 1e-12
    T = np.zeros_like(G)
    np.divide(G, den, out=T, where=defined)
    np.fill_diagonal(T, 1.0)
    return T, defined


def kendall_tau_b(x, y) -> float:
    T, d = tau_b_matrix(np.vstack([x, y]))
    return float(T[0, 1]) if d[0, 1] else 0.0


def systemic_tau(Z: np.ndarray, alive=None, return_details: bool = False):
    """tau_s = mean of tau_b over all unordered pairs of (alive) modules."""
    Z = np.asarray(Z, dtype=float)
    if alive is not None:
        Z = Z[np.asarray(alive, dtype=bool)]
    N = Z.shape[0]
    if N < 2:
        raise ValueError("tau_s needs at least two modules")
    T, d = tau_b_matrix(Z)
    iu = np.triu_indices(N, 1)
    ts = float(T[iu].mean())
    if not return_details:
        return ts
    return ts, {"T": T, "defined": d, "frac_undefined": float(1 - d[iu].mean()),
                "lower_bound": -1.0 / (N - 1)}
