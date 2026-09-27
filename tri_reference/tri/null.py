"""No-dependence references for tau_s.

1. Circular-shift surrogates of OBSERVED streams: each row except the first is
   rotated by an independent random lag in [m, W-m].  This keeps every
   module's marginal firing distribution and (cyclic) autocorrelation and
   destroys cross-module alignment.
2. Synthetic reference streams: Bernoulli streams with prescribed marginal
   rates and optional within-stream persistence, either independent (null) or
   driven by a shared latent state (dependence of known strength).
3. Analytical approximations:
   - binary streams (bin width 1): tau_b equals the phi coefficient, so under
     independence SD(tau_b) ~ 1/sqrt(W) and SD(tau_s) ~ 1/sqrt(W*M), M = N(N-1)/2
     (pairs are uncorrelated under mutual independence).
   - untied continuous streams: Var(tau) = 2(2W+5) / (9W(W-1))  (Kendall).
   These hold for stationary, serially independent streams only.
"""
from __future__ import annotations
import numpy as np
from .tau import systemic_tau


def circular_shift_surrogate(Z, rng, min_shift=1):
    Z = np.asarray(Z)
    N, W = Z.shape
    out = Z.copy()
    hi = W - min_shift
    if hi < min_shift:
        raise ValueError("window too short for requested min_shift")
    for i in range(1, N):
        out[i] = np.roll(Z[i], int(rng.integers(min_shift, hi + 1)))
    return out


def surrogate_null(Z, n_surr, rng, min_shift=1):
    return np.array([systemic_tau(circular_shift_surrogate(Z, rng, min_shift))
                     for _ in range(n_surr)])


def analytic_null_sd(W, N, kind="binary"):
    M = N * (N - 1) / 2
    if kind == "binary":
        return 1.0 / np.sqrt(W * M)
    if kind == "untied":
        return np.sqrt(2 * (2 * W + 5) / (9 * W * (W - 1)) / M)
    raise ValueError(kind)


def _latent(W, rng, persist):
    """Binary Markov latent with P(stay) = persist (0.5 = serially independent)."""
    c = np.empty(W, dtype=np.int8)
    c[0] = rng.integers(0, 2)
    flips = rng.random(W) >= persist
    for k in range(1, W):
        c[k] = c[k - 1] ^ flips[k]
    return c


def simulate_streams(rates, W, rng, coupling=0.0, persist=0.5, independent=False):
    """Bernoulli streams with marginal rate p_i: P(fire) = p_i (1 + a (2c - 1)).

    independent=True gives each module its own latent (same marginal and
    autocorrelation, no cross-dependence): the reference null.
    """
    rates = np.asarray(rates, float)
    N = len(rates)
    a = float(coupling)
    if np.any(rates * (1 + a) > 1):
        raise ValueError("rates*(1+coupling) must be <= 1")
    if independent:
        C = np.vstack([_latent(W, rng, persist) for _ in range(N)])
    else:
        C = np.tile(_latent(W, rng, persist), (N, 1))
    P = rates[:, None] * (1 + a * (2 * C - 1))
    return (rng.random((N, W)) < P).astype(float)


def null_curve(rates, W_grid, n_sim, rng, persist=0.5, coupling=0.0):
    """tau_s distribution for independent streams with the same marginals."""
    rows = []
    for W in W_grid:
        v, und = [], []
        for _ in range(n_sim):
            Z = simulate_streams(rates, W, rng, coupling, persist, independent=True)
            ts, d = systemic_tau(Z, return_details=True)
            v.append(ts); und.append(d["frac_undefined"])
        v = np.array(v)
        rows.append(dict(W=int(W), mean=float(v.mean()), sd=float(v.std(ddof=1)),
                         q025=float(np.quantile(v, .025)), q95=float(np.quantile(v, .95)),
                         q975=float(np.quantile(v, .975)), frac_undefined=float(np.mean(und)),
                         sd_analytic_binary=float(analytic_null_sd(W, len(rates)))))
    return rows


def bias_curve(rates, coupling, W_grid, n_sim, rng, persist=0.5, W_ref=2000, n_ref=50):
    """Small-window bias: E[tau_s(W)] - E[tau_s(W_ref)] for dependent streams."""
    ref = np.mean([systemic_tau(simulate_streams(rates, W_ref, rng, coupling, persist))
                   for _ in range(n_ref)])
    rows = []
    for W in W_grid:
        v = np.array([systemic_tau(simulate_streams(rates, W, rng, coupling, persist))
                      for _ in range(n_sim)])
        rows.append(dict(W=int(W), mean=float(v.mean()), ref=float(ref), bias=float(v.mean() - ref),
                         rel_bias=float((v.mean() - ref) / ref) if ref != 0 else float("nan")))
    return rows


def w_min(rates, coupling, W_grid, n_sim, rng, persist=0.5, alpha=0.05, power=0.8):
    """Smallest W in W_grid at which tau_s under dependence exceeds the
    (1-alpha) quantile of the independence null with probability >= power."""
    table = []
    found = None
    for W in W_grid:
        null = np.array([systemic_tau(simulate_streams(rates, W, rng, coupling, persist, True))
                         for _ in range(n_sim)])
        alt = np.array([systemic_tau(simulate_streams(rates, W, rng, coupling, persist))
                        for _ in range(n_sim)])
        q = np.quantile(null, 1 - alpha)
        pw = float(np.mean(alt > q))
        table.append(dict(W=int(W), null_q=float(q), power=pw))
        if found is None and pw >= power:
            found = int(W)
    return found, table
