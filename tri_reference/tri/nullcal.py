"""Pre-specified null calibration of tau_s (independent of the XOR task).

  python -m tri.nullcal --out results_full/nullcal.json [--n-sim 2000]
Reports, for N = 7 modules and several marginal firing rates:
  (a) the tau_s distribution for independent streams with the same marginals
      (serially independent and persistent), with the analytic 1/sqrt(W M) SD;
  (b) the small-window bias E[tau_s(W)] - E[tau_s(2000)] under dependence;
  (c) W_min: smallest W at which dependent streams exceed the null 95th
      percentile with probability >= 0.8.
"""
from __future__ import annotations
import argparse, json
import numpy as np
from .null import null_curve, bias_curve, w_min

W_GRID = [10, 20, 50, 100, 200, 400]
RATE_SETS = {"sparse": [0.02] * 7, "moderate": [0.1] * 7, "dense": [0.25] * 7,
             "mixed": [0.5, 0.5, 0.5, 0.5, 0.1, 0.1, 0.05]}
COUPLINGS = [0.1, 0.2, 0.4]
PERSIST = [0.5, 0.9]


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--n-sim", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=20260927)
    a = ap.parse_args(argv)
    rng = np.random.default_rng(a.seed)
    rep = dict(W_grid=W_GRID, n_sim=a.n_sim, null={}, bias={}, w_min={})
    for name, rates in RATE_SETS.items():
        for ps in PERSIST:
            key = f"{name}|persist={ps}"
            rep["null"][key] = null_curve(rates, W_GRID, a.n_sim, rng, persist=ps)
            for cp in COUPLINGS:
                if max(rates) * (1 + cp) > 1:
                    continue
                k2 = f"{key}|coupling={cp}"
                rep["bias"][k2] = bias_curve(rates, cp, W_GRID, max(50, a.n_sim // 4), rng, persist=ps)
                rep["w_min"][k2] = w_min(rates, cp, W_GRID, max(50, a.n_sim // 4), rng, persist=ps)
    json.dump(rep, open(a.out, "w"), indent=1)
    print(f"wrote {a.out}")


if __name__ == "__main__":
    main()
