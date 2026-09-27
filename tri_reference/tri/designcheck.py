"""Pre-specified design check for the tau_s window (run on TUNING seeds only).

For each candidate (W ticks, bin width b) in the pre-specified order, the
intact_none arm is simulated once per tuning seed and, over the calibration
period, we compute the baseline mean of tau_s and the circular-shift null
(shift >= one trial; window and surrogate counts are those in the config).
The first candidate for which the baseline mean lies
outside the central 95% null interval in >= 80% of tuning seeds is recorded
(criterion F5). Amendment 0.1.2 fixes the confirmatory window at the value in
the configuration; this check does not replace it, and a signed mean that
stays inside its null does not void the baseline-edge alarm. No outcome
(post-lesion) data are used.
"""
from __future__ import annotations
import json, math
import numpy as np
from .sim import run_deg_arm, rngs_for
from .tau import systemic_tau
from .null import circular_shift_surrogate

CANDIDATES = [(140, 1), (280, 1), (280, 2), (560, 7), (700, 14), (1400, 14)]


def binned(hist_rows, e, W, b):
    Z = hist_rows[:, e - W:e].astype(float)
    return Z.reshape(Z.shape[0], W // b, b).sum(axis=2) if b > 1 else Z


def check_seed(cfg, seed, candidates=CANDIDATES, n_windows=None, n_surr=None):
    nc = cfg["null"]
    if n_windows is None:
        n_windows = int(nc["n_windows"])
    if n_surr is None:
        n_surr = int(nc["n_surrogates"])
    r = run_deg_arm(cfg, "E1", "intact_none", seed, keep_hist=True)
    T, tl = cfg["task"]["T"], cfg["timeline"]
    H = r["hist"][r["alive_final"]]
    rng = rngs_for(seed)["surrogate"]
    out = []
    for W, b in candidates:
        ends = np.arange(tl["cal_start"] * T + T, tl["cal_end"] * T + 1, T)
        ends = ends[ends >= W]
        base = np.array([systemic_tau(binned(H, e, W, b)) for e in ends])
        pick = rng.choice(ends, size=min(n_windows, len(ends)), replace=False)
        ms = max(1, math.ceil(T / b))
        nul = np.array([systemic_tau(circular_shift_surrogate(binned(H, e, W, b), rng, ms))
                        for e in pick for _ in range(n_surr)])
        q = np.quantile(nul, [.025, .975])
        out.append(dict(W=W, b=b, base_mu=float(base.mean()), base_sd=float(base.std(ddof=1)),
                        null_mu=float(nul.mean()), null_q025=float(q[0]), null_q975=float(q[1]),
                        passes=bool(base.mean() < q[0] or base.mean() > q[1])))
    return out


def run(cfg, seeds, out_path):
    per = {s: check_seed(cfg, s) for s in seeds}
    chosen = None
    summary = []
    for k, (W, b) in enumerate(CANDIDATES):
        frac = float(np.mean([per[s][k]["passes"] for s in seeds]))
        summary.append(dict(W=W, b=b, pass_fraction=frac))
        if chosen is None and frac >= 0.8:
            chosen = dict(W=W, bin=b)
    rep = dict(rule="first candidate with baseline outside null 95% in >= 80% of tuning seeds",
               candidates=summary, chosen=chosen, per_seed={str(s): v for s, v in per.items()})
    json.dump(rep, open(out_path, "w"), indent=1)
    return rep
