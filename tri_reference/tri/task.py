"""Regenerative XOR task as event streams.

Trial n: bits (b1, b2) ~ Uniform{0,1}^2, label y = b1 XOR b2.
Each trial lasts T ticks of the common event index.  Four input modules use a
one-hot code per bit:  I1 <- b1, I2 <- not b1, I3 <- b2, I4 <- not b2
(with swap_b2=True: I3 <- not b2, I4 <- b2, i.e. real concept drift).
An input module whose code is active fires at each tick with probability
r_on, otherwise with probability r_off (background events).
The desired output event stream is d(k) = y for ticks k >= readout_start of
the trial and 0 before.
"""
from __future__ import annotations
import numpy as np


def regime_of_trial(n, block, regimes):
    return (n // block) % len(regimes)


def generate(n_trials, T, block, regimes, rng):
    bits = rng.integers(0, 2, size=(n_trials, 2))
    y = bits[:, 0] ^ bits[:, 1]
    reg = np.array([regime_of_trial(n, block, regimes) for n in range(n_trials)])
    U = rng.random((n_trials, T, 4))           # pre-drawn: identical across arms
    X = np.zeros((n_trials, T, 4), dtype=np.int8)
    for n in range(n_trials):
        R = regimes[reg[n]]
        b1, b2 = bits[n]
        if R.get("swap_b2", False):
            code = np.array([b1, 1 - b1, 1 - b2, b2])
        else:
            code = np.array([b1, 1 - b1, b2, 1 - b2])
        p = np.where(code == 1, R["r_on"], R["r_off"])
        X[n] = (U[n] < p[None, :]).astype(np.int8)
    return dict(bits=bits, y=y, regime=reg, X=X)


def desired_stream(y_trial, T, readout_start):
    d = np.zeros(T)
    d[readout_start:] = y_trial
    return d
