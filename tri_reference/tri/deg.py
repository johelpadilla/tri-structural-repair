"""Dynamic Event Graph (DEG) of discrete-time leaky integrate-and-fire modules.

Module j (non-input) at tick k:
    u_j(k) = lam * u_j(k-1) + sum_i W[i, j] * s_i(k-1) + sigma * xi_j(k)
    s_j(k) = 1[u_j(k) >= theta_j];  if s_j(k) = 1 then u_j(k) <- 0
Input modules copy the task's input events.  Edges carry fixed weights
+-w0 (excitatory / inhibitory); structure and thresholds change, weights do
not.  Edges are admissible only from a lower to a higher layer
(input 0 < hidden 1 < bridge (between parents) < output 2): the DEG stays
feed-forward, so every event reaches the output in a bounded number of ticks.
"""
from __future__ import annotations
import numpy as np

KIND_IN, KIND_HID, KIND_OUT, KIND_BRIDGE = 0, 1, 2, 3


class DEG:
    def __init__(self, cfg, rng_theta):
        self.cfg = cfg
        n_in, n_hid, n_out = cfg["n_in"], cfg["n_hid"], cfg["n_out"]
        self.nmax = n_in + n_hid + n_out + cfg["max_bridges"]
        self.kind = np.full(self.nmax, -1)
        self.layer = np.zeros(self.nmax)
        self.alive = np.zeros(self.nmax, bool)
        self.born = np.zeros(self.nmax, int)
        self.theta = np.zeros(self.nmax)
        self.u = np.zeros(self.nmax)
        self.s = np.zeros(self.nmax)
        self.W = np.zeros((self.nmax, self.nmax))
        self.names = [""] * self.nmax
        idx = 0
        for a in range(n_in):
            self._add_module(idx, KIND_IN, 0.0, f"I{a+1}"); idx += 1
        for a in range(n_hid):
            self._add_module(idx, KIND_HID, 1.0, f"H{a+1}"); idx += 1
        for a in range(n_out):
            self._add_module(idx, KIND_OUT, 2.0, "O" if n_out == 1 else f"O{a+1}"); idx += 1
        lo, hi = cfg["theta_init_hidden"]
        for j in np.where(self.kind == KIND_HID)[0]:
            self.theta[j] = rng_theta.uniform(lo, hi)
        lo, hi = cfg["theta_init_output"]
        for j in np.where(self.kind == KIND_OUT)[0]:
            self.theta[j] = rng_theta.uniform(lo, hi)
        w0 = cfg["w0"]
        for (a, b, sgn) in cfg["initial_edges"]:
            self.W[self.index(a), self.index(b)] = sgn * w0
        self.syn_ops = 0
        self.last_spike_trial = {}

    # --- bookkeeping -------------------------------------------------------
    def _add_module(self, idx, kind, layer, name, born=0):
        self.kind[idx] = kind; self.layer[idx] = layer; self.alive[idx] = True
        self.names[idx] = name; self.born[idx] = born
        self.u[idx] = 0.0; self.s[idx] = 0.0

    def index(self, name):
        return self.names.index(name)

    @property
    def inputs(self):
        return np.where(self.alive & (self.kind == KIND_IN))[0]

    @property
    def output(self):
        return int(np.where(self.kind == KIND_OUT)[0][0])

    def plastic(self):
        return np.where(self.alive & (self.kind != KIND_IN))[0]

    def edges(self):
        return [(int(i), int(j)) for i, j in zip(*np.nonzero(self.W))]

    def n_edges(self):
        return int(np.count_nonzero(self.W))

    def admissible(self, i, j):
        return (self.alive[i] and self.alive[j] and i != j and self.kind[j] != KIND_IN
                and self.layer[i] < self.layer[j])

    # --- dynamics ----------------------------------------------------------
    def reset_state(self):
        self.u[:] = 0.0; self.s[:] = 0.0

    def step(self, x_in, noise):
        cfg = self.cfg
        drive = self.s @ self.W
        self.syn_ops += int(np.count_nonzero(self.W[self.s > 0]))
        u = cfg["leak"] * self.u + drive + cfg["noise_sd"] * noise
        s = np.zeros(self.nmax)
        nonin = self.alive & (self.kind != KIND_IN)
        fire = nonin & (u >= self.theta)
        s[fire] = 1.0
        u[fire] = 0.0
        u[~self.alive] = 0.0
        s[self.inputs] = x_in
        self.u, self.s = u, s
        return s

    # --- structural operations --------------------------------------------
    def add_edge(self, i, j, sgn):
        if not self.admissible(i, j):
            return False
        self.W[i, j] = sgn * self.cfg["w0"]
        return True

    def remove_edge(self, i, j):
        had = self.W[i, j] != 0
        self.W[i, j] = 0.0
        return bool(had)

    def lesion_edges(self, frac, rng):
        E = self.edges()
        k = int(np.floor(frac * len(E)))
        pick = rng.choice(len(E), size=k, replace=False)
        removed = [E[p] for p in sorted(pick)]
        for (i, j) in removed:
            self.W[i, j] = 0.0
        return removed

    def spawn_bridge(self, i, j, sgn_out, t_born, theta):
        free = np.where(~self.alive & (self.kind == -1))[0]
        if len(free) == 0:
            free = np.where(~self.alive & (self.kind == KIND_BRIDGE))[0]
        if len(free) == 0:
            return None
        a, b = (i, j) if self.layer[i] < self.layer[j] else (j, i)
        if self.layer[a] == self.layer[b] or self.kind[b] == KIND_IN:
            return None
        k = int(free[0])
        self.W[k, :] = 0.0; self.W[:, k] = 0.0
        self._add_module(k, KIND_BRIDGE, 0.5 * (self.layer[a] + self.layer[b]), f"B{k}", born=t_born)
        self.theta[k] = theta
        self.last_spike_trial.pop(k, None)
        self.W[a, k] = +self.cfg["w0"]
        self.W[k, b] = sgn_out * self.cfg["w0"]
        return k

    def remove_module(self, k):
        if self.kind[k] in (KIND_IN, KIND_OUT):
            return False
        self.alive[k] = False
        self.W[k, :] = 0.0; self.W[:, k] = 0.0
        self.u[k] = 0.0; self.s[k] = 0.0
        return True
