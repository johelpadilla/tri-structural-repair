"""LSTM baselines (PyTorch, CPU).

lstm      online prequential training (predict, then one Adam step per trial);
          at the lesion trial a random 50% of every weight matrix is zeroed and
          kept at zero (masked); training continues = "retrained after lesion".
lstm_ewc  same, plus Elastic Weight Consolidation (Kirkpatrick et al. 2017):
          at the end of every drift block the diagonal empirical Fisher is
          estimated on the last `fisher_trials` trials and a quadratic penalty
          (lambda/2) * sum_k F_k (w_k - w*_k)^2 is added (Fishers accumulate).
Approximate cost is counted as multiply-accumulates (MACs): forward per tick
4H(I+H) + H; a training step is counted as 3x forward (forward + backward).
"""
from __future__ import annotations
import numpy as np

try:
    import torch
    import torch.nn as nn
    HAVE_TORCH = True
except Exception:  # pragma: no cover
    HAVE_TORCH = False


def run_lstm_arm(cfg, experiment, arm, seed):
    if not HAVE_TORCH:
        raise RuntimeError("PyTorch is required for LSTM baselines: pip install torch")
    from .sim import rngs_for
    from .task import generate
    tk, tl, lc = cfg["task"], cfg["timeline"], cfg["lstm"]
    R = rngs_for(seed)
    regimes = tk["E1_regimes"] if experiment == "E1" else tk["E2_regimes"]
    n_trials, T = tk["n_trials"], tk["T"]
    data = generate(n_trials, T, tk["block"], regimes, R["task"])
    X = torch.tensor(data["X"], dtype=torch.float32)
    y = torch.tensor(data["y"], dtype=torch.float32)
    torch.manual_seed(int(R["lstm"].integers(2**31 - 1)))
    torch.set_num_threads(1)
    H, I = lc["hidden"], 4
    lstm, head = nn.LSTM(I, H, batch_first=True), nn.Linear(H, 1)
    params = list(lstm.parameters()) + list(head.parameters())
    opt = torch.optim.Adam(params, lr=lc["lr"])
    lossf = nn.BCEWithLogitsLoss()
    masks = None
    lesion_trial = tl["lesion_trial"] if experiment == "E1" else None
    use_ewc = arm == "lstm_ewc"
    lam = float(lc["ewc_lambda"])
    fisher = [torch.zeros_like(p) for p in params]
    anchor = [p.detach().clone() for p in params]
    have_anchor = False
    n_par = sum(p.numel() for p in params)
    mac_fwd = T * 4 * H * (I + H) + H
    correct = np.zeros(n_trials, dtype=np.int8)
    macs = np.zeros(n_trials, dtype=np.int64)
    total = 0
    for n in range(n_trials):
        if lesion_trial is not None and n == lesion_trial:
            g = torch.Generator().manual_seed(int(R["lesion"].integers(2**31 - 1)))
            masks = []
            with torch.no_grad():
                for p in params:
                    if p.dim() >= 2:
                        m = (torch.rand(p.shape, generator=g) >= lc["lesion_frac"]).float()
                        p.mul_(m)
                    else:
                        m = torch.ones_like(p)
                    masks.append(m)
        xb = X[n:n + 1]
        h, _ = lstm(xb)
        logit = head(h[:, -1, :]).squeeze()
        correct[n] = int((logit.item() > 0) == bool(y[n].item()))
        loss = lossf(logit, y[n])
        if use_ewc and have_anchor:
            pen = sum((f * (p - a) ** 2).sum() for f, p, a in zip(fisher, params, anchor))
            loss = loss + 0.5 * lam * pen
            total += 2 * n_par
        opt.zero_grad(); loss.backward()
        if masks is not None:
            for p, m in zip(params, masks):
                if p.grad is not None:
                    p.grad.mul_(m)
        opt.step()
        if masks is not None:
            with torch.no_grad():
                for p, m in zip(params, masks):
                    p.mul_(m)
        total += 3 * mac_fwd
        macs[n] = total
        if use_ewc and (n + 1) % tk["block"] == 0:
            nf = min(lc["fisher_trials"], n + 1)
            F = [torch.zeros_like(p) for p in params]
            for q in range(n + 1 - nf, n + 1):
                hq, _ = lstm(X[q:q + 1]); lq = lossf(head(hq[:, -1, :]).squeeze(), y[q])
                gr = torch.autograd.grad(lq, params)
                for Fi, gi in zip(F, gr):
                    Fi += gi.detach() ** 2 / nf
            total += nf * 3 * mac_fwd
            fisher = [f + Fi for f, Fi in zip(fisher, F)]
            anchor = [p.detach().clone() for p in params]
            have_anchor = True
    return dict(experiment=experiment, arm=arm, seed=int(seed), correct=correct, macs=macs,
                lesion_trial=lesion_trial, regime=data["regime"], n_trials=n_trials, n_params=n_par)
