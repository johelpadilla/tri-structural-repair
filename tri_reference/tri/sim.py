"""Simulation of one DEG arm (TRI, ablations, intact references).

Arms
  tri            tau_s-triggered (two-sided baseline band), targeted regeneration
  tri_onesided   as tri, but triggers only when tau_s < mu - kappa*sigma
  tri_matrix     as tri, triggered by D(t) = mean |T_ij(t) - T_ij_base| (cancellation-proof)
  random_rewire  yoked: same trigger trials and same operation types as the tri run
                 of the same seed, random targets
  random_time    targeted regeneration, same NUMBER of triggers as the tri run of the
                 same seed, trigger trials drawn uniformly at random
  none           lesion, no structural control (threshold plasticity only)
  perf           targeted regeneration triggered by running error rate (uses labels)
  intact_tri     no lesion, tri controller on (false alarms / cost when nothing is wrong)
  intact_none    no lesion, no structural control (ceiling reference)
All arms share the module model, OEC threshold plasticity, the input sequence,
the module noise and the lesion (common random numbers per seed).
"""
from __future__ import annotations
import numpy as np
from .deg import DEG, KIND_IN, KIND_OUT, KIND_BRIDGE, KIND_HID
from .task import generate
from .tau import tau_b_matrix, kendall_tau_b
from .null import circular_shift_surrogate
from .tau import systemic_tau

STREAMS = ["task", "lesion", "noise", "theta", "oec", "ctrl", "surrogate", "lstm"]


def rngs_for(seed):
    ss = np.random.SeedSequence([int(seed), 20260927])
    return {name: np.random.default_rng(s) for name, s in zip(STREAMS, ss.spawn(len(STREAMS)))}


def window_matrix(hist, g, W, b):
    Z = hist[:, g - W:g].astype(float)
    if b > 1:
        Z = Z[:, : (W // b) * b].reshape(Z.shape[0], W // b, b).sum(axis=2)
    return Z


def pair_matrix(Z, alive):
    T, d = tau_b_matrix(Z)
    T[~alive, :] = 0.0; T[:, ~alive] = 0.0
    return T


def tau_s_from(T, alive):
    idx = np.where(alive)[0]
    sub = T[np.ix_(idx, idx)]
    iu = np.triu_indices(len(idx), 1)
    return float(sub[iu].mean())


def regulated_alive(alive, base):
    """Modules that enter the regulated mean of tau_s.

    Before calibration every live module is included. Afterwards only modules
    that were alive at baseline are included. A bridge spawned later stays out
    of the mean, so a silent bridge cannot pull tau_s toward zero by adding
    undefined pairs.
    """
    alive = np.asarray(alive, dtype=bool)
    if base is None:
        return alive
    return np.asarray(base["alive"], dtype=bool) & alive


def run_deg_arm(cfg, experiment, arm, seed, yoke=None, keep_hist=False):
    tk, md, oc, tc, tl, cc, nc = (cfg["task"], cfg["model"], cfg["oec"], cfg["tau"],
                                  cfg["timeline"], cfg["controller"], cfg["null"])
    R = rngs_for(seed)
    regimes = tk["E1_regimes"] if experiment == "E1" else tk["E2_regimes"]
    n_trials, T, rs = tk["n_trials"], tk["T"], tk["readout_start"]
    data = generate(n_trials, T, tk["block"], regimes, R["task"])
    X, y = data["X"], data["y"]
    deg = DEG(md, R["theta"])
    nmax = deg.nmax
    noise = R["noise"].standard_normal((n_trials * T, nmax))
    n_oec = n_trials // oc["block_trials"] + 1
    oec_u, oec_z = R["oec"].random(n_oec), R["oec"].standard_normal(n_oec)
    lesion_rng = R["lesion"]
    ctrl_rng = R["ctrl"]
    W, b = tc["W"], tc["bin"]
    lesion_trial = tl["lesion_trial"] if (experiment == "E1" and not arm.startswith("intact")) else None
    lo_th, hi_th = md["theta_bounds"]

    hist = np.zeros((nmax, n_trials * T), dtype=np.int8)
    correct = np.zeros(n_trials, dtype=np.int8)
    tau_s = np.full(n_trials, np.nan)
    dev_D = np.full(n_trials, np.nan)
    dev_edge = np.full(n_trials, np.nan)
    n_edges = np.zeros(n_trials, dtype=np.int16)
    n_alive = np.zeros(n_trials, dtype=np.int16)
    syn_ops = np.zeros(n_trials, dtype=np.int64)
    tau_cost = np.zeros(n_trials, dtype=np.int64)
    ops, triggers, silent_alarms = [], [], []
    cal = dict(ts=[], T=[], D=[], err=[], rates=[])
    base = None
    refractory_until = -1
    entrain_until = -1
    pending_check = None
    lesion_removed = []
    null_info = {}

    # random_time schedule: same count as the yoked tri run, uniform over [cal_end, n_trials)
    rt_set = set()
    if arm == "random_time":
        k = int(yoke["n_triggers"]) if yoke else 0
        pool = np.arange(tl["cal_end"], n_trials)
        if k > 0:
            rt_set = set(int(v) for v in ctrl_rng.choice(pool, size=min(k, len(pool)), replace=False))
    rr_ops = {}
    if arm == "random_rewire" and yoke:
        for t0 in yoke["triggers"]:
            rr_ops.setdefault(int(t0), [])
        for o in yoke["ops"]:
            if o["type"] != "prune":
                rr_ops.setdefault(int(o["trial"]), []).append(o)

    oec_state = dict(pending=None, parent=None, k=0)
    g = 0
    for n in range(n_trials):
        if lesion_trial is not None and n == lesion_trial:
            lesion_removed = deg.lesion_edges(tl["lesion_frac"], lesion_rng)
        # ---- run one trial on the event index --------------------------------
        for t in range(T):
            s = deg.step(X[n, t], noise[g])
            hist[:, g] = s.astype(np.int8)
            g += 1
        spk = np.where(hist[:, g - T:g].any(axis=1))[0]
        for k in spk:
            deg.last_spike_trial[int(k)] = n
        out = deg.output
        c_out = int(hist[out, g - T + rs:g].sum())
        yhat = int(c_out >= tk["c_star"])
        correct[n] = int(yhat == y[n])          # prequential: predict, then adapt
        syn_ops[n] = deg.syn_ops
        n_edges[n] = deg.n_edges(); n_alive[n] = int(deg.alive.sum())
        # ---- tau_s ---------------------------------------------------------
        Tm = None
        if g >= W and n % tc["every_trials"] == 0:
            Z = window_matrix(hist, g, W, b)
            monitor = regulated_alive(deg.alive, base)
            Tm = pair_matrix(Z, monitor)
            tau_s[n] = tau_s_from(Tm, monitor)
            na = int(np.count_nonzero(monitor))
            tau_cost[n] = na * (na - 1) // 2 * (W // b)   # regulated pairs, binary/phi path
            if base is not None:
                m = base["alive"] & deg.alive
                iu = np.triu_indices(nmax, 1)
                mask = m[iu[0]] & m[iu[1]]
                dev_D[n] = float(np.abs(Tm[iu] - base["T"][iu])[mask].mean())
                present = [(i, j) for i, j, _ in base.get("edges", [])
                           if deg.alive[i] and deg.alive[j]]
                if present:
                    dev_edge[n] = float(np.mean([
                        abs(Tm[i, j] - base["T"][i, j]) for i, j in present]))
        # ---- OEC: ordinal error correction of thresholds ----------------------
        if (n + 1) % oc["block_trials"] == 0:
            B = oc["block_trials"]
            d = np.zeros(B * T)
            for q in range(B):
                d[q * T + rs:(q + 1) * T] = y[n - B + 1 + q]
            sO = hist[out, g - B * T:g].astype(float)
            score = kendall_tau_b(d, sO)
            st = oec_state
            if st["pending"] is not None:
                j, old = st["pending"]
                if score < st["parent"]:
                    deg.theta[j] = old
                    st["parent"] = (1 - oc["parent_decay"]) * st["parent"] + oc["parent_decay"] * score
                else:
                    st["parent"] = score
            else:
                st["parent"] = score
            pl = deg.plastic()
            j = int(pl[int(oec_u[st["k"]] * len(pl)) % len(pl)])
            st["pending"] = (j, deg.theta[j])
            deg.theta[j] = float(np.clip(deg.theta[j] + oc["sigma"] * oec_z[st["k"]], lo_th, hi_th))
            st["k"] += 1
        # ---- calibration -------------------------------------------------------
        err_run = 1 - correct[max(0, n - cc["perf_window_trials"] + 1):n + 1].mean()
        if tl["cal_start"] <= n < tl["cal_end"] and Tm is not None:
            cal["ts"].append(tau_s[n]); cal["T"].append(Tm); cal["err"].append(err_run)
            cal["rates"].append(hist[:, g - T:g].mean(axis=1))
        if n == tl["cal_end"] - 1:
            Tb = np.mean(cal["T"], axis=0)
            rates = np.mean(cal["rates"], axis=0)
            hid = (deg.kind == KIND_HID) & deg.alive
            base = dict(mu=float(np.mean(cal["ts"])), sd=float(np.std(cal["ts"], ddof=1)),
                        T=Tb, alive=deg.alive.copy(), rates=rates,
                        bridge_rate=float(rates[hid].mean()) if hid.any() else 0.05,
                        err_mu=float(np.mean(cal["err"])), err_sd=float(np.std(cal["err"], ddof=1)),
                        theta_bridge=float(np.median(deg.theta[hid])) if hid.any() else 1.5)
            iu = np.triu_indices(nmax, 1)
            m = base["alive"]; mask = m[iu[0]] & m[iu[1]]
            Ds = [float(np.abs(Tc[iu] - Tb[iu])[mask].mean()) for Tc in cal["T"]]
            base["D_mu"], base["D_sd"] = float(np.mean(Ds)), float(np.std(Ds, ddof=1))
            # Snapshot the wiring before any restorative action on this trial.
            base["edges"] = [(int(i), int(j), int(np.sign(deg.W[i, j])))
                             for i, j in zip(*np.nonzero(deg.W))
                             if deg.alive[i] and deg.alive[j]]
            if base["edges"]:
                De = [float(np.mean([abs(Tc[i, j] - Tb[i, j]) for i, j, _ in base["edges"]]))
                      for Tc in cal["T"]]
                base["D_edge_mu"] = float(np.mean(De))
                base["D_edge_sd"] = float(np.std(De, ddof=1))
            else:
                base["D_edge_mu"], base["D_edge_sd"] = 0.0, 0.0
            # no-dependence reference: circular-shift surrogates of calibration windows
            srng = R["surrogate"]
            ends = srng.choice(np.arange(tl["cal_start"] * T + W, tl["cal_end"] * T + 1),
                               size=nc["n_windows"], replace=False)
            nul = []
            for e in ends:
                Zc = window_matrix(hist[base["alive"]], e, W, b)
                ms = max(1, -(-nc["min_shift_ticks"] // b))       # shift >= one trial, in bins
                for _ in range(nc["n_surrogates"]):
                    nul.append(systemic_tau(circular_shift_surrogate(Zc, srng, ms)))
            nul = np.array(nul)
            null_info = dict(mean=float(nul.mean()), sd=float(nul.std(ddof=1)),
                             q025=float(np.quantile(nul, .025)), q975=float(np.quantile(nul, .975)))
            null_info["baseline_outside_null95"] = bool(base["mu"] < null_info["q025"] or
                                                        base["mu"] > null_info["q975"])
        # ---- entrainment: rate homeostasis toward baseline rates ---------------
        if base is not None and n <= entrain_until:
            r_now = hist[:, g - T:g].mean(axis=1)
            for j in deg.plastic():
                if deg.kind[j] == KIND_OUT:
                    continue
                tgt = base["bridge_rate"] if deg.kind[j] == KIND_BRIDGE else base["rates"][j]
                deg.theta[j] = float(np.clip(deg.theta[j] + cc["eta_entrain"] * (r_now[j] - tgt), lo_th, hi_th))
        # ---- structural controller --------------------------------------------
        if base is None or arm in ("none", "intact_none"):
            continue
        # follow-up check of a targeted rewiring: bridge if insufficient
        if pending_check is not None and n >= pending_check["due"] and Tm is not None:
            a, c = pending_check["pair"]
            dnow = abs(Tm[a, c] - base["T"][a, c])
            if dnow > cc["bridge_if_reduction_below"] * pending_check["d0"] and arm != "random_rewire":
                sg = 1 if base["T"][a, c] >= 0 else -1
                k = deg.spawn_bridge(a, c, sg, n, base["theta_bridge"])
                if k is not None:
                    ops.append(dict(trial=n, type="bridge", pair=[deg.names[a], deg.names[c]], new=deg.names[k], sign=int(sg)))
                    entrain_until = n + cc["entrain_trials"]
            pending_check = None
        fire = False
        if n > refractory_until and Tm is not None:
            mu, sd, kap = base["mu"], base["sd"], cc["kappa"]
            if arm in ("tri", "intact_tri"):
                fire = abs(tau_s[n] - mu) > kap * sd
            elif arm == "tri_onesided":
                fire = tau_s[n] < mu - kap * sd
            elif arm == "tri_matrix":
                fire = (np.isfinite(dev_edge[n]) and
                        dev_edge[n] > base.get("D_edge_mu", 0.0) + kap * max(base.get("D_edge_sd", 0.0), 1e-6))
            elif arm == "perf":
                fire = err_run > base["err_mu"] + kap * max(base["err_sd"], 1e-3)
        if arm == "random_time":
            fire = n in rt_set and Tm is not None
        if arm == "random_rewire":
            if n in rr_ops:
                triggers.append(n)
                for op in rr_ops[n]:
                    rec = random_op(deg, op, ctrl_rng, n, base, cc)
                    if rec: ops.append(rec)
                ops.extend(prune_silent(deg, n, cc))
                entrain_until = n + cc["entrain_trials"]
            continue
        if fire:
            # E1 only puts a cut baseline edge back. An alarm that finds the
            # baseline wiring already intact does not become a yoked repair:
            # the window can stay high after the edges are back, and copying
            # those empty alarms would hand the random-time arm extra restores.
            recs, chk = targeted_regeneration(
                deg, Tm, base, n, cc, allow_new_edges=(experiment == "E2"))
            acted = any(r.get("type") != "prune" for r in recs)
            if acted:
                triggers.append(n)
                ops.extend(recs)
                if chk is not None:
                    pending_check = chk
            else:
                silent_alarms.append(n)
            refractory_until = n + cc["refractory_trials"]
            entrain_until = n + cc["entrain_trials"]

    return dict(experiment=experiment, arm=arm, seed=int(seed), correct=correct, tau_s=tau_s,
                dev_D=dev_D, dev_edge=dev_edge, n_edges=n_edges, n_alive=n_alive,
                syn_ops=syn_ops, tau_cost=tau_cost,
                ops=ops, triggers=triggers, silent_alarms=silent_alarms,
                base=_jsonable_base(base), null=null_info,
                lesion_removed=[[deg.names[i], deg.names[j]] for i, j in lesion_removed],
                lesion_trial=lesion_trial, regime=data["regime"], n_trials=n_trials,
                **({"hist": hist, "alive_final": deg.alive.copy()} if keep_hist else {}))


def _jsonable_base(base):
    if base is None:
        return None
    return {k: (float(v) if isinstance(v, (float, np.floating)) else None)
            for k, v in base.items() if k in ("mu", "sd", "err_mu", "err_sd", "D_mu", "D_sd", "bridge_rate", "theta_bridge")}


def _restore_baseline_edge(deg, Tm, base, n):
    """Put back one baseline edge that is currently missing.

    The edge restored is the missing baseline edge with the largest drop in
    absolute concordance. Pairs that were not edges at calibration are not
    candidates. The sign is the sign stored at calibration.
    """
    edges = base.get("edges") or []
    Tb = base.get("T")
    if Tb is None or not edges:
        return None
    best = None
    for i, j, sg in edges:
        if not (deg.alive[i] and deg.alive[j]) or deg.W[i, j] != 0:
            continue
        if not deg.admissible(i, j):
            continue
        loss = abs(float(Tb[i, j])) - abs(float(Tm[i, j]))
        if best is None or loss > best[0]:
            best = (loss, int(i), int(j), int(sg))
    if best is None:
        return None
    _, i, j, sg = best
    if not deg.add_edge(i, j, sg):
        return None
    return dict(trial=n, type="add", pair=[deg.names[i], deg.names[j]], sign=sg, restore=True)


def targeted_regeneration(deg, Tm, base, n, cc, allow_new_edges=True):
    """One regenerative event.

    A missing baseline edge is restored before any new edge is considered.
    In the lesion experiment new edges are not allowed: once the baseline
    wiring is complete the controller waits, so a lagging window cannot
    recruit a pair that was never an edge. Concept drift (E2) may still
    add or bridge.
    """
    recs = []
    restored = _restore_baseline_edge(deg, Tm, base, n)
    if restored is not None:
        recs.append(restored)
        recs.extend(prune_silent(deg, n, cc))
        return recs, None
    if not allow_new_edges:
        recs.extend(prune_silent(deg, n, cc))
        return recs, None
    nmax = deg.nmax
    m = base["alive"] & deg.alive
    Tb = base["T"]
    L = np.abs(Tb) - np.abs(Tm)
    best, pair = 0.0, None
    worst, xpair = 0.0, None
    for i in range(nmax):
        for j in range(i + 1, nmax):
            if not (m[i] and m[j]) or deg.layer[i] == deg.layer[j]:
                continue
            if L[i, j] > best:
                best, pair = L[i, j], (i, j)
            if -L[i, j] > worst:
                worst, xpair = -L[i, j], (i, j)
    chk = None
    if pair is not None:
        i, j = pair
        a, c = (i, j) if deg.layer[i] < deg.layer[j] else (j, i)
        sg = 1 if Tb[a, c] >= 0 else -1
        if deg.W[a, c] == 0 and deg.admissible(a, c):
            deg.add_edge(a, c, sg)
            recs.append(dict(trial=n, type="add", pair=[deg.names[a], deg.names[c]], sign=sg))
            chk = dict(due=n + cc["check_after_trials"], pair=(a, c), d0=abs(Tm[a, c] - Tb[a, c]))
        else:
            k = deg.spawn_bridge(a, c, sg, n, base["theta_bridge"])
            if k is not None:
                recs.append(dict(trial=n, type="bridge", pair=[deg.names[a], deg.names[c]], new=deg.names[k], sign=int(sg)))
    elif xpair is not None:
        i, j = xpair
        a, c = (i, j) if deg.layer[i] < deg.layer[j] else (j, i)
        if deg.W[a, c] != 0:
            deg.remove_edge(a, c)
            recs.append(dict(trial=n, type="remove", pair=[deg.names[a], deg.names[c]]))
    recs.extend(prune_silent(deg, n, cc))
    return recs, chk


def prune_silent(deg, n, cc):
    """Remove controller-spawned bridge modules with zero events in the last P trials."""
    recs = []
    P = cc["prune_silent_trials"]
    for k in deg.plastic():
        if deg.kind[k] != KIND_BRIDGE:        # original modules are never pruned
            continue
        if n - deg.born[k] < P:
            continue
        if deg.last_spike_trial.get(int(k), -10**9) < n - P:
            deg.remove_module(k)
            recs.append(dict(trial=n, type="prune", module=deg.names[k]))
    return recs


def _op_field(op, key):
    return op.get(key) if isinstance(op, dict) else None


def random_op(deg, op, rng, n, base, cc):
    """Replay one TRI operation on a random admissible target.

    The sign is the sign recorded for that TRI operation. Only the target is
    random, which is the contrast named in the protocol.
    """
    ty = op["type"] if isinstance(op, dict) else op
    nmax = deg.nmax
    if ty == "add":
        cand = [(i, j) for i in range(nmax) for j in range(nmax) if deg.admissible(i, j) and deg.W[i, j] == 0]
        if not cand:
            return None
        i, j = cand[int(rng.integers(len(cand)))]
        recorded = _op_field(op, "sign")
        sg = int(recorded) if recorded is not None else int(rng.choice([-1, 1]))
        deg.add_edge(i, j, sg)
        return dict(trial=n, type="add", pair=[deg.names[i], deg.names[j]], sign=sg, random=True)
    if ty == "remove":
        E = deg.edges()
        if not E:
            return None
        i, j = E[int(rng.integers(len(E)))]
        deg.remove_edge(i, j)
        return dict(trial=n, type="remove", pair=[deg.names[i], deg.names[j]], random=True)
    if ty == "bridge":
        cand = [(i, j) for i in range(nmax) for j in range(nmax) if deg.admissible(i, j)]
        if not cand:
            return None
        i, j = cand[int(rng.integers(len(cand)))]
        recorded = _op_field(op, "sign")
        sg = int(recorded) if recorded is not None else int(rng.choice([-1, 1]))
        k = deg.spawn_bridge(i, j, sg, n, base["theta_bridge"])
        return None if k is None else dict(trial=n, type="bridge", pair=[deg.names[i], deg.names[j]], new=deg.names[k], sign=sg, random=True)
    if ty == "prune":
        recs = prune_silent(deg, n, cc)
        return recs[0] if recs else None
    return None
