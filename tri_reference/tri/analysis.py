"""Pre-specified confirmatory analysis.

For every contrast (arm X vs arm Y) the unit of analysis is the seed; arms are
paired by seed (common random numbers).  We report the mean paired difference
X - Y of the endpoint with a 95% BCa bootstrap CI (10,000 resamples), Cohen's
d_z, a two-sided Wilcoxon signed-rank p-value with Holm correction across the
primary contrasts, a TOST equivalence decision (90% CI within +-SESOI) and, for
non-inferiority contrasts, whether the lower 95% bound exceeds -SESOI.
"""
from __future__ import annotations
import argparse, json, os
import numpy as np
from scipy import stats


def load(path):
    return [json.loads(l) for l in open(os.path.join(path, "summary.jsonl"))]


def paired(rows, exp, x, y, key):
    A = {r["seed"]: r.get(key) for r in rows if r["experiment"] == exp and r["arm"] == x}
    B = {r["seed"]: r.get(key) for r in rows if r["experiment"] == exp and r["arm"] == y}
    seeds = sorted(s for s in A if s in B and A[s] is not None and B[s] is not None)
    return seeds, np.array([A[s] - B[s] for s in seeds], float)


def ci(d, level, n_boot, rng):
    if len(d) < 3 or np.allclose(d, d[0]):
        return (float("nan"), float("nan")), "undefined (n<3 or no variance)"
    try:
        r = stats.bootstrap((d,), np.mean, confidence_level=level, n_resamples=n_boot,
                            method="BCa", random_state=rng)
        return (float(r.confidence_interval.low), float(r.confidence_interval.high)), "BCa"
    except Exception:
        r = stats.bootstrap((d,), np.mean, confidence_level=level, n_resamples=n_boot,
                            method="percentile", random_state=rng)
        return (float(r.confidence_interval.low), float(r.confidence_interval.high)), "percentile"


def holm(ps):
    ps = np.asarray(ps, float); m = len(ps); order = np.argsort(ps)
    adj = np.empty(m); run = 0.0
    for k, i in enumerate(order):
        run = max(run, min(1.0, (m - k) * ps[i])); adj[i] = run
    return adj


def _finite_pair(bounds):
    return bounds is not None and len(bounds) == 2 and all(np.isfinite(bounds))


def superiority_verdict(ci95, ci90, sesoi):
    """H2, H3 and H5. A wide interval that clears neither bar is inconclusive."""
    if not (_finite_pair(ci95) and _finite_pair(ci90)):
        return "undefined"
    lo, hi = ci95
    a, b = ci90
    if lo > sesoi:
        return "supported"
    if (a > -sesoi and b < sesoi) or hi < 0:
        return "refuted"
    return "inconclusive"


def noninferiority_verdict(ci95, sesoi):
    """H4. Support and refutation partition the defined intervals."""
    if not _finite_pair(ci95):
        return "undefined"
    if ci95[0] > -sesoi:
        return "supported"
    return "refuted"


def detection_verdict(detected_fraction, false_alarms_per_1000, max_fa):
    """H1. Exactly half the seeds, or too many same-regime alarms, stays inconclusive."""
    if detected_fraction is None or not np.isfinite(detected_fraction):
        return "undefined"
    if detected_fraction < 0.5:
        return "refuted"
    fa_ok = (false_alarms_per_1000 is not None and np.isfinite(false_alarms_per_1000)
             and false_alarms_per_1000 <= max_fa)
    if detected_fraction > 0.5 and fa_ok:
        return "supported"
    return "inconclusive"


def central_claim_verdict(targeting, timing, f5_uninformative):
    if f5_uninformative:
        return "uninformative"
    if targeting == "refuted" and timing == "refuted":
        return "refuted"
    return "not refuted"


def contrast(rows, exp, x, y, key, A, rng):
    seeds, d = paired(rows, exp, x, y, key)
    out = dict(contrast=f"{x} - {y}", endpoint=key, n=len(d))
    if len(d) == 0:
        return out
    c95, meth = ci(d, 0.95, A["n_boot"], rng)
    c90, _ = ci(d, 0.90, A["n_boot"], rng)
    sd = d.std(ddof=1) if len(d) > 1 else float("nan")
    p = stats.wilcoxon(d).pvalue if len(d) >= 6 and not np.allclose(d, 0) else float("nan")
    e = A["sesoi"]
    out.update(mean_diff=float(d.mean()), ci95=c95, ci90=c90, ci_method=meth,
               d_z=float(d.mean() / sd) if sd and sd > 0 else float("nan"), p_wilcoxon=float(p),
               equivalent_TOST=bool(c90[0] > -e and c90[1] < e) if np.isfinite(c90[0]) else None,
               superior_by_sesoi=bool(c95[0] > e) if np.isfinite(c95[0]) else None,
               noninferior=bool(c95[0] > -e) if np.isfinite(c95[0]) else None)
    return out


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", required=True)
    ap.add_argument("--config", default=os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "config", "preregistered.json"))
    ap.add_argument("--design-check", default=None)
    a = ap.parse_args(argv)
    cfg = json.load(open(a.config)); A = cfg["analysis"]
    arm = A.get("primary_arm", "tri")
    rows = [r for r in load(a.results) if "skipped" not in r]
    man = json.load(open(os.path.join(a.results, "manifest.json")))
    rng = np.random.default_rng(12345)
    rep = dict(mode=man["mode"], NOTE=man.get("NOTE", ""), primary=[], secondary=[], E2=[], checks={})
    noninf = {tuple(p) for p in A["noninferiority"]}
    for (x, y) in A["primary_contrasts"]:
        c = contrast(rows, "E1", x, y, "post_acc", A, rng)
        if (x, y) in noninf:
            c["verdict"] = noninferiority_verdict(c.get("ci95"), A["sesoi"])
            c["rule"] = "noninferiority"
        else:
            c["verdict"] = superiority_verdict(c.get("ci95"), c.get("ci90"), A["sesoi"])
            c["rule"] = "superiority"
        rep["primary"].append(c)
    ps = [c.get("p_wilcoxon", float("nan")) for c in rep["primary"]]
    if all(np.isfinite(ps)):
        for c, pa in zip(rep["primary"], holm(ps)): c["p_holm"] = float(pa)
    for (x, y) in A["secondary_contrasts"]:
        rep["secondary"].append(contrast(rows, "E1", x, y, "post_acc", A, rng))
        cost = contrast(rows, "E1", x, y, "compute_post", A, rng)
        cost["units_note"] = ("Event-driven arms count synaptic events, module updates and tau_s pair operations. "
                              "LSTM arms count multiply-accumulates. The difference is not a single scale.")
        rep["secondary"].append(cost)
    for (x, y) in A["primary_contrasts"] + [[arm, "lstm"], [arm, "lstm_ewc"]]:
        rep["E2"].append(contrast(rows, "E2", x, y, "adapt_acc", A, rng))
        rep["E2"].append(contrast(rows, "E2", x, y, "retention_acc", A, rng))
    det = cfg.get("detection", {})
    latency_max = int(det.get("latency_trials", 50))
    fa_max = float(det.get("false_alarms_per_1000_max", 20))
    f5_min = float(A.get("f5_min_fraction_outside_null", 0.8))
    primary_rows = [r for r in rows if r["experiment"] == "E1" and r["arm"] == arm]
    mean_rows = [r for r in rows if r["experiment"] == "E1" and r["arm"] == "tri"]
    if primary_rows:
        detected = float(np.mean([r.get("detect_latency") is not None and r["detect_latency"] <= latency_max for r in primary_rows]))
        fa = float(np.mean([r["false_alarms_per_1000"] for r in primary_rows]))
        rep["checks"]["F6_lesion_detected_fraction"] = detected
        rep["checks"]["false_alarms_per_1000_mean"] = fa
        rep["checks"]["H1_verdict"] = detection_verdict(detected, fa, fa_max)
        rep["checks"]["H1_arm"] = arm
    if mean_rows:
        outside = float(np.mean([bool(r.get("null", {}).get("baseline_outside_null95")) for r in mean_rows]))
        rep["checks"]["F5_baseline_outside_null95_fraction"] = outside
        rep["checks"]["F5_signed_mean_uninformative"] = bool(outside < f5_min)
    if a.design_check and os.path.exists(a.design_check):
        dc = json.load(open(a.design_check))
        rep["checks"]["design_check_chosen"] = dc.get("chosen")
        rep["checks"]["F5_no_window"] = dc.get("chosen") is None
    by_name = {c.get("contrast"): c.get("verdict") for c in rep["primary"]}
    # The signed mean can sit inside its null while the edge alarm still sees a
    # lesion. That fact is recorded above. It does not void the primary claim.
    rep["central_claim"] = central_claim_verdict(
        by_name.get(f"{arm} - random_rewire"), by_name.get(f"{arm} - random_time"), False)
    thr = float(A.get("sensitivity_pre_acc_min", 0.8))
    good = sorted(r["seed"] for r in rows if r["experiment"] == "E1" and r["arm"] == "none"
                  and r.get("pre_acc") is not None and r["pre_acc"] >= thr)
    sens_rows = [r for r in rows if r["experiment"] != "E1" or r["seed"] in set(good)]
    rep["sensitivity"] = dict(rule=f"E1 seeds whose no-regeneration pre_acc is at least {thr}",
                              n_seeds=len(good), seeds=good, primary=[])
    for (x, y) in A["primary_contrasts"]:
        c = contrast(sens_rows, "E1", x, y, "post_acc", A, rng)
        if (x, y) in noninf:
            c["verdict"] = noninferiority_verdict(c.get("ci95"), A["sesoi"])
            c["rule"] = "noninferiority"
        else:
            c["verdict"] = superiority_verdict(c.get("ci95"), c.get("ci90"), A["sesoi"])
            c["rule"] = "superiority"
        rep["sensitivity"]["primary"].append(c)
    def _jsonable(o):
        if isinstance(o, tuple):
            return [_jsonable(x) for x in o]
        if isinstance(o, np.ndarray):
            return o.tolist()
        if isinstance(o, (np.floating, np.integer)):
            return o.item()
        raise TypeError(type(o).__name__)

    json.dump(rep, open(os.path.join(a.results, "analysis.json"), "w"), indent=1, default=_jsonable)
    banner = "SMOKE TEST - NOT EVIDENCE - " if man["mode"] == "smoke" else ""
    print(banner + "primary contrasts (E1, post_acc):")
    for c in rep["primary"]:
        print(" ", {k: c.get(k) for k in ("contrast", "n", "mean_diff", "ci95", "verdict", "p_holm")})
    print(banner + "H1:", rep["checks"].get("H1_verdict"), " central claim:", rep.get("central_claim"))
    print(banner + "checks:", rep["checks"])
    return rep


if __name__ == "__main__":
    main()
