"""Pre-specified per-run endpoints (one number per seed x arm)."""
from __future__ import annotations
import numpy as np


def _moving(x, w):
    c = np.cumsum(np.insert(x.astype(float), 0, 0.0))
    out = np.full(len(x), np.nan)
    out[w - 1:] = (c[w:] - c[:-w]) / w
    return out


def summarize(res, cfg):
    tl, tk, cc = cfg["timeline"], cfg["task"], cfg["controller"]
    corr = res["correct"]
    s = dict(experiment=res["experiment"], arm=res["arm"], seed=res["seed"])
    if res["experiment"] == "E1":
        tL = tl["lesion_trial"]            # nominal for intact arms
        pre, post = tl["pre_window"], tl["post_window"]
        s["pre_acc"] = float(corr[tL - pre:tL].mean())
        s["post_acc"] = float(corr[tL:tL + post].mean())            # PRIMARY
        mv = _moving(corr, 50)
        target = 0.9 * s["pre_acc"]
        idx = np.where(mv[tL + 49:tL + post] >= target)[0]
        s["recovery_trials"] = int(idx[0] + 49) if len(idx) else int(post)
        s["recovery_censored"] = bool(len(idx) == 0)
        if "macs" in res:
            s["compute_post"] = int(res["macs"][tL + post - 1] - res["macs"][tL - 1])
            s["compute_kind"] = "LSTM MACs (train)"
        else:
            T = tk["T"]
            syn = int(res["syn_ops"][tL + post - 1] - res["syn_ops"][tL - 1])
            upd = int(res["n_alive"][tL:tL + post].sum()) * T
            tau = int(res["tau_cost"][tL:tL + post].sum())
            s.update(compute_post=syn + upd + tau, compute_syn=syn, compute_update=upd, compute_tau=tau,
                     compute_kind="DEG synaptic events + module updates + tau_s pair-ops")
            trig = np.array(res["triggers"], dtype=int)
            silent = np.array(res.get("silent_alarms", []), dtype=int)
            alarms = np.concatenate([trig, silent]) if len(silent) else trig
            ce = tl["cal_end"]
            s["n_triggers"] = int(len(trig))
            s["n_silent_alarms"] = int(len(silent))
            s["false_alarms_per_1000"] = float(np.sum((alarms >= ce) & (alarms < tL)) / (tL - ce) * 1000)
            if res.get("lesion_trial") is None:
                dur = int(res["n_trials"]) - int(ce)
                if dur > 0:
                    s["alarms_after_calibration_per_1000"] = float(np.sum(trig >= ce) / dur * 1000)
            post_trig = trig[trig >= tL]
            s["detect_latency"] = int(post_trig[0] - tL) if len(post_trig) else None
            s["n_ops_post"] = int(sum(1 for o in res["ops"] if tL <= o["trial"] < tL + post))
            b = res["base"]
            if b is not None and b.get("sd") is not None:
                ts = res["tau_s"]
                inside = np.abs(ts - b["mu"]) <= cc["kappa"] * b["sd"]
                run = 0; ret = None
                start = tL + (s["detect_latency"] or 0)
                for n in range(start, min(tL + post, len(ts))):
                    run = run + 1 if inside[n] else 0
                    if run >= 10:
                        ret = n - 9 - tL; break
                s["tau_return_trials"] = ret
                s["tau_base_mu"], s["tau_base_sd"] = b["mu"], b["sd"]
            s["null"] = res.get("null", {})
            s["final_edges"] = int(res["n_edges"][-1]); s["final_modules"] = int(res["n_alive"][-1])
            s["lesion_removed"] = res.get("lesion_removed", [])
    else:
        blk, n = tk["block"], res["n_trials"]
        starts = list(range(blk, n, blk))
        s["adapt_acc"] = float(np.mean([corr[a:a + 200].mean() for a in starts]))      # E2 primary
        ret = [a for a in starts if (a // blk) % 2 == 0]
        s["retention_acc"] = float(np.mean([corr[a:a + 200].mean() for a in ret])) if ret else None
        s["overall_acc"] = float(corr.mean())
        if "triggers" in res:
            s["n_triggers"] = int(len(res["triggers"]))
    return s
