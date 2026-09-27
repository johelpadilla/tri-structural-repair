"""Checks for the 2026-09-27 protocol amendment. These do not run the confirmatory study."""
import json
import os
import numpy as np
from tri.analysis import (
    central_claim_verdict, detection_verdict, noninferiority_verdict, superiority_verdict,
)
from tri.deg import DEG
from tri.sim import pair_matrix, random_op, regulated_alive, targeted_regeneration, tau_s_from
from tri.tau import systemic_tau

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CFG = json.load(open(os.path.join(HERE, "config", "preregistered.json")))


def test_config_amendment_timeline():
    tl = CFG["timeline"]
    assert (tl["cal_start"], tl["cal_end"], tl["lesion_trial"]) == (2000, 2500, 2900)
    assert tl["pre_window"] == 400 and tl["lesion_trial"] - tl["cal_end"] == 400
    assert CFG["detection"]["false_alarms_per_1000_max"] == 20
    assert CFG["detection"]["latency_trials"] == 50
    assert CFG["version"] == "0.1.2-draft"
    assert CFG["tau"]["W"] == 280 and CFG["tau"]["bin"] == 1
    assert CFG["analysis"]["primary_arm"] == "tri_matrix"
    assert CFG["analysis"]["primary_contrasts"][0] == ["tri_matrix", "random_rewire"]
    assert "tri_matrix" in CFG["arms_E2"]


def test_silent_bridge_does_not_enter_the_regulated_mean():
    Z = np.tile(np.arange(20.0), (7, 1))
    bridge = np.vstack([Z, np.zeros(20)])
    assert abs(systemic_tau(Z) - 1.0) < 1e-9
    assert abs(systemic_tau(bridge) - 21 / 28) < 1e-9
    base = {"alive": np.array([True] * 7 + [False])}
    mask = regulated_alive(np.array([True] * 8), base)
    T = pair_matrix(bridge, mask)
    assert abs(tau_s_from(T, mask) - 1.0) < 1e-9


def test_yoked_add_reuses_the_recorded_sign():
    deg = DEG(CFG["model"], np.random.default_rng(1))
    base = {"theta_bridge": 1.0}
    rec = random_op(deg, {"type": "add", "sign": -1}, np.random.default_rng(2), 10, base, CFG["controller"])
    assert rec is not None and rec["sign"] == -1
    i, j = deg.index(rec["pair"][0]), deg.index(rec["pair"][1])
    assert deg.W[i, j] == -CFG["model"]["w0"]


def test_restore_puts_back_the_cut_baseline_edge():
    deg = DEG(CFG["model"], np.random.default_rng(0))
    i, j = deg.index("I4"), deg.index("H1")
    a, c = deg.index("I3"), deg.index("H1")
    deg.remove_edge(i, j)
    nmax = deg.nmax
    Tb = np.zeros((nmax, nmax))
    Tm = np.zeros((nmax, nmax))
    Tb[i, j] = Tb[j, i] = 0.4
    Tb[a, c] = Tb[c, a] = 0.9
    base = {
        "T": Tb,
        "alive": deg.alive.copy(),
        "edges": [(i, j, 1)],
        "theta_bridge": 1.5,
    }
    recs, chk = targeted_regeneration(deg, Tm, base, 10, CFG["controller"], allow_new_edges=False)
    assert deg.W[i, j] == CFG["model"]["w0"]
    assert deg.W[a, c] == 0.0
    assert chk is None
    assert recs[0]["pair"] == ["I4", "H1"] and recs[0]["sign"] == 1 and recs[0]["restore"] is True


def test_intact_baseline_wiring_is_not_expanded_in_e1():
    deg = DEG(CFG["model"], np.random.default_rng(0))
    nmax = deg.nmax
    before = deg.W.copy()
    Tb = np.zeros((nmax, nmax))
    Tm = np.zeros((nmax, nmax))
    a, c = deg.index("I3"), deg.index("H1")
    Tb[a, c] = Tb[c, a] = 0.9
    edges = [(int(u), int(v), 1) for u, v in zip(*np.nonzero(before))]
    base = {"T": Tb, "alive": deg.alive.copy(), "edges": edges, "theta_bridge": 1.5}
    recs, chk = targeted_regeneration(deg, Tm, base, 10, CFG["controller"], allow_new_edges=False)
    assert np.array_equal(deg.W, before)
    assert chk is None
    assert recs == []


def test_decision_rules_match_the_amendment():
    e = 0.05
    assert superiority_verdict((0.06, 0.10), (0.07, 0.09), e) == "supported"
    assert superiority_verdict((-0.02, 0.04), (-0.01, 0.03), e) == "refuted"
    assert superiority_verdict((-0.10, -0.01), (-0.08, -0.02), e) == "refuted"
    assert superiority_verdict((0.00, 0.20), (-0.02, 0.15), e) == "inconclusive"
    assert superiority_verdict((float("nan"), 0.1), (0.0, 0.1), e) == "undefined"
    assert noninferiority_verdict((-0.04, 0.02), e) == "supported"
    assert noninferiority_verdict((-0.05, 0.02), e) == "refuted"
    assert detection_verdict(0.49, 0.0, 20) == "refuted"
    assert detection_verdict(0.50, 0.0, 20) == "inconclusive"
    assert detection_verdict(0.80, 20.0, 20) == "supported"
    assert detection_verdict(0.80, 20.1, 20) == "inconclusive"
    assert central_claim_verdict("refuted", "refuted", False) == "refuted"
    assert central_claim_verdict("supported", "refuted", False) == "not refuted"
    assert central_claim_verdict("refuted", "refuted", True) == "uninformative"
