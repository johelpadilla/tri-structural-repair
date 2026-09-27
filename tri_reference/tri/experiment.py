"""Command-line runner.

  python -m tri.experiment --mode smoke                      # 2 seeds, short timeline
  python -m tri.experiment --mode tuning --out results_tuning  # EWC lambda on tuning seeds
  python -m tri.experiment --mode full --out results_full --jobs 8 \
         [--tuning-from results_tuning/tuning.json]
  python -m tri.analysis --results results_full
"""
from __future__ import annotations
import argparse, copy, hashlib, json, os, platform, sys, time
from multiprocessing import Pool
import numpy as np
from .sim import run_deg_arm
from .metrics import summarize
from . import __version__

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_CFG = os.path.join(HERE, "config", "preregistered.json")
LSTM_ARMS = ("lstm", "lstm_ewc")


def smoke_overrides(cfg):
    c = copy.deepcopy(cfg)
    c["task"].update(n_trials=600, block=150)
    c["timeline"].update(cal_start=150, cal_end=210, lesion_trial=250, pre_window=40, post_window=200)
    c["null"].update(n_windows=10, n_surrogates=5)
    c["lstm"].update(fisher_trials=100)
    return c


def run_one(cfg, exp, arm, seed, yoke):
    if arm in LSTM_ARMS:
        from .lstm import run_lstm_arm
        return run_lstm_arm(cfg, exp, arm, seed)
    return run_deg_arm(cfg, exp, arm, seed, yoke)


def run_seed(args):
    cfg, exp, arms, seed, outdir = args
    rows, yoke = [], None
    primary = cfg.get("analysis", {}).get("primary_arm", "tri")
    order = ([primary] if primary in arms else []) + [a for a in arms if a != primary]
    for arm in order:
        if arm in LSTM_ARMS:
            try:
                import torch  # noqa: F401
            except Exception:
                rows.append(dict(experiment=exp, arm=arm, seed=seed, skipped="torch not installed"))
                continue
        t0 = time.time()
        res = run_one(cfg, exp, arm, seed, yoke)
        if arm == primary:
            yoke = dict(n_triggers=len(res["triggers"]), triggers=res["triggers"], ops=res["ops"])
        s = summarize(res, cfg)
        s["runtime_s"] = round(time.time() - t0, 2)
        rows.append(s)
        np.savez_compressed(os.path.join(outdir, "raw", f"{exp}_{arm}_s{seed}.npz"),
                            **{k: np.asarray(v) for k, v in res.items()
                               if isinstance(v, np.ndarray)})
        with open(os.path.join(outdir, "raw", f"{exp}_{arm}_s{seed}_ops.json"), "w") as f:
            json.dump(dict(ops=res.get("ops", []), triggers=res.get("triggers", []),
                           null=res.get("null", {}), base=res.get("base")), f)
    return rows


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["smoke", "full", "tuning"], required=True)
    ap.add_argument("--config", default=DEFAULT_CFG)
    ap.add_argument("--out", default=None)
    ap.add_argument("--seeds", type=int, nargs="*")
    ap.add_argument("--experiments", nargs="*", default=["E1", "E2"])
    ap.add_argument("--jobs", type=int, default=1)
    ap.add_argument("--tuning-from", default=None)
    a = ap.parse_args(argv)
    raw = open(a.config, "rb").read()
    cfg = json.loads(raw)
    if a.mode == "smoke":
        cfg = smoke_overrides(cfg)
        seeds = a.seeds or [1, 2]
    elif a.mode == "tuning":
        lo, hi = cfg["seeds"]["tuning"]; seeds = a.seeds or list(range(lo, hi + 1))
    else:
        lo, hi = cfg["seeds"]["confirmatory"]; seeds = a.seeds or list(range(lo, hi + 1))
    if a.tuning_from:
        tj = json.load(open(a.tuning_from))
        cfg["lstm"]["ewc_lambda"] = tj["ewc_lambda"]
        # Amendment 0.1.2 fixes the tau window. The design check still records
        # whether the signed mean leaves its null; it does not replace W.
    out = a.out or f"results_{a.mode}"
    os.makedirs(os.path.join(out, "raw"), exist_ok=True)
    manifest = dict(mode=a.mode, version=__version__, config_sha256=hashlib.sha256(raw).hexdigest(),
                    seeds=seeds, experiments=a.experiments, python=sys.version, platform=platform.platform(),
                    numpy=np.__version__, started=time.strftime("%Y-%m-%dT%H:%M:%S%z"),
                    NOTE=("SMOKE TEST ONLY - NOT EVIDENCE" if a.mode == "smoke" else ""))
    rows = []
    if a.mode == "tuning":
        from .designcheck import run as design_run
        dc = design_run(cfg, seeds, os.path.join(out, "design_check.json"))
        print("design check:", dc["candidates"], "chosen:", dc["chosen"])
        for lam in cfg["lstm"]["ewc_lambda_grid"]:
            c = copy.deepcopy(cfg); c["lstm"]["ewc_lambda"] = lam
            jobs = [(c, "E2", ["lstm_ewc"], s, out) for s in seeds]
            with Pool(a.jobs) as p:
                for r in p.map(run_seed, jobs):
                    for x in r: x["ewc_lambda"] = lam
                    rows.extend(r)
        best = max(cfg["lstm"]["ewc_lambda_grid"],
                   key=lambda L: np.mean([r["retention_acc"] for r in rows if r.get("ewc_lambda") == L]))
        json.dump(dict(ewc_lambda=best, rule="max mean E2 retention_acc on tuning seeds",
                       tau=dc["chosen"]),
                  open(os.path.join(out, "tuning.json"), "w"), indent=1)
    else:
        for exp in a.experiments:
            arms = cfg["arms_E1"] if exp == "E1" else cfg["arms_E2"]
            jobs = [(cfg, exp, arms, s, out) for s in seeds]
            if a.jobs > 1:
                with Pool(a.jobs) as p:
                    for r in p.map(run_seed, jobs): rows.extend(r)
            else:
                for j in jobs: rows.extend(run_seed(j))
    with open(os.path.join(out, "summary.jsonl"), "w") as f:
        for r in rows: f.write(json.dumps(r, default=float) + "\n")
    manifest["finished"] = time.strftime("%Y-%m-%dT%H:%M:%S%z")
    json.dump(manifest, open(os.path.join(out, "manifest.json"), "w"), indent=1)
    print(f"[{a.mode}] wrote {len(rows)} rows to {out}/summary.jsonl")
    return rows


if __name__ == "__main__":
    main()
