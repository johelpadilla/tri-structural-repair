# Reference implementation for: Systemic Tau as a homeostatic trigger for structural self-repair in event-driven networks

This package (`tri-reference`, version 0.1.2.dev0) is the reference implementation for the article
*Systemic Tau as a homeostatic trigger for structural self-repair in event-driven networks*
(https://doi.org/10.5281/zenodo.23001549). Systemic Tau (τs) is the mean pairwise Kendall τb of module
activity on a common event index. The operative alarm is the mean absolute change of τb on the edges present
at calibration. In the lesion experiment an alarm restores the missing baseline edge with the largest drop in
absolute concordance and adds no other edge. The signed-mean arm remains in the experiment as a control.

## What the package implements

| Module | Content | Article reference |
|---|---|---|
| `tri/tau.py` | Kendall τb on a common event index; Systemic Tau τs as the mean pairwise τb; exact binary path (τb = φ) | Methods |
| `tri/null.py` | Circular-shift surrogates; rate-matched independent streams; analytical null SD; small-window bias; minimum window | Methods |
| `tri/nullcal.py` | Null calibration of τs for N = 7 modules, independent of the task | Methods |
| `tri/task.py` | Regenerative XOR task as event streams; covariate drift (E1) and polarity-swap concept drift (E2) | Methods |
| `tri/deg.py` | Dynamic Event Graph of discrete-time leaky integrate-and-fire modules; edge addition and removal, bridge genesis, pruning, lesion | Methods |
| `tri/sim.py` | TRI controller (two-sided band on the baseline module set, source-pair identification, structural operations), ordinal error correction, threshold re-entrainment and all ablations | Methods |
| `tri/lstm.py` | LSTM baseline retrained online after the lesion; LSTM with elastic weight consolidation | Methods |
| `tri/metrics.py` | Per-seed endpoints, including post-lesion accuracy | Methods |
| `tri/designcheck.py` | Prespecified choice of window and bin width on tuning seeds (criterion F5) | Methods |
| `tri/analysis.py` | Paired BCa bootstrap intervals, d_z, Wilcoxon signed-rank tests with Holm correction, TOST equivalence, non-inferiority and falsification decisions | Methods |
| `tri/experiment.py` | Command-line runner for the tuning and confirmatory steps; writes manifests | Methods |
| `config/preregistered.json` | Every parameter, seed, arm and contrast | Table 3 |

Arms in experiment E1: `tri_matrix` (primary; mean absolute change of τb on the edges present at calibration), `tri` (signed-mean band, kept as a control), `random_rewire` (yoked to `tri_matrix`: same operation types and signs, random admissible targets), `random_time`
(same action at the same number of uniformly random trials), `none` (no structural regeneration),
`perf` (same action, triggered by running error), `tri_onesided` (one-sided band on the signed mean),
`intact_tri`, `intact_none`, `lstm` and `lstm_ewc`. Experiment E2 uses
`tri_matrix`, `tri`, `random_rewire`, `random_time`, `none`, `perf`, `lstm` and `lstm_ewc`.
In E1 an alarm restores the missing baseline edge with the largest drop in absolute concordance and does not add any other edge. An alarm that finds that wiring already intact counts as a false alarm and is not copied into the yoked arms. In E2 the baseline wiring is intact, so the controller may still add or bridge.

## Installation

Python 3.10 or later is required. NumPy and SciPy are the core dependencies; PyTorch is needed only for the
LSTM baselines.

```bash
cd tri_reference
python3 -m venv .venv && . .venv/bin/activate
pip install -e ".[test]"
pip install torch --index-url https://download.pytorch.org/whl/cpu   # LSTM baselines
```

## Tests

```bash
python -m pytest -q tests
```

The unit tests check τb against SciPy's implementation in the presence of ties, the identity τb = φ for
binary streams, the range of τs and its lower bound −1/(N−1), the convention for constant streams, the
preservation of marginals by circular-shift surrogates, and the agreement between the binary and general
computational paths.

`smoke_test.sh` provides a quick installation check: it runs the unit tests, a small null calibration and a
shortened two-seed pass through every arm and the analysis script. It confirms only that the installation
executes end to end and is not part of the prespecified protocol.

```bash
PYTHON=python ./smoke_test.sh
```

## Full prespecified run

Run the steps in this order, and do not change `config/preregistered.json` after step 1 except through a
dated amendment recorded before step 2.

```bash
# 0. record the config hash (it is also written to every manifest.json)
sha256sum config/preregistered.json

# 1. tuning seeds 1001-1005 only: design check for (W, bin) and EWC lambda
python -m tri.experiment --mode tuning --out results_tuning --jobs 5

# 2. null calibration of tau_s (independent of the task)
python -m tri.nullcal --out results_nullcal.json --n-sim 2000

# 3. confirmatory runs: seeds 1-30, experiments E1 and E2, all arms
python -m tri.experiment --mode full --tuning-from results_tuning/tuning.json \
       --out results_full --jobs 8

# 4. pre-specified analysis (writes results_full/analysis.json)
python -m tri.analysis --results results_full
```

The same sequence as a single command:

```bash
cd tri_reference && pip install -e ".[test,lstm]" && python -m pytest -q tests && \
python -m tri.experiment --mode tuning --out results_tuning --jobs 5 && \
python -m tri.nullcal --out results_nullcal.json --n-sim 2000 && \
python -m tri.experiment --mode full --tuning-from results_tuning/tuning.json --out results_full --jobs 8 && \
python -m tri.analysis --results results_full
```

The design check records whether the signed mean leaves its surrogate null. Amendment 0.1.2 does not treat a
signed mean that stays inside that null as a veto of the matrix alarm, and it does not replace the window of
280 ticks. The completed local confirmatory run is `results_confirmatory/`. Seeds 1–30 have already been used
there, for the event-driven arms and for the LSTM arms. Do not regenerate that directory. The EWC strength
used for `lstm_ewc` in that run is 10, chosen as the maximum mean E2 retention on tuning seeds 1001–1005 and
recorded in `results_confirmatory/ewc_tuning.json`. The value 100 stored in the configuration is the
pre-selection default.

## Configuration

All parameters are fixed in `config/preregistered.json` (27 September 2026): task
timing and drift regimes, module model, initial wiring, the τ window (280 ticks, bin width 1), calibration period
(trials 2,000–2,499, inside one input regime), controller settings, threshold search, lesion timing
(trial 2,900) and fraction, the same-regime false-alarm window (trials 2,500–2,899; the detection rule also requires at
most 20 false alarms per 1,000 trials), null-calibration sizes, LSTM settings, seeds, arms, the primary
endpoint, the smallest effect size of interest (0.05) and the primary, non-inferiority and secondary
contrasts. The primary arm is `tri_matrix`. The window is fixed in that file; the design check on
tuning seeds still records whether the signed mean leaves its surrogate null, and it does not replace the
window. EWC strength remains the value with the largest mean E2 retention on the tuning seeds. The design
check uses the null counts in the configuration (100 windows × 20 surrogates). Every run records the
SHA-256 hash of the configuration file in its manifest. After calibration, the regulated mean of τs uses
only modules that were alive at baseline, and the matrix alarm uses only the edges present then.

## Seeds

Confirmatory seeds are 1–30; tuning seeds are 1001–1005 and are never used in confirmatory analyses. Each
seed spawns independent random streams for the input sequence, the lesion, module noise, initial thresholds,
threshold proposals, controller randomness, surrogates and LSTM initialization, so that all arms of a seed
receive identical inputs, noise and lesioned edges.

## Outputs

| Path | Content |
|---|---|
| `results_tuning/design_check.json` | Candidate windows, per-seed null comparisons and the adopted (W, bin) |
| `results_tuning/tuning.json` | Adopted EWC strength and window (`"tau"`), read by the confirmatory run |
| `results_nullcal.json` | Null distribution of τs, small-window bias and minimum window |
| `results_full/raw/` | Per-run trajectories (`.npz`) and logs of structural operations and triggers (`_ops.json`) |
| `results_full/summary.jsonl` | One row of endpoints per experiment, arm and seed |
| `results_full/manifest.json` | Mode, package version, configuration hash, Python and NumPy versions, platform, seeds and timestamps |
| `results_full/analysis.json` | Prespecified contrasts, intervals, detection decision, and the sensitivity analysis restricted to seeds whose unrepaired pre-lesion accuracy is at least 0.8 |

## Scope

The module model, threshold-search rule and controller are one concrete instantiation of the framework.
Because τs averages signed concordances, concordant and anti-concordant pairs can cancel; the design check,
the `tri_matrix` arm and criterion F5 address this. Computational cost is counted as abstract operations
(synaptic events, module updates and τs pair operations for event-driven arms; multiply–accumulate
operations for the LSTMs), not as energy measured on hardware.

## Citation

Padilla-Villanueva, J. (2026). Systemic Tau as a homeostatic trigger for structural self-repair in event-driven
networks. https://doi.org/10.5281/zenodo.23001549

The reference implementation is archived at https://doi.org/10.5281/zenodo.23001551.
ORCID: [0000-0002-5797-6931](https://orcid.org/0000-0002-5797-6931).

## License

MIT
