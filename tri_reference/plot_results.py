"""Confirmatory figures for protocol 0.1.2. Reads results; writes nothing else."""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
RES = Path(__file__).resolve().parent / "results_confirmatory"
OUT = ROOT / "figures"
OUT.mkdir(exist_ok=True)

ROWS = [json.loads(line) for line in (RES / "summary.jsonl").read_text().splitlines() if line.strip()]
ROWS = [r for r in ROWS if "skipped" not in r]
ANALYSIS = json.loads((RES / "analysis.json").read_text())

EVENT = [
    ("intact_none", "Unlesioned", "Sin lesión", "#009E73"),
    ("tri_matrix", "Matrix alarm", "Alarma por aristas", "#0072B2"),
    ("perf", "Error alarm", "Alarma por error", "#E69F00"),
    ("tri", "Signed mean", "Media con signo", "#56B4E9"),
    ("random_time", "Random time", "Tiempo al azar", "#CC79A7"),
    ("none", "No repair", "Sin reparar", "#666666"),
    ("random_rewire", "Random target", "Blanco al azar", "#D55E00"),
    ("tri_onesided", "One-sided mean", "Media unilateral", "#882255"),
]
LSTM = [
    ("lstm", "LSTM", "LSTM", "#000000"),
    ("lstm_ewc", "LSTM-EWC", "LSTM-EWC", "#44AA99"),
]


def e1(arm):
    return [r for r in ROWS if r["experiment"] == "E1" and r["arm"] == arm]


def present(spec):
    return [s for s in spec if e1(s[0])]


def draw(lang, path):
    en = lang == "en"
    event = present(EVENT)
    lstm = present(LSTM)
    fig, axes = plt.subplots(2, 2, figsize=(12.6, 8.8), dpi=160)
    fig.subplots_adjust(left=0.07, right=0.985, top=0.96, bottom=0.08, hspace=0.48, wspace=0.42)

    ax = axes[0, 0]
    rng = np.random.default_rng(0)
    for i, (arm, en_lab, es_lab, color) in enumerate(event):
        ys = np.array([r["post_acc"] for r in e1(arm)])
        x = i + rng.uniform(-0.16, 0.16, size=len(ys))
        ax.scatter(x, ys, s=16, color=color, alpha=0.75, linewidths=0, zorder=3)
        ax.plot([i - 0.28, i + 0.28], [ys.mean(), ys.mean()], color="black", lw=1.6, zorder=4)
    ax.axhline(0.5, color="0.45", ls="--", lw=0.8)
    ax.set_xticks(range(len(event)))
    ax.set_xticklabels([s[1] if en else s[2] for s in event], rotation=35, ha="right", fontsize=8)
    ax.set_ylim(0.28, 1.02)
    ax.set_ylabel("Accuracy after the lesion" if en else "Exactitud después de la lesión")
    ax.set_title("A. Post-lesion accuracy" if en else "A. Exactitud tras la lesión", loc="left", fontsize=11)
    ax.tick_params(axis="y", labelsize=8)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    ax = axes[0, 1]
    wanted = [
        "tri_matrix - random_rewire",
        "tri_matrix - random_time",
        "tri_matrix - none",
        "tri_matrix - perf",
    ]
    labels_en = [
        "Random target",
        "Random time",
        "No repair",
        "Error alarm",
    ]
    labels_es = [
        "Blanco al azar",
        "Tiempo al azar",
        "Sin reparar",
        "Alarma por error",
    ]
    by = {c["contrast"]: c for c in ANALYSIS["primary"]}
    ypos = np.arange(len(wanted))[::-1]
    for y, key in zip(ypos, wanted):
        c = by[key]
        lo, hi = c["ci95"]
        mu = c["mean_diff"]
        ax.plot([lo, hi], [y, y], color="#0072B2", lw=1.8)
        ax.plot(mu, y, "o", color="#0072B2", ms=5)
    ax.axvline(0, color="0.2", lw=0.8)
    ax.axvline(0.05, color="0.55", ls=":", lw=0.8)
    ax.axvline(-0.05, color="0.55", ls=":", lw=0.8)
    ax.set_yticks(ypos)
    ax.set_yticklabels(labels_en if en else labels_es, fontsize=8)
    ax.set_xlabel("Matrix alarm minus the named arm" if en else "Alarma por aristas menos el brazo nombrado")
    ax.set_title("B. Paired differences" if en else "B. Diferencias pareadas", loc="left", fontsize=11)
    ax.set_xlim(-0.12, 0.40)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    ax = axes[1, 0]
    for arm, en_lab, es_lab, color, ls in (
        ("tri_matrix", "Matrix alarm", "Alarma por aristas", "#0072B2", "-"),
        ("perf", "Error alarm", "Alarma por error", "#E69F00", "-"),
        ("random_time", "Random time", "Tiempo al azar", "#CC79A7", "-"),
    ):
        lat = sorted(r["detect_latency"] for r in e1(arm) if r.get("detect_latency") is not None)
        if not lat:
            continue
        y = np.arange(1, len(lat) + 1) / 30
        ax.step(lat, y, where="post", color=color, lw=1.6, label=en_lab if en else es_lab)
    ax.axvline(50, color="0.35", ls="--", lw=0.8)
    ax.set_xlim(0, 2500)
    ax.set_ylim(0, 1.05)
    ax.set_xlabel("Trials from the lesion to the first repair" if en else "Ensayos desde la lesión hasta la primera reparación")
    ax.set_ylabel("Fraction of seeds" if en else "Fracción de semillas")
    ax.set_title("C. Latency of the first repair" if en else "C. Latencia de la primera reparación", loc="left", fontsize=11)
    ax.legend(frameon=False, fontsize=8, loc="lower right")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    ax = axes[1, 1]
    if lstm:
        width = 0.36
        pre_m, post_m = [], []
        for arm, *_ in lstm:
            xs = e1(arm)
            pre_m.append(np.mean([r["pre_acc"] for r in xs]))
            post_m.append(np.mean([r["post_acc"] for r in xs]))
        x = np.arange(len(lstm))
        ax.bar(x - width / 2, pre_m, width, color="#BBBBBB")
        ax.bar(x + width / 2, post_m, width, color="#444444")
        for i, (arm, *_) in enumerate(lstm):
            ys = [r["post_acc"] for r in e1(arm)]
            ax.scatter(np.full(len(ys), i + width / 2), ys, s=12, color="#0072B2", zorder=3, linewidths=0)
        ax.set_xticks(x)
        labels = []
        for s in lstm:
            name = s[1] if en else s[2]
            labels.append(name + ("\nbefore / after" if en else "\nantes / después"))
        ax.set_xticklabels(labels, fontsize=8)
        ax.set_ylim(0, 1.05)
        ax.axhline(0.5, color="0.45", ls="--", lw=0.8)
        ax.set_ylabel("Accuracy" if en else "Exactitud")
    else:
        ax.text(0.5, 0.5, "LSTM arms were not run" if en else "Los brazos LSTM no se ejecutaron", ha="center", va="center")
        ax.set_axis_off()
    ax.set_title("D. Accuracy under weight masking" if en else "D. Exactitud bajo el enmascaramiento de pesos", loc="left", fontsize=11)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    fig.savefig(path, dpi=160)
    fig.savefig(path.with_suffix(".pdf"))
    plt.close(fig)
    print("wrote", path)


if __name__ == "__main__":
    draw("en", OUT / "fig_results_en.png")
    draw("es", OUT / "fig_results_es.png")
