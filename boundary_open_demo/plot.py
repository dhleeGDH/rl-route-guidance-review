"""Generate the two Section V figures from results.npz.

Fig V-1  OD trip completion rate over training, four conditions, mean +/- std over seeds.
Fig V-2  Final OD trip completion rate by condition (bar chart, mean +/- std).

Styling is grayscale-safe: each condition has a distinct line style and marker,
and the bars use distinct hatches, so the figures remain readable in black and white.
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# condition -> (label, color, linestyle, marker, hatch)
STYLE = {
    "closed_time_min": ("Boundary-closed, travel-time-min (typical)",
                        "#08519c", "-", "o", "//"),
    "open_time_min":   ("Boundary-open, travel-time-min (typical)",
                        "#a50f15", "--", "s", "xx"),
    "open_aligned":    ("Boundary-open, destination-aligned (control)",
                        "#238b45", "-.", "^", ".."),
    "closed_aligned":  ("Boundary-closed, destination-aligned",
                        "#54278f", ":", "D", "\\\\"),
}
ORDER = ["closed_time_min", "open_time_min", "open_aligned", "closed_aligned"]


def load(path="results.npz"):
    d = np.load(path)
    out = {}
    for key in ORDER:
        out[key] = {
            "steps": d[f"{key}__steps"],
            "comp": d[f"{key}__comp"],   # [seeds, n_eval]
            "ret": d[f"{key}__ret"],
        }
    return out


def fig_v1(data, path="fig_v1_completion_curves.png"):
    fig, ax = plt.subplots(figsize=(7.0, 4.2))
    for key in ORDER:
        label, color, ls, mk, _ = STYLE[key]
        steps = data[key]["steps"]
        comp = data[key]["comp"] * 100.0
        mean, std = comp.mean(0), comp.std(0)
        ax.plot(steps, mean, ls, color=color, marker=mk, markersize=4,
                markevery=max(1, len(steps) // 12), label=label, linewidth=1.8)
        ax.fill_between(steps, mean - std, mean + std, color=color, alpha=0.12)
    ax.set_xlabel("Training episodes")
    ax.set_ylabel("OD trip completion rate (%)")
    ax.set_ylim(-3, 103)
    ax.grid(True, alpha=0.3)
    ax.legend(loc="center right", fontsize=8, framealpha=0.9)
    fig.tight_layout()
    fig.savefig(path, dpi=300)
    print("wrote", path)


def fig_v2(data, path="fig_v2_final_completion.png"):
    fig, ax = plt.subplots(figsize=(7.0, 4.0))
    finals = [data[k]["comp"][:, -1] * 100.0 for k in ORDER]
    means = [f.mean() for f in finals]
    stds = [f.std() for f in finals]
    x = np.arange(len(ORDER))
    for i, key in enumerate(ORDER):
        label, color, _, _, hatch = STYLE[key]
        ax.bar(x[i], means[i], yerr=stds[i], color=color, alpha=0.85,
               hatch=hatch, edgecolor="black", linewidth=0.8, capsize=4)
        ax.text(x[i], means[i] + stds[i] + 2, f"{means[i]:.0f}%",
                ha="center", va="bottom", fontsize=9)
    ax.set_xticks(x)
    ax.set_xticklabels(["closed\ntime-min\n(typical)", "open\ntime-min\n(typical)",
                        "open\naligned\n(control)", "closed\naligned"], fontsize=8)
    ax.set_ylabel("Final OD trip completion rate (%)")
    ax.set_ylim(0, 108)
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=300)
    print("wrote", path)


if __name__ == "__main__":
    data = load()
    fig_v1(data)
    fig_v2(data)
