#!/usr/bin/env python3
# figures for the Practical Work report. writes into docs/report/figures/.
# every panel is drawn from a committed CSV or a cached embedding dump.

import re
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import torch
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
M = ROOT / "outputs" / "metrics"
OUT = ROOT / "docs" / "report" / "figures"
OUT.mkdir(parents=True, exist_ok=True)

# Figures are drawn at roughly the size they are printed at, so a 9-pt label in the
# figure lands at about 9 pt on the page. Scaling a large figure down in LaTeX instead
# shrinks the labels with it, which is how figures end up with unreadable axes.
plt.rcParams.update({"font.size": 8, "axes.titlesize": 8.5, "axes.labelsize": 8,
                     "xtick.labelsize": 7, "ytick.labelsize": 7, "legend.fontsize": 7,
                     "figure.titlesize": 9})

COND = ["cosine", "none", "per_type", "per_type_median", "global", "global_median"]
LABEL = {"cosine": "cosine\n(continuous)", "none": "Hamming\nnone", "per_type": "per_type\n(mean)",
         "per_type_median": "per_type\n(median)", "global": "global\n(mean)",
         "global_median": "global\n(median)"}
SRC = {
 ("SPRINT", "LIT-PCBA"): {"cosine": "lit_pcba_results_cosine_20260824_232705.csv",
   "none": "lit_pcba_results_hamming_20260824_233650.csv",
   "per_type": "lit_pcba_results_hamming_per_type_20260825_130648.csv",
   "per_type_median": "lit_pcba_results_hamming_per_type_median_20260826_235432.csv",
   "global": "lit_pcba_results_hamming_global_20260824_235309.csv",
   "global_median": "lit_pcba_results_hamming_global_median_20260826_235624.csv"},
 ("SPRINT", "DUD-E"): {"cosine": "sprint_dude_results_cosine_20260826_120525.csv",
   "none": "sprint_dude_results_hamming_20260826_122228.csv",
   "per_type": "sprint_dude_results_hamming_per_type_20260826_124015.csv",
   "per_type_median": "sprint_dude_results_hamming_per_type_median_20260826_235721.csv",
   "global": "sprint_dude_results_hamming_global_20260826_130116.csv",
   "global_median": "sprint_dude_results_hamming_global_median_20260826_235817.csv"},
 ("DrugCLIP", "LIT-PCBA"): {"cosine": "drugclip_pcba_cosine_20260825_004540.csv",
   "none": "drugclip_pcba_hamming_20260825_004551.csv",
   "per_type": "drugclip_pcba_hamming_per_type_20260825_125817.csv",
   "per_type_median": "drugclip_pcba_hamming_per_type_median_20260826_174109.csv",
   "global": "drugclip_pcba_hamming_global_20260825_004612.csv",
   "global_median": "drugclip_pcba_hamming_global_median_20260826_174140.csv"},
 ("DrugCLIP", "DUD-E"): {"cosine": "drugclip_dude_cosine_20260825_150207.csv",
   "none": "drugclip_dude_hamming_20260825_150214.csv",
   "per_type": "drugclip_dude_hamming_per_type_20260825_150221.csv",
   "per_type_median": "drugclip_dude_hamming_per_type_median_20260826_174154.csv",
   "global": "drugclip_dude_hamming_global_20260825_150228.csv",
   "global_median": "drugclip_dude_hamming_global_median_20260826_174214.csv"},
}


def per_target(csv, col="auroc"):
    d = pd.read_csv(M / csv)
    d = d[d["target"].astype(str).str.lower() != "mean"]
    return d[col].dropna().values


def fig_centering():
    """Continuous vs the best binarized condition, per model and benchmark.

    The full six-condition grid is Table 4; what a table cannot show is the per-target
    spread, which is what this is for.
    """
    best = {("SPRINT", "LIT-PCBA"): "per_type", ("SPRINT", "DUD-E"): "per_type_median",
            ("DrugCLIP", "LIT-PCBA"): "global", ("DrugCLIP", "DUD-E"): "global_median"}
    fig, ax = plt.subplots(figsize=(4.9, 2.3))
    pos, labels = [], []
    for i, key in enumerate([("SPRINT", "LIT-PCBA"), ("SPRINT", "DUD-E"),
                             ("DrugCLIP", "LIT-PCBA"), ("DrugCLIP", "DUD-E")]):
        cont = per_target(SRC[key]["cosine"])
        binr = per_target(SRC[key][best[key]])
        b = ax.boxplot([cont, binr], positions=[i * 2.4, i * 2.4 + 0.85], widths=0.72,
                       patch_artist=True, medianprops=dict(color="black", lw=1.2),
                       flierprops=dict(marker="o", ms=2, alpha=0.5))
        b["boxes"][0].set_facecolor("#9ecae1"); b["boxes"][1].set_facecolor("#fdd0a2")
        pos.append(i * 2.4 + 0.42)
        labels.append(f"{key[0]}\n{key[1]}\n({best[key]})")
    ax.axhline(0.5, ls=":", lw=1, color="grey")
    ax.set_xticks(pos); ax.set_xticklabels(labels, fontsize=7)
    ax.set_ylabel("AUROC per target")
    ax.set_title("Continuous (blue) vs. best binarized condition (orange)", fontsize=9.5)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout(); fig.savefig(OUT / "centering_sweep.png", dpi=200)
    print("  centering_sweep.png")


def fig_lambda():
    """the trained head's lambda curve, DrugCLIP on DUD-E."""
    rows = []
    for f in sorted((M / "drugclip_dude_heads").glob("head_lam*_none.csv")):
        m = re.search(r"lam(\d+)p(\d+)_none", f.name)
        lam = float(f"{m.group(1)}.{m.group(2)}")
        d = pd.read_csv(f); d = d[d["target"] != "mean"]
        rows.append((lam, d.auroc.mean(), d.bedroc.mean(), d["ef_0.005"].mean()))
    rows.sort()
    lam, au, be, ef = map(np.array, zip(*rows))
    x = np.where(lam == 0, 0.01, lam)          # 0 cannot sit on a log axis
    fig, axes = plt.subplots(1, 3, figsize=(4.9, 1.8))
    for ax, y, name, base in zip(axes, [au, be, ef],
                                 ["AUROC", "BEDROC", "EF 0.5%"], [au[0], be[0], ef[0]]):
        ax.plot(x, y, "o-", color="#e6550d", lw=1.6, ms=4)
        ax.axhline(base, ls="--", lw=1, color="grey", label="$\\lambda=0$")
        ax.axvline(3.0, ls=":", lw=1, color="#3182bd", label="$\\lambda=3$")
        ax.set_xscale("log"); ax.set_xlabel("$\\lambda$"); ax.set_title(name, fontsize=9.5)
        ax.grid(alpha=0.3)
        ax.set_xticks([0.01, 0.1, 1, 10, 100])
        ax.set_xticklabels(["0", "0.1", "1", "10", "100"])
        ax.margins(y=0.18)   # the lambda=3 AUROC point otherwise sits on the top spine
    axes[0].legend(fontsize=8)
    # no suptitle: at this width it was clipped at both edges, and the report caption
    # already says which model, benchmark and target count this is.
    fig.tight_layout()
    fig.savefig(OUT / "lambda_sweep.png", dpi=200, bbox_inches="tight")
    print("  lambda_sweep.png")


def fig_distributions():
    """why the two encoders differ: where the mean sits in each dimension."""
    fig, axes = plt.subplots(1, 2, figsize=(6.0, 2.3))
    for ax, (name, path, col) in zip(axes, [
            ("DrugCLIP (128-d)", M / "drugclip_dude/embeddings/aa2ar/mol_reps.npy", "#3182bd"),
            ("SPRINT (1024-d)", M / "sprint_dude/embeddings/aa2ar/mol_reps.npy", "#e6550d")]):
        X = np.load(path).astype(np.float64)
        pct = [(X[:, j] < X[:, j].mean()).mean() * 100 for j in range(X.shape[1])]
        ax.hist(pct, bins=40, color=col, alpha=0.85)
        ax.axvline(np.mean(pct), color="black", lw=1.4,
                   label=f"median dim: {np.median(pct):.1f}th pct")
        ax.axvline(50, ls="--", lw=1, color="grey")
        ax.set_xlim(0, 100)
        ax.set_xlabel("percentile of the mean within its dimension")
        ax.set_title(f"{name}, target aa2ar, n={X.shape[0]:,} molecules", fontsize=9)
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3)
    axes[0].set_ylabel("number of dimensions")
    fig.suptitle("Mean-centering splits DrugCLIP's dimensions in half, but not SPRINT's",
                 fontsize=10.5)
    fig.tight_layout()
    fig.savefig(OUT / "dimension_statistics.png", dpi=200)
    print("  dimension_statistics.png")


def fig_two_routes():
    """the central synthesis: the two routes move AUROC alike but EF oppositely."""
    cols = ["auroc", "bedroc", "ef_0.005"]
    cos = [per_target("drugclip_dude_cosine_20260825_150207.csv", c).mean() for c in cols]
    glo = [per_target("drugclip_dude_hamming_global_20260825_150228.csv", c).mean() for c in cols]
    l0 = [per_target("drugclip_dude_heads/head_lam0p00_none.csv", c).mean() for c in cols]
    l3 = [per_target("drugclip_dude_heads/head_lam3p00_none.csv", c).mean() for c in cols]
    post = [(g - c) / c * 100 for g, c in zip(glo, cos)]
    trained = [(b - a) / a * 100 for b, a in zip(l3, l0)]

    fig, ax = plt.subplots(figsize=(5.2, 2.5))
    x = np.arange(3); w = 0.36
    ax.bar(x - w/2, post, w, color="#3182bd", label="post-hoc centering (no training)")
    ax.bar(x + w/2, trained, w, color="#e6550d", label="trained head, $\\lambda=3$")
    ax.axhline(0, color="black", lw=0.9)
    for xi, (a, b) in enumerate(zip(post, trained)):
        ax.text(xi - w/2, a + (0.6 if a >= 0 else -1.6), f"{a:+.1f}%", ha="center", fontsize=9)
        ax.text(xi + w/2, b + (0.6 if b >= 0 else -1.6), f"{b:+.1f}%", ha="center", fontsize=9)
    ax.set_xticks(x); ax.set_xticklabels(["AUROC", "BEDROC", "EF 0.5%"])
    ax.set_ylabel("change vs. its own baseline (%)")
    ax.set_title("Two routes to a binary code, DrugCLIP on DUD-E (100 targets)", fontsize=9.5)
    ax.legend(fontsize=8, loc="upper left"); ax.grid(axis="y", alpha=0.3)
    ax.set_ylim(min(post + trained) - 4, max(post + trained) + 5)
    fig.tight_layout(); fig.savefig(OUT / "two_routes.png", dpi=200)
    print("  two_routes.png")


def fig_tanh():
    """the degenerate case L2-normalisation prevents, measured on real embeddings."""
    X = np.load(M / "drugclip_dude/embeddings/aa2ar/mol_reps.npy")[:4000]
    x = torch.from_numpy(X.astype(np.float32))
    W = torch.eye(x.shape[1])
    base = (x @ W >= 0).numpy()
    scales = np.logspace(0, 2.2, 25)
    hu, fu, hn, fn = [], [], [], []
    for s in scales:
        y = torch.tanh(x @ (W * s))                      # unnormalised tanh head
        hu.append((((y - torch.sign(y)) ** 2).sum(1).mean() / x.shape[1]).item())
        fu.append(((y.numpy() >= 0) != base).mean())
        z = x @ (W * s)                                  # L2-normalised head
        z = z / z.norm(dim=1, keepdim=True).clamp_min(1e-12)
        hn.append((((z - torch.sign(z)) ** 2).sum(1).mean() / x.shape[1]).item())
        fn.append(((z.numpy() >= 0) != base).mean())
    fig, axes = plt.subplots(1, 2, figsize=(6.0, 2.3))
    axes[0].plot(scales, hu, "o-", color="#e6550d", ms=2.6, label="unnormalised $\\tanh$")
    axes[0].plot(scales, hn, "s-", color="#3182bd", ms=2.6, label="L2-normalised")
    axes[0].set_xscale("log"); axes[0].set_yscale("log")
    axes[0].set_ylabel("$L_{\\rm hash}$ per dimension")
    axes[0].set_title("The hashing loss collapses...", fontsize=9.5)
    axes[1].plot(scales, np.array(fu) * 100, "o-", color="#e6550d", ms=2.6, label="unnormalised $\\tanh$")
    axes[1].plot(scales, np.array(fn) * 100, "s-", color="#3182bd", ms=2.6, label="L2-normalised")
    axes[1].set_xscale("log"); axes[1].set_ylim(-2, 50)
    axes[1].set_ylabel("% of bits changed")
    axes[1].set_title("...but no bit actually changes", fontsize=9.5)
    for ax in axes:
        ax.set_xlabel("weight scale applied to $W$"); ax.legend(fontsize=8); ax.grid(alpha=0.3)
    fig.suptitle("Why the head output must be L2-normalised", fontsize=10.5)
    fig.tight_layout(); fig.savefig(OUT / "tanh_degeneracy.png", dpi=200)
    print("  tanh_degeneracy.png")


if __name__ == "__main__":
    fig_centering()
    fig_lambda()
    print(f"  -> {OUT}")
