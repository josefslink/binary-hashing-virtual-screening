#!/usr/bin/env python3
# summarise the hashing sweeps: one figure per (benchmark, model), plus a comparison table.
# --benchmark {lit_pcba,dude,all} --model {SPRINT,DrugCLIP,all}

from __future__ import annotations

import argparse
import csv
import re
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

METRICS = ["auroc", "bedroc", "ef_0.005", "ef_0.01", "ef_0.05"]
METRIC_LABELS = {
    "auroc": "AUROC",
    "bedroc": "BEDROC",
    "ef_0.005": "EF 0.5%",
    "ef_0.01": "EF 1%",
    "ef_0.05": "EF 5%",
}

CONDITIONS = ["cosine", "none", "per_type", "per_type_median", "global", "global_median"]
CONDITION_TITLES = {
    "cosine": "Continuous (cosine)",
    "none": "Binary, no centering",
    "per_type": "Binary + per_type mean centering",
    "per_type_median": "Binary + per_type median centering",
    "global": "Binary + global mean centering",
    "global_median": "Binary + global median centering",
}
# mean and median of the same centering share a hue; median is the lighter of the pair
CONDITION_COLORS = {
    "cosine": "#4C78A8",
    "none": "#F58518",
    "per_type": "#54A24B",
    "per_type_median": "#88D27A",
    "global": "#B279A2",
    "global_median": "#D4A6C8",
}

# (benchmark, model) -> {condition: anchored filename regex}, matching exactly the naming
# scheme benchmark_sprint.py and evaluate_drugclip_hamming.py write.
SERIES = {
    ("lit_pcba", "SPRINT"): {
        "cosine": r"^lit_pcba_results_cosine_\d{8}_\d{6}\.csv$",
        "none": r"^lit_pcba_results_hamming_\d{8}_\d{6}\.csv$",
        "per_type": r"^lit_pcba_results_hamming_per_type_\d{8}_\d{6}\.csv$",
        "per_type_median": r"^lit_pcba_results_hamming_per_type_median_\d{8}_\d{6}\.csv$",
        "global": r"^lit_pcba_results_hamming_global_\d{8}_\d{6}\.csv$",
        "global_median": r"^lit_pcba_results_hamming_global_median_\d{8}_\d{6}\.csv$",
    },
    ("lit_pcba", "DrugCLIP"): {
        "cosine": r"^drugclip_pcba_cosine_\d{8}_\d{6}\.csv$",
        "none": r"^drugclip_pcba_hamming_\d{8}_\d{6}\.csv$",
        "per_type": r"^drugclip_pcba_hamming_per_type_\d{8}_\d{6}\.csv$",
        "per_type_median": r"^drugclip_pcba_hamming_per_type_median_\d{8}_\d{6}\.csv$",
        "global": r"^drugclip_pcba_hamming_global_\d{8}_\d{6}\.csv$",
        "global_median": r"^drugclip_pcba_hamming_global_median_\d{8}_\d{6}\.csv$",
    },
    ("dude", "SPRINT"): {
        "cosine": r"^sprint_dude_results_cosine_\d{8}_\d{6}\.csv$",
        "none": r"^sprint_dude_results_hamming_\d{8}_\d{6}\.csv$",
        "per_type": r"^sprint_dude_results_hamming_per_type_\d{8}_\d{6}\.csv$",
        "per_type_median": r"^sprint_dude_results_hamming_per_type_median_\d{8}_\d{6}\.csv$",
        "global": r"^sprint_dude_results_hamming_global_\d{8}_\d{6}\.csv$",
        "global_median": r"^sprint_dude_results_hamming_global_median_\d{8}_\d{6}\.csv$",
    },
    ("dude", "DrugCLIP"): {
        "cosine": r"^drugclip_dude_cosine_\d{8}_\d{6}\.csv$",
        "none": r"^drugclip_dude_hamming_\d{8}_\d{6}\.csv$",
        "per_type": r"^drugclip_dude_hamming_per_type_\d{8}_\d{6}\.csv$",
        "per_type_median": r"^drugclip_dude_hamming_per_type_median_\d{8}_\d{6}\.csv$",
        "global": r"^drugclip_dude_hamming_global_\d{8}_\d{6}\.csv$",
        "global_median": r"^drugclip_dude_hamming_global_median_\d{8}_\d{6}\.csv$",
    },
}

# One figure per (benchmark, model). The original filenames are kept so existing
# references to them keep resolving.
FIGURES = {
    ("lit_pcba", "SPRINT"): {
        "suptitle": "SPRINT on LIT-PCBA: cost of post-hoc binary hashing",
        "ylabel": "Mean over LIT-PCBA targets",
        "filename": "lit_pcba_hashing_sweep.png",
    },
    ("lit_pcba", "DrugCLIP"): {
        "suptitle": "DrugCLIP on LIT-PCBA: cost of post-hoc binary hashing",
        "ylabel": "Mean over LIT-PCBA targets",
        "filename": "lit_pcba_hashing_sweep_drugclip.png",
    },
    ("dude", "DrugCLIP"): {
        "suptitle": "DrugCLIP on DUD-E: cost of post-hoc binary hashing",
        "ylabel": "Mean over DUD-E targets",
        "filename": "dude_hashing_sweep.png",
    },
    ("dude", "SPRINT"): {
        "suptitle": "SPRINT on DUD-E: cost of post-hoc binary hashing",
        "ylabel": "Mean over DUD-E targets",
        "filename": "dude_hashing_sweep_sprint.png",
    },
}
BENCHMARKS = sorted({b for b, _m in FIGURES})
MODELS = sorted({m for _b, m in FIGURES})

# Transcribed from the papers, re-verified against the primary sources on 2026-08-25.
# AUROC/BEDROC are reported on a 0-100 scale in all three papers and divided by 100 here.
# These are the only rows in the comparison table that do not come from a result CSV.
PUBLISHED = [
    # benchmark, label, auroc, bedroc, ef0.5, ef1, ef5, citation
    ("lit_pcba", "SPRINT (16M), published", 0.734, 0.123, 15.90, 10.78, 5.29,
     "arXiv:2411.15418 Table 2"),
    ("lit_pcba", "DrugCLIP, published (own paper)", 0.5717, 0.0623, 8.56, 5.51, 2.27,
     "arXiv:2310.06367 Table 3"),
    ("lit_pcba", "DrugCLIP, retrained by DrugHash authors", 0.5636, 0.0678, 7.77, 5.66, 2.32,
     "arXiv:2407.19790 Table 2"),
    ("lit_pcba", "DrugHash, published", 0.5458, 0.0714, 9.65, 6.14, 2.42,
     "arXiv:2407.19790 Table 2"),
    ("dude", "DrugCLIP_ZS, published", 0.8093, 0.5052, 38.07, 31.89, 10.66,
     "arXiv:2310.06367 Table 1"),
    # DrugHash reports its own DrugCLIP baseline, and it is NOT DrugCLIP's published row.
    # DrugHash measures its gain against this one, so it is the correct denominator for any
    # "what does trained hashing buy" claim. Carrying both also restores symmetry with
    # LIT-PCBA, where two DrugCLIP reference rows were already listed.
    ("dude", "DrugCLIP, retrained by DrugHash authors", 0.7945, 0.4782, 37.86, 30.76, 10.10,
     "arXiv:2407.19790 Table 1"),
    ("dude", "DrugHash, published", 0.8373, 0.5716, 43.03, 37.18, 12.07,
     "arXiv:2407.19790 Table 1"),
]

DEFAULT_METRICS_DIR = Path(__file__).resolve().parent.parent / "outputs" / "metrics"
DEFAULT_VIZ_DIR = Path(__file__).resolve().parent.parent / "outputs" / "visualizations"


def resolve_condition(
    condition: str,
    benchmark: str = "lit_pcba",
    model: str = "SPRINT",
    metrics_dir: Path | None = None,
) -> str | None:
    """Newest CSV whose filename matches this (benchmark, model, condition) exactly."""
    # "newest" reads the _YYYYMMDD_HHMMSS filename stamp, not mtime, as the primary key
    # (mtime is only the tiebreak). A fresh git checkout writes every tracked CSV in the
    # same instant, so mtime ordering alone once resolved SPRINT LIT-PCBA per_type to the
    # pre-fix degenerate run (0.6949, since retracted) instead of the
    # post-fix 0.6662, and that number made it into the committed comparison table.
    base_dir = metrics_dir or DEFAULT_METRICS_DIR
    patterns = SERIES.get((benchmark, model))
    if not patterns or condition not in patterns:
        return None
    pattern = re.compile(patterns[condition])
    matches = [p for p in base_dir.glob("*.csv") if pattern.match(p.name)]
    if not matches:
        return None
    return str(max(matches, key=_run_order))


_STAMP = re.compile(r"_(\d{8}_\d{6})\.csv$")


def _run_order(path: Path) -> tuple[str, int]:
    stamp = _STAMP.search(path.name)
    return (stamp.group(1) if stamp else "", path.stat().st_mtime_ns)


def _valid_rows(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    if "error" in df.columns:
        df = df[df["error"].isna() | (df["error"] == "")]  # failed targets, not missing-metric rows
    return df.dropna(subset=METRICS, how="all")  # keep a row that has any metric; drop only all-NaN


def load_and_summarize(path: str) -> pd.Series:
    return _valid_rows(path)[METRICS].mean()  # unweighted across targets, same as the figure ylabel


def count_valid_targets(path: str) -> int:
    """Rows that survive the error/NaN filter, not raw CSV rows."""
    return len(_valid_rows(path))


def resolve_series(benchmark: str, model: str, metrics_dir: Path | None = None) -> dict[str, str]:
    paths = {}
    for condition in CONDITIONS:
        resolved = resolve_condition(condition, benchmark, model, metrics_dir)
        if resolved:
            paths[condition] = resolved
    return paths


def warn_on_partial_runs(paths: dict[str, str]) -> None:
    """Differing target counts across conditions means one is a partial (debug) run."""
    counts = {c: count_valid_targets(p) for c, p in paths.items()}
    if len(set(counts.values())) > 1:
        print("  ! conditions disagree on target count -- one of these is probably a partial run:")
        for condition, count in counts.items():
            print(f"      {condition:9} n={count:<4} {Path(paths[condition]).name}")


def plot_sweep(paths: dict[str, str], out_path: Path, benchmark: str, model: str) -> str:
    spec = FIGURES[(benchmark, model)]
    conditions = [c for c in CONDITIONS if c in paths]
    summaries = {c: load_and_summarize(paths[c]) for c in conditions}
    n_targets = {c: count_valid_targets(paths[c]) for c in conditions}

    fig, axes = plt.subplots(1, len(METRICS), figsize=(4.0 * len(METRICS), 5.2))  # one y-axis per metric
    for ax, metric in zip(axes, METRICS):
        values = [summaries[c][metric] for c in conditions]
        bars = ax.bar(range(len(conditions)), values,
                      color=[CONDITION_COLORS[c] for c in conditions])
        ax.set_title(METRIC_LABELS[metric], fontsize=12)
        ax.set_xticks(range(len(conditions)))
        ax.set_xticklabels([])  # filenames live in the legend; ticks can't hold them
        ax.grid(axis="y", linestyle="--", alpha=0.4)
        ax.set_axisbelow(True)
        ax.set_ylim(0, max(values) * 1.18 if max(values) > 0 else 1)  # headroom for the bar labels
        for bar, value in zip(bars, values):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                    f"{value:.3f}" if value < 1 else f"{value:.2f}",  # AUROC/BEDROC vs EF
                    ha="center", va="bottom", fontsize=9)
    axes[0].set_ylabel(spec["ylabel"])

    handles = [plt.Rectangle((0, 0), 1, 1, color=CONDITION_COLORS[c]) for c in conditions]
    labels = [f"{CONDITION_TITLES[c]}  —  {Path(paths[c]).name}  (n={n_targets[c]})"
              for c in conditions]
    fig.legend(handles, labels, loc="lower center", ncol=1, frameon=False, fontsize=9)
    fig.suptitle(spec["suptitle"], fontsize=14)
    fig.tight_layout(rect=(0, 0.02 + 0.035 * len(conditions), 1, 0.96))  # legend is one row per condition

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=200)
    plt.close(fig)
    return str(out_path)


def write_comparison_table(out_csv: Path, metrics_dir: Path | None = None) -> str:
    """Reproduced rows (from CSVs) and published rows (from papers) in one table."""
    rows = []
    for (benchmark, model), _patterns in SERIES.items():
        paths = resolve_series(benchmark, model, metrics_dir)
        for condition in CONDITIONS:
            path = paths.get(condition)
            if not path:
                continue
            summary = load_and_summarize(path)
            rows.append({
                "benchmark": benchmark,
                "model": model,
                "scoring": "cosine (continuous)" if condition == "cosine"
                           else f"hamming ({condition})",
                "auroc": round(float(summary["auroc"]), 6),  # 6 dp; EF is 4, as in the committed csv
                "bedroc": round(float(summary["bedroc"]), 6),
                "ef_0.005": round(float(summary["ef_0.005"]), 4),
                "ef_0.01": round(float(summary["ef_0.01"]), 4),
                "ef_0.05": round(float(summary["ef_0.05"]), 4),
                "n_targets": count_valid_targets(path),
                "source": "reproduced",
                "source_ref": Path(path).name,
            })
    for benchmark, label, auroc, bedroc, ef5, ef1, ef50, ref in PUBLISHED:
        rows.append({
            "benchmark": benchmark,
            "model": label.split(",")[0],  # "SPRINT (16M)" / "DrugCLIP"; full label goes in scoring
            "scoring": label,
            "auroc": auroc, "bedroc": bedroc,
            "ef_0.005": ef5, "ef_0.01": ef1, "ef_0.05": ef50,
            "n_targets": "",
            "source": "paper",
            "source_ref": ref,
        })

    order = {"lit_pcba": 0, "dude": 1}
    # lit_pcba then dude; reproduced above paper so the table reads as "ours, then the citations"
    rows.sort(key=lambda r: (order.get(r["benchmark"], 9), r["source"] == "paper", r["model"]))

    out_csv.parent.mkdir(parents=True, exist_ok=True)
    fields = ["benchmark", "model", "scoring", "auroc", "bedroc",
              "ef_0.005", "ef_0.01", "ef_0.05", "n_targets", "source", "source_ref"]
    with open(out_csv, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    return str(out_csv)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Summarise the hashing sweeps: one figure per (benchmark, model), plus a comparison table."
    )
    parser.add_argument("--benchmark", choices=BENCHMARKS + ["all"], default="all",
                        help="Which benchmark's figure(s) to render (default: all)")
    parser.add_argument("--model", choices=MODELS + ["all"], default="all",
                        help="Which model's figure(s) to render (default: all)")
    parser.add_argument("--default", dest="cosine", help="Continuous/cosine CSV")
    parser.add_argument("--binary", dest="none", help="Hamming, no centering CSV")
    parser.add_argument("--binary-mean-centering", dest="per_type", help="Hamming, per_type CSV")
    parser.add_argument("--binary-global", dest="global_", help="Hamming, global CSV")
    parser.add_argument("--output", dest="output", default=None, help="Output PNG path")
    # flag alone writes outputs/metrics/stage1_comparison.csv; pass a path to override
    parser.add_argument("--comparison-csv", dest="comparison_csv", nargs="?",
                        const=str(DEFAULT_METRICS_DIR / "stage1_comparison.csv"), default=None,
                        help="Also write the reproduced-vs-published comparison table")
    parser.add_argument("--no-figure", action="store_true", help="Only write the table")
    args = parser.parse_args()

    # median variants have no CLI flag; they only appear when auto-resolved from SERIES
    explicit = {"cosine": args.cosine, "none": args.none,
                "per_type": args.per_type, "global": args.global_}
    if any(explicit.values()) and args.benchmark == "all":
        args.benchmark = "lit_pcba"  # explicit paths only make sense for one benchmark

    if not args.no_figure:
        wanted = [
            (b, m) for (b, m) in FIGURES
            if args.benchmark in ("all", b) and args.model in ("all", m)
        ]
        for benchmark, model in sorted(wanted):
            paths: dict[str, str] = {}
            for condition in CONDITIONS:
                chosen = explicit.get(condition) or resolve_condition(condition, benchmark, model)
                if chosen:
                    paths[condition] = chosen
            if len(paths) < 2:  # one bar is not a sweep; skip rather than half-render
                print(f"  ! [{benchmark}/{model}] fewer than two conditions found — figure skipped")
                continue
            if len(paths) < len(CONDITIONS):
                missing = [c for c in CONDITIONS if c not in paths]
                print(f"  ! [{benchmark}/{model}] no CSV for {missing} — panel(s) omitted")
            warn_on_partial_runs(paths)
            # --output is one path; --benchmark all would overwrite. omit it for the usual sweep
            out_path = Path(args.output) if args.output else DEFAULT_VIZ_DIR / FIGURES[(benchmark, model)]["filename"]
            print(f"[{benchmark}/{model}] conditions plotted:")
            for condition, path in paths.items():
                print(f"  {condition:9} {Path(path).name}")
            print(f"  -> {plot_sweep(paths, out_path, benchmark, model)}")

    if args.comparison_csv:
        print(f"comparison table -> {write_comparison_table(Path(args.comparison_csv))}")


if __name__ == "__main__":
    main()
