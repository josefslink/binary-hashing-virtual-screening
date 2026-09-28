from __future__ import annotations

import argparse
import warnings
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import pytorch_lightning as pl
import torch

from src.core.config import load_yaml_config
from src.core.wrapper import SprintWrapper
from src.evaluation.evaluator import evaluate_screening_targets, list_dude_targets, save_results_csv
from ultrafast.datamodules import DTIDataModule


def get_available_checkpoints(checkpoint_dir: str | Path) -> dict[str, str]:
    checkpoints = {}
    checkpoint_path = Path(checkpoint_dir)
    if checkpoint_path.exists():
        for ckpt in checkpoint_path.glob("*.ckpt"):
            checkpoints[ckpt.stem] = str(ckpt)  # keyed by filename minus .ckpt, e.g. "sprint"
    return checkpoints


def visualize_lit_pcba_csv(csv_path: str) -> str | None:
    csv_file = Path(csv_path)
    if not csv_file.exists():
        print(f"Error: CSV not found: {csv_path}")
        return None

    df = pd.read_csv(csv_file)
    if df.empty:
        print("Error: CSV is empty. Nothing to visualize.")
        return None

    if "error" in df.columns:
        df = df[df["error"].isna() | (df["error"] == "")]  # drop targets that failed to score

    metric_cols = ["auroc", "bedroc", "ef_0.005", "ef_0.01", "ef_0.05"]
    missing_cols = [column for column in metric_cols + ["target"] if column not in df.columns]
    if missing_cols:
        print(f"Error: CSV missing required columns: {missing_cols}")
        return None

    df = df.dropna(subset=metric_cols, how="all")  # rows where every metric is NaN, not just one
    if df.empty:
        print("Error: no valid metric rows to visualize.")
        return None

    df = df.sort_values(by="auroc", ascending=False)  # best target first, left to right
    fig, axes = plt.subplots(3, 2, figsize=(14, 12))  # 6 slots for 5 metrics, one spare
    axes = axes.flatten()

    for index, metric in enumerate(metric_cols):
        ax = axes[index]
        ax.bar(df["target"], df[metric])
        ax.set_title(metric.upper())
        ax.set_xlabel("Target")
        ax.set_ylabel(metric.upper())
        ax.tick_params(axis="x", rotation=45, labelsize=8)
        ax.grid(axis="y", linestyle="--", alpha=0.4)

    for index in range(len(metric_cols), len(axes)):
        axes[index].axis("off")  # hide the unused 6th subplot

    fig.tight_layout()
    output_file = str(csv_file.with_suffix("")) + "_plots.png"  # next to the CSV it came from
    fig.savefig(output_file, dpi=200)
    plt.close(fig)
    print(f"Saved visualization to: {output_file}")
    return output_file


def benchmark_standard_dataset(
    checkpoint_path: str,
    dataset: str,
    batch_size: int,
    num_workers: int,
    device: int,
) -> dict:
    # SprintWrapper isn't a LightningModule itself; .model is the wrapped one Trainer needs
    model = SprintWrapper.from_checkpoint(checkpoint_path, device=torch.device(f"cuda:{device}") if torch.cuda.is_available() else torch.device("cpu")).model
    # each branch is a DTIDataModule preset; the task name is the only thing that differs
    if dataset == "davis":
        datamodule = DTIDataModule(task="davis", batch_size=batch_size, num_workers=num_workers)
    elif dataset == "biosnap":
        datamodule = DTIDataModule(task="biosnap", batch_size=batch_size, num_workers=num_workers)
    elif dataset == "bindingdb":
        datamodule = DTIDataModule(task="bindingdb", batch_size=batch_size, num_workers=num_workers)
    elif dataset == "merged":
        datamodule = DTIDataModule(task="merged", batch_size=batch_size, num_workers=num_workers)
    else:
        raise ValueError(f"Unknown dataset: {dataset}")

    datamodule.setup(stage="test")
    trainer = pl.Trainer(
        devices=[device],  # a list of device indices, not a device count
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
        max_epochs=1,
        logger=False,
        enable_progress_bar=True,
    )
    results = trainer.test(model, datamodule=datamodule, verbose=True)
    return results[0] if results else {}  # one dict per dataloader; there's only ever one


def main() -> None:
    # pandas/matplotlib/sklearn imports install warnings filters that swallow the
    # scorer's single-row centering warning; re-assert it so sweeps surface it.
    warnings.filterwarnings("always", message=".*centering skipped.*")
    parser = argparse.ArgumentParser(description="Benchmark SPRINT models with shared evaluation logic.")
    parser.add_argument("--config", type=str, default=None)
    parser.add_argument("--checkpoint", type=str, help="Path to a checkpoint. If omitted, auto-select the default.")
    parser.add_argument("--dataset", type=str, default=None, choices=["davis", "biosnap", "bindingdb", "merged", "lit_pcba", "dude"])
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--device", type=int, default=None)
    parser.add_argument("--num-workers", type=int, default=None)
    parser.add_argument("--pcba-dir", type=str, default=None, dest="pcba_dir")
    parser.add_argument("--dude-dir", type=str, default=None, dest="dude_dir",
                        help="DUD-E directory (default data/DUDe): *_actives.tsv, *_decoys.tsv, dude_targets.fasta")
    parser.add_argument("--target-protein-id", type=str, default=None, dest="target_protein_id")
    parser.add_argument("--visualize-csv", type=str, default=None, dest="visualize_csv")
    parser.add_argument("--metric", type=str, choices=["cosine", "hamming"], default=None)
    parser.add_argument("--binarize-center", choices=["none", "per_type", "per_type_median", "global", "global_median"], default=None, dest="binarize_center")
    parser.add_argument("--plot-binarized-dist", action="store_true", dest="plot_binarized_dist")
    parser.add_argument("--list", action="store_true")
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Write the results CSV here instead of the default "
             "outputs/metrics/lit_pcba_results_<condition>_<timestamp>.csv. Use this for "
             "control or regression runs so they cannot become the newest match that "
             "plot_metrics.py resolves for a condition.",
    )
    args = parser.parse_args()

    config_values = {}
    if args.config:
        config_values.update(load_yaml_config(args.config))
    cli_overrides = {
        "checkpoint_path": args.checkpoint,
        "dataset": args.dataset,
        "batch_size": args.batch_size,
        "device": args.device,
        "num_workers": args.num_workers,
        "pcba_dir": args.pcba_dir,
        "target_protein_id": args.target_protein_id,
        "metric": args.metric,
        "binarize_center": args.binarize_center,
        "dude_dir": args.dude_dir,
    }
    # every CLI flag above defaults to None, so this only overrides a config value when
    # the flag was actually passed
    for key, value in cli_overrides.items():
        if value is not None:
            config_values[key] = value

    # re-apply the same defaults argparse would have used, now that config values won
    args.checkpoint = config_values.get("checkpoint_path")
    args.dataset = config_values.get("dataset", "davis")
    # cast in case a yaml config supplied these as strings; argparse would have done this
    args.batch_size = int(config_values.get("batch_size", 64))
    args.device = int(config_values.get("device", 0))
    args.num_workers = int(config_values.get("num_workers", 0))
    args.pcba_dir = config_values.get("pcba_dir", "data/lit_pcba")
    args.dude_dir = config_values.get("dude_dir", "data/DUDe")
    args.target_protein_id = config_values.get("target_protein_id", "all")
    args.metric = config_values.get("metric", "cosine")
    args.binarize_center = config_values.get("binarize_center", "none")

    if args.visualize_csv:
        visualize_lit_pcba_csv(args.visualize_csv)
        return

    base_dir = Path(__file__).resolve().parent.parent  # repo root, one level above scripts/
    checkpoint_dir = base_dir / "checkpoints" / "sprint_official"
    available = get_available_checkpoints(checkpoint_dir)

    if args.list:
        print("Available checkpoints:")
        for name, path in available.items():
            print(f"  {name}: {path}")
        if not available:
            print("  No checkpoints found")
        return

    if args.checkpoint:
        checkpoint_to_use = args.checkpoint
    else:
        if not available:
            raise SystemExit(f"No checkpoints found in {checkpoint_dir}")
        # prefer the one literally named "sprint"
        checkpoint_to_use = available.get("sprint", next(iter(available.values())))

    if args.dataset in ("lit_pcba", "dude"):
        wrapper = SprintWrapper.from_checkpoint(checkpoint_to_use, device=torch.device(f"cuda:{args.device}") if torch.cuda.is_available() else torch.device("cpu"))
        data_dir = args.pcba_dir if args.dataset == "lit_pcba" else args.dude_dir
        if str(args.target_protein_id).lower() == "all":  # sentinel for every target in data_dir
            if args.dataset == "dude":
                targets = list_dude_targets(data_dir)  # filtered against dude_targets.fasta
            else:
                # one dir per target
                targets = sorted(path.name for path in Path(data_dir).iterdir() if path.is_dir())
        else:
            targets = [t.strip() for t in str(args.target_protein_id).split(",") if t.strip()]
        print(f"Dataset: {args.dataset}  |  data dir: {data_dir}  |  {len(targets)} target(s)")

        results = evaluate_screening_targets(
            model_wrapper=wrapper,
            data_dir=data_dir,
            targets=targets,
            dataset=args.dataset,
            target_featurizer=wrapper.model.args.target_featurizer,  # the checkpoint's own, no CLI override
            metric=args.metric,
            binarize_center=None if args.binarize_center == "none" else args.binarize_center,
            plot_binarized_dist=args.plot_binarized_dist,
            checkpoint_name=Path(checkpoint_to_use).name,
        )

        # filename encodes the scoring condition so plot_metrics.py can tell runs apart
        suffix = f"_{args.metric}"
        if args.metric == "hamming" and args.binarize_center != "none":
            suffix += f"_{args.binarize_center}"
        # sprint_ prefix keeps this apart from drugclip's own dude results
        stem = "lit_pcba_results" if args.dataset == "lit_pcba" else "sprint_dude_results"
        output_file = (
            Path(args.output)
            if args.output
            else base_dir / "outputs" / "metrics" / f"{stem}{suffix}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        )
        output_path = save_results_csv(results, output_file)
        print(f"Saved results to: {output_path}")
        print(pd.DataFrame(results).to_string(index=False))
        return

    results = benchmark_standard_dataset(
        checkpoint_path=checkpoint_to_use,
        dataset=args.dataset,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        device=args.device,
    )
    for key, value in results.items():
        if isinstance(value, float):
            print(f"{key}: {value:.4f}")
        else:  # non-numeric entries print as-is
            print(f"{key}: {value}")


if __name__ == "__main__":
    main()
