# run SPRINT DTI benchmarks on BIOSNAP, DAVIS, and/or BindingDB.
# results go to outputs/metrics/sprint_dti_results_<timestamp>.csv
from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path

import pandas as pd
import pytorch_lightning as pl
import torch

from src.core.config import load_yaml_config
from src.core.wrapper import SprintWrapper
from ultrafast.datamodules import DTIDataModule, get_task_dir
from ultrafast.utils import get_featurizer

# paper AUPR. the gap is expected, not a bug: released checkpoints are LIT-PCBA
# screening models (paper Table 2) and no DTI checkpoint was published, so this is a
# screening model zero-shot on DTI. BIOSNAP lands at AUPR 0.574 against the paper's
# 0.936. per-task training is out of scope; this run is one documented data point.
SUPPORTED_DATASETS = ["biosnap", "davis", "bindingdb"]
PAPER_REFERENCE = {
    "biosnap":   {"aupr": 0.936},
    "davis":     {"aupr": 0.507},
    "bindingdb": {"aupr": 0.718},
}


def get_available_checkpoints(checkpoint_dir: Path) -> dict[str, str]:
    checkpoints = {}
    if checkpoint_dir.exists():
        for ckpt in checkpoint_dir.glob("*.ckpt"):
            checkpoints[ckpt.stem] = str(ckpt)  # keyed by filename minus .ckpt, e.g. "sprint"
    return checkpoints


def run_dti_dataset(
    checkpoint_path: str,
    dataset: str,
    batch_size: int,
    num_workers: int,
    device: int,
    featurizer_batch_size: int = 16,
) -> dict[str, float]:
    """Load a SPRINT checkpoint and run trainer.test() on one DTI dataset."""
    # featurizer_batch_size: sequences per GPU call during the one-time SaProt LMDB
    # write. 32/64 roughly halve write time on 8 GB VRAM; falls back to sequential on OOM.
    resolved_device = (
        torch.device(f"cuda:{device}") if torch.cuda.is_available() else torch.device("cpu")
    )  # device index is ignored outright on a machine with no GPU
    wrapper = SprintWrapper.from_checkpoint(checkpoint_path, device=resolved_device)
    model = wrapper.model

    # Build featurizers the same way train.py does: use the names stored in the
    # checkpoint's args so they always match what the model was trained with.
    task_dir = get_task_dir(dataset)
    # drug side parallelizes the one-time featurization over CPU workers; target side
    # (SaProt) batches on the GPU instead, hence the different kwargs below
    drug_featurizer = get_featurizer(
        model.args.drug_featurizer,
        save_dir=task_dir,
        n_jobs=num_workers or 1,  # 0 means something else to joblib; fall back to 1
        ext="lmdb",
    )
    target_featurizer = get_featurizer(
        model.args.target_featurizer,
        save_dir=task_dir,
        batch_size=featurizer_batch_size,
        ext="lmdb",
    )

    datamodule = DTIDataModule(
        data_dir=task_dir,
        drug_featurizer=drug_featurizer,
        target_featurizer=target_featurizer,
        device=resolved_device,
        batch_size=batch_size,
        shuffle=False,  # eval, not training - order doesn't matter
        num_workers=num_workers,
    )
    datamodule.prepare_data()  # writes the LMDB feature caches to disk if they don't exist yet
    datamodule.setup(stage="test")  # loads the split csvs and wires the featurizers to them

    trainer = pl.Trainer(
        devices=[device] if torch.cuda.is_available() else 1,  # gpu wants an index list, cpu wants a count
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
        max_epochs=1,
        logger=False,
        enable_progress_bar=True,
    )
    results = trainer.test(model, datamodule=datamodule, verbose=True)
    raw = results[0] if results else {}  # one dict per dataloader; there's only ever one
    # Strip the "test/" prefix that Lightning adds to logged metric keys.
    return {k.removeprefix("test/"): v for k, v in raw.items()}


def main() -> None:
    parser = argparse.ArgumentParser(description="SPRINT DTI benchmark (BIOSNAP / DAVIS / BindingDB).")
    parser.add_argument("--config", type=str, default=None, help="YAML config (same one used for LIT-PCBA is fine).")
    parser.add_argument("--checkpoint", type=str, default=None, help="Checkpoint path. Auto-selected if omitted.")
    parser.add_argument(
        "--dataset",
        type=str,
        default="all",
        choices=SUPPORTED_DATASETS + ["all"],
        help="Dataset to evaluate, or 'all' (default).",
    )
    parser.add_argument("--batch-size", type=int, default=None, dest="batch_size")
    parser.add_argument("--device", type=int, default=None)
    parser.add_argument("--num-workers", type=int, default=None, dest="num_workers")
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Write the results CSV here instead of the default "
             "outputs/metrics/sprint_dti_results_<timestamp>.csv.",
    )
    parser.add_argument(
        "--featurizer-batch-size",
        type=int,
        default=None,
        dest="featurizer_batch_size",
        help="Sequences per GPU call when building the target LMDB (default 16). "
             "Increase to 32 or 64 to speed up the one-time SaProt write on 8 GB VRAM.",
    )
    args = parser.parse_args()

    # Merge config file → CLI overrides (CLI wins).
    config: dict = {}
    if args.config:
        config.update(load_yaml_config(args.config))
    cli_overrides = {
        "checkpoint_path": args.checkpoint,
        "batch_size": args.batch_size,
        "device": args.device,
        "num_workers": args.num_workers,
        "featurizer_batch_size": args.featurizer_batch_size,
    }
    for key, value in cli_overrides.items():
        if value is not None:
            config[key] = value

    checkpoint_path: str | None = config.get("checkpoint_path")
    batch_size = int(config.get("batch_size", 64))
    device = int(config.get("device", 0))
    num_workers = int(config.get("num_workers", 0))
    featurizer_batch_size = int(config.get("featurizer_batch_size", 16))

    base_dir = Path(__file__).resolve().parent.parent
    if checkpoint_path is None:
        checkpoint_dir = base_dir / "checkpoints" / "sprint_official"
        available = get_available_checkpoints(checkpoint_dir)
        if not available:
            raise SystemExit(f"No checkpoints found in {checkpoint_dir}")
        # prefer the one literally named "sprint"
        checkpoint_path = available.get("sprint", next(iter(available.values())))

    datasets = SUPPORTED_DATASETS if args.dataset == "all" else [args.dataset]
    checkpoint_name = Path(checkpoint_path).name

    rows: list[dict] = []
    for dataset in datasets:
        print(f"\n{'='*60}")
        print(f"  Dataset   : {dataset.upper()}")
        print(f"  Checkpoint: {checkpoint_name}")
        print(f"{'='*60}")
        metrics = run_dti_dataset(
            checkpoint_path=checkpoint_path,
            dataset=dataset,
            batch_size=batch_size,
            num_workers=num_workers,
            device=device,
            featurizer_batch_size=featurizer_batch_size,
        )
        row = {"dataset": dataset, "checkpoint": checkpoint_name, **metrics}
        ref = PAPER_REFERENCE.get(dataset, {})
        row["paper_aupr"] = ref.get("aupr")
        rows.append(row)

        # skip non-float entries
        metric_str = "  ".join(f"{k}={v:.4f}" for k, v in metrics.items() if isinstance(v, float))
        print(f"  Results: {metric_str}")
        if ref.get("aupr") is not None:
            repro_aupr = metrics.get("aupr", float("nan"))  # nan if this run didn't log an aupr
            print(f"  Paper AUPR: {ref['aupr']:.3f}  |  Delta: {repro_aupr - ref['aupr']:+.3f}")

    output_dir = base_dir / "outputs" / "metrics"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_file = (
        Path(args.output)
        if args.output
        else output_dir / f"sprint_dti_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
    )
    df = pd.DataFrame(rows)
    df.to_csv(output_file, index=False)

    print(f"\n{'='*60}")
    print(df.to_string(index=False))
    print(f"\nSaved to: {output_file}")


if __name__ == "__main__":
    main()
