#!/usr/bin/env python3
# dump DrugCLIP encoder outputs for LIT-PCBA or DUD-E without touching third_party/.
# run in .venv_drugclip; score with
# scripts/evaluate_drugclip_hamming.py --embeddings-dir <this script's output>

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.core.drugclip_capture import (  # noqa: E402
    LabelCapture,
    ProjectionCapture,
    empty_cache_before,
    load_drugclip_model,
)
from src.core.embeddings_io import save_target_embeddings  # noqa: E402
from src.core.vendor_compat import drugclip_workdir, install_drugclip_cpu_compat  # noqa: E402

# DrugCLIP's own cal_metrics uses alpha=80.5 for BEDROC; the same value the published
# tables and every DrugCLIP row in the report use.
DRUGCLIP_BEDROC_ALPHA = 80.5

TASK_LAYOUT = {
    # task: (subdirectory under the data root, molecule lmdb, pocket lmdb)
    "PCBA": ("lit_pcba", "mols.lmdb", "pockets.lmdb"),
    "DUDE": ("DUD-E/raw/all", "mols.lmdb", "pocket.lmdb"),
}


def discover_targets(data_root: Path, test_task: str) -> tuple[list[str], list[str]]:
    """A target is usable only if it has both its molecule and pocket LMDB."""
    subdir, mol_name, pocket_name = TASK_LAYOUT[test_task]
    root = data_root / subdir
    if not root.is_dir():
        raise FileNotFoundError(f"{test_task} data directory not found: {root}")
    usable, skipped = [], []
    for path in sorted(p for p in root.iterdir() if p.is_dir()):
        if (path / mol_name).exists() and (path / pocket_name).exists():
            usable.append(path.name)
        else:
            skipped.append(path.name)
    return usable, skipped


def _cal_metrics():
    """DrugCLIP's own metric function, imported from the (unmodified) vendored task."""
    from unimol.tasks.drugclip import cal_metrics

    return cal_metrics


def dump_target(task, model, target: str, test_task: str, out_dir: Path) -> dict[str, object]:
    """Run DrugCLIP's unmodified eval loop and persist what its encoders produced."""
    # hooks on mol_project/pocket_project capture what the vendored eval loops discard
    mol_capture = ProjectionCapture(model.mol_project)
    pocket_capture = ProjectionCapture(model.pocket_project)
    cache_hook = empty_cache_before(model.pocket_model)
    label_capture = None
    try:
        with LabelCapture(task) as label_capture, torch.no_grad():
            if test_task == "DUDE":
                auc, bedroc, ef, _re, res_single, labels = task.test_dude_target(target, model)
                labels = np.asarray(labels, dtype=np.int32)
            else:
                auc, bedroc, ef, _re = task.test_pcba_target(target, model)
                res_single = None
                labels = label_capture.as_array()
    finally:
        mol_capture.close()
        pocket_capture.close()
        cache_hook.remove()

    mol_reps = mol_capture.stacked()
    pocket_reps = pocket_capture.stacked()

    if len(mol_reps) != len(labels):
        raise RuntimeError(f"{target}: captured {len(mol_reps)} mol reps but {len(labels)} labels")
    if pocket_reps.shape[0] == 0:
        raise RuntimeError(f"{target}: captured no pocket reps")

    # captured arrays must reproduce what DrugCLIP itself scored, checked per target
    recomputed = (pocket_reps @ mol_reps.T).max(axis=0)
    if res_single is not None:
        # DUD-E test_dude_target returns a score vector: compare element-wise
        max_dev = float(
            np.max(np.abs(recomputed.astype(np.float64) - np.asarray(res_single, dtype=np.float64)))
        )
        auc_dev = 0.0
    else:
        # LIT-PCBA test_pcba_target returns only scalars: re-score via cal_metrics, compare AUC
        max_dev = float("nan")
        our_auc, _bedroc, _ef, _re = _cal_metrics()(labels, recomputed, DRUGCLIP_BEDROC_ALPHA)
        auc_dev = float(abs(our_auc - auc))

    target_dir = save_target_embeddings(out_dir, target, mol_reps, pocket_reps, labels)
    # Molecule names make a row-level leakage sensitivity analysis possible: without them a
    # dump cannot be mapped back to SMILES, and 2.56% of DUD-E actives turn out to be in
    # DrugCLIP's training ligands (2026-08-27). The LIT-PCBA dump has always saved these;
    # the DUD-E path did not, so DUD-E must be re-dumped before that analysis can be run.
    if label_capture is not None and label_capture.names:
        (target_dir / "mol_names.txt").write_text("\n".join(map(str, label_capture.names)) + "\n")

    return {
        "target": target,
        "auroc": float(auc),
        "bedroc": float(bedroc),
        "ef_0.005": float(ef["0.005"]),
        "ef_0.01": float(ef["0.01"]),
        "ef_0.02": float(ef["0.02"]),
        "ef_0.05": float(ef["0.05"]),
        "n_molecules": int(len(labels)),
        "n_actives": int(labels.sum()),
        "n_pockets": int(pocket_reps.shape[0]),
        "capture_max_dev": max_dev,
        "capture_auc_dev": auc_dev,
    }


def main(args) -> None:
    use_cuda = torch.cuda.is_available() and not args.cpu
    if use_cuda:
        torch.cuda.set_device(args.device_id)
    else:
        for shim in install_drugclip_cpu_compat():
            print(f"vendor_compat: {shim}")

    task, model = load_drugclip_model(args, use_cuda)

    # Absolute-ise our own paths before chdir'ing: DrugCLIP resolves its LMDBs relative
    # to the cwd ("./data/lit_pcba/<target>/mols.lmdb").
    data_root = Path(args.data).resolve()
    out_dir = Path(args.embeddings_dir).resolve()
    native_csv = Path(args.native_csv).resolve() if args.native_csv else None
    workdir = Path(args.workdir).resolve()

    usable, skipped = discover_targets(data_root, args.test_task)
    if args.targets:
        wanted = [t.strip() for t in args.targets.split(",") if t.strip()]
        unknown = [t for t in wanted if t not in usable]
        if unknown:
            raise SystemExit(f"Unknown or incomplete {args.test_task} target(s): {unknown}")
        targets = wanted
    else:
        targets = usable[: args.limit] if args.limit else usable

    print(f"Task:           {args.test_task}")
    print(f"Data root:      {data_root}")
    print(f"Embeddings out: {out_dir}")
    if skipped:
        print(f"Skipping {len(skipped)} target(s) missing an LMDB: {', '.join(skipped)}")
    print(f"Dumping {len(targets)} target(s)\n")

    out_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, object]] = []
    with drugclip_workdir(data_root, workdir):
        for index, target in enumerate(targets, start=1):
            print(f"[{index}/{len(targets)}] {target}")
            try:
                rows.append(dump_target(task, model, target, args.test_task, out_dir))
            except Exception as exc:  # one bad target must not sink the run
                print(f"  ! {target}: {exc}")
                rows.append({"target": target, "error": str(exc), "auroc": None})
            if use_cuda:
                torch.cuda.empty_cache()

    frame = pd.DataFrame(rows)
    print()
    print(frame.to_string(index=False))
    valid = frame["auroc"].dropna() if "auroc" in frame else pd.Series(dtype=float)
    if not valid.empty:
        print(f"\nnative auc mean over {len(valid)} target(s): {valid.mean()}")
    if "capture_max_dev" in frame:
        devs = frame["capture_max_dev"].dropna()
        if not devs.empty and not devs.isna().all():
            print(f"max deviation of captured reps vs DrugCLIP's own scores: {devs.max():.3e}")
            print("(0.0 means the hooks captured exactly the arrays the native path scored)")
    if "capture_auc_dev" in frame:
        devs = frame["capture_auc_dev"].dropna()
        if not devs.empty:
            print(f"max |AUC(recomputed) - AUC(native)|: {devs.max():.3e}")
    if native_csv:
        native_csv.parent.mkdir(parents=True, exist_ok=True)
        frame.to_csv(native_csv, index=False)
        print(f"\nSaved native metrics to: {native_csv}")


def cli_main() -> None:
    from unicore import distributed_utils, options

    parser = options.get_validation_parser()
    parser.add_argument("--test-task", type=str, default="PCBA", choices=sorted(TASK_LAYOUT))
    parser.add_argument("--targets", type=str, default=None, help="Comma-separated subset")
    parser.add_argument("--limit", type=int, default=None, help="Dump only the first N targets")
    parser.add_argument(
        "--embeddings-dir",
        type=str,
        default=str(REPO_ROOT / "outputs/metrics/drugclip_pcba/embeddings"),
        dest="embeddings_dir",
    )
    parser.add_argument(
        "--workdir",
        type=str,
        default=str(REPO_ROOT / "outputs/drugclip_workdir"),
        help="Disposable cwd holding a `data` symlink; replaces the symlink that used to "
        "live inside third_party/drugclip.",
    )
    parser.add_argument("--native-csv", type=str, default=None, dest="native_csv")
    options.add_model_args(parser)
    args = options.parse_args_and_arch(parser)
    distributed_utils.call_main(args, main)


if __name__ == "__main__":
    cli_main()
