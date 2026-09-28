#!/usr/bin/env python3
# dump SPRINT co-embeddings per target so a scoring condition is a re-score, not a re-run.
# run in .venv: dump_sprint_embeddings.py --dataset {lit_pcba,dude}

from __future__ import annotations

import argparse
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import torch

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.core.embeddings_io import save_target_embeddings  # noqa: E402
from src.core.wrapper import SprintWrapper  # noqa: E402
from src.evaluation.evaluator import (  # noqa: E402
    get_screening_embeddings,
    list_dude_targets,
)

DATASET_DEFAULT_DIR = {"lit_pcba": "data/lit_pcba", "dude": "data/DUDe"}
# 1024-d float32 is ~4 KB/molecule (~11 GB LIT-PCBA / ~6 GB DUD-E), gitignored
DATASET_DEFAULT_OUT = {
    "lit_pcba": "outputs/metrics/sprint_pcba/embeddings",
    "dude": "outputs/metrics/sprint_dude/embeddings",
}


def discover_targets(dataset: str, data_dir: Path) -> list[str]:
    if dataset == "dude":
        return list_dude_targets(data_dir)
    return sorted(p.name for p in data_dir.iterdir() if p.is_dir())


def dump_target(wrapper, data_dir: Path, target: str, dataset: str, out_dir: Path,
                dtype: str = "float32") -> dict:
    """Embed one target and persist exactly what the scoring chain consumes."""
    # SprintWrapper.embed_items never persisted them; every condition used to re-run the
    # full model (~15 min LIT-PCBA, ~20 min DUD-E)
    embeddings = get_screening_embeddings(
        model_wrapper=wrapper,
        data_dir=data_dir,
        target=target,
        target_featurizer=wrapper.model.args.target_featurizer,
        dataset=dataset,
    )
    mol_reps = np.asarray(embeddings["drug_embeddings"], dtype=dtype)
    pocket_reps = np.asarray(embeddings["target_embeddings"], dtype=dtype)
    if pocket_reps.ndim == 1:
        pocket_reps = pocket_reps[None, :]
    labels = np.asarray(embeddings["labels"], dtype=np.int32)

    if len(mol_reps) != len(labels):
        raise RuntimeError(f"{target}: {len(mol_reps)} mol reps but {len(labels)} labels")
    if pocket_reps.shape[0] == 0:
        raise RuntimeError(f"{target}: no target embeddings")
    # A dump of zeros is what a silently-failed featurization looks like; refuse to save it
    # rather than let it score as a plausible number (see errors.md, the DUD-E 'X#' bug).
    if not np.any(pocket_reps):
        raise RuntimeError(f"{target}: all target embeddings are zero — featurization failed")

    # same <target>/{mol_reps,pocket_reps,labels}.npy contract as the DrugCLIP dumps
    save_target_embeddings(out_dir, target, mol_reps, pocket_reps, labels)

    return {
        "target": target,
        "n_molecules": int(len(labels)),
        "n_actives": int(labels.sum()),
        "n_pockets": int(pocket_reps.shape[0]),
        "dim": int(mol_reps.shape[1]),
        "mb": round((mol_reps.nbytes + pocket_reps.nbytes + labels.nbytes) / 1e6, 1),
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Dump SPRINT co-embeddings per target so a scoring condition is a re-score, not a re-run."
    )
    parser.add_argument("--dataset", choices=sorted(DATASET_DEFAULT_DIR), default="lit_pcba")
    parser.add_argument("--data-dir", default=None, dest="data_dir")
    parser.add_argument("--embeddings-dir", default=None, dest="embeddings_dir")
    parser.add_argument("--targets", default=None, help="Comma-separated subset")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--checkpoint", default="checkpoints/sprint_official/sprint.ckpt")
    parser.add_argument("--device", type=int, default=0)
    # float32 by design, not the DrugCLIP dumps' float16: DrugCLIP runs under model.half(),
    # so float16 loses nothing there. SPRINT runs float32, and a float16 cache is lossy
    # right at the >= 0 threshold - measured 0.003% of centered values within one fp16
    # ULP on ADRB2, enough to move EF 0.5% by 1.18. --dtype float16 is for disk pressure
    # only; those re-scored numbers are not the committed ones.
    parser.add_argument("--dtype", choices=["float32", "float16"], default="float32",
                        help="float32 preserves SPRINT's native precision so a re-score "
                             "reproduces the benchmark path exactly; float16 halves the "
                             "size but flips threshold-adjacent bits")
    parser.add_argument("--overwrite", action="store_true",
                        help="Re-dump targets that already have all three arrays. Without "
                             "this, an interrupted dump resumes where it stopped.")
    parser.add_argument("--manifest", default=None, help="Write a per-target summary CSV here")
    args = parser.parse_args()

    data_dir = Path(args.data_dir or DATASET_DEFAULT_DIR[args.dataset])
    out_dir = Path(args.embeddings_dir or DATASET_DEFAULT_OUT[args.dataset]).resolve()

    targets = discover_targets(args.dataset, data_dir)
    if args.targets:
        wanted = [t.strip() for t in args.targets.split(",") if t.strip()]
        unknown = [t for t in wanted if t not in targets]
        if unknown:
            raise SystemExit(f"Unknown {args.dataset} target(s): {unknown}")
        targets = wanted
    elif args.limit:
        targets = targets[: args.limit]

    device = torch.device(f"cuda:{args.device}") if torch.cuda.is_available() else torch.device("cpu")
    wrapper = SprintWrapper.from_checkpoint(args.checkpoint, device=device)

    print(f"Dataset:        {args.dataset}")
    print(f"Data dir:       {data_dir}")
    print(f"Embeddings out: {out_dir}  (dtype {args.dtype})")
    print(f"Dumping {len(targets)} target(s)\n")
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    started = datetime.now()
    required = ("mol_reps.npy", "pocket_reps.npy", "labels.npy")
    for index, target in enumerate(targets, start=1):
        # Resumable: a full dump is 10-25 min and this session lost two of them to
        # interruptions. A target counts as done only when all three arrays exist, so a
        # target killed mid-write is redone rather than left half-written.
        if not args.overwrite and all((out_dir / target / name).exists() for name in required):
            print(f"[{index}/{len(targets)}] {target} — already dumped, skipping", flush=True)
            rows.append({"target": target, "skipped": True})
            continue
        print(f"[{index}/{len(targets)}] {target}", flush=True)
        try:
            rows.append(dump_target(wrapper, data_dir, target, args.dataset, out_dir, args.dtype))
        except Exception as exc:  # one bad target must not sink the dump
            print(f"  ! {target}: {exc}", flush=True)
            rows.append({"target": target, "error": str(exc)})

    frame = pd.DataFrame(rows)
    print()
    print(frame.to_string(index=False))
    if "mb" in frame:
        print(f"\ntotal dumped: {frame['mb'].sum() / 1000:.2f} GB in {datetime.now() - started}")
    if args.manifest:
        Path(args.manifest).parent.mkdir(parents=True, exist_ok=True)
        frame.to_csv(args.manifest, index=False)
        print(f"manifest -> {args.manifest}")


if __name__ == "__main__":
    main()
