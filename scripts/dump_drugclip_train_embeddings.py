#!/usr/bin/env python3
# dump paired (pocket, molecule) embeddings from DrugCLIP's training split.
# for train_drugclip_hash_head.py; never mix with the LIT-PCBA/DUD-E test dumps.
# run in .venv_drugclip with the same flags as the evaluation dumps.

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.core.drugclip_capture import ProjectionCapture, load_drugclip_model  # noqa: E402
from src.core.vendor_compat import (  # noqa: E402
    drugclip_workdir,
    ensure_drugclip_split_alias,
)

# must not start with "train": that branch redraws a conformer per epoch. any other
# name takes the deterministic path, so row i is LMDB record i.
ALIAS_SPLIT = "pairs"


def dump_split(task, model, split: str, out_dir: Path, batch_size: int, scale: float) -> dict:
    task.load_dataset(split)
    dataset = task.datasets[split]
    loader = torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, shuffle=False, collate_fn=dataset.collater
    )

    # same ProjectionCapture hooks as the test dumps; training forward is the same three lines
    mol_capture = ProjectionCapture(model.mol_project)
    pocket_capture = ProjectionCapture(model.pocket_project)
    smi_names: list[str] = []
    pocket_names: list[str] = []
    max_dev = 0.0

    try:
        with torch.no_grad():
            for index, sample in enumerate(loader):
                device = next(model.parameters()).device
                net_input = {
                    k: (v.to(device) if torch.is_tensor(v) else v)
                    for k, v in sample["net_input"].items()
                }
                smi = list(sample["smi_name"])
                poc = list(sample["pocket_name"])
                out = model(**net_input, smi_list=smi, pocket_list=poc, features_only=True)
                smi_names.extend(smi)
                pocket_names.extend(poc)

                # rebuild (P @ M.T) * exp(logit_scale) with the in-batch duplicate mask
                # and diff against forward()'s own ba_predict
                ba = out[0] if isinstance(out, (tuple, list)) else out
                if torch.is_tensor(ba) and ba.dim() == 2 and ba.shape[0] == ba.shape[1]:
                    m = mol_capture.batches[-1].astype(np.float64)
                    p = pocket_capture.batches[-1].astype(np.float64)
                    logits = (p @ m.T) * scale
                    pd_ = np.equal.outer(np.array(poc), np.array(poc)).astype(np.float64)
                    md_ = np.equal.outer(np.array(smi), np.array(smi)).astype(np.float64)
                    mask = pd_ + md_ - 2 * np.eye(len(poc))
                    logits = mask * -1e6 + logits
                    dev = float(np.max(np.abs(logits - ba.detach().cpu().numpy().astype(np.float64))))
                    max_dev = max(max_dev, dev)
                if index % 50 == 0:
                    print(f"    batch {index}/{len(loader)}", flush=True)
    finally:
        mol_capture.close()
        pocket_capture.close()

    mol_reps = mol_capture.stacked()
    pocket_reps = pocket_capture.stacked()
    if len(mol_reps) != len(pocket_reps) or len(mol_reps) != len(smi_names):
        raise RuntimeError(
            f"{split}: {len(mol_reps)} mol / {len(pocket_reps)} pocket / {len(smi_names)} names"
        )

    out_dir.mkdir(parents=True, exist_ok=True)
    np.save(out_dir / "mol_reps.npy", mol_reps)
    np.save(out_dir / "pocket_reps.npy", pocket_reps)
    np.save(out_dir / "smi_names.npy", np.array(smi_names))
    np.save(out_dir / "pocket_names.npy", np.array(pocket_names))
    return {
        "split": split,
        "pairs": int(len(mol_reps)),
        "unique_pockets": int(len(set(pocket_names))),
        "unique_smiles": int(len(set(smi_names))),
        "dim": int(mol_reps.shape[1]),
        "max_logit_dev": max_dev,
    }


def main(args) -> None:
    use_cuda = torch.cuda.is_available() and not args.cpu
    if use_cuda:
        torch.cuda.set_device(args.device_id)

    source = Path(args.train_lmdb).resolve()
    alias = ensure_drugclip_split_alias(source, args.alias_dir, ALIAS_SPLIT, source.parent)
    args.data = str(alias)

    task, model = load_drugclip_model(args, use_cuda)

    scale = float(model.logit_scale.exp().detach().cpu())
    print(f"logit_scale.exp() = {scale:.8f}  (1/scale = {1/scale:.6f}, the paper's tau)")

    out_root = Path(args.out_dir).resolve()
    manifest = []
    with drugclip_workdir(alias, args.workdir):
        info = dump_split(task, model, ALIAS_SPLIT, out_root / args.tag, args.batch_size, scale)
        info["source_lmdb"] = str(source)
        manifest.append(info)

    print()
    for row in manifest:
        print(row)
    (out_root / args.tag / "meta.json").write_text(
        json.dumps(
            {
                "checkpoint": str(Path(args.path).resolve()),
                "logit_scale_exp": scale,
                "max_pocket_atoms": args.max_pocket_atoms,
                "seed": args.seed,
                "fp16": bool(args.fp16),
                "splits": manifest,
            },
            indent=2,
        )
    )
    print(f"\nmeta -> {out_root / args.tag / 'meta.json'}")
    # Threshold reasoning: the model runs in fp16 and logits are (P@M.T)*exp(logit_scale),
    # so |logit| <= ~14 and one fp16 step at that magnitude is ~0.008. This is a MAX over
    # every batch (8,271 batches x 64 entries for the full split), so a few multiples of
    # the unit spacing is expected; 0.069 was measured and is fine. A genuine misalignment
    # would give deviations of order 14, not 0.07 - and is better caught by the in-batch
    # retrieval check (97.9% at B=48 against 2.1% chance) than by this bound.
    if manifest and manifest[0]["max_logit_dev"] > 0.5:
        print(f"WARNING: max logit deviation {manifest[0]['max_logit_dev']:.3e} — far above "
              "fp16 noise; check row alignment before using this cache")


def cli_main() -> None:
    from unicore import distributed_utils, options

    parser = options.get_validation_parser()
    parser.add_argument("--train-lmdb", required=True, dest="train_lmdb",
                        help="e.g. data/drugclip/train_no_test_af/train.lmdb")
    parser.add_argument("--tag", default="train", help="subdirectory name for this dump")
    parser.add_argument("--alias-dir", dest="alias_dir",
                        default=str(REPO_ROOT / "outputs/drughash/split_alias"))
    parser.add_argument("--workdir", default=str(REPO_ROOT / "outputs/drugclip_workdir"))
    parser.add_argument("--out-dir", dest="out_dir",
                        default=str(REPO_ROOT / "outputs/drughash/train_embeddings"))
    options.add_model_args(parser)  # same flags as the eval dumps; a mismatch is what the head would measure
    args = options.parse_args_and_arch(parser)
    distributed_utils.call_main(args, main)


if __name__ == "__main__":
    cli_main()
