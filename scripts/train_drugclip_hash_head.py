#!/usr/bin/env python3
# train a hash head on frozen DrugCLIP embeddings and sweep DrugHash's lambda grid.
# L = L_c + lambda * L_hash (Eq. 9); encoders frozen, so a lower bound, never a reproduction.

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.hashing.head import HashHeadPair, code_diagnostics  # noqa: E402
from src.hashing.losses import (  # noqa: E402
    DRUGCLIP_LOGIT_SCALE,
    duplicate_mask,
    hash_quantization_loss,
    in_batch_softmax_loss,
    quantization_quality,
)


def load_cache(path: Path):
    mol = np.load(path / "mol_reps.npy").astype(np.float32)  # dump writes fp16
    pocket = np.load(path / "pocket_reps.npy").astype(np.float32)
    smi = np.load(path / "smi_names.npy", allow_pickle=True)  # object arrays of names, not floats
    poc = np.load(path / "pocket_names.npy", allow_pickle=True)
    return mol, pocket, smi, poc


def pocket_disjoint_split(pocket_names, frac: float, seed: int):
    """Hold out `frac` of the *unique pockets*, not of the pairs."""
    uniq = np.unique(pocket_names)
    rng = np.random.default_rng(seed)
    # at least one pocket even if frac rounds to 0
    held = set(rng.permutation(uniq)[: max(1, int(len(uniq) * frac))].tolist())
    mask = np.array([p in held for p in pocket_names])
    return ~mask, mask  # train, val


def evaluate(head, mol, pocket, poc_names, smi_names, device, batch_size, scale, lam):
    head.eval()
    n = len(mol)
    tot_c = tot_h = 0.0
    steps = 0
    with torch.no_grad():
        # drop a leftover <batch: in-batch softmax is only well-defined on a full B x B
        for start in range(0, n - batch_size + 1, batch_size):
            sl = slice(start, start + batch_size)
            ym, yp = head(mol[sl], pocket[sl])
            mask = duplicate_mask(poc_names[sl], smi_names[sl], device)
            tot_c += in_batch_softmax_loss(ym, yp, mask, scale).item()
            tot_h += hash_quantization_loss(ym, yp).item()
            steps += 1
    head.train()
    steps = max(steps, 1)  # val smaller than one batch would divide by zero
    return tot_c / steps, tot_h / steps, (tot_c + lam * tot_h) / steps


def train_one(lam: float, data, args, device):
    mol, pocket, smi, poc = data
    # pocket-disjoint inner val: lambda is never selected on LIT-PCBA/DUD-E
    tr, va = pocket_disjoint_split(poc, args.val_frac, args.seed)
    print(f"  train pairs {tr.sum()}, inner-val pairs {va.sum()} "
          f"({len(np.unique(poc[va]))} held-out pockets)")

    t_mol = torch.from_numpy(mol).to(device)
    t_pk = torch.from_numpy(pocket).to(device)
    idx_tr = np.flatnonzero(tr)
    idx_va = np.flatnonzero(va)

    # identity_init: step 0 matches post-hoc none. lambda=0 still trains L_c, so bits can move.
    head = HashHeadPair(mol.shape[1], tie=args.tie, identity_init=True).to(device)
    opt = torch.optim.Adam(head.parameters(), lr=args.lr)
    rng = np.random.default_rng(args.seed)

    # bit_flip_fraction vs these bits. an unnormalised head drives L_hash to 0 by
    # rescaling W, loss falling with zero bits changed - a clean lambda curve without
    # this proves nothing. mol side only: screening codes, and code_diagnostics, are mol.
    with torch.no_grad():
        base_bits = (head.mol(t_mol[idx_va]).cpu().numpy() >= 0).astype(np.uint8)

    history = []
    best = {"val_total": float("inf")}
    for epoch in range(args.epochs):
        perm = rng.permutation(idx_tr)
        ep_c = ep_h = 0.0
        grad_c = grad_h = 0.0
        steps = 0
        # same leftover drop as evaluate
        for start in range(0, len(perm) - args.batch_size + 1, args.batch_size):
            sel = perm[start : start + args.batch_size]
            ym, yp = head(t_mol[sel], t_pk[sel])
            mask = duplicate_mask(poc[sel], smi[sel], device)
            lc = in_batch_softmax_loss(ym, yp, mask, args.scale)
            lh = hash_quantization_loss(ym, yp, args.hash_target)

            # grad_ratio: ||lambda grad L_hash|| / ||grad L_c||. below ~0.1 the hash
            # term is not in play, which is why the paper's grid looks flat on a frozen
            # encoder. first batch of the epoch only; retain_graph for the real step.
            if steps == 0:
                opt.zero_grad(); lc.backward(retain_graph=True)
                grad_c = sum(p.grad.norm().item() ** 2 for p in head.parameters() if p.grad is not None) ** 0.5
                opt.zero_grad(); (lam * lh).backward(retain_graph=True)
                grad_h = sum(p.grad.norm().item() ** 2 for p in head.parameters() if p.grad is not None) ** 0.5

            opt.zero_grad()
            (lc + lam * lh).backward()
            torch.nn.utils.clip_grad_norm_(head.parameters(), args.clip_norm)
            opt.step()
            ep_c += lc.item(); ep_h += lh.item(); steps += 1

        vc, vh, vt = evaluate(head, t_mol[idx_va], t_pk[idx_va], poc[idx_va], smi[idx_va],
                              device, args.batch_size, args.scale, lam)
        with torch.no_grad():
            ym = head.mol(t_mol[idx_va])
            # n_distinct_codes: L_hash has no diversity pressure, so high lambda buys
            # quantization by collapsing the code (4013 -> 3366 across the sweep)
            diag = code_diagnostics((ym.cpu().numpy() >= 0).astype(np.uint8), base_bits)
            diag["quant_quality"] = quantization_quality(ym)
        row = {"epoch": epoch, "train_c": ep_c / max(steps, 1), "train_h": ep_h / max(steps, 1),
               "val_c": vc, "val_h": vh, "val_total": vt,
               "grad_ratio": (grad_h / grad_c) if grad_c else float("nan"), **diag}
        history.append(row)
        if vt < best["val_total"]:
            # clone: later steps would otherwise mutate the saved tensors in place
            best = {**row, "state": {k: v.detach().cpu().clone() for k, v in head.state_dict().items()}}
        if epoch % max(1, args.epochs // 5) == 0 or epoch == args.epochs - 1:
            print(f"    ep{epoch:3d} Lc {row['train_c']:.4f} Lh {row['train_h']:.4f} "
                  f"val {vt:.4f} | flip {row['bit_flip_fraction']:.3f} "
                  f"grad_ratio {row['grad_ratio']:.4f} quant {row['quant_quality']:.4f} "
                  f"codes {row['n_distinct_codes']}", flush=True)

    head.load_state_dict(best["state"])  # best val checkpoint, not the last epoch
    return head, history, {k: v for k, v in best.items() if k != "state"}


def main() -> None:
    p = argparse.ArgumentParser(
        description="Train a hash head on frozen DrugCLIP embeddings and sweep DrugHash's lambda grid."
    )
    # dump_drugclip_train_embeddings.py's cache; LIT-PCBA/DUD-E stay untouched
    p.add_argument("--train-cache", default="outputs/drughash/train_embeddings/train", dest="train_cache")
    p.add_argument("--out-dir", default="outputs/drughash/heads", dest="out_dir")
    p.add_argument("--lambdas", default="0,0.03,0.1,0.2,0.4,0.6,0.8,1.0")
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--batch-size", type=int, default=48, dest="batch_size")
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--clip-norm", type=float, default=1.0, dest="clip_norm")
    # 13.99 from checkpoint_best; tau = 1/scale
    p.add_argument("--scale", type=float, default=DRUGCLIP_LOGIT_SCALE)
    # pm1 is the paper's b=+-1
    p.add_argument("--hash-target", choices=["pm1", "pm1_over_sqrt_d"], default="pm1", dest="hash_target")
    p.add_argument("--val-frac", type=float, default=0.10, dest="val_frac")  # of unique pockets, not of pairs
    p.add_argument("--tie", action="store_true")  # one shared head; ITQ reference, not the default
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--overwrite", action="store_true", help="Retrain lambdas that already have a head")
    args = p.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    data = load_cache(Path(args.train_cache))
    print(f"cache: {len(data[0])} pairs, {len(np.unique(data[3]))} unique pockets, dim {data[0].shape[1]}")
    out = Path(args.out_dir); out.mkdir(parents=True, exist_ok=True)

    summary = []
    for lam in [float(x) for x in args.lambdas.split(",")]:
        tag = f"lam{lam:.2f}".replace(".", "p")  # lam0.03 -> lam0p03, safe as a filename
        print(f"\n=== lambda = {lam} ({tag}) ===", flush=True)
        if (out / f"{tag}.pt").exists() and not args.overwrite:
            hist = json.loads((out / f"{tag}_history.json").read_text())
            best = min(hist, key=lambda r: r["val_total"])
            summary.append({"lam": lam, "tag": tag, **best})
            print(f"  already trained, skipping (best val {best['val_total']:.4f})", flush=True)
            continue
        head, history, best = train_one(lam, data, args, device)
        torch.save({"state_dict": head.state_dict(), "lam": lam, "args": vars(args)}, out / f"{tag}.pt")
        (out / f"{tag}_history.json").write_text(json.dumps(history, indent=2))
        summary.append({"lam": lam, "tag": tag, **best})
        print(f"  best epoch {best['epoch']}: val {best['val_total']:.4f}, "
              f"flip {best['bit_flip_fraction']:.3f}, codes {best['n_distinct_codes']}")

    (out / "sweep_summary.json").write_text(json.dumps(summary, indent=2))
    print(f"\nsummary -> {out / 'sweep_summary.json'}")


if __name__ == "__main__":
    main()
