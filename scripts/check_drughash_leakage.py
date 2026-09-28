#!/usr/bin/env python3
# check evaluation-set ligand leakage against DrugCLIP's training SMILES.
# run: .venv_drugclip/bin/python scripts/check_drughash_leakage.py
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent


# RDKit-canonical SMILES; unparseable rows are dropped, not counted as overlap
def canonical(smiles_iter):
    from rdkit import Chem, RDLogger

    RDLogger.DisableLog("rdApp.*")  # otherwise every unparseable active spams the log
    out = set()
    for s in smiles_iter:
        mol = Chem.MolFromSmiles(s)
        if mol is not None:
            out.add(Chem.MolToSmiles(mol))
    return out


def dude_actives(dude_dir: Path):
    """One SMILES per compound id, matching evaluator.load_dude_target."""
    for path in sorted(dude_dir.glob("*_actives.tsv")):
        target = path.name[: -len("_actives.tsv")]
        seen, smis = set(), []
        for line in path.read_text().splitlines():
            parts = line.split("\t")
            if len(parts) < 2 or parts[0] in seen:
                continue
            seen.add(parts[0])
            smis.append(parts[1])
        yield target, smis


def main() -> int:
    train_cache = REPO / "outputs/drughash/train_embeddings/train/smi_names.npy"
    if not train_cache.exists():
        print(f"missing {train_cache}; run scripts/dump_drugclip_train_embeddings.py first")
        return 1
    train = canonical(set(np.load(train_cache, allow_pickle=True).tolist()))  # dump may repeat a SMILES
    print(f"DrugCLIP training ligands: {len(train)} canonical SMILES\n")

    report = {"train_ligands": len(train), "benchmarks": {}}

    total = hits = 0
    per_target = {}
    # a leaked active is partly remembered, not retrieved, which inflates that benchmark
    for target, smis in dude_actives(REPO / "data/DUDe"):
        cs = canonical(smis)
        overlap = cs & train
        total += len(cs)
        hits += len(overlap)
        per_target[target] = {"actives": len(cs), "leaked": len(overlap)}
    pct = 100 * hits / total if total else 0.0
    print(f"DUD-E actives:    {total} canonical, {hits} in training ({pct:.2f}%), "
          f"{sum(1 for v in per_target.values() if v['leaked'])} / {len(per_target)} targets affected")
    report["benchmarks"]["dude"] = {"actives": total, "leaked": hits, "pct": round(pct, 3),
                                    "per_target": per_target}

    # LIT-PCBA, for comparison
    lit_total = lit_hits = 0
    lit_per = {}
    for d in sorted((REPO / "data/lit_pcba").iterdir()):
        f = d / "actives.smi"
        if not f.is_file():
            continue
        # .smi first token; no compound-id dedup (unlike DUD-E)
        cs = canonical(line.split()[0] for line in f.read_text().splitlines() if line.strip())
        overlap = cs & train
        lit_total += len(cs)
        lit_hits += len(overlap)
        lit_per[d.name] = {"actives": len(cs), "leaked": len(overlap)}
    lit_pct = 100 * lit_hits / lit_total if lit_total else 0.0
    print(f"LIT-PCBA actives: {lit_total} canonical, {lit_hits} in training ({lit_pct:.2f}%), "
          f"{sum(1 for v in lit_per.values() if v['leaked'])} / {len(lit_per)} targets affected")
    report["benchmarks"]["lit_pcba"] = {"actives": lit_total, "leaked": lit_hits,
                                        "pct": round(lit_pct, 3), "per_target": lit_per}

    # "not checked", not "no overlap": DUD-E names and DrugCLIP PDB ids don't match as strings
    print("\nTARGET-side leakage: NOT CHECKED. DUD-E pockets are named by DUD-E code and "
          "DrugCLIP's by PDB id, so a string intersection is vacuous. Needs a DUD-E -> PDB "
          "mapping that is not available offline.")
    report["target_side"] = "not checked - no DUD-E to PDB mapping available offline"

    out = REPO / "outputs/metrics/drughash_leakage.json"  # audit_report_numbers.py reads this
    out.write_text(json.dumps(report, indent=2))
    print(f"\nreport -> {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
