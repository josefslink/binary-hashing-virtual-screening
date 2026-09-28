#!/usr/bin/env python3
# build DrugCLIP-format LIT-PCBA LMDBs (mols.lmdb/pocket.lmdb) from raw LIT-PCBA.

from __future__ import annotations

import argparse
import copy
import multiprocessing as mp
import os
import pickle
from functools import partial
from pathlib import Path

import lmdb
import numpy as np
from rdkit import Chem, RDLogger
from rdkit.Chem import AllChem
from tqdm import tqdm

RDLogger.DisableLog("rdApp.*")


# this file once lived at third_party/drugclip/py_scripts/write_lit_pcba.py (moved 2026-08-26);
# it is ours, not upstream's. helpers below (read_mol2_*, read_smi_mol, add_charges,
# gen_conformation, convert_2d_mol_to_data, pocket_parser, write_lmdb) are modelled on
# DrugCLIP's write_dude_multi.py - do not restructure them.


def _read_mol2_atom_block(path: str | Path) -> tuple[list[list[float]], list[str]]:
    coords: list[list[float]] = []
    atom_types: list[str] = []
    in_atom_section = False
    with open(path, "r", encoding="utf-8") as handle:
        for line in handle:
            if line.startswith("@<TRIPOS>ATOM"):
                in_atom_section = True
                continue
            if line.startswith("@<TRIPOS>") and in_atom_section:
                break
            if not in_atom_section:
                continue
            parts = line.split()
            if len(parts) < 6:
                continue
            # mol2 cols: id name x y z type ...
            coords.append([float(parts[2]), float(parts[3]), float(parts[4])])
            atom_type = parts[5].split(".", 1)[0]  # drop the hybridization suffix (C.3, N.ar)
            atom_types.append(atom_type)
    return coords, atom_types


def read_mol2_protein(path: str | Path) -> dict[str, object]:
    coord, atom_type = _read_mol2_atom_block(path)
    return {
        "coord": np.array(coord),
        "atom_type": atom_type,
    }


def read_mol2_ligand(path: str | Path) -> dict[str, object]:
    coord, atom_type = _read_mol2_atom_block(path)
    return {
        "coord": np.array(coord),
        "atom_type": atom_type,
        "mol": Chem.MolFromMol2File(str(path)),  # only the ligand needs a full rdkit mol
    }


def read_smi_mol(path: str | Path) -> list[Chem.Mol]:
    with open(path, "r", encoding="utf-8") as handle:
        smis = [line.split(" ")[0].strip() for line in handle if line.strip()]  # rows are "SMILES id"
    return [Chem.MolFromSmiles(smi) for smi in smis]


# unused in this file - upstream only calls it from the SDF-ligand path, and our
# actives/inactives come from plain SMILES (read_smi_mol) instead. Kept for parity with
# write_dude_multi.py, per the helpers note above.
def add_charges(mol: Chem.Mol):
    mol = copy.deepcopy(mol)
    mol.UpdatePropertyCache(strict=False)
    problems = Chem.DetectChemistryProblems(mol)
    if not problems:
        Chem.SanitizeMol(mol)
        return mol

    for problem in problems:
        if problem.GetType() != "AtomValenceException":
            continue
        atom = mol.GetAtomWithIdx(problem.GetAtomIdx())
        if atom.GetAtomicNum() == 7 and atom.GetFormalCharge() == 0 and atom.GetExplicitValence() == 4:
            atom.SetFormalCharge(1)  # tetravalent neutral N (e.g. protonated amine) needs the +
        if atom.GetAtomicNum() == 6 and atom.GetExplicitValence() == 5:
            for bond in atom.GetBonds():
                if bond.GetBondType() == Chem.rdchem.BondType.DOUBLE:
                    # pentavalent C: a double bond was likely miscounted
                    bond.SetBondType(Chem.rdchem.BondType.SINGLE)
                    break
        if atom.GetAtomicNum() == 8 and atom.GetFormalCharge() == 0 and atom.GetExplicitValence() == 3:
            atom.SetFormalCharge(1)  # trivalent neutral O (e.g. oxocarbenium) needs the +
        if atom.GetAtomicNum() == 5 and atom.GetFormalCharge() == 0 and atom.GetExplicitValence() == 4:
            atom.SetFormalCharge(-1)  # tetravalent neutral B (borate-like) needs the -
    try:
        Chem.SanitizeMol(mol)
    except Exception:
        return None  # still broken after the fixes above; drop it rather than crash the batch
    return mol


def gen_conformation(mol: Chem.Mol, num_conf: int = 1, num_worker: int = 1):
    try:
        mol = Chem.AddHs(mol)  # explicit Hs give the embedder a realistic geometry
        # pruneRmsThresh collapses near-duplicate conformers; maxAttempts retries hard
        # molecules instead of giving up early
        AllChem.EmbedMultipleConfs(
            mol,
            numConfs=num_conf,
            numThreads=num_worker,
            pruneRmsThresh=1,
            maxAttempts=10000,
            useRandomCoords=False,
        )
        try:
            AllChem.MMFFOptimizeMoleculeConfs(mol, numThreads=num_worker)
        except Exception:
            pass  # some atom types have no MMFF params; keep the raw embedded geometry
        mol = Chem.RemoveHs(mol)
    except Exception:
        return None
    if mol.GetNumConformers() == 0:
        return None  # embedding can fail silently and leave zero conformers
    return mol


def convert_2d_mol_to_data(mol: Chem.Mol, num_conf: int = 1, num_worker: int = 1):
    mol = gen_conformation(mol, num_conf=num_conf, num_worker=num_worker)
    if mol is None:
        return None
    coords = [np.array(mol.GetConformer(i).GetPositions()) for i in range(mol.GetNumConformers())]
    atom_types = [atom.GetSymbol() for atom in mol.GetAtoms()]
    return {"coords": coords, "atom_types": atom_types, "smi": Chem.MolToSmiles(mol), "mol": mol}


def get_pocket_atom_indices(protein: dict[str, object], ligand: dict[str, object], raid: float = 6.0):
    # brute-force atom-atom distance scan, atom-level rather than upstream's residue-level
    # get_different_raid. "raid" is upstream's own spelling of radius (angstrom), kept as-is.
    protein_coord = protein["coord"]
    ligand_coord = ligand["coord"]
    pocket_atom_idx = set()
    for i in range(len(protein_coord)):
        for j in range(len(ligand_coord)):
            if np.linalg.norm(protein_coord[i] - ligand_coord[j]) < raid:
                pocket_atom_idx.add(i)
    return sorted(pocket_atom_idx)


def pocket_parser(protein_path: str | Path, ligand_path: str | Path, pocket_index: int, raid: float = 6.0):
    protein = read_mol2_protein(protein_path)
    ligand = read_mol2_ligand(ligand_path)
    pocket_atom_idx = get_pocket_atom_indices(protein, ligand, raid=raid)
    pocket_atom_type = [protein["atom_type"][i] for i in pocket_atom_idx]
    pocket_coord = [protein["coord"][i] for i in pocket_atom_idx]
    pocket_name = Path(protein_path).parent.name  # LIT-PCBA layout: <target>/<id>_protein.mol2
    return {
        "pocket": pocket_name,
        "pocket_index": pocket_index,
        "pocket_atoms": pocket_atom_type,
        "pocket_coordinates": pocket_coord,
    }


def write_lmdb(data: list[dict[str, object]], lmdb_path: Path) -> None:
    lmdb_path.parent.mkdir(parents=True, exist_ok=True)
    if lmdb_path.exists():
        lmdb_path.unlink()
    env = lmdb.open(
        str(lmdb_path),
        subdir=False,
        readonly=False,
        lock=False,
        readahead=False,
        meminit=False,
        max_readers=64,
        map_size=1099511627776,  # 1 TiB virtual cap, not real disk use; lmdb grows the file lazily
    )
    with env.begin(write=True) as txn:
        for index, item in enumerate(data):
            txn.put(str(index).encode("ascii"), pickle.dumps(item))
    env.close()


def build_mols_lmdb(target_dir: Path, num_worker: int) -> int:
    active_path = target_dir / "actives.smi"
    inactive_path = target_dir / "inactives.smi"
    # drop unparsable SMILES
    molecules = [mol for mol in read_smi_mol(active_path) + read_smi_mol(inactive_path) if mol is not None]
    data = []
    process_count = min(32, os.cpu_count() or 1)  # more workers just contend over the same embedding work
    with mp.Pool(process_count) as pool:
        convert_payload = partial(convert_2d_mol_to_data, num_conf=1, num_worker=num_worker)
        for payload in tqdm(
            pool.imap_unordered(convert_payload, molecules),
            total=len(molecules),
            desc=f"{target_dir.name}: molecules",
        ):
            if payload is not None:
                # mols.lmdb schema: atoms/coordinates keys; pocket.lmdb below uses different names
                data.append({"atoms": payload["atom_types"], "coordinates": payload["coords"], "smi": payload["smi"], "mol": payload["mol"]})
    write_lmdb(data, target_dir / "mols.lmdb")
    return len(data)


def build_pocket_lmdb(target_dir: Path) -> int:
    pocket_files = sorted(target_dir.glob("*_protein.mol2"))  # one file per pocket in this target
    pockets = []
    for pocket_index, protein_path in enumerate(tqdm(pocket_files, desc=f"{target_dir.name}: pockets")):
        ligand_path = protein_path.with_name(protein_path.name.replace("_protein.mol2", "_ligand.mol2"))
        if not ligand_path.exists():
            raise FileNotFoundError(f"Missing ligand file for {protein_path.name}: {ligand_path.name}")
        pockets.append(pocket_parser(protein_path, ligand_path, pocket_index))
    write_lmdb(pockets, target_dir / "pocket.lmdb")
    return len(pockets)


def target_is_complete(target_dir: Path) -> bool:
    mols_path = target_dir / "mols.lmdb"
    pocket_path = target_dir / "pocket.lmdb"
    return mols_path.is_file() and pocket_path.is_file()


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate DrugCLIP Lit-PCBA LMDB caches from raw target folders.")
    # The report's DrugCLIP LIT-PCBA numbers come from LMDBs this wrote under data/drugclip/lit_pcba/
    parser.add_argument("--pcba-dir", type=str, default="./data/lit_pcba", help="Root Lit-PCBA directory")
    parser.add_argument("--target", type=str, default=None, help="Optional single target to rebuild")
    parser.add_argument("--num-worker", type=int, default=1, help="RDKit conformer worker threads")
    args = parser.parse_args()

    pcba_dir = Path(args.pcba_dir)
    if args.target:
        target_dirs = [pcba_dir / args.target]
    else:
        target_dirs = sorted(path for path in pcba_dir.iterdir() if path.is_dir())

    for target_dir in target_dirs:
        if not target_dir.exists():
            raise FileNotFoundError(f"Target directory not found: {target_dir}")
        if target_is_complete(target_dir):
            print(f"{target_dir.name}: already done, skipping")
            continue
        mol_count = build_mols_lmdb(target_dir, num_worker=args.num_worker)
        pocket_count = build_pocket_lmdb(target_dir)
        print(f"{target_dir.name}: wrote {mol_count} molecules and {pocket_count} pockets")


if __name__ == "__main__":
    main()
