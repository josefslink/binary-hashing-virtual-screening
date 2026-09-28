from __future__ import annotations

from pathlib import Path

import numpy as np

# numpy + pathlib only: this gets imported from both .venv and .venv_drugclip.


def save_target_embeddings(
    out_dir: Path,
    target: str,
    mol_reps: np.ndarray,
    pocket_reps: np.ndarray,
    labels: np.ndarray,
) -> Path:
    """Write <out_dir>/<target>/{mol_reps,pocket_reps,labels}.npy, the contract the scorer reads."""
    # one place for the three filenames because evaluate_drugclip_hamming.py reads them back
    # by name (REQUIRED_ARRAYS) and discover_targets() decides a target is usable by their
    # presence. both dump scripts wrote this block verbatim before it was extracted.
    target_dir = Path(out_dir) / target
    target_dir.mkdir(parents=True, exist_ok=True)
    np.save(target_dir / "mol_reps.npy", mol_reps)
    np.save(target_dir / "pocket_reps.npy", pocket_reps)
    np.save(target_dir / "labels.npy", labels)
    return target_dir
