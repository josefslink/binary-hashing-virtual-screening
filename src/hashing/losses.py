from __future__ import annotations

import numpy as np
import torch
from torch import Tensor
from torch.nn import functional as F

# exp(logit_scale) read from checkpoint_best.pt; 1/13.9922 = 0.0715, the paper's tau.
DRUGCLIP_LOGIT_SCALE = 13.99218750


def duplicate_mask(pocket_names, smi_names, device) -> Tensor:
    """Upstream's in-batch duplicate mask: pocket_dup + mol_dup - 2*I."""
    # the same pocket (or ligand) can appear twice in a batch; those off-diagonal pairs
    # are not negatives. Mask them to -1e6 before the softmax, or the loss trains the
    # head to push genuine duplicates apart.
    pockets = np.asarray(pocket_names)
    smiles = np.asarray(smi_names)
    pocket_dup = np.equal.outer(pockets, pockets).astype(np.float32)
    mol_dup = np.equal.outer(smiles, smiles).astype(np.float32)
    mask = pocket_dup + mol_dup - 2.0 * np.eye(len(pockets), dtype=np.float32)
    return torch.from_numpy(mask).to(device)


def in_batch_softmax_loss(
    mol_emb: Tensor,
    pocket_emb: Tensor,
    mask: Tensor,
    scale: float = DRUGCLIP_LOGIT_SCALE,
) -> Tensor:
    """DrugCLIP's in_batch_softmax, symmetrised over both directions."""
    logits = (pocket_emb @ mol_emb.T) * scale
    logits = mask * -1e6 + logits
    labels = torch.arange(len(logits), device=logits.device)
    return 0.5 * (F.cross_entropy(logits, labels) + F.cross_entropy(logits.T, labels))


def hash_quantization_loss(
    mol_emb: Tensor,
    pocket_emb: Tensor,
    target: str = "pm1",
) -> Tensor:
    """DrugHash Eq. 8: mean squared distance to the sign target b = sign(y), b detached."""
    # target sets the scale of b and isn't cosmetic: with b=+-1 vs b=+-1/sqrt(d) the
    # gradient wrt ||y||_1 differs by sqrt(d) = 11.3x at d=128, enough to shift the
    # paper's lambda grid by an order of magnitude. "pm1" follows the paper's text.
    if target == "pm1":
        b_mol, b_pocket = torch.sign(mol_emb).detach(), torch.sign(pocket_emb).detach()
    elif target == "pm1_over_sqrt_d":
        d = mol_emb.shape[-1] ** 0.5
        b_mol = (torch.sign(mol_emb) / d).detach()
        b_pocket = (torch.sign(pocket_emb) / d).detach()
    else:
        raise ValueError(f"Unknown hash target {target!r}")

    n, d = mol_emb.shape
    total = ((pocket_emb - b_pocket) ** 2).sum() + ((mol_emb - b_mol) ** 2).sum()
    return total / (n * d)


def quantization_quality(emb: Tensor) -> float:
    """||y||_1 / sqrt(d): 1.0 means every coordinate is already +-1/sqrt(d)."""
    # watch this across a lambda sweep instead of the raw loss - on the unit sphere
    # L_hash spans only ~0.036 on a base of ~0.85, too flat to read progress from
    with torch.no_grad():
        return float((emb.abs().sum(dim=-1) / emb.shape[-1] ** 0.5).mean())
