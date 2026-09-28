from __future__ import annotations

import numpy as np
import torch
from torch import Tensor, nn


class HashHead(nn.Module):
    """128-d -> 128-d, L2-normalised. Identity-initialised by default."""

    # unnormalised tanh (y = tanh(Wx+b)) is degenerate here: L_c is a cosine loss and
    # scale-invariant, so the head can drive L_hash to ~0 by scaling W until tanh
    # saturates (y -> sign(Wx+b), same bits as at init) - a clean-looking lambda sweep
    # with zero bits flipped. L2-normalising removes that escape: with ||y||=1,
    # minimising the hash loss means maximising ||y||_1 on the unit sphere, a real
    # rotation that can't be faked by rescaling.
    #
    # identity_init: at step 0 sign(head(x)) == sign(x), so an untrained head must
    # reproduce the committed post-hoc numbers exactly under --binarize-center none
    # (verified, max |delta| = 0 over 15 targets) - the regression test for this class.

    def __init__(
        self,
        dim: int = 128,
        out_dim: int | None = None,
        bias: bool = True,
        identity_init: bool = True,
    ) -> None:
        super().__init__()
        out_dim = out_dim or dim
        self.linear = nn.Linear(dim, out_dim, bias=bias)
        if identity_init and out_dim == dim:
            with torch.no_grad():
                self.linear.weight.copy_(torch.eye(dim))
                if bias:
                    self.linear.bias.zero_()

    def forward(self, x: Tensor) -> Tensor:
        y = self.linear(x)
        return y / y.norm(dim=-1, keepdim=True).clamp_min(1e-12)


class HashHeadPair(nn.Module):
    """Separate heads for the two modalities, mirroring DrugCLIP's two projectors."""

    # tie=True shares one head - the ITQ case: a shared orthogonal map leaves every
    # inner product (and L_c) invariant, so the objective reduces to quantization alone.
    # Reference point, not the default.

    def __init__(self, dim: int = 128, tie: bool = False, **kwargs) -> None:
        super().__init__()
        self.mol = HashHead(dim, **kwargs)
        self.pocket = self.mol if tie else HashHead(dim, **kwargs)
        self.tied = tie

    def forward(self, mol: Tensor, pocket: Tensor) -> tuple[Tensor, Tensor]:
        return self.mol(mol), self.pocket(pocket)


def apply_head(
    mol_reps: np.ndarray,
    pocket_reps: np.ndarray,
    head: HashHeadPair,
    device: str = "cpu",
    batch_size: int = 65536,
) -> tuple[np.ndarray, np.ndarray]:
    """Map cached embeddings through a trained head."""
    # cached arrays are float16; compute in float32 and return float32 rather than
    # casting back - a needless lossy round-trip right before a >= 0 threshold, the same
    # trap as the original SPRINT embedding cache (scripts/dump_sprint_embeddings.py)
    head = head.to(device).eval()
    out = []
    with torch.no_grad():
        for name, arr in (("mol", mol_reps), ("pocket", pocket_reps)):
            module = head.mol if name == "mol" else head.pocket
            chunks = []
            for start in range(0, len(arr), batch_size):
                block = torch.from_numpy(
                    np.ascontiguousarray(arr[start : start + batch_size], dtype=np.float32)
                ).to(device)
                chunks.append(module(block).cpu().numpy())
            out.append(np.concatenate(chunks, axis=0).astype(np.float32))
    return out[0], out[1]


def code_diagnostics(mol_bits: np.ndarray, reference_bits: np.ndarray | None = None) -> dict:
    """Numbers that reveal a degenerate hash, reported per lambda."""
    # L_hash has no diversity pressure - every hypercube corner minimises it equally, only
    # L_c stops every molecule collapsing onto one code. A lambda whose hash loss fell
    # while bit_flip_fraction stayed ~0 changed nothing and isn't a real data point.
    bits = mol_bits.astype(np.uint8)
    info = {
        "mean_bit_density": float(bits.mean()),
        "per_bit_density_sd": float(bits.mean(axis=0).std()),
        "n_distinct_codes": int(len(np.unique(bits, axis=0))),
        "n_molecules": int(len(bits)),
    }
    if reference_bits is not None:
        info["bit_flip_fraction"] = float((bits != reference_bits.astype(np.uint8)).mean())
    return info
