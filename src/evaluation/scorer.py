from __future__ import annotations

import warnings

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

# per_type/global meant mean-centering until the median variants were added on
# 2026-08-26; every result file written before then is mean. Median centers exactly at
# the >= 0 threshold, mean only does for a symmetric distribution. Table 4 in the report
# shows that it buys nothing on DrugCLIP and swings SPRINT in both directions.
CENTERING_MODES: dict[str, tuple[str, str]] = {
    "per_type": ("per_type", "mean"),
    "per_type_median": ("per_type", "median"),
    "global": ("global", "mean"),
    "global_median": ("global", "median"),
}

_STATISTICS = {"mean": np.mean, "median": np.median}


def resolve_centering(center: str | None) -> tuple[str, str] | None:
    """center string -> (scope, statistic), or None for uncentered."""
    if center is None or center == "none":
        return None
    try:
        return CENTERING_MODES[center]
    except KeyError:
        # old if/elif chain had no fallthrough guard here: a typo fell through to
        # uncentered silently, still written to a file named after the intended condition
        raise ValueError(
            f"Unknown binarize_center {center!r}; expected one of "
            f"{['none', *sorted(CENTERING_MODES)]}"
        ) from None


PROJECTIONS = ("gaussian", "truncate", "variance")


def project_pair(
    drug_embeddings: np.ndarray,
    target_embeddings: np.ndarray,
    code_length: int,
    method: str = "gaussian",
    seed: int = 0,
) -> tuple[np.ndarray, np.ndarray]:
    """Reduce both matrices to code_length dims before binarizing, same projection for both."""
    if method not in PROJECTIONS:
        raise ValueError(f"Unknown projection {method!r}; expected one of {list(PROJECTIONS)}")
    dim = drug_embeddings.shape[1]
    if target_embeddings.shape[1] != dim:
        raise ValueError("Drug and target embeddings must share a dimension.")
    if code_length > dim:
        raise ValueError(f"code_length {code_length} exceeds the embedding dimension {dim}")
    if code_length == dim:
        return drug_embeddings, target_embeddings

    drug = drug_embeddings.astype(np.float32, copy=False)
    target = target_embeddings.astype(np.float32, copy=False)

    if method == "gaussian":
        # standard LSH construction, seeded for reproducibility
        rng = np.random.default_rng(seed)
        matrix = rng.normal(size=(dim, code_length)).astype(np.float32) / np.sqrt(code_length)
        return drug @ matrix, target @ matrix
    if method == "truncate":
        # dimension order of a learned embedding is arbitrary - weak baseline
        return drug[:, :code_length], target[:, :code_length]
    # highest-variance dims, ranked on the drug matrix (the larger one); data-dependent
    keep = np.argsort(drug.var(axis=0))[::-1][:code_length]
    keep.sort()
    return drug[:, keep], target[:, keep]


def _center_per_dimension(
    embeddings: np.ndarray,
    name: str,
    statistic: str = "mean",
    mode: str = "per_type",
) -> np.ndarray:
    """Subtract the per-dimension mean/median; skip (and warn) if degenerate."""
    if embeddings.shape[0] < 2:
        # centering a single row by its own mean/median is exactly 0 -> an all-ones code
        # once thresholded at >= 0, and the row drops out of Hamming ranking silently.
        # Same for the median: the median of one row is that row.
        warnings.warn(
            f"{mode} centering skipped for {name} with {embeddings.shape[0]} row(s): "
            f"centering a single row by its own {statistic} produces a constant all-ones "
            "code. Binarizing it uncentered instead.",
            RuntimeWarning,
            stacklevel=3,
        )
        return embeddings
    return embeddings - _STATISTICS[statistic](embeddings, axis=0, keepdims=True)


def per_type_centering_skipped(embeddings: np.ndarray, center: str | None) -> bool:
    """True if per_type centering would hit the single-row degenerate case."""
    # the RuntimeWarning above isn't reliable in scripts - pandas/matplotlib/sklearn imports
    # add warnings filters that swallow it, so callers writing result files use this instead
    resolved = resolve_centering(center)
    return resolved is not None and resolved[0] == "per_type" and embeddings.shape[0] < 2


def apply_binarization(embeddings: np.ndarray, center: str | None = None) -> np.ndarray:
    """Binarize one matrix on its own. Not the single-matrix form of binarize_pair:
    under `global` this subtracts one scalar over the whole matrix, whereas binarize_pair
    subtracts a per-dimension vector shared by both sides. Only the per_type scopes agree.
    Used by tests/test_scorer.py to pin the centering arithmetic; no scoring path calls it."""
    resolved = resolve_centering(center)
    adjusted = embeddings.astype(np.float32, copy=True)
    if resolved is not None:
        scope, statistic = resolved
        if scope == "global":
            adjusted = adjusted - _STATISTICS[statistic](adjusted)
        else:
            adjusted = _center_per_dimension(adjusted, "embeddings", statistic, center)
    return (adjusted >= 0).astype(np.uint8)


def binarize_pair(
    drug_embeddings: np.ndarray,
    target_embeddings: np.ndarray,
    center: str | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    resolved = resolve_centering(center)
    drug_array = drug_embeddings.astype(np.float32, copy=True)
    target_array = target_embeddings.astype(np.float32, copy=True)

    if resolved is not None:
        scope, statistic = resolved
        if scope == "global":
            # one per-dimension reference shared by both matrices - not degenerate for a
            # single-row matrix, since the reference comes from the concatenation
            reference = _STATISTICS[statistic](
                np.concatenate([drug_array, target_array], axis=0), axis=0, keepdims=True
            )
            drug_array = drug_array - reference
            target_array = target_array - reference
        else:
            drug_array = _center_per_dimension(drug_array, "drug embeddings", statistic, center)
            target_array = _center_per_dimension(target_array, "target embeddings", statistic, center)

    return (drug_array >= 0).astype(np.uint8), (target_array >= 0).astype(np.uint8)


def compute_similarity_matrix(
    query_embeddings: np.ndarray,
    candidate_embeddings: np.ndarray,
    metric: str = "cosine",
) -> np.ndarray:
    if query_embeddings.ndim != 2 or candidate_embeddings.ndim != 2:
        raise ValueError("Embeddings must be 2D arrays.")
    if query_embeddings.shape[1] != candidate_embeddings.shape[1]:
        raise ValueError("Embedding dimensions must match.")

    if metric == "hamming":
        xor = np.logical_xor(query_embeddings[:, None, :], candidate_embeddings[None, :, :])
        hamming_distance = xor.sum(axis=2).astype(np.float32)
        return 1.0 - (hamming_distance / query_embeddings.shape[1])

    if metric == "cosine":
        return cosine_similarity(query_embeddings, candidate_embeddings)

    raise ValueError(f"Unsupported metric: {metric}")


def reduce_max_similarity(similarity_matrix: np.ndarray) -> np.ndarray:
    if similarity_matrix.ndim != 2:
        raise ValueError("Similarity matrix must be 2D.")
    return np.max(similarity_matrix, axis=0)
