# guard tests for src/evaluation/scorer.py, the file every published number flows through
# run: .venv/bin/python -m pytest tests/test_scorer.py -q

from __future__ import annotations

import warnings

import numpy as np
import pytest

from src.evaluation.scorer import (
    CENTERING_MODES,
    apply_binarization,
    binarize_pair,
    per_type_centering_skipped,
    resolve_centering,
)


# mode resolution

def test_unknown_mode_raises_rather_than_silently_skipping_centering():
    # The pre-2026-08-26 if/elif chain returned uncentered codes for any unknown string,
    # which is indistinguishable from center="none" in the output but not in the filename.
    with pytest.raises(ValueError, match="Unknown binarize_center"):
        resolve_centering("per_type_mediam")  # typo, deliberately
    with pytest.raises(ValueError):
        binarize_pair(np.zeros((4, 3)), np.zeros((2, 3)), center="nonsense")


def test_none_and_none_string_both_mean_no_centering():
    assert resolve_centering(None) is None
    assert resolve_centering("none") is None


def test_historical_modes_still_resolve_to_mean():
    # Renaming these would invalidate every committed CSV filename.
    assert resolve_centering("per_type") == ("per_type", "mean")
    assert resolve_centering("global") == ("global", "mean")
    assert set(CENTERING_MODES) == {"per_type", "per_type_median", "global", "global_median"}


# the statistic actually applied

def test_per_type_is_mean_centering():
    rng = np.random.default_rng(0)
    drugs = rng.normal(size=(50, 8))
    targets = rng.normal(size=(5, 8))
    d_bits, t_bits = binarize_pair(drugs, targets, center="per_type")
    np.testing.assert_array_equal(d_bits, (drugs - drugs.mean(0, keepdims=True) >= 0))
    np.testing.assert_array_equal(t_bits, (targets - targets.mean(0, keepdims=True) >= 0))


def test_per_type_median_is_median_centering():
    rng = np.random.default_rng(1)
    drugs = rng.normal(size=(50, 8))
    targets = rng.normal(size=(5, 8))
    d_bits, t_bits = binarize_pair(drugs, targets, center="per_type_median")
    np.testing.assert_array_equal(d_bits, (drugs - np.median(drugs, 0, keepdims=True) >= 0))
    np.testing.assert_array_equal(t_bits, (targets - np.median(targets, 0, keepdims=True) >= 0))


def test_global_median_shares_one_reference_over_the_concatenation():
    rng = np.random.default_rng(2)
    drugs = rng.normal(size=(9, 4))
    targets = rng.normal(size=(3, 4))
    reference = np.median(np.concatenate([drugs, targets], 0), axis=0, keepdims=True)
    d_bits, t_bits = binarize_pair(drugs, targets, center="global_median")
    np.testing.assert_array_equal(d_bits, (drugs - reference >= 0))
    np.testing.assert_array_equal(t_bits, (targets - reference >= 0))


def test_mean_and_median_differ_on_a_skewed_distribution():
    # If these ever agree the test fixture is wrong, not the code.
    rng = np.random.default_rng(3)
    drugs = rng.exponential(size=(200, 6))          # strongly right-skewed
    targets = rng.exponential(size=(7, 6))
    mean_bits, _ = binarize_pair(drugs, targets, center="per_type")
    median_bits, _ = binarize_pair(drugs, targets, center="per_type_median")
    assert not np.array_equal(mean_bits, median_bits)
    # Median splits each dimension in half by construction; the mean does not on a skew.
    assert abs(median_bits.mean() - 0.5) < abs(mean_bits.mean() - 0.5)


# the single-row guard

@pytest.mark.parametrize("mode", ["per_type", "per_type_median"])
def test_single_row_guard_fires_for_both_statistics(mode):
    """A one-row matrix centered by its own reference is exactly zero -> all-ones code; this silently dropped FEN1/OPRK1 before the 2026-08-25 fix."""
    drugs = np.random.default_rng(4).normal(size=(20, 5))
    single_target = np.array([[0.4, -0.2, 0.9, -1.1, 0.0]])

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        _, t_bits = binarize_pair(drugs, single_target, center=mode)

    assert any("centering skipped" in str(w.message) for w in caught), "guard did not warn"
    # Uncentered, not all-ones: the sign of the raw row survives.
    np.testing.assert_array_equal(t_bits, (single_target >= 0))
    assert t_bits.min() == 0, "all-ones code means the guard did not fire"
    assert per_type_centering_skipped(single_target, mode) is True


def test_skipped_predicate_matches_reality_for_multi_row_and_global():
    rows = np.random.default_rng(5).normal(size=(6, 5))
    single = rows[:1]
    assert per_type_centering_skipped(rows, "per_type") is False
    assert per_type_centering_skipped(rows, "per_type_median") is False
    assert per_type_centering_skipped(single, "none") is False
    # global takes its reference over the concatenation, so it is never degenerate.
    assert per_type_centering_skipped(single, "global") is False
    assert per_type_centering_skipped(single, "global_median") is False


# the >= 0 threshold

def test_exact_median_row_lands_on_bit_one():
    """Odd row count -> one row sits exactly on the median; >= 0 sends it to bit 1, biasing density upward on small matrices."""
    column = np.array([[-1.0], [0.0], [1.0]])          # median is exactly 0.0
    bits = apply_binarization(column, center="per_type_median")
    assert bits.ravel().tolist() == [0, 1, 1]


def test_apply_binarization_global_is_the_scalar_grand_statistic():
    values = np.array([[1.0, 2.0], [3.0, 100.0]])
    np.testing.assert_array_equal(
        apply_binarization(values, center="global"), (values - values.mean() >= 0)
    )
    np.testing.assert_array_equal(
        apply_binarization(values, center="global_median"), (values - np.median(values) >= 0)
    )


# dimensionality reduction for the bit-budget sweep

def test_projection_is_shared_between_the_two_matrices():
    """Independent projections would leave drugs/targets in different random spaces with matching shapes and no error raised."""
    from src.evaluation.scorer import project_pair

    rng = np.random.default_rng(0)
    shared = rng.normal(size=(30, 16))
    d, t = project_pair(shared, shared, code_length=8, method="gaussian", seed=7)
    # Same input through the same projection must give the same output.
    np.testing.assert_allclose(d, t, rtol=0, atol=0)


def test_projection_preserves_relative_geometry_on_structured_embeddings():
    """Low-rank + noise fixture, not isotropic vectors - on pure noise the correlation is only ~0.61 (JL distortion at k=128), which would measure nothing."""
    from src.evaluation.scorer import project_pair

    rng = np.random.default_rng(1)
    basis = rng.normal(size=(16, 256))                    # 16 latent directions
    drugs = rng.normal(size=(200, 16)) @ basis + 0.1 * rng.normal(size=(200, 256))
    targets = rng.normal(size=(4, 16)) @ basis + 0.1 * rng.normal(size=(4, 256))
    drugs /= np.linalg.norm(drugs, axis=1, keepdims=True)
    targets /= np.linalg.norm(targets, axis=1, keepdims=True)

    before = targets @ drugs.T
    pd_, pt_ = project_pair(drugs, targets, code_length=128, method="gaussian", seed=0)
    pd_ /= np.linalg.norm(pd_, axis=1, keepdims=True)
    pt_ /= np.linalg.norm(pt_, axis=1, keepdims=True)
    after = pt_ @ pd_.T
    assert np.corrcoef(before.ravel(), after.ravel())[0, 1] > 0.95


def test_projection_shapes_and_guards():
    from src.evaluation.scorer import project_pair

    drugs = np.zeros((5, 32), dtype=np.float32)
    targets = np.zeros((2, 32), dtype=np.float32)
    for method in ("gaussian", "truncate", "variance"):
        d, t = project_pair(drugs, targets, 8, method=method)
        assert d.shape == (5, 8) and t.shape == (2, 8)
    # identity when the requested length is the native one
    d, t = project_pair(drugs, targets, 32)
    assert d.shape == (5, 32)
    with pytest.raises(ValueError, match="exceeds the embedding dimension"):
        project_pair(drugs, targets, 64)
    with pytest.raises(ValueError, match="Unknown projection"):
        project_pair(drugs, targets, 8, method="pca")
    with pytest.raises(ValueError, match="share a dimension"):
        project_pair(drugs, np.zeros((2, 16)), 8)


def test_gaussian_projection_is_seeded_and_reproducible():
    from src.evaluation.scorer import project_pair

    rng = np.random.default_rng(2)
    drugs, targets = rng.normal(size=(10, 64)), rng.normal(size=(3, 64))
    a, _ = project_pair(drugs, targets, 16, seed=42)
    b, _ = project_pair(drugs, targets, 16, seed=42)
    c, _ = project_pair(drugs, targets, 16, seed=43)
    np.testing.assert_array_equal(a, b)
    assert not np.array_equal(a, c)
