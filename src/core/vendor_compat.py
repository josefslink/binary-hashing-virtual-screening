# Runtime shims for third_party/{sprint,drugclip,uni-core} - pristine upstream, never
# edited. Each shim replaces an old in-tree edit; docs/VENDOR_COMPAT.md has the full
# inventory and history of each item.

from __future__ import annotations

import os
from contextlib import contextmanager

# Hyperparameters absent from checkpoints trained before the key was introduced.
# `sigmoid_scalar` postdates `sprint.ckpt`; 5 is upstream's own train.py CLI default.
SPRINT_MISSING_ARG_DEFAULTS: dict[str, object] = {"sigmoid_scalar": 5}

_sprint_installed = False


def install_sprint_compat() -> list[str]:
    """Key LMDB feature caches on the prepared sequence, not the raw one. Idempotent."""
    # upstream hashes the raw sequence on read and again on write, but SaProtFeaturizer's
    # prepare_string truncates anything over 2046 chars, so a long sequence is stored
    # under one key and looked up under another. Symptom: KeyError on LIT-PCBA's long
    # targets (MTORC1, PKM2). Hashing the prepared sequence on both sides fixes it.
    global _sprint_installed
    if _sprint_installed:
        return []

    from ultrafast.featurizers import Featurizer

    original_call = Featurizer.__call__
    original_process_lmdb = Featurizer.process_lmdb

    def __call__(self, seq: str):
        # Only the LMDB branch is redirected. `prepare_string` is idempotent, so callers
        # that already prepared their sequence are unaffected, and the in-memory cache in
        # the branch below still keys on whatever the caller actually passed.
        if self.ext == "lmdb" and self.db is not None:
            seq = self.prepare_string(seq)
        return original_call(self, seq)

    def process_lmdb(self, seq_list):
        # Preparing up front makes upstream's `{md5(seq): seq}` become
        # `{md5(prepare_string(seq)): prepare_string(seq)}`, the key `__call__` now looks
        # up. The stored *values* also become prepared strings, which changes nothing:
        # `SaProtFeaturizer._transform` calls `prepare_string` on everything it receives
        # and the function is idempotent. Upstream's `moltype == "target"` digit filter
        # runs on the prepared list instead of the raw one, with the same outcome:
        # `prepare_string` only interleaves "#" and truncates; it never introduces digits.
        return original_process_lmdb(self, [self.prepare_string(seq) for seq in seq_list])

    Featurizer.__call__ = __call__
    Featurizer.process_lmdb = process_lmdb
    _sprint_installed = True
    return ["Featurizer.__call__", "Featurizer.process_lmdb"]


def apply_sprint_checkpoint_defaults(model) -> list[str]:
    """Fill in hyperparameters older checkpoints predate; returns what it had to supply."""
    # sprint.ckpt predates sigmoid_scalar - DrugTargetCoembeddingLightning.forward raised
    # ConfigAttributeError reading it. Set the default on the loaded args, not the
    # vendored model, so the fix stays in our code.
    args = getattr(model, "args", None)
    if args is None:
        return []

    try:  # SPRINT's args is an OmegaConf DictConfig, which may be in struct mode.
        from omegaconf import OmegaConf

        if OmegaConf.is_config(args):
            OmegaConf.set_struct(args, False)
    except ImportError:  # pragma: no cover - omegaconf ships with SPRINT
        pass

    applied: list[str] = []
    for key, value in SPRINT_MISSING_ARG_DEFAULTS.items():
        try:
            present = key in args
        except TypeError:
            present = hasattr(args, key)
        if present:
            continue
        try:
            args[key] = value
        except TypeError:
            setattr(args, key, value)
        applied.append(f"{key}={value}")
    return applied


def install_drugclip_cpu_compat() -> list[str]:
    """Redirect move_to_cuda to move_to_cpu so DrugCLIP's eval loop can run on CPU."""
    # GPU runs are unaffected; call this only for CPU runs. Known limitation inherited
    # from upstream: BindingAffinityModel.__init__ builds logit_scale with an explicit
    # device="cuda", so *constructing* the model still needs a CUDA device to exist even
    # when inference runs on CPU. No published result in this repo used the CPU path.
    import unicore.utils

    if getattr(unicore.utils.move_to_cuda, "_vendor_compat", False):
        return []

    original = unicore.utils.move_to_cuda

    def move_to_cuda(sample, device=None):
        return unicore.utils.move_to_cpu(sample)

    move_to_cuda._vendor_compat = True
    move_to_cuda._original = original
    unicore.utils.move_to_cuda = move_to_cuda
    return ["unicore.utils.move_to_cuda -> move_to_cpu"]


def ensure_drugclip_workdir(data_dir: str | os.PathLike, workdir: str | os.PathLike):
    """Symlink workdir/data -> data_dir; DrugCLIP resolves its LMDBs relative to cwd."""
    from pathlib import Path

    data_path = Path(data_dir).resolve()
    if not data_path.is_dir():
        raise FileNotFoundError(f"DrugCLIP data directory not found: {data_path}")

    work_path = Path(workdir).resolve()
    work_path.mkdir(parents=True, exist_ok=True)
    link = work_path / "data"
    if link.is_symlink() or link.exists():
        if link.is_symlink() and link.resolve() == data_path:
            return work_path
        link.unlink()
    link.symlink_to(data_path, target_is_directory=True)
    return work_path


def ensure_drugclip_split_alias(
    source_lmdb: str | os.PathLike,
    alias_dir: str | os.PathLike,
    split: str,
    dict_dir: str | os.PathLike,
):
    """Alias one LMDB under a split name that skips DrugCLIP's train-resampling branch."""
    # load_dataset branches on split.startswith("train"): that branch redraws a conformer
    # per epoch and reshuffles rows via SortDataset -> ResamplingDataset, wrong for a
    # fixed cache. Any other split name takes the deterministic branch instead, so
    # dataset[i] is LMDB record i. Pass e.g. split="pairs", not "trainpairs".
    from pathlib import Path

    if split.startswith("train"):
        raise ValueError(
            f"split {split!r} starts with 'train', which selects DrugCLIP's "
            "conformer-resampling branch and defeats the point of the alias"
        )

    source = Path(source_lmdb).resolve()
    if not source.exists():
        raise FileNotFoundError(f"source LMDB not found: {source}")

    alias = Path(alias_dir).resolve()
    alias.mkdir(parents=True, exist_ok=True)

    # setup_task loads both dictionaries from args.data, so they must live here too.
    dicts = Path(dict_dir).resolve()
    for name in ("dict_mol.txt", "dict_pkt.txt"):
        link = alias / name
        if not link.exists():
            link.symlink_to(dicts / name)

    target = alias / f"{split}.lmdb"
    if target.is_symlink() or target.exists():
        if target.is_symlink() and target.resolve() == source:
            return alias
        target.unlink()
    target.symlink_to(source)
    return alias


@contextmanager
def drugclip_workdir(data_dir: str | os.PathLike, workdir: str | os.PathLike):
    """ensure_drugclip_workdir plus a chdir, restored on exit."""
    from pathlib import Path

    work_path = ensure_drugclip_workdir(data_dir, workdir)
    previous = Path.cwd()
    os.chdir(work_path)
    try:
        yield work_path
    finally:
        os.chdir(previous)
