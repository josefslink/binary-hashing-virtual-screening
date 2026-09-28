# capture DrugCLIP encoder outputs (mol_reps/pocket_reps/labels) without editing third_party/.

from __future__ import annotations

from typing import Any

import numpy as np
import torch


# test_pcba_target/test_dude_target return only scalars; hook mol_project/pocket_project
class ProjectionCapture:
    """Forward hook replaying model.<x>_project's own L2-normalize-then-numpy steps."""

    def __init__(self, module: torch.nn.Module) -> None:
        self.batches: list[np.ndarray] = []
        self._handle = module.register_forward_hook(self._on_forward)

    def _on_forward(self, module, inputs, output) -> None:  # noqa: ARG002
        normalized = output / output.norm(dim=-1, keepdim=True)
        self.batches.append(normalized.detach().cpu().numpy())

    def stacked(self) -> np.ndarray:
        if not self.batches:
            return np.empty((0, 0), dtype=np.float16)
        return np.concatenate(self.batches, axis=0)

    def close(self) -> None:
        self._handle.remove()


class LabelCapture:
    """Wraps task.load_mols_dataset so its collater also records labels and SMILES names."""

    # batch order is the DataLoader's, i.e. the same order ProjectionCapture sees the
    # molecule projections in, so the two line up index for index. Use as a context
    # manager; the task is restored on exit.

    def __init__(self, task: Any) -> None:
        self.task = task
        self.labels: list[int] = []
        self.names: list[str] = []
        self._original = task.load_mols_dataset

    def __enter__(self) -> "LabelCapture":
        def load_mols_dataset(*args, **kwargs):
            dataset = self._original(*args, **kwargs)
            original_collater = dataset.collater

            def collater(samples):
                batch = original_collater(samples)
                try:
                    target = batch["target"]
                except (KeyError, TypeError):
                    target = None
                if target is not None:
                    self.labels.extend(np.asarray(target.detach().cpu()).ravel().tolist())
                try:
                    names = batch["smi_name"]
                except (KeyError, TypeError):
                    names = None
                if names is not None:
                    self.names.extend(list(names))
                return batch

            try:
                dataset.collater = collater
            except AttributeError as exc:  # pragma: no cover - defensive
                raise RuntimeError(
                    "Cannot wrap the DrugCLIP mol dataset collater; label capture would "
                    "silently return nothing. Refusing to continue."
                ) from exc
            return dataset

        self.task.load_mols_dataset = load_mols_dataset
        return self

    def __exit__(self, *exc_info) -> None:
        # Remove the instance attribute so the class method is reachable again, rather
        # than leaving a bound method shadowing it.
        try:
            del self.task.load_mols_dataset
        except AttributeError:
            self.task.load_mols_dataset = self._original

    def as_array(self) -> np.ndarray:
        return np.asarray(self.labels, dtype=np.int32)


def load_drugclip_model(args: Any, use_cuda: bool):
    """Checkpoint -> task -> model, shared by the test and train embedding dumps."""
    # import here, not at module level: unicore is only in .venv_drugclip, and this
    # module also gets imported from .venv where unicore happens to be installed too,
    # so a top-level import wouldn't even fail there and would hide the real boundary.
    from unicore import checkpoint_utils, tasks

    state = checkpoint_utils.load_checkpoint_to_cpu(args.path)
    task = tasks.setup_task(args)
    model = task.build_model(args)
    model.load_state_dict(state["model"], strict=False)
    if args.fp16 and use_cuda:
        model.half()
    if use_cuda:
        model.cuda()
    model.eval()
    return task, model


def empty_cache_before(module: torch.nn.Module):
    """Run torch.cuda.empty_cache() once, right before module's first forward pass."""
    # reproduces a fix that used to sit between test_pcba_target's molecule and pocket
    # loops: the molecule phase leaves GiB of cached allocator blocks behind, and the
    # pocket encoder's pair-representation float cast is what then failed to allocate.
    # Hooking the pocket encoder puts the release at exactly that boundary.
    state = {"done": False}

    def _pre_hook(module, inputs):  # noqa: ARG001
        if not state["done"]:
            state["done"] = True
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    return module.register_forward_pre_hook(_pre_hook)
