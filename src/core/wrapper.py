from __future__ import annotations

from abc import ABC, abstractmethod
from functools import partial
from pathlib import Path
from typing import Iterable

import numpy as np
import torch
from torch.utils.data import DataLoader

from ultrafast.datamodules import EmbedInMemoryDataset, embed_collate_fn
from ultrafast.model import DrugTargetCoembeddingLightning
from ultrafast.utils import get_featurizer

from src.core.vendor_compat import apply_sprint_checkpoint_defaults, install_sprint_compat

# Installed at import time so every SPRINT entry point gets the fix before any featurizer
# touches an LMDB cache; see src/core/vendor_compat.py for what this replaces.
install_sprint_compat()


class BaseModelWrapper(ABC):
    @abstractmethod
    def embed_items(
        self,
        items: Iterable[str],
        sample_type: str,
        save_dir: str | Path,
        batch_size: int | None = None,
    ) -> np.ndarray:
        raise NotImplementedError


class SprintWrapper(BaseModelWrapper):
    def __init__(self, model: DrugTargetCoembeddingLightning, device: torch.device) -> None:
        self.model = model
        self.device = device
        self.model.to(device)
        self.model.eval()

    @classmethod
    def from_checkpoint(
        cls,
        checkpoint_path: str | Path,
        device: torch.device | None = None,
    ) -> "SprintWrapper":
        model = DrugTargetCoembeddingLightning.load_from_checkpoint(str(checkpoint_path))
        supplied = apply_sprint_checkpoint_defaults(model)
        if supplied:
            # Printed rather than silent: which hyperparameters a checkpoint lacks is a
            # fact about that checkpoint, and it changes what the model computes.
            print(f"vendor_compat: checkpoint predates {', '.join(supplied)}; using defaults")
        resolved_device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        return cls(model=model, device=resolved_device)

    def embed_items(
        self,
        items: Iterable[str],
        sample_type: str,
        save_dir: str | Path,
        batch_size: int | None = None,
    ) -> np.ndarray:
        if sample_type not in {"drug", "target"}:
            raise ValueError(f"Unsupported sample_type: {sample_type}")

        featurizer_name = (
            self.model.args.drug_featurizer if sample_type == "drug" else self.model.args.target_featurizer
        )
        # drug featurizer is cheap per item, target's isn't
        featurizer_batch_size = 2048 * 8 if sample_type == "drug" else 16
        # mirrors the featurizer's own default
        loader_batch_size = batch_size or (2048 if sample_type == "drug" else 16)

        featurizer = get_featurizer(
            featurizer_name,
            save_dir=str(save_dir),
            batch_size=featurizer_batch_size,
            ext="lmdb",  # caches through the LMDB path install_sprint_compat() patches
        ).to(self.device)

        dataset = EmbedInMemoryDataset(list(items), featurizer)  # materialize; the dataset needs len()
        collate_fn = partial(embed_collate_fn, moltype=sample_type)  # "moltype" is ultrafast's name for this
        dataloader = DataLoader(dataset, batch_size=loader_batch_size, shuffle=False, collate_fn=collate_fn)

        embeddings: list[np.ndarray] = []
        with torch.no_grad():
            for batch in dataloader:
                batch = batch.to(self.device)
                embedding = self.model.embed(batch, sample_type=sample_type)
                embeddings.append(embedding.detach().cpu().numpy())

        if not embeddings:
            return np.empty((0, 0), dtype=np.float32)  # empty items, but still a typed array

        return np.concatenate(embeddings, axis=0)
