from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.core.wrapper import BaseModelWrapper
from src.evaluation.scorer import (
    binarize_pair,
    compute_similarity_matrix,
    per_type_centering_skipped,
    reduce_max_similarity,
)


def load_lit_pcba_target(pcba_dir: str | Path, target: str) -> tuple[np.ndarray, set[str], list[str]]:
    target_folder = Path(pcba_dir) / target
    if not target_folder.is_dir():
        raise ValueError(f"Target protein {target} not found in {pcba_dir}")

    active_smiles: set[str] = set()
    all_smiles: list[str] = []

    # .smi is "SMILES [id]"; actives first so tie_break=input_order favours them on equal scores
    with open(target_folder / "actives.smi", encoding="utf-8") as handle:
        for line in handle:
            smiles = line.strip().split(" ")[0]
            active_smiles.add(smiles)
            all_smiles.append(smiles)

    with open(target_folder / "inactives.smi", encoding="utf-8") as handle:
        for line in handle:
            smiles = line.strip().split(" ")[0]
            all_smiles.append(smiles)

    return np.array(all_smiles), active_smiles, [str(target_folder)]


def load_target_sequences(pcba_dir: str | Path, target: str, target_featurizer: str) -> list[str]:
    pcba_path = Path(pcba_dir)
    # SaProt wants 3Di-tokenised sequences; the other featurizers want residue strings
    toks_file = "saprot_sequence_dict.json" if target_featurizer == "SaProtFeaturizer" else "lit_pcba_sequence_dict.json"
    with open(pcba_path / toks_file, encoding="utf-8") as handle:
        sequence_map = json.load(handle)
    return sequence_map[target]  # a list: LIT-PCBA targets can have several pockets


def load_dude_target(dude_dir: str | Path, target: str) -> tuple[np.ndarray, set[str], list[str]]:
    """Molecules for one DUD-E target, from `<t>_actives.tsv` / `<t>_decoys.tsv`."""
    dude_path = Path(dude_dir)
    actives_file = dude_path / f"{target}_actives.tsv"
    decoys_file = dude_path / f"{target}_decoys.tsv"
    if not actives_file.is_file() or not decoys_file.is_file():
        raise ValueError(f"DUD-E target {target} not found in {dude_path}")

    active_smiles: set[str] = set()
    all_smiles: list[str] = []
    seen_ids: set[str] = set()
    # actives file first so the SMILES order matches LIT-PCBA's actives-then-inactives
    for path, is_active in ((actives_file, True), (decoys_file, False)):
        with open(path, encoding="utf-8") as handle:
            for line in handle:
                parts = line.rstrip("\n").split("\t")
                if len(parts) < 2 or not parts[1]:
                    continue
                compound_id, smiles = parts[0].strip(), parts[1].strip()
                # DUD-E ships several protonation/tautomer rows per compound id (aa2ar:
                # 797 active rows, 482 ids) - keep one per id, matching DrugCLIP's LMDBs
                # (one entry per compound), so counts and AUROC/EF are comparable
                if compound_id in seen_ids:
                    continue
                seen_ids.add(compound_id)
                if is_active:
                    active_smiles.add(smiles)
                all_smiles.append(smiles)

    return np.array(all_smiles), active_smiles, [str(dude_path)]


def _read_fasta(path: str | Path) -> dict[str, str]:
    sequences: dict[str, str] = {}
    name = None
    chunks: list[str] = []
    with open(path, encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line.startswith(">"):
                if name is not None:
                    sequences[name] = "".join(chunks)
                name = line[1:].split()[0]  # first token; matches the tsv prefix used as target id
                chunks = []
            elif line:
                chunks.append(line)
    if name is not None:
        sequences[name] = "".join(chunks)
    return sequences


def load_dude_sequences(dude_dir: str | Path, target: str, target_featurizer: str) -> list[str]:
    # DUD-E has no 3Di tokens (unlike LIT-PCBA's saprot_sequence_dict.json), so
    # SaProtFeaturizer runs masked-structure here. Measured: masking 3Di on LIT-PCBA too
    # moves mean AUROC 0.7253 -> 0.7340, so this isn't costing the DUD-E run anything.
    sequences = _read_fasta(Path(dude_dir) / "dude_targets.fasta")
    if target not in sequences:
        raise ValueError(f"No sequence for DUD-E target {target} in dude_targets.fasta")
    sequence = sequences[target]
    if target_featurizer == "SaProtFeaturizer":
        sequence = to_saprot_tokens(sequence)
    return [sequence]


SAPROT_RESIDUES = set("ACDEFGHIKLMNPQRSTVWY")


def to_saprot_tokens(sequence: str) -> str:
    # SaProt has no 'X#' token - prepare_string maps unknown residue X to X#, which isn't
    # in the 446-token vocab, so the tokenizer raises and upstream's _transform_single
    # catches it and returns a zero tensor. Target silently scores plausibly with no
    # error. DUD-E hits this on most targets (aa2ar has 32 X residues); map to '##'
    # ("both unknown") instead. LIT-PCBA is unaffected - its sequences have no X.
    return "".join(f"{c}#" if c in SAPROT_RESIDUES else "##" for c in sequence.upper())


def list_dude_targets(dude_dir: str | Path) -> list[str]:
    """Targets that have actives, decoys and a sequence."""
    dude_path = Path(dude_dir)
    sequences = _read_fasta(dude_path / "dude_targets.fasta")
    targets = []
    # all three must exist; a target missing the fasta is omitted here, not failed mid-sweep
    for actives in sorted(dude_path.glob("*_actives.tsv")):
        name = actives.name[: -len("_actives.tsv")]
        if (dude_path / f"{name}_decoys.tsv").is_file() and name in sequences:
            targets.append(name)
    return targets


# dataset -> (molecule loader, sequence loader); both share a signature so the
# evaluation path below is dataset-agnostic
DATASET_LOADERS = {
    "lit_pcba": (load_lit_pcba_target, load_target_sequences),
    "dude": (load_dude_target, load_dude_sequences),
}


def maybe_save_bit_density_plot(
    drug_embeddings: np.ndarray,
    target_embeddings: np.ndarray,
    output_path: str | Path,
    target: str,
) -> None:
    # per-bit fraction of 1s, drugs and targets pooled
    ones_ratio = np.concatenate([drug_embeddings, target_embeddings], axis=0).mean(axis=0)
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.hist(ones_ratio, bins=40, color="#4C78A8", alpha=0.85)
    ax.set_title(f"Binarized Bit Density (1s) - {target}")
    ax.set_xlabel("Fraction of 1s per bit")
    ax.set_ylabel("Count")
    ax.grid(axis="y", linestyle="--", alpha=0.4)
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def get_screening_embeddings(
    model_wrapper: BaseModelWrapper,
    data_dir: str | Path,
    target: str,
    target_featurizer: str,
    output_dir: str | Path | None = None,
    dataset: str = "lit_pcba",
) -> dict[str, object]:
    """Drug and target embeddings for one virtual-screening target."""
    if dataset not in DATASET_LOADERS:
        raise ValueError(f"Unknown dataset {dataset!r}; expected one of {sorted(DATASET_LOADERS)}")
    load_molecules, load_sequences = DATASET_LOADERS[dataset]
    all_smiles, active_smiles, _ = load_molecules(data_dir, target)
    target_sequences = load_sequences(data_dir, target, target_featurizer)

    if output_dir:
        target_output_dir = Path(output_dir)
    elif dataset == "dude":
        # DUD-E has no per-target directory, unlike LIT-PCBA
        target_output_dir = Path(data_dir) / "features" / target
    else:
        target_output_dir = Path(data_dir) / target  # LIT-PCBA: LMDB cache lives in the target folder
    target_output_dir.mkdir(parents=True, exist_ok=True)

    drug_embeddings = model_wrapper.embed_items(all_smiles, sample_type="drug", save_dir=target_output_dir)
    target_embeddings = model_wrapper.embed_items(target_sequences, sample_type="target", save_dir=target_output_dir)

    # a single sequence can come back (D,); the similarity matrix needs (n_pockets, D)
    if target_embeddings.ndim == 1:
        target_embeddings = np.expand_dims(target_embeddings, axis=0)

    # upstream featurizers swallow a tokenizer failure and return a zero tensor, which
    # scores as a plausible number while the target contributes nothing (see to_saprot_tokens)
    if target_embeddings.size and not np.any(target_embeddings):
        raise RuntimeError(
            f"{dataset} target {target!r}: all target embeddings are zero. This is what a "
            "silently-failed featurization looks like — check the featurizer output above "
            "for a tokenization error rather than trusting the metrics."
        )

    # dumped as labels.npy; evaluate_drugclip_hamming.py scores from that, not SMILES
    labels = np.array([1 if smiles in active_smiles else 0 for smiles in all_smiles], dtype=np.int32)
    return {
        "all_smiles": all_smiles,
        "active_smiles": active_smiles,
        "labels": labels,
        "drug_embeddings": drug_embeddings,
        "target_embeddings": target_embeddings,
        "target_output_dir": target_output_dir,
    }


def calculate_ranking_metrics(
    scores: list[tuple[float, int]],
    alpha: float = 85.0,  # SPRINT/Truchon default; DrugCLIP DUD-E passes 80.5 from its own scorer
    tie_break: str = "input_order",
    tie_break_seed: int = 0,
) -> dict[str, float]:
    """Rank-based virtual-screening metrics; alpha is BEDROC's early-recognition parameter."""
    # tie_break default is input_order so committed numbers reproduce: molecules load
    # actives-first and sorted() is stable, so equal scores favour actives. Worth ~0.006
    # AUROC at SPRINT's native 1024 bits, but large at short codes - MAPK1 went
    # 0.7264 -> 0.6503 at 64 bits once ties were broken fairly. Use tie_break="random"
    # for any code-length sweep.
    from ultrafast.utils import CalcAUC, CalcBEDROC, CalcEnrichment

    if tie_break == "random":
        import numpy as _np

        order = _np.random.default_rng(tie_break_seed).permutation(len(scores))
        scores = [scores[i] for i in order]
    elif tie_break != "input_order":
        raise ValueError(f"Unknown tie_break {tie_break!r}; expected 'input_order' or 'random'")

    # CalcAUC/CalcBEDROC expect (similarity, label) pairs in descending score order
    ordered_scores = sorted(scores, key=lambda item: item[0], reverse=True)
    efs = CalcEnrichment(ordered_scores, 1, [0.005, 0.01, 0.05])
    return {
        "auroc": CalcAUC(ordered_scores, 1),
        "bedroc": CalcBEDROC(ordered_scores, 1, alpha),
        "ef_0.005": efs[0],
        "ef_0.01": efs[1],
        "ef_0.05": efs[2],
    }


def evaluate_screening_target(
    model_wrapper: BaseModelWrapper,
    data_dir: str | Path,
    target: str,
    target_featurizer: str,
    metric: str = "cosine",
    binarize_center: str | None = None,
    plot_binarized_dist: bool = False,
    output_dir: str | Path | None = None,
    dataset: str = "lit_pcba",
) -> dict[str, float]:
    embeddings = get_screening_embeddings(
        model_wrapper=model_wrapper,
        data_dir=data_dir,
        target=target,
        target_featurizer=target_featurizer,
        output_dir=output_dir,
        dataset=dataset,
    )
    all_smiles = embeddings["all_smiles"]
    active_smiles = embeddings["active_smiles"]
    drug_embeddings = embeddings["drug_embeddings"]
    target_embeddings = embeddings["target_embeddings"]
    target_output_dir = embeddings["target_output_dir"]

    similarity_metric = metric
    # captured before binarization: binarize_pair may leave a single-row matrix
    # uncentered, and that has to reach the CSV, not just a warning stream that
    # third-party imports can filter away
    n_target_rows = int(target_embeddings.shape[0])
    uncentered_target = per_type_centering_skipped(target_embeddings, binarize_center) if metric == "hamming" else False
    if metric == "hamming":
        drug_embeddings, target_embeddings = binarize_pair(
            drug_embeddings,
            target_embeddings,
            center=binarize_center,
        )
        if plot_binarized_dist:
            maybe_save_bit_density_plot(
                drug_embeddings,
                target_embeddings,
                target_output_dir / f"{dataset}_{target}_binarized_bit_density.png",
                target,
            )
    # pockets are the query, molecules the candidates (same orientation as the DrugCLIP scorer)
    similarity_matrix = compute_similarity_matrix(target_embeddings, drug_embeddings, metric=similarity_metric)
    # max over pockets; a multi-pocket LIT-PCBA target contributes one score per molecule
    max_similarities = reduce_max_similarity(similarity_matrix)

    scores = []
    for index, smiles in enumerate(all_smiles):
        scores.append((float(max_similarities[index]), 1 if smiles in active_smiles else 0))

    metrics = calculate_ranking_metrics(scores)
    metrics["target"] = target
    metrics["n_molecules"] = int(len(all_smiles))
    metrics["n_actives"] = int(sum(label for _score, label in scores))
    metrics["n_pockets"] = n_target_rows
    # a d-bit code admits at most d+1 distinct Hamming similarities, so a short code
    # makes the ranking's top one big tie block rather than an ordering
    metrics["n_distinct_scores"] = int(len(np.unique(max_similarities)))
    metrics["per_type_uncentered_target"] = uncentered_target
    return metrics


def evaluate_screening_targets(
    model_wrapper: BaseModelWrapper,
    data_dir: str | Path,
    targets: list[str],
    target_featurizer: str,
    metric: str = "cosine",
    binarize_center: str | None = None,
    plot_binarized_dist: bool = False,
    checkpoint_name: str | None = None,
    dataset: str = "lit_pcba",
) -> list[dict[str, object]]:
    results: list[dict[str, object]] = []
    for target in targets:
        try:
            target_result = evaluate_screening_target(
                model_wrapper=model_wrapper,
                data_dir=data_dir,
                target=target,
                target_featurizer=target_featurizer,
                metric=metric,
                binarize_center=binarize_center,
                plot_binarized_dist=plot_binarized_dist,
                dataset=dataset,
            )
            if checkpoint_name:
                target_result["checkpoint"] = checkpoint_name
            results.append(target_result)
        except Exception as exc:  # one bad target must not abort a 15-target sweep
            results.append(
                {
                    "target": target,
                    "checkpoint": checkpoint_name,
                    "error": str(exc),
                    "auroc": None,
                    "bedroc": None,
                    "ef_0.005": None,
                    "ef_0.01": None,
                    "ef_0.05": None,
                }
            )
    return results


def save_results_csv(results: list[dict[str, object]], output_file: str | Path) -> Path:
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(results).to_csv(output_path, index=False)
    return output_path
