# score dumped embeddings with cosine or binarized Hamming retrieval.
# model-agnostic despite the name; scores SPRINT dumps too.
# run: evaluate_drugclip_hamming.py --embeddings-dir <dump>

from __future__ import annotations

import argparse
import warnings
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

from src.core.config import load_yaml_config
from src.evaluation.evaluator import calculate_ranking_metrics, save_results_csv
from src.evaluation.scorer import (
    project_pair,
    binarize_pair,
    compute_similarity_matrix,
    per_type_centering_skipped,
    reduce_max_similarity,
)

# DrugCLIP's own cal_metrics uses 80.5; SPRINT's calculate_ranking_metrics default is 85.0
DRUGCLIP_BEDROC_ALPHA = 80.5
REQUIRED_ARRAYS = ("mol_reps.npy", "pocket_reps.npy", "labels.npy")


def discover_targets(embeddings_dir: str | Path) -> list[str]:
    root = Path(embeddings_dir)
    if not root.is_dir():
        raise SystemExit(f"Embeddings directory not found: {root}. Run scripts/benchmark_drugclip.py first.")
    # all three arrays must exist; a target killed mid-dump is skipped, not scored half-written
    targets = sorted(
        path.name
        for path in root.iterdir()
        if path.is_dir() and all((path / name).exists() for name in REQUIRED_ARRAYS)
    )
    if not targets:
        raise SystemExit(f"No target directories with dumped embeddings under {root}.")
    return targets


def evaluate_target(
    embeddings_dir: str | Path,
    target: str,
    metric: str = "hamming",
    binarize_center: str | None = None,
    alpha: float = DRUGCLIP_BEDROC_ALPHA,
    code_length: int | None = None,
    projection: str = "gaussian",
    projection_seed: int = 0,
    tie_break: str = "input_order",
    head=None,
) -> dict[str, object]:
    target_dir = Path(embeddings_dir) / target
    missing = [name for name in REQUIRED_ARRAYS if not (target_dir / name).exists()]
    if missing:
        raise FileNotFoundError(f"{target}: missing dumped arrays {missing} in {target_dir}")

    mol_reps = np.load(target_dir / "mol_reps.npy")
    pocket_reps = np.load(target_dir / "pocket_reps.npy")
    labels = np.load(target_dir / "labels.npy")

    if len(labels) != len(mol_reps):
        raise ValueError(f"{target}: {len(labels)} labels but {len(mol_reps)} molecule embeddings")

    drug_embeddings, target_embeddings = mol_reps, pocket_reps
    # A trained hash head maps embeddings to embeddings, so it slots in ahead of the
    # existing chain and everything downstream is unchanged.
    if head is not None:
        from src.hashing.head import apply_head

        drug_embeddings, target_embeddings = apply_head(drug_embeddings, target_embeddings, head)
    # Project before centering and before sign(): the code length IS the projected
    # dimension, and centering is only meaningful in the space the bits are taken in.
    if code_length is not None and code_length < mol_reps.shape[1]:
        drug_embeddings, target_embeddings = project_pair(
            drug_embeddings, target_embeddings, code_length, projection, projection_seed
        )
    if metric == "hamming":
        drug_embeddings, target_embeddings = binarize_pair(
            drug_embeddings, target_embeddings, center=binarize_center
        )

    # Pockets are the query and molecules the candidates, so the max reduces over
    # pockets -- same orientation as evaluate_lit_pcba_target in the SPRINT path.
    similarity_matrix = compute_similarity_matrix(target_embeddings, drug_embeddings, metric=metric)
    max_similarities = reduce_max_similarity(similarity_matrix)

    scores = [(float(score), int(label)) for score, label in zip(max_similarities, labels)]
    metrics = calculate_ranking_metrics(scores, alpha=alpha, tie_break=tie_break)
    metrics["target"] = target
    metrics["n_molecules"] = int(len(labels))
    metrics["n_actives"] = int(labels.sum())
    metrics["n_pockets"] = int(pocket_reps.shape[0])
    metrics["code_length"] = int(drug_embeddings.shape[1])  # after projection, so this is bits
    metrics["tie_break"] = tie_break
    metrics["head"] = "none" if head is None else "applied"
    # "none" when the native dim was kept, even though --projection defaulted to gaussian
    metrics["projection"] = projection if (code_length or 0) and code_length < mol_reps.shape[1] else "none"
    # a 128-bit code admits only 129 Hamming values, so scores tie heavily and early
    # enrichment depends on calculate_ranking_metrics' stable sort. low EF here is a
    # property of the metric's resolution, not necessarily a pipeline defect
    metrics["n_distinct_scores"] = int(len(np.unique(max_similarities)))
    # Recorded rather than merely warned about: see per_type_centering_skipped.
    metrics["per_type_uncentered_target"] = bool(
        metric == "hamming" and per_type_centering_skipped(target_embeddings, binarize_center)
    )
    return metrics


def main() -> None:
    # Re-assert after third-party imports: pandas/matplotlib/sklearn add filters that
    # otherwise swallow the scorer's single-row centering warning.
    warnings.filterwarnings("always", message=".*centering skipped.*")
    parser = argparse.ArgumentParser(
        description="Score a dumped embedding directory with cosine or binarized Hamming "
                    "retrieval. Model-agnostic: any dump following the "
                    "<target>/{mol_reps,pocket_reps,labels}.npy contract works, so this "
                    "scores SPRINT dumps (scripts/dump_sprint_embeddings.py) as well as "
                    "DrugCLIP ones. Set `alpha` per model - SPRINT rows use 85.0, DrugCLIP "
                    "rows 80.5. Mixing them corrupts the BEDROC column and nothing else, silently.")
    parser.add_argument("--config", type=str, default="configs/drugclip/pcba_hamming.yaml")
    parser.add_argument("--embeddings-dir", type=str, default=None, dest="embeddings_dir")
    parser.add_argument("--target", type=str, default=None, help='Target name, or "all"')
    parser.add_argument("--metric", type=str, choices=["cosine", "hamming"], default=None)
    parser.add_argument("--binarize-center", choices=["none", "per_type", "per_type_median", "global", "global_median"], default=None, dest="binarize_center")
    parser.add_argument("--alpha", type=float, default=None,
                        help="BEDROC early-recognition parameter. Falls back to the config's "
                             "`alpha`, then to DrugCLIP's 80.5. SPRINT dumps must use 85.0.")
    parser.add_argument("--code-length", type=int, default=None, dest="code_length",
                        help="Project both matrices to this many dimensions before "
                             "binarizing, i.e. the number of bits in the code. Omit for the "
                             "model's native dimension (SPRINT 1024, DrugCLIP 128).")
    parser.add_argument("--projection", choices=["gaussian", "truncate", "variance"],
                        default="gaussian", help="How to reduce dimensions (see scorer.project_pair)")
    parser.add_argument("--projection-seed", type=int, default=0, dest="projection_seed")
    parser.add_argument("--head", default=None,
                        help="Trained hash head checkpoint (scripts/train_drugclip_hash_head.py). "
                             "Applied to the loaded arrays before centering/sign.")
    parser.add_argument("--tie-break", choices=["input_order", "random"], default="input_order",
                        dest="tie_break",
                        help="How to order equal scores. Molecules are stored actives-first, "
                             "so the default stable sort resolves every tie in favour of the "
                             "actives. Use 'random' for any short-code sweep.")
    parser.add_argument("--output", type=str, default=None, help="Output CSV path")
    args = parser.parse_args()

    config_values = load_yaml_config(args.config) if args.config else {}
    cli_overrides = {
        "embeddings_dir": args.embeddings_dir,
        "target_protein_id": args.target,
        "metric": args.metric,
        "binarize_center": args.binarize_center,
        "alpha": args.alpha,
    }
    for key, value in cli_overrides.items():
        if value is not None:
            config_values[key] = value

    embeddings_dir = config_values.get("embeddings_dir")
    if not embeddings_dir:
        results_dir = config_values.get("results_dir", "outputs/metrics/drugclip_pcba")
        # dump convention: <results_dir>/embeddings
        embeddings_dir = str(Path(results_dir) / "embeddings")
    target_protein_id = str(config_values.get("target_protein_id", "all"))
    metric = config_values.get("metric", "hamming")
    binarize_center = config_values.get("binarize_center", "none")
    center = None if binarize_center in (None, "none") else binarize_center
    # Alpha is model-specific and getting it wrong silently corrupts only the BEDROC
    # column, so let the config carry it rather than relying on a flag being remembered.
    alpha = float(config_values.get("alpha", DRUGCLIP_BEDROC_ALPHA))

    head = None
    if args.head:
        import torch

        from src.hashing.head import HashHeadPair

        # checkpoint stores args + state_dict, not a bare tensor dict
        blob = torch.load(args.head, map_location="cpu", weights_only=False)
        head = HashHeadPair(128, tie=blob["args"].get("tie", False))  # DrugCLIP's dim, not SPRINT's 1024
        head.load_state_dict(blob["state_dict"])
        head.eval()
        print(f"Hash head: {args.head} (lambda={blob['lam']})")

    if target_protein_id.lower() == "all":
        targets = discover_targets(embeddings_dir)
    else:
        targets = [target_protein_id]

    print(f"Embeddings dir: {embeddings_dir}")
    print(f"Metric: {metric} | binarize_center: {center} | BEDROC alpha: {alpha}")
    print(f"Targets ({len(targets)}): {', '.join(targets)}")

    results: list[dict[str, object]] = []
    for target in targets:
        try:
            results.append(
                evaluate_target(
                    embeddings_dir=embeddings_dir,
                    target=target,
                    metric=metric,
                    binarize_center=center,
                    alpha=alpha,
                    code_length=args.code_length,
                    projection=args.projection,
                    projection_seed=args.projection_seed,
                    tie_break=args.tie_break,
                    head=head,
                )
            )
        except Exception as exc:
            # Mirrors evaluate_lit_pcba_targets: one bad target must not sink the sweep.
            print(f"  ! {target}: {exc}")
            results.append(
                {
                    "target": target,
                    "error": str(exc),
                    "auroc": None,
                    "bedroc": None,
                    "ef_0.005": None,
                    "ef_0.01": None,
                    "ef_0.05": None,
                }
            )

    if args.output:
        output_file = Path(args.output)
    else:
        suffix = f"_{metric}"
        if metric == "hamming" and center:
            suffix += f"_{center}"
        # Configurable so a DUD-E run does not get filed under a drugclip_pcba_* name.
        prefix = config_values.get("output_prefix", "drugclip_pcba")
        base_dir = Path(__file__).resolve().parent.parent
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file = base_dir / "outputs" / "metrics" / f"{prefix}{suffix}_{timestamp}.csv"

    output_path = save_results_csv(results, output_file)
    frame = pd.DataFrame(results)
    print()
    print(frame.to_string(index=False))
    if "auroc" in frame.columns:
        valid = frame["auroc"].dropna()  # error rows must not pull the printed mean down
        if not valid.empty:
            print(f"\nauc mean over {len(valid)} target(s): {valid.mean()}")
    print(f"\nSaved results to: {output_path}")


if __name__ == "__main__":
    main()
