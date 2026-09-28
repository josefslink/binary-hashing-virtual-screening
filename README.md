# Binary Screening Codes: SPRINT, DrugCLIP, and a DrugHash Approximation

This repository reproduces two structure-aware virtual screening models, SPRINT and
DrugCLIP, on the LIT-PCBA and DUD-E benchmarks, and then measures what binarizing their
embeddings into hash codes costs in retrieval quality. DrugHash, the paper that proposes
training hash codes into such a model, publishes neither code nor weights, so part of this
work is an approximation of it: a small hash head trained on frozen DrugCLIP encoders.

The full write-up is [`docs/report/report.pdf`](docs/report/report.pdf). Every number in
it is re-derived from the committed CSVs by
[`scripts/audit_report_numbers.py`](scripts/audit_report_numbers.py), which runs from a
fresh clone and exits non-zero on any mismatch.

## Headline results

All of these are read directly from CSVs committed under `outputs/metrics/`.

| Model, benchmark | Metric | Reproduced | Published |
|---|---|---|---|
| SPRINT, LIT-PCBA | AUROC | 0.7253 | 0.7340 |
| DrugCLIP, LIT-PCBA | AUROC | 0.5832 | 0.5717 |
| DrugCLIP, DUD-E | AUROC | 0.8071 | 0.8093 |

Both AUROC reproductions land within one to two points of their papers. BEDROC is less
close: DrugCLIP's DUD-E BEDROC comes back at 0.4733 against a published 0.5052, 3.2 points
low. Beyond reproduction, two findings came out of this project:

- Binarizing embeddings pushes SPRINT and DrugCLIP in opposite directions, and which
  reference you subtract matters as much as the direction does. A shared per-dimension
  reference (`global`) lifts DrugCLIP 0.019 AUROC above its own continuous baseline while
  costing SPRINT 0.28-0.30; a per-type reference (`per_type`) costs SPRINT far less, 0.06
  to 0.12. The pattern holds on both benchmarks, over 117 SPRINT targets and 115 DrugCLIP
  targets.
- Post-hoc binarization and a trained hashing objective preserve different halves of the
  ranking: centering recovers global ranking quality (AUROC) while losing early
  enrichment, and a hash head trained on frozen DrugCLIP encoders does the opposite,
  improving BEDROC and EF 0.5% while AUROC barely moves. The trained-head result is a
  lower bound, not a reproduction of DrugHash, because the encoders were kept frozen
  (see "Hardware used" below for why).

The report's limitations section lists every caveat that qualifies these numbers. The
trained-head figures in particular are a lower bound rather than a reproduction of
DrugHash, and a measurable fraction of the DUD-E actives appear in the released
checkpoint's training set.

## Setup

Two virtual environments are needed because SPRINT and DrugCLIP pull in incompatible
dependency stacks (different pinned versions of PyTorch, fairseq-style packages, etc.):

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e .
deactivate

python -m venv .venv_drugclip
source .venv_drugclip/bin/activate
pip install -e .
deactivate
```

`pip install -e .` installs this project's own `src/` and `scripts/` packages plus the
shared dependencies in `pyproject.toml` (torch, pandas, numpy, pytorch-lightning,
matplotlib, scikit-learn, rdkit, lmdb, tqdm, PyYAML) into whichever environment is
active. SPRINT-specific and
DrugCLIP-specific packages (e.g. the vendored `ultrafast` and `unimol`/`unicore` code
under `third_party/`, and the SPRINT/DrugCLIP third-party pip dependencies) need to be
installed into the matching venv separately, following the upstream instructions in
`third_party/sprint/README.md` and `third_party/drugclip/README.md`.

Data and checkpoints are **not in git** (`data/` is roughly 190 GB, `checkpoints/` roughly
7 GB) and must be obtained from upstream:

- LIT-PCBA: `third_party/sprint/data/download_pcba.sh` downloads the benchmark from
  `drugdesign.unistra.fr` plus a set of SaProt structure-token sequences via `gdown`.
- DUD-E: `third_party/sprint/data/DUDe/download_full_tsv.sh` downloads the full DUD-E
  table from MIT's ConPLex mirror.
- SPRINT checkpoints (`checkpoints/sprint_official/`): Google Drive links in
  `third_party/sprint/checkpoints/README.md`.
- DrugCLIP checkpoint and training/test data (`checkpoints/drugclip/`,
  `data/drugclip/`): Google Drive link in `third_party/drugclip/README.md`.

## Upstream provenance

`third_party/` holds pristine, unmodified checkouts of three upstream repositories, pinned
at these exact commits:

| Project | Repository | Commit |
|---|---|---|
| SPRINT | https://github.com/abhinadduri/panspecies-dti.git | `ce5bc698fcedba5b37cbaf60f876a88ea4cdb051` |
| DrugCLIP | https://github.com/bowen-gao/DrugCLIP.git | `7a3a3fa33673f8668c811790f2e4681c98af44ef` |
| Uni-Core | https://github.com/dptech-corp/Uni-Core.git | `ace6fae1c8479a9751f2bb1e1d6e4047427bc134` |

Nothing under `third_party/` is edited. Every fix this project needs on top of upstream
(featurizer defaults, a CPU compatibility path, capturing encoder outputs without touching
the training loops, and similar) is applied at run time by a shim in
[`src/core/vendor_compat.py`](src/core/vendor_compat.py) instead of being patched into the
vendored tree. The full inventory of what each shim replaces, and why the replacement is
behaviour-identical to the original in-tree edit, is in
[`docs/VENDOR_COMPAT.md`](docs/VENDOR_COMPAT.md); per-repo pinning details are in
[`docs/vendored/`](docs/vendored/).

This was verified directly: all three repositories were re-cloned fresh at the commits
above and the whole Stage 1 pipeline was re-run against those clones. 17 artefacts came
back bit-identical to the committed results, including a DrugCLIP LIT-PCBA embedding dump
of 362,198,460 elements across `mol_reps`, `pocket_reps`, and `labels`. This is the basis
for the claim that "delete `third_party/`, re-clone at the pins above, reproduce every
number" actually holds. All three pins were still the upstream default-branch HEAD when
this was last checked, so a plain clone reproduces them today; the pins are recorded so
that stays true after upstream moves. Per-repo details are in
[`docs/vendored/`](docs/vendored/).

## Reproducing the results

The key structural point: encoder outputs are dumped to disk once per model and benchmark,
and every hashing condition (no centering, per-type, global, and their median variants) is
then a re-score of those cached arrays, not a fresh model pass. A SPRINT model pass takes
about 10 minutes per condition; a DrugCLIP dump takes 30-45 minutes once, after which all
four scoring conditions together take about a minute.

```bash
# SPRINT on LIT-PCBA: continuous baseline + three hashing conditions (~10 min each), in .venv
python scripts/benchmark_sprint.py --config configs/sprint/pcba_cosine.yaml
for c in none per_type per_type_median global global_median; do
  python scripts/benchmark_sprint.py --config configs/sprint/pcba_hamming.yaml \
    --metric hamming --binarize-center "$c"
done

# DrugCLIP on LIT-PCBA: dump encoder outputs once (~45 min, .venv_drugclip),
# then score four ways (~1 min total, .venv)
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
.venv_drugclip/bin/python scripts/dump_drugclip_embeddings.py data/drugclip \
  --test-task PCBA --user-dir third_party/drugclip/unimol --valid-subset test \
  --task drugclip --loss in_batch_softmax --arch drugclip \
  --path checkpoints/drugclip/checkpoint_best.pt \
  --max-pocket-atoms 256 --batch-size 4 --num-workers 4 --seed 1 \
  --ddp-backend=c10d --log-interval 100 --log-format simple \
  --fp16 --fp16-init-scale 4 --fp16-scale-window 256 \
  --embeddings-dir outputs/metrics/drugclip_pcba/embeddings
python scripts/evaluate_drugclip_hamming.py --config configs/drugclip/pcba_hamming.yaml \
  --target all --metric cosine
for c in none per_type per_type_median global global_median; do
  python scripts/evaluate_drugclip_hamming.py --config configs/drugclip/pcba_hamming.yaml \
    --target all --metric hamming --binarize-center "$c"
done

# Figures + the machine-generated comparison table (writes
# outputs/visualizations/{lit_pcba,dude}_hashing_sweep.png and
# outputs/metrics/stage1_comparison.csv)
python scripts/plot_metrics.py --comparison-csv
```

```bash
# DUD-E (DrugCLIP): dump all 100 targets (~30 min, .venv_drugclip), then score four ways (~1 min, .venv)
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
.venv_drugclip/bin/python scripts/dump_drugclip_embeddings.py data/drugclip \
  --test-task DUDE --user-dir third_party/drugclip/unimol --valid-subset test \
  --task drugclip --loss in_batch_softmax --arch drugclip \
  --path checkpoints/drugclip/checkpoint_best.pt \
  --max-pocket-atoms 256 --batch-size 4 --num-workers 4 --seed 1 \
  --ddp-backend=c10d --log-interval 100 --log-format simple \
  --fp16 --fp16-init-scale 4 --fp16-scale-window 256 \
  --embeddings-dir outputs/metrics/drugclip_dude/embeddings \
  --native-csv outputs/metrics/drugclip_dude_native.csv

python scripts/evaluate_drugclip_hamming.py --config configs/drugclip/dude_hamming.yaml \
  --target all --metric cosine
for c in none per_type per_type_median global global_median; do
  python scripts/evaluate_drugclip_hamming.py --config configs/drugclip/dude_hamming.yaml \
    --target all --metric hamming --binarize-center "$c"
done
```

SPRINT runs in `.venv`. The DrugCLIP dump runs in `.venv_drugclip`. The DrugCLIP scoring
pass runs in `.venv` (it needs `ultrafast.utils` for the metric implementations).

The three remaining experiments in the report are driven as follows. All of them run in
`.venv` except the training-set dump, which needs `.venv_drugclip`.

```bash
# Trained hash head (the DrugHash approximation). Dump the checkpoint's own training
# split once, sweep lambda, then re-score DUD-E through a chosen head.
.venv_drugclip/bin/python scripts/dump_drugclip_train_embeddings.py \
  --train-lmdb data/drugclip/train.lmdb --tag train \
  --out-dir outputs/drughash/train_embeddings
python scripts/train_drugclip_hash_head.py \
  --lambdas 0,0.03,0.1,0.2,0.4,0.6,0.8,1.0,3.0,10.0,100.0 \
  --epochs 30 --batch-size 48 --lr 1e-3 --clip-norm 1.0 --seed 1 \
  --out-dir outputs/drughash/heads
for c in none global; do
  python scripts/evaluate_drugclip_hamming.py --config configs/drugclip/dude_hamming.yaml \
    --target all --metric hamming --binarize-center "$c" \
    --head outputs/drughash/heads/lam3p00.pt \
    --output outputs/metrics/drugclip_dude_heads/head_lam3p00_$c.csv
done

# Training-set leakage. Writes outputs/metrics/drughash_leakage.json.
python scripts/check_drughash_leakage.py

# Code length under a fair tie-break. Writes outputs/metrics/sprint_truncation/.
for b in pcba dude; do
  for k in 1024 512 256 128 64; do
    python scripts/evaluate_drugclip_hamming.py \
      --embeddings-dir outputs/metrics/sprint_$b/embeddings --target all \
      --metric hamming --binarize-center per_type --alpha 85.0 \
      --code-length $k --projection truncate --tie-break random \
      --output outputs/metrics/sprint_truncation/sprint_${b}_truncate_${k}bit.csv
  done
done
```

The `--tie-break` flag matters. The default, `input_order`, keeps Python's stable sort, and
because molecules load actives-first it resolves every tied score in favour of the actives.
Short codes produce many ties, so the truncation numbers are only meaningful under
`random`; that is what the report's truncation table uses.

Run `scripts/selftest.sh` for a fast (about a minute) end-to-end check that both
environments are present, `src/` imports correctly under each, and the unit tests pass.
For just the unit tests, which cover the centering and projection logic in
`src/evaluation/scorer.py`, install the test extra and run pytest directly:

```bash
pip install -e '.[dev]'
python -m pytest tests/
```

`pip install -e .` alone does not pull in pytest.

## Repository layout

| Path | Contents |
|---|---|
| `src/core/` | Model wrappers for SPRINT and DrugCLIP, config loading, embedding I/O, and `vendor_compat.py` (the runtime shims described above) |
| `src/evaluation/` | Similarity scoring, binarization/centering logic (`scorer.py`), and the evaluation driver (`evaluator.py`) |
| `src/hashing/` | The trained hash head (`head.py`) and its loss functions (`losses.py`), used for the DrugHash approximation |
| `scripts/` | Benchmark entry points, embedding dumpers, the trained hash head trainer, figure generation, and `audit_report_numbers.py`, which checks the report's tables against the CSVs in `outputs/metrics/` |
| `tests/` | Unit tests for `src/evaluation/scorer.py` |
| `configs/` | YAML defaults for the SPRINT and DrugCLIP benchmark entry points; see `configs/README.md` for what each one is for |
| `outputs/metrics/` | Committed result CSVs (66 files) plus the leakage JSON, backing every number in the report |
| `outputs/drughash/heads/` | `sweep_summary.json`, the per-lambda training diagnostics behind Figure 2 |
| `outputs/visualizations/` | Exploratory sweep figures produced by `plot_metrics.py`; the two figures the report actually uses live in `docs/report/figures/` |
| `third_party/` | Pristine upstream checkouts of SPRINT, DrugCLIP, and Uni-Core (see "Upstream provenance") |
| `docs/VENDOR_COMPAT.md` | The inventory of runtime shims applied over the vendored code |
| `docs/report/` | The LaTeX source and PDF of the report |
| `data/`, `checkpoints/` | Not in git; see "Setup" |

## Hardware used

A single consumer workstation: an NVIDIA RTX 2070 with 8.1 GB of VRAM, an AMD Ryzen 5 3600
with 12 threads, and 15 GB of system memory. Both SPRINT and DrugCLIP were run for
inference only; the only component actually trained in this project is the hash head in
`src/hashing/`. This is also why the DrugHash approximation freezes the DrugCLIP encoders
rather than fine-tuning them end to end as DrugHash itself does: end-to-end training of
Uni-Mol-based encoders is out of reach on 8 GB of VRAM.

## Acknowledgements

This project builds directly on three upstream projects, used here as pristine, pinned
checkouts under `third_party/`:

- **SPRINT** ([abhinadduri/panspecies-dti](https://github.com/abhinadduri/panspecies-dti)),
  MIT licence.
- **DrugCLIP** ([bowen-gao/DrugCLIP](https://github.com/bowen-gao/DrugCLIP)). Its `LICENSE`
  file splits the project: the source code is Apache License 2.0; the model weights and
  any output results are CC BY-NC 4.0 (non-commercial, attribution required).
- **Uni-Core** ([dptech-corp/Uni-Core](https://github.com/dptech-corp/Uni-Core)), MIT
  licence, a dependency of DrugCLIP's Uni-Mol-based encoders.

See each project's `LICENSE` file under `third_party/` for the full text.
