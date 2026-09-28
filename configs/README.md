Configuration files for stable benchmark entrypoints.

- `sprint/` contains SPRINT benchmark defaults.
  - `pcba_cosine.yaml` — continuous baseline; `pcba_hamming.yaml` — binarized/Hamming
    sweep. `--binarize-center` takes `{none, per_type, per_type_median, global,
    global_median}`: `per_type` centers each embedding matrix by its own per-dimension
    reference, `global` shares one reference over the concatenation, and the `_median`
    variants use the column median instead of the mean. `per_type` / `global` keep their
    historical meaning (mean), so every CSV written before 2026-08-26 still describes the
    same computation.
  - `pcba_hamming_rescore.yaml` / `dude_hamming_rescore.yaml` — score a dump from
    `scripts/dump_sprint_embeddings.py` instead of re-running the model. These carry
    `alpha: 85.0`; SPRINT rows use BEDROC alpha 85.0 and DrugCLIP rows 80.5, and mixing
    them corrupts only the BEDROC column, silently.
- `drugclip/` contains DrugCLIP benchmark defaults.
  - `pcba.yaml` — the native reproduction path (`scripts/benchmark_drugclip.py`, which
    drives DrugCLIP's own `unimol/test.py` as a subprocess).
  - `pcba_hamming.yaml` / `dude_hamming.yaml` — the binarized/Hamming scoring pass
    (`scripts/evaluate_drugclip_hamming.py`) over embeddings dumped by
    `scripts/dump_drugclip_embeddings.py --test-task {PCBA,DUDE}`.

`data_dir` points at `data/drugclip` directly. DrugCLIP resolves its LMDBs relative to the
cwd, so the entry points run from a generated `outputs/drugclip_workdir/` holding a `data`
symlink — `third_party/` is a pristine upstream checkout and carries no symlink of its own.
See [`docs/VENDOR_COMPAT.md`](../docs/VENDOR_COMPAT.md).
