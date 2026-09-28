# Vendored trees are never edited

`third_party/{sprint,drugclip,uni-core}` are **byte-for-byte upstream checkouts** at the
base commits recorded in [`docs/vendored/*_VENDORED_FROM.md`](vendored/). Nothing in this
project modifies them. Everything this project needs on top of upstream is applied at run
time, from our own code, and is inventoried below.

Written 2026-08-25, when the last in-tree edits were removed.

## Why

Edits inside a vendored tree are invisible in three ways that matter here: they do not
show up in a diff of *this* project's code, they silently become merge conflicts the
moment upstream is re-synced, and — as the retired DrugCLIP patch set showed —
the file that documents them drifts out of date without anything failing. A shim in
`src/` is ordinary reviewable code that lives next to its own tests and its own reasons.

## Where the shims live

| File | Covers |
|---|---|
| [`src/core/vendor_compat.py`](../src/core/vendor_compat.py) | SPRINT featurizer + checkpoint defaults; DrugCLIP CPU path and working directory |
| [`src/core/drugclip_capture.py`](../src/core/drugclip_capture.py) | Capturing DrugCLIP encoder outputs from outside the vendored loops |
| [`scripts/dump_drugclip_embeddings.py`](../scripts/dump_drugclip_embeddings.py) | The entry point that assembles the DrugCLIP capture for LIT-PCBA and DUD-E |

## Inventory

### SPRINT

| Was | Now | Why it is equivalent |
|---|---|---|
| `ultrafast/featurizers.py` — `Featurizer.__call__` hashed the raw sequence; patched to hash `prepare_string(seq)` | `install_sprint_compat()` wraps `__call__` and prepares the sequence before delegating, on the LMDB branch only | `prepare_string` is idempotent, and the wrapper changes only which string is hashed — the same change the edit made |
| `ultrafast/featurizers.py` — `process_lmdb` keyed the LMDB by the raw sequence | the same wrapper prepares `seq_list` up front, so upstream's `{md5(seq): seq}` becomes `{md5(prepared): prepared}` | `SaProtFeaturizer._transform` calls `prepare_string` on everything it receives anyway; the `moltype == "target"` digit filter is unaffected because `prepare_string` only interleaves `#` and truncates |
| `ultrafast/model.py` — `sigmoid_scalar` read with an `in self.args` fallback to 5 | `apply_sprint_checkpoint_defaults(model)`, called by `SprintWrapper.from_checkpoint`, sets the default on the loaded args and prints what it had to supply | the model reads the same value from the same place; the default (5) is upstream's own `train.py` CLI default |
| `ultrafast/featurizers.py` — an added `OneHotFeaturizer` class | **deleted, not relocated** | it was never referenced anywhere in the repo |
| `ultrafast/featurizers.py` — `process_lmdb`'s batch loop bound changed from `len(sorted_ids)` to `len(seq_list)` | **deleted, not relocated** | it was a defect, not a fix: `sorted_ids` is keyed by hash, so it is *shorter* than `seq_list` whenever two sequences collide after truncation, and the extra iterations index past its end. Upstream's bound is correct |

The featurizer shim is installed at import time by `src/core/wrapper.py`, which every
SPRINT entry point in the repo imports, so it is in force before any featurizer opens a
cache. **The existing LMDB caches stay valid** — the shim reproduces the key scheme they
were written under, so no re-featurization is required. Verified by re-running the
LIT-PCBA cosine baseline against its committed CSV (see below).

### DrugCLIP

| Was | Now | Why it is equivalent |
|---|---|---|
| `unimol/tasks/drugclip.py` — a `save_target_reps()` method persisting `mol_reps` / `pocket_reps` / `labels`, called from `test_pcba_target` | `ProjectionCapture` forward hooks on `model.mol_project` / `model.pocket_project`, plus `LabelCapture` wrapping the mol dataset's collater | the hook applies the same L2 normalisation the loop applies on the next line, and the collater sees the same batches in the same order; checked per target by re-scoring the captured arrays through DrugCLIP's own `cal_metrics` |
| `unimol/tasks/drugclip.py` — `@torch.no_grad()` on `test_pcba_target` and `test_dude_target` | `with torch.no_grad():` around the call in `dump_drugclip_embeddings.py` | a context manager and a decorator are the same mechanism |
| `unimol/tasks/drugclip.py` — `torch.cuda.empty_cache()` inserted between the mol and pocket loops | `empty_cache_before(model.pocket_model)`, a one-shot forward pre-hook | fires at the same boundary: after the last molecule batch, before the pocket encoder allocates |
| `unimol/tasks/drugclip.py` — `move_sample_to_model_device()` replacing `unicore.utils.move_to_cuda` | `install_drugclip_cpu_compat()` redirects that one function, and is called only for CPU runs | on GPU upstream is already right; `torch.cuda.set_device()` steers `move_to_cuda` at a non-zero device |
| `unimol/tasks/drugclip.py` — a `--pcba-targets` flag to evaluate a subset | `--targets` on `dump_drugclip_embeddings.py`, which iterates targets itself | the script drives `test_pcba_target` per target and never calls the `test_pcba` loop the flag filtered |
| `unimol/tasks/drugclip.py` — `pocket_names` accumulated across batches instead of overwritten | **not needed** | it existed to make `save_target_reps` correct; nothing reads pocket names now |
| `unimol/models/drugclip.py`, `unimol/retrieval.py`, `unimol/test.py` — device/fp16 guards for CPU runs | **dropped** | `dump_drugclip_embeddings.py` loads the checkpoint itself and never goes through `unimol/test.py`'s fp16 branch. See the limitation below |
| `third_party/drugclip/data -> ../../data/drugclip` symlink | `drugclip_workdir()` chdirs into a disposable directory containing exactly one entry, `data -> <data root>` | DrugCLIP's loops hardcode cwd-relative paths (`./data/lit_pcba/...`); they need *a* cwd with a `data` entry, not that specific one |
| `third_party/drugclip/data/dict_mol.txt`, `dict_pkt.txt` — deleted locally | restored from `data/drugclip/`, tracked with `git add -f` | they are upstream repo content |

**Known limitation, inherited from upstream.** `BindingAffinityModel.__init__` creates
`logit_scale` with an explicit `device="cuda"`, so *constructing* the DrugCLIP model needs
a CUDA device to exist even when inference runs on CPU. On a genuinely CUDA-less host
upstream DrugCLIP cannot be built at all. This project used to paper over that in-tree; it
no longer does. No published result here used the CPU path.

## The rule going forward

1. Never edit anything under `third_party/`. If a fix is needed, it goes in `src/` and
   gets a row in the inventory above.
2. If a fix genuinely cannot be expressed from outside, say so explicitly in
   `changes.md` and add the row anyway, describing what was done instead.
3. After any re-vendoring, check
   `git ls-files --others --ignored --exclude-standard third_party/` — a bare `data/` or
   `checkpoints/` rule in `.gitignore` matches at any depth and will silently drop source
   packages with those names (this happened; see `changes.md` 2026-08-25).
