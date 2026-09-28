# Vendoring provenance

- **Upstream:** https://github.com/bowen-gao/DrugCLIP.git
- **Base commit:** `7a3a3fa33673f8668c811790f2e4681c98af44ef` (2026-01-26, "save results")
- **Vendored:** 2026-08-24, in-tree (nested `.git` removed; history not preserved).

## Local modifications on top of the base commit

**None. This tree is a byte-for-byte upstream checkout.**

Every fix this project needs on top of upstream is applied at run time from
`src/core/vendor_compat.py` (and, for DrugCLIP, `src/core/drugclip_capture.py`) instead of
being edited into the tree. The inventory — what each shim replaces, and why the
replacement is behaviour-identical — is [`docs/VENDOR_COMPAT.md`](../VENDOR_COMPAT.md).

The edits that once lived in this tree were removed on 2026-08-25 and are recoverable
from git history; they are **no longer applied**.

To re-sync with a newer upstream: clone the repo above at the desired commit and replace
this directory wholesale. Nothing here needs merging, which is the point.
