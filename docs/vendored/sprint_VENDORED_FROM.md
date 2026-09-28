# Vendoring provenance

- **Upstream:** https://github.com/abhinadduri/panspecies-dti.git (SPRINT / ultrafast)
- **Base commit:** `ce5bc698fcedba5b37cbaf60f876a88ea4cdb051` (2026-01-21, "Merge pull request #50 from katztyler01/benchmark_mpp")
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
