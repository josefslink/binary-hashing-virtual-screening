#!/usr/bin/env bash
# end-to-end check that the repo still works. run from the repo root:
#   scripts/selftest.sh            fast checks only, ~1 min
#   scripts/selftest.sh --full     adds a GPU re-dump of two DUD-E targets, ~5 min
set -uo pipefail
cd "$(git rev-parse --show-toplevel)"
# unicore prints "fused_* is not installed corrected" to STDOUT on every import - harmless,
# and not suppressible with 2>/dev/null, which is why the checks below send both streams
# to /dev/null rather than just stderr.
FULL=0; [ "${1:-}" = "--full" ] && FULL=1
pass=0; fail=0
ok(){ printf '  PASS  %s\n' "$1"; pass=$((pass+1)); }
no(){ printf '  FAIL  %s\n' "$1"; fail=$((fail+1)); }
hdr(){ printf '\n%s\n' "$1"; }

echo "=================================================="
echo " repo self-test   $(date -u '+%Y-%m-%d %H:%M UTC')"
echo " commit $(git rev-parse --short HEAD)  branch $(git branch --show-current)"
echo "=================================================="

hdr "1. environments"
for v in .venv .venv_drugclip; do
  [ -x "$v/bin/python" ] && ok "$v present ($($v/bin/python -V 2>&1))" || no "$v missing"
done

hdr "2. imports"
if .venv/bin/python -c "
import src.core.wrapper, src.core.config, src.core.embeddings_io, src.core.vendor_compat
import src.evaluation.scorer, src.evaluation.evaluator, src.hashing.head, src.hashing.losses" 2>/dev/null
then ok "src imports under .venv"; else no "src import failed under .venv"; fi
# unimol is not a top-level module: unicore loads it at runtime from --user-dir
# third_party/drugclip/unimol, which is how every dump command invokes it.
if .venv_drugclip/bin/python -c "
import argparse, src.core.drugclip_capture, unicore
from unicore import options, utils
ns = argparse.Namespace(user_dir='third_party/drugclip/unimol')
utils.import_user_module(ns)
from unimol.tasks.drugclip import cal_metrics" >/dev/null 2>&1
then ok "drugclip stack loads via --user-dir under .venv_drugclip"
else no "drugclip stack failed to load via --user-dir"; fi

hdr "3. unit tests"
if out=$(.venv/bin/python -m pytest tests/ -q 2>&1); then ok "$(echo "$out" | tail -1)"
else no "pytest failed"; echo "$out" | tail -12 | sed 's/^/        /'; fi

hdr "4. every entry point starts"
n=0; broken=0
for f in scripts/*.py; do
  case "$(basename "$f")" in dump_drugclip*|benchmark_drugclip*|write_drugclip*) V=.venv_drugclip;; *) V=.venv;; esac
  n=$((n+1))
  timeout 90 "$V/bin/python" "$f" --help >/dev/null 2>&1 || { echo "        broken: $f"; broken=$((broken+1)); }
done
[ $broken -eq 0 ] && ok "all $n scripts import and build their parser" || no "$broken of $n scripts do not start"

hdr "5. published numbers still reproduce from their CSVs"
if out=$(.venv/bin/python scripts/audit_report_numbers.py 2>&1); then ok "$(echo "$out" | tail -1)"
else no "audit failed"; echo "$out" | tail -12 | sed 's/^/        /'; fi

hdr "6. re-scoring cached embeddings is bit-identical"
E=outputs/metrics/drugclip_dude/embeddings
if [ -d "$E" ]; then
  t=$(mktemp -d)
  if .venv/bin/python scripts/evaluate_drugclip_hamming.py --embeddings-dir "$E" \
       --target all --metric hamming --binarize-center global --alpha 80.5 \
       --output "$t/c.csv" >/dev/null 2>&1 \
     && out=$(.venv/bin/python scripts/compare_result_csvs.py "$t/c.csv" \
              outputs/metrics/drugclip_dude_hamming_global_20260825_150228.csv 2>&1)
  then ok "$(echo "$out" | head -1)"; else no "re-score differs"; echo "$out" | sed 's/^/        /'; fi
  rm -rf "$t"
else printf '  SKIP  %s\n' "no cached DUD-E embeddings (run the dump first)"; fi

hdr "7. vendored code is untouched"
if [ -z "$(git status --porcelain third_party/)" ] && [ -z "$(git diff HEAD --stat -- third_party/)" ]
then ok "third_party/ matches the committed upstream checkout"; else no "third_party/ was modified"; fi

hdr "8. figures and the comparison table regenerate"
b=$(mktemp -d); cp outputs/metrics/stage1_comparison.csv "$b/" 2>/dev/null
if .venv/bin/python scripts/plot_metrics.py --comparison-csv >/dev/null 2>&1; then
  if diff -q "$b/stage1_comparison.csv" outputs/metrics/stage1_comparison.csv >/dev/null 2>&1
  then ok "4 figures + stage1_comparison.csv regenerate identically"
  else no "stage1_comparison.csv changed on regeneration"; fi
else no "plot_metrics.py failed"; fi
rm -rf "$b"

hdr "9. report builds"
if [ -x docs/report/build.sh ]; then
  if docs/report/build.sh >/dev/null 2>&1 && [ -f docs/report/build/main.pdf ]
  then ok "main.pdf, $(pdfinfo docs/report/build/main.pdf 2>/dev/null | awk '/^Pages/{print $2}') pages"
  else no "report build failed"; fi
else printf '  SKIP  %s\n' "docs/report/build.sh not present"; fi

hdr "10. large inputs present (not in git; needed for a full re-run)"
for p in data/lit_pcba data/DUDe checkpoints; do
  [ -e "$p" ] && printf '  PASS  %-22s %s\n' "$p" "$(du -sh "$p" 2>/dev/null | cut -f1)" && pass=$((pass+1)) \
              || printf '  SKIP  %-22s absent - download required for a full re-run\n' "$p"
done

if [ $FULL -eq 1 ]; then
hdr "11. GPU re-dump reproduces committed embeddings (--full)"
  t=$(mktemp -d)
  if .venv_drugclip/bin/python scripts/dump_drugclip_embeddings.py --test-task DUDE \
       --targets aa2ar,abl1 --embeddings-dir "$t" >/dev/null 2>&1; then
    d=0
    for tg in aa2ar abl1; do for a in mol_reps pocket_reps labels; do
      cmp -s "$t/$tg/$a.npy" "$E/$tg/$a.npy" || d=$((d+1))
    done; done
    [ $d -eq 0 ] && ok "6 arrays bit-identical to the committed dump" || no "$d of 6 arrays differ"
  else no "dump did not run (GPU required)"; fi
  rm -rf "$t"
fi

echo
echo "=================================================="
printf " %d passed, %d failed\n" "$pass" "$fail"
[ $fail -eq 0 ] && echo " RESULT: OK" || echo " RESULT: PROBLEMS - see FAIL lines above"
echo "=================================================="
exit $([ $fail -eq 0 ] && echo 0 || echo 1)
