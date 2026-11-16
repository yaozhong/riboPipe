#!/usr/bin/env bash
# Reproduce the four main figures of the RiboPipe paper from paper_results/.
# Usage: bash reproduce/figures/make_all.sh [OUTDIR] [PYTHON]
#   OUTDIR  default: reproduce/figures/out
#   PYTHON  default: python3 (needs numpy and matplotlib; the published figures were made with matplotlib 3.10.9, numpy 2.2.6, Python 3.10)
# Each script checks the md5 of its inputs against paper_results/MANIFEST.json and writes <fig>.pdf, <fig>.png (300 dpi)
# and <fig>_numbers_drawn.json into OUTDIR.
set -u
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
OUT="${1:-$HERE/out}"
PY="${2:-python3}"
mkdir -p "$OUT"
fail=0
for f in fig1 fig2 fig3 fig4; do
  "$PY" "$HERE/$f.py" "$OUT"
  rc=$?
  if [ $rc -ne 0 ] || [ ! -s "$OUT/$f.pdf" ] || [ ! -s "$OUT/${f}_numbers_drawn.json" ]; then
    echo "FAILED: $f (rc=$rc)" >&2; fail=1
  else
    echo "ok: $f -> $OUT/$f.pdf"
  fi
done
exit $fail
