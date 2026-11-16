#!/usr/bin/env bash
# Fetch the official baseline code at the exact commits used for the RiboPipe paper, verify the md5 of every file the
# paper used, and apply our documented adaptations (baselines/patches/*.diff).  The official repositories carry no
# licence file, so their code is NOT redistributed with RiboPipe; this script downloads it from the authors' GitHub.
#
#   bash baselines/fetch_upstream.sh [TARGET_DIR]        (default: baselines/upstream; wrappers read
#                                                         RIBOPIPE_BASELINE_UPSTREAM if set)
# Result:
#   TARGET/RiboMIMO  TARGET/Riboexp  TARGET/ribogl  TARGET/iXnos   sparse checkouts at the pinned commits
#   TARGET/patched/ribomimo/{model.py,utils.py}                      official model.py + 2-line compatibility patch
#   TARGET/patched/riboexp/riboexp_{models,dataio,nntools,run}_py3.py  py2/torch0.3 -> py3/torch2 ports + protocol adapter
# RiboGL is imported verbatim (no patch).  iXnos is fetched for reference only (our PyTorch re-implementation,
# baselines/ixnos.py, is what runs).  Exit status 0 only if every md5 matches.
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
T="${1:-${RIBOPIPE_BASELINE_UPSTREAM:-$HERE/upstream}}"
mkdir -p "$T"
fail=0
# safe.directory=* only for these commands: shared/network file systems may map file ownership to another uid
g() { git -c safe.directory="*" "$@"; }

fetch() {   # name url commit sparse-paths...
  local name=$1 url=$2 commit=$3; shift 3
  local d="$T/$name"
  if [ -d "$d/.git" ] && [ "$(g -C "$d" rev-parse HEAD 2>/dev/null)" = "$commit" ]; then
    echo "[fetch] $name already at $commit"; return 0
  fi
  rm -rf "$d"; g init -q "$d"
  g -C "$d" remote add origin "$url"
  g -C "$d" sparse-checkout init --no-cone >/dev/null
  g -C "$d" sparse-checkout set --no-cone "$@"
  g -C "$d" fetch -q --depth 1 --filter=blob:none origin "$commit"
  g -C "$d" checkout -q FETCH_HEAD
  [ "$(g -C "$d" rev-parse HEAD)" = "$commit" ] || { echo "[fetch] $name: HEAD != $commit"; exit 1; }
  echo "[fetch] $name $url @ $commit"
}

check() {   # file expected-md5
  local got; got=$(md5sum "$1" | cut -d' ' -f1)
  if [ "$got" = "$2" ]; then echo "  md5 OK   $2  ${1#$T/}"; else echo "  md5 FAIL $got != $2  ${1#$T/}"; fail=1; fi
}

apply() {   # source target patch expected-md5
  mkdir -p "$(dirname "$2")"; cp "$1" "$2"
  patch -s --no-backup-if-mismatch "$2" < "$HERE/patches/$3"
  check "$2" "$4"
}

# ---- official repositories (commit = HEAD of the copy used for the paper; file md5s recorded in the paper's
#      baseline_code/*/CHANGES.md and CODE_MD5.txt)
fetch RiboMIMO https://github.com/tiantz17/RiboMIMO 094c693d4df45f4b29f5a3209b31c452a39092a5 /src/ /Readme.md
fetch Riboexp  https://github.com/Liuxg16/Riboexp   8edf66bd4db6a2ec14e9600cee58fb7ee8bff6c5 /code/ /Readme.md
fetch ribogl   https://github.com/vam-sin/ribogl    f3722b60c33427484e9fbf483758e41a0736add5 /src/ /README.md
fetch iXnos    https://github.com/lareaulab/iXnos   1ea2337dd24d1e6fa3859f08573e99f02512810f /iXnos/ /reproduce_scripts/ /README.md

echo "[verify] official files used by the paper"
check "$T/RiboMIMO/src/model.py"                              6210949e839255a555bbd381396cbed5
check "$T/RiboMIMO/src/runRiboMIMO.py"                        516e1699d2ba248425fff70b83eaaedf
check "$T/RiboMIMO/src/utils.py"                              202d9eb40a2d77311b4e8f842fd70adb
check "$T/RiboMIMO/Readme.md"                                 014285507e067c1443fb3ecf9f633a80
check "$T/Riboexp/code/models.py"                             b991e2bb9f98001b527e45a65a6f5a96
check "$T/Riboexp/code/dataio.py"                             5580513b96e0da51b7aa03eb372ec62d
check "$T/Riboexp/code/runRiboexp.py"                         5b989ba715c0578a454890993dabce46
check "$T/Riboexp/code/options.py"                            fe1be2a3e6dea37a438e35d07333ba12
check "$T/Riboexp/code/nntools/nntools.py"                    3ee636dd861ee8788d2ec7bcd5701d9a
check "$T/ribogl/src/models/GNN/gnn_utils.py"                 50003f1cb814696bb67ec917c3ac99fb
check "$T/ribogl/src/preprocessing/graph_processing_utils.py" 6488011191e45b4d6f088fa5ccb3455c
check "$T/ribogl/src/preprocessing/rna_struct_utils.py"       3b79f28f753e64ae2e3bbd51d4af1e9e

echo "[patch] documented adaptations -> $T/patched (md5 must equal the files that produced the paper's runs)"
P="$T/patched"
apply "$T/RiboMIMO/src/model.py"           "$P/ribomimo/model.py"               ribomimo_model.py.diff       375c7a0e8796092cc45d908be2bf4779
cp "$T/RiboMIMO/src/utils.py" "$P/ribomimo/utils.py"; check "$P/ribomimo/utils.py" 202d9eb40a2d77311b4e8f842fd70adb
apply "$T/Riboexp/code/models.py"          "$P/riboexp/riboexp_models_py3.py"   riboexp_models_py3.py.diff   60aea21eaeef05cb1bb56eaf687612e7
apply "$T/Riboexp/code/dataio.py"          "$P/riboexp/riboexp_dataio_py3.py"   riboexp_dataio_py3.py.diff   03bc80e7814960b6d79812e468bff8bd
apply "$T/Riboexp/code/nntools/nntools.py" "$P/riboexp/riboexp_nntools_py3.py"  riboexp_nntools_py3.py.diff  b38776a12efca1dab3d3922aa543ae64
apply "$T/Riboexp/code/runRiboexp.py"      "$P/riboexp/riboexp_run_py3.py"      riboexp_run_py3.py.diff      14526114e8965279ec18c6127409c6c4

if [ $fail -ne 0 ]; then echo "[fetch_upstream] FAILED: md5 mismatch"; exit 1; fi
echo "[fetch_upstream] OK: all files match the versions used for the paper"
