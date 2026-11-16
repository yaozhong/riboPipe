#!/usr/bin/env python3
"""Download the folding-energy (MFE) tracks of the released datasets (GitHub release assets) into data/mfe/ and
check their MD5 against data/RELEASE_MANIFEST.json.

    python tools/download_mfe.py                      # all released datasets
    python tools/download_mfe.py GSE133393_WT         # one dataset
The tracks can also be recomputed from sequence with ViennaRNA (ribopipe.prepare.mfe_track); the downloaded files
are the exact values used in the paper (ViennaRNA 2.7.0)."""
import os, sys, json, hashlib, urllib.request

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BASE = "https://github.com/yaozhong/riboPipe/releases/download/v2.0.0/"


def md5(p):
    h = hashlib.md5()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 22), b""): h.update(b)
    return h.hexdigest()


def main(tags):
    man = json.load(open(os.path.join(REPO, "data", "RELEASE_MANIFEST.json")))["assets"]
    out = os.environ.get("RIBOPIPE_MFE", os.path.join(REPO, "data", "mfe")); os.makedirs(out, exist_ok=True)
    files = [f for f in sorted(man) if not tags or f.replace("_mfe.npz", "") in tags]
    if not files: raise SystemExit(f"no MFE asset for {tags}; available: {sorted(man)}")
    for f in files:
        p = os.path.join(out, f)
        if os.path.exists(p) and md5(p) == man[f]:
            print(f"[ok] {f} already present"); continue
        print(f"[get] {BASE + f}", flush=True)
        urllib.request.urlretrieve(BASE + f, p + ".part"); os.replace(p + ".part", p)
        if md5(p) != man[f]: raise SystemExit(f"MD5 mismatch for {f}: {md5(p)} != {man[f]}")
        print(f"[ok] {f} md5 {man[f]}")


if __name__ == "__main__":
    main(sys.argv[1:])
