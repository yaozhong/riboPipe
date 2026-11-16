"""Integrity of the released data in data/ (runs in CI; no MFE assets needed)."""
import os, json
import numpy as np
import pytest
from ribopipe import data as D

ROOT = D.data_dir()
MAN = os.path.join(ROOT, "RELEASE_MANIFEST.json")
pytestmark = pytest.mark.skipif(not os.path.exists(MAN), reason="release data not present")


def test_manifest_md5():
    man = json.load(open(MAN))
    assert man["source_manifest_md5"] == D.SOURCE_MANIFEST_MD5
    for rel, h in man["files"].items():
        assert D.md5_file(os.path.join(ROOT, rel)) == h, rel


@pytest.mark.parametrize("tag", [t for t in D.HUMAN + D.NONHUMAN if os.path.isdir(os.path.join(ROOT, t))])
def test_dataset(tag):
    parts = ["partI", "partII"] if tag in D.HUMAN else ["partI"]
    for part in parts:
        ds = D.load_dataset(tag, part, load_mfe=False)
        assert sorted(sum((ds.test_ids(f) for f in ds.folds), [])) == sorted(ds.ids)
        for f in ds.folds:
            tr, va, te = set(ds.train_ids(f)), set(ds.val_ids(f)), set(ds.test_ids(f))
            assert not (tr & va) and not (tr & te) and not (va & te)
            assert not ({ds.unit[t] for t in te} & {ds.unit[t] for t in tr | va})      # units never cross the test split
        for t in ds.ids[:200]:
            c = ds.counts[t]; assert len(ds.seq[t]) == 3 * len(c) and c.sum() > 0
            if tag in D.HUMAN: assert np.array_equal(ds.reps(t).mean(0), c)
            assert ds.mfe_valid(t) == len(ds.seq[t]) - 29
        if part == "partI":
            cov = [ds.counts[t].sum() / len(ds.seq[t]) for t in ds.ids]
            if tag in D.HUMAN: assert min(cov) > 0.5
