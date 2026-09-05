"""P3: frozen gene-level folds reconstruct the exact gene_folds partition.

Data-free: builds a synthetic universe + ENST->ENSG map, runs gene_folds, writes it
in the frozen-file schema, and asserts load_frozen_folds round-trips it. Also checks the
four committed reproduce/folds/*.json parse and are internally consistent.
"""
import glob
import json
import os

from ribopipe.folds import gene_folds, load_frozen_folds

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def test_frozen_roundtrip_matches_gene_folds(tmp_path):
    # synthetic universe: 40 transcripts, 2 isoforms/gene, plus 5 unmapped
    ids = [f"ENST{t:08d}" for t in range(40)]
    e2g = {k: f"ENSG{i // 2:08d}" for i, k in enumerate(ids)}
    ids += [f"NOMAP{t:04d}" for t in range(5)]  # unmapped -> always train

    folds, n_genes, n_unmapped = gene_folds(ids, e2g=e2g, n_folds=5, seed=0)
    assert n_unmapped == 5
    test_folds = [te for _tr, te in folds]
    mapped = {k for te in test_folds for k in te}
    unmapped = [k for k in ids if k not in mapped]

    blob = {"tag": "SYN", "seed": 0, "n_folds": 5, "n_tx": len(ids),
            "n_genes": n_genes, "n_unmapped": n_unmapped,
            "test_folds": test_folds, "unmapped": unmapped}
    p = tmp_path / "cv5_folds_SYN.json"
    json.dump(blob, open(p, "w"))

    loaded, all_ids, meta = load_frozen_folds(str(p))
    assert meta["tag"] == "SYN" and len(loaded) == 5
    assert set(all_ids) == set(ids)
    for (tr0, te0), (tr1, te1) in zip(folds, loaded):
        assert set(te0) == set(te1)
        assert set(tr0) == set(tr1)          # train = other folds' test + unmapped
    # every unmapped id is in every train fold, never in any test fold
    for tr, te in loaded:
        assert all(u in tr for u in unmapped)
        assert not any(u in te for u in unmapped)


def test_committed_fold_files_are_consistent():
    files = glob.glob(os.path.join(REPO, "reproduce", "folds", "cv5_folds_*.json"))
    assert files, "no committed frozen fold files found"
    for f in files:
        folds, all_ids, meta = load_frozen_folds(f)
        assert meta["n_folds"] == len(folds)
        # universe size matches recorded n_tx; test folds are a disjoint cover of mapped ids
        assert len(all_ids) == meta["n_tx"]
        seen = set()
        for _tr, te in folds:
            assert not (seen & set(te)), "test folds must be disjoint"
            seen |= set(te)
