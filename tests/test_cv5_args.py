"""P1b: the cv5 CLI must accept --backbone / --no-nt / --no-struct / --target so that
reproduce/run_cv5.sh (headline) and the README feature-ablation commands run as written.

Exercised without GPU/ViennaRNA on tiny synthetic data.
"""
import json
import os
import tempfile

import numpy as np

CODONS = [a + b + c for a in "ACGT" for b in "ACGT" for c in "ACGT"]


def _synthetic(dir_path, n_tx=30, seed=5):
    rng = np.random.RandomState(seed)
    obj, e2g = {}, {}
    for t in range(n_tx):
        L = int(rng.randint(45, 80))
        seq = "".join(CODONS[i] for i in rng.randint(0, 64, size=L))
        base = np.array([(i % 7) + 1 for i in rng.randint(0, 64, size=L)], np.float32)
        cnt = base + rng.poisson(2.0, size=L).astype(np.float32)
        cnt[rng.rand(L) < 0.3] = 0.0
        k = f"ENST{t:08d}"
        obj[k] = {"cds": {"sequence": seq, "avg_count": cnt}}
        e2g[k] = f"ENSG{t // 2:08d}"  # 2 isoforms per gene -> gene-level folds are meaningful
    npz = os.path.join(dir_path, "syn.npz")
    np.savez(npz, **{k: np.array(v, dtype=object) for k, v in obj.items()})
    e2g_path = os.path.join(dir_path, "e2g.json")
    json.dump(e2g, open(e2g_path, "w"))
    return npz, e2g_path, list(obj.keys())


def test_cli_cv5_parser_has_ablation_flags():
    """The exact flags reproduce/run_cv5.sh and the README ablations pass must parse."""
    import argparse
    import ribopipe.cli as cli

    # Build the same parser main() builds, then parse the headline + ablation commands.
    # We call main() with a parse that stops before dispatch by using --help-free argv
    # through the public parser: easiest is to assert argparse accepts them.
    for argv in (
        ["cv5", "--npz", "x", "--enst2ensg", "y", "--backbone", "cnn", "--loss", "huber"],
        ["cv5", "--npz", "x", "--enst2ensg", "y", "--no-nt", "--no-struct"],
        ["cv5", "--npz", "x", "--enst2ensg", "y", "--backbone", "bilstm",
         "--target", "covmean0_log"],
    ):
        # main() would dispatch and fail on the fake npz; we only want to prove the
        # parser accepts the flags, so intercept at run_cv5.
        import ribopipe.cv5 as cv5mod
        captured = {}

        def fake_run_cv5(*a, **k):
            captured.update(k)
            return {"ok": True}

        orig = cv5mod.run_cv5
        try:
            cv5mod.run_cv5 = fake_run_cv5
            # main() does `from .cv5 import run_cv5`, so patching the module attr suffices;
            # stub np.load so the fake npz path yields a trivial id list without I/O.
            np_load_orig = np.load
            np.load = lambda *a, **k: type("Z", (), {"files": ["ENST00000000"]})()
            try:
                cli.main(argv)
            finally:
                np.load = np_load_orig
        finally:
            cv5mod.run_cv5 = orig
        assert "backbone" in captured and "use_nt" in captured and "target" in captured


def test_run_cv5_honours_target_and_backbone():
    import ribopipe

    with tempfile.TemporaryDirectory() as d:
        npz, e2g, ids = _synthetic(d)
        summary = ribopipe.run_cv5(
            npz, ids, enst2ensg_path=e2g,
            methods=["ribopipe"], backbone="cnn",
            use_nt=True, use_struct=False, target="covmean0_log",
            n_folds=5, epochs=2, patience=3, device="cpu", verbose=False,
        )
        assert "ribopipe" in summary["results"]
        assert summary["n_folds"] == 5
        assert sum(summary["results"]["ribopipe"]["n_per_fold"]) > 0
