"""Regression test for loading a ``covmean0_log`` CNN checkpoint (the shipped-checkpoint
path in reproduce/predict_with_checkpoint.py).

Guards two bugs that made the released checkpoints unloadable:

* ``predict_from_checkpoint`` raised ``StopIteration`` locating the CNN first conv
  (``dict.get`` default was eagerly evaluated and no key ends in ``.c1.weight``);
* ``RiboDataset`` rejected ``target='covmean0_log'`` with ``ValueError``.

Runs on CPU with tiny synthetic data -- no GPU, no ViennaRNA, no real datasets.
"""
import os
import tempfile

import numpy as np

CODONS = [a + b + c for a in "ACGT" for b in "ACGT" for c in "ACGT"]


def _make_synthetic(dir_path, n_tx=24, seed=1):
    rng = np.random.RandomState(seed)
    obj = {}
    for t in range(n_tx):
        L = int(rng.randint(40, 80))
        seq = "".join(CODONS[i] for i in rng.randint(0, 64, size=L))
        base = np.array([(i % 7) + 1 for i in rng.randint(0, 64, size=L)], np.float32)
        cnt = base + rng.poisson(2.0, size=L).astype(np.float32)
        cnt[rng.rand(L) < 0.3] = 0.0  # zero-coverage codons
        obj[f"ENST{t:08d}"] = {"cds": {"sequence": seq, "avg_count": cnt}}
    path = os.path.join(dir_path, "syn.npz")
    np.savez(path, **{k: np.array(v, dtype=object) for k, v in obj.items()})
    return path, list(obj.keys())


def test_cnn_covmean0_log_checkpoint_roundtrip():
    from ribopipe.train import train_on_ids, save_checkpoint
    from ribopipe.predict import predict_from_checkpoint

    with tempfile.TemporaryDirectory() as d:
        npz, ids = _make_synthetic(d)
        tr, val, te = ids[:14], ids[14:19], ids[19:]

        # headline backbone (CNN) trained on the paper target
        model = train_on_ids(
            npz, tr, val_ids=val, backbone="cnn",
            use_nt=True, use_struct=False, target="covmean0_log",
            epochs=2, patience=5, batch_size=8, loss_name="huber",
            device="cpu", verbose=False,
        )
        cfg = getattr(model, "_ribopipe_config", {})
        assert cfg.get("target") == "covmean0_log"

        ckpt = os.path.join(d, "cnn_covmean.pt")
        save_checkpoint(model, ckpt)

        # This is exactly what reproduce/predict_with_checkpoint.py does; it used to
        # raise StopIteration, then ValueError. It must now return predictions.
        preds, scores = predict_from_checkpoint(ckpt, npz, te, device="cpu")
        assert len(preds) > 0
        for p in preds.values():
            assert p.ndim == 1 and len(p) > 0 and np.isfinite(p).all()
