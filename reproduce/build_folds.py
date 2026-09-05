#!/usr/bin/env python3
r"""Freeze the paper's gene-level 5-fold evaluation universe + fold assignments.

The paper's headline benchmark does NOT run on every transcript in a dataset NPZ. Its
evaluation universe is, per dataset:

    universe = T_high  INTERSECT  REP

* **T_high** — the sample's high-coverage transcripts (normalised CDS coverage > 0.5).
  This is the frozen gene-level split shipped with the manuscript
  (``.../splits/<TAG>/gene_80_20.json``; its ``train`` + ``test`` together are T_high).
* **REP** — the gene-longest representative set (one transcript per gene),
  ``rep_ids.json`` (committed here as ``reproduce/rep_ids.json``). Intersecting removes
  isoform pseudo-replication so no gene contributes more than one transcript.

The universe is then partitioned into ``--n-folds`` folds by
:func:`ribopipe.folds.gene_folds` with **seed 0** (all isoforms of a gene share a fold;
unmapped transcripts always go to TRAIN). Because a gene-longest universe has ~1
transcript per gene, this is effectively a frozen per-transcript 5-fold split.

This script writes one ``reproduce/folds/cv5_folds_<TAG>.json`` per dataset. Those files
are committed, so the universe and folds are reproducible **without** re-deriving them.
``ribopipe cv5 --folds reproduce/folds/cv5_folds_<TAG>.json`` consumes them directly.

Provenance / determinism: seed=0, ``gene_folds`` round-robin after ``sorted(genes)`` +
``RandomState(0).shuffle``; the ENST->ENSG map is ``reproduce/enst2ensg_grch38.json.gz``.
Regenerating with the same inputs reproduces byte-identical fold files.

Usage (defaults point at the paper's data layout; override for your own):

    python reproduce/build_folds.py \
        --splits-dir /path/to/20260619_riboPipe_final/data/splits \
        --npz-dir    /path/to/Psite_human_processed_npz \
        --rep-ids    reproduce/rep_ids.json \
        --enst2ensg  reproduce/enst2ensg_grch38.json.gz \
        --out-dir    reproduce/folds
"""
import argparse
import gzip
import json
import os
import sys

# TAG -> (split-json subdir, NPZ basename without extension)
DATASETS = {
    "TX9_WT":       ("DMSO_TX9P1",   "TX9_HEK_WT_DMSO_codon_summary_withSeq_rawCount"),
    "GSE233886_WT": ("GSE233886_WT", "GSE233886_HEK293F_HEK293F_WT_DMSO_codon_summary_withSeq_rawCount"),
    "GSE133393_WT": ("GSE133393_WT", "GSE133393_HEK_HEK293_WT_codon_summary_withSeq_rawCount"),
    "PRJNA_iPS":    ("PRJNA_iPS",    "PRJNA976655_human_iPS_409B2_codon_summary_withSeq_rawCount"),
}


def _load_map(path):
    opener = gzip.open if path.endswith(".gz") else open
    with opener(path, "rt") as f:
        return json.load(f)


def build(tag, split_json, rep_ids, e2g, n_folds, seed):
    from ribopipe.folds import gene_folds

    rep = set(rep_ids)
    rep_stem = {x.split(".")[0] for x in rep}
    sp = json.load(open(split_json))
    base = list(sp["train"]) + list(sp["test"])          # T_high (train + test)
    universe = [k for k in base if k in rep or k.split(".")[0] in rep_stem]

    folds, n_genes, n_unmapped = gene_folds(universe, e2g=e2g, n_folds=n_folds, seed=seed)
    test_folds = [te for _tr, te in folds]
    # unmapped ids appear in every train and never a test fold -> record them explicitly
    mapped = {k for te in test_folds for k in te}
    unmapped = [k for k in universe if k not in mapped]
    return {
        "tag": tag,
        "seed": seed,
        "n_folds": n_folds,
        "universe_def": "T_high (gene_80_20 train+test) INTERSECT REP (gene-longest), "
                        "folded by ribopipe.folds.gene_folds(seed=%d)" % seed,
        "n_tx": len(universe),
        "n_genes": n_genes,
        "n_unmapped": n_unmapped,
        "test_folds": test_folds,
        "unmapped": unmapped,
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    here = os.path.dirname(os.path.abspath(__file__))
    ap.add_argument("--splits-dir", required=True,
                    help="dir with <split-subdir>/gene_80_20.json per dataset (T_high source)")
    ap.add_argument("--rep-ids", default=os.path.join(here, "rep_ids.json"))
    ap.add_argument("--enst2ensg", default=os.path.join(here, "enst2ensg_grch38.json.gz"))
    ap.add_argument("--out-dir", default=os.path.join(here, "folds"))
    ap.add_argument("--n-folds", type=int, default=5)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()

    # allow running from a source checkout without installing the package
    sys.path.insert(0, os.path.dirname(here))

    rep_ids = _load_map(a.rep_ids)
    e2g = _load_map(a.enst2ensg)
    os.makedirs(a.out_dir, exist_ok=True)

    for tag, (subdir, _npz) in DATASETS.items():
        split_json = os.path.join(a.splits_dir, subdir, "gene_80_20.json")
        if not os.path.isfile(split_json):
            print(f"[skip] {tag}: no split json at {split_json}", flush=True)
            continue
        blob = build(tag, split_json, rep_ids, e2g, a.n_folds, a.seed)
        out = os.path.join(a.out_dir, f"cv5_folds_{tag}.json")
        json.dump(blob, open(out, "w"), indent=0)
        print(f"[{tag}] {blob['n_tx']} tx, {blob['n_genes']} genes, "
              f"{blob['n_unmapped']} unmapped, folds={[len(f) for f in blob['test_folds']]} -> {out}",
              flush=True)


if __name__ == "__main__":
    main()
