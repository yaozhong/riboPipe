#!/usr/bin/env python3
"""Write an example input for `ribopipe prepare` (CDS FASTA + per-codon counts TSV + transcript-to-gene map) from a
released dataset, to try the own-data workflow end to end.

    python examples/make_example_input.py --dataset GSE133393_WT --n 400 --out example_input
    ribopipe prepare --fasta example_input/cds.fa --counts example_input/counts.tsv \
                     --genes example_input/tx2gene.tsv --out mydata --tag Example
"""
import os, argparse
import numpy as np
from ribopipe import data as D

ap = argparse.ArgumentParser()
ap.add_argument("--dataset", default="GSE133393_WT"); ap.add_argument("--n", type=int, default=400)
ap.add_argument("--out", default="example_input"); ap.add_argument("--seed", type=int, default=0)
a = ap.parse_args()
ds = D.load_dataset(a.dataset, "partII" if a.dataset in D.HUMAN else "partI", load_mfe=False)
ids = sorted(np.random.RandomState(a.seed).choice(ds.ids, min(a.n, len(ds.ids)), replace=False))
os.makedirs(a.out, exist_ok=True)
with open(os.path.join(a.out, "cds.fa"), "w") as f:
    for t in ids: f.write(f">{t}\n{ds.seq[t]}\n")
reps = ds.reps(ids[0]) is not None
with open(os.path.join(a.out, "counts.tsv"), "w") as f:
    f.write("transcript\tcodon\t" + ("rep1\trep2" if reps else "count") + "\n")
    for t in ids:
        x = ds.reps(t) if reps else ds.counts[t][None]
        for i in np.flatnonzero(x.sum(0) > 0):
            f.write(f"{t}\t{i}\t" + "\t".join(str(int(v)) if reps else repr(float(v)) for v in x[:, i]) + "\n")
with open(os.path.join(a.out, "tx2gene.tsv"), "w") as f:
    for t in ids: f.write(f"{t}\t{ds.unit[t]}\n")
print(f"wrote {len(ids)} transcripts of {a.dataset} to {a.out}/ (cds.fa, counts.tsv, tx2gene.tsv)")
