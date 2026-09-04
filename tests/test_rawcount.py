"""Smoke test for the raw-read -> P-site -> codon-count converter (ribopipe.rawcount).

Builds a tiny synthetic CDS FASTA + BAM, runs the converter, and checks the P-site
codon assignment and the output CSV schema. Skipped automatically where pysam is not
installed (it is an optional `[raw]` dependency).
"""
import os
import tempfile

import pytest

pysam = pytest.importorskip("pysam")

from ribopipe import rawcount  # noqa: E402


def _write_fasta(path, seqs):
    with open(path, "w") as f:
        for k, s in seqs.items():
            f.write(f">{k}\n{s}\n")


def _write_bam(path, contigs, reads, read_len=29):
    header = {"HD": {"VN": "1.0"},
              "SQ": [{"SN": n, "LN": L} for n, L in contigs]}
    with pysam.AlignmentFile(path, "wb", header=header) as out:
        for i, (rid, start) in enumerate(reads):
            a = pysam.AlignedSegment()
            a.query_name = f"r{i}"
            a.reference_id = rid
            a.reference_start = start
            a.query_sequence = "A" * read_len
            a.cigar = [(0, read_len)]
            a.flag = 0
            a.mapping_quality = 60
            a.query_qualities = pysam.qualitystring_to_array("I" * read_len)
            out.write(a)
    pysam.index(path)


def test_rawcount_psite_and_schema():
    with tempfile.TemporaryDirectory() as d:
        fa = os.path.join(d, "cds.fa")
        _write_fasta(fa, {"tx1": "ACG" * 30, "tx2": "GCA" * 20})  # 30 and 20 codons

        bam = os.path.join(d, "s1.bam")
        # offset 12: ref_start 0 -> codon 4; ref_start 30 -> codon 14; tx2 start 6 -> codon 6
        reads = [(0, 0)] * 5 + [(0, 30)] * 3 + [(1, 6)] * 2
        _write_bam(bam, [("tx1", 90), ("tx2", 60)], reads)

        out_csv = os.path.join(d, "counts.csv")
        col = "HEK293T|WT|DMSO|rep1|mono|x"
        rawcount.bams_to_codon_csv([bam], [col], fa, out_csv, default_offset=12)

        import pandas as pd
        df = pd.read_csv(out_csv)

        # required schema columns present
        for c in ("transcript", "region", "start", "end", "from_cds_start", "from_cds_stop", col):
            assert c in df.columns

        def cnt(tx, codon):
            return int(df[(df.transcript == tx) & (df.from_cds_start == codon)][col].iloc[0])

        assert cnt("tx1", 4) == 5
        assert cnt("tx1", 14) == 3
        assert cnt("tx2", 6) == 2
        assert (df.transcript == "tx1").sum() == 30       # one row per codon
        assert df[col].sum() == 10                        # all reads assigned
        assert (df["region"] == "cds").all()

        # the output must validate against the preprocess schema
        from ribopipe.preprocess.schema import validate_csv_columns
        sample_cols = validate_csv_columns(df.columns)
        assert sample_cols == [col]
