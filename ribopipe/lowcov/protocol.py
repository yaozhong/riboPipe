"""Frozen protocol of the crossover analysis (PARTII_CROSSOVER_v2.json, shipped next to this module).

The bin edges are also embedded here so that the analysis works when the JSON is not installed; the test suite
checks that both agree and that the JSON is the frozen file (md5 below).
"""
import hashlib
import json
import os

PROTOCOL_JSON = os.path.join(os.path.dirname(os.path.abspath(__file__)), "PARTII_CROSSOVER_v2.json")
PROTOCOL_MD5 = "39ae5628b8d123b104450f1ca2ee7f0b"
EDGES_V2 = [6.67e-05, 0.0001012, 0.0001537, 0.0002333, 0.0003542, 0.0005378, 0.0008165, 0.0012396, 0.0018821, 0.0028574,
            0.0043383, 0.0065866, 0.01, 0.0151824, 0.0230506, 0.0349964, 0.0531329, 0.0806686, 0.1224745, 0.1859459,
            0.2823108, 0.4286161, 0.6507428, 0.9879849, 1.5]
MIN_N = 30          # per-role N_scored for a valid bin
B_BOOT = 2000       # similarity-unit bootstrap resamples
FAIL_FRAC = 0.10    # bootstrap localisation failure fraction above which D* is NOT_LOCATABLE


def edges(path=None):
    """Bin edges in reads/codon. Without a path: the frozen v2 edges (the shipped JSON is checked if present)."""
    if path:
        return list(json.load(open(path))["edges_reads_per_codon"])
    if os.path.exists(PROTOCOL_JSON):
        m5 = hashlib.md5(open(PROTOCOL_JSON, "rb").read()).hexdigest()
        assert m5 == PROTOCOL_MD5, f"{PROTOCOL_JSON} was modified (md5 {m5}); pass a protocol path explicitly"
        assert json.load(open(PROTOCOL_JSON))["edges_reads_per_codon"] == EDGES_V2
    return list(EDGES_V2)
