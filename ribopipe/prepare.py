"""Prepare your own Ribo-seq data in the release format, with the paper's rules.

Input
  --fasta    A-site-aligned CDS sequences (one record per transcript; length a multiple of 3, start to stop codon)
  --counts   per-codon A-site counts, tab-separated with a header: transcript, codon (0-based), then one column per
             replicate (e.g. rep1 rep2). Codons absent from the file count 0.
  --genes    optional tab-separated transcript -> gene map; transcripts of one gene always share a similarity unit
Output (directory <out>/<tag>/ plus <out>/MFE_NORM_FOLDWISE.json and <out>/mfe/<tag>_mfe.npz), read by
``ribopipe.data.load_dataset(tag, data=<out>)``.

Rules (as in the paper, PaperData_v1.0)
  admission    sequence of ACGT only, length = 3L, at least one read
  high / low   Part I universe (training and benchmark) = coverage sum(c) / (3L) > 0.5 with c the replicate mean;
               Part II universe = all admitted transcripts, strata high and low
  units        transcripts of different genes are linked when their CDS 31-mer containment is >= 0.2 (shared k-mers
               over the smaller k-mer set; k-mers occurring in more than 200 transcripts ignored); units are the
               connected components over genes
  folds        unit ids sorted, RandomState(0) permutation, round-robin into 5 folds; validation = for each fold the
               sorted training units shuffled with RandomState(42), first 10 %
  MFE          ViennaRNA RNA.fold MFE of the 30-nt window starting at every nucleotide (last 29 positions 0);
               normalisation per fold = mean and SD over the valid windows of the fold's training transcripts
"""
import os, json, argparse, collections, time
import numpy as np

K, MAXOCC, TH, SEED, VSEED, VFRAC = 31, 200, 0.2, 0, 42, 0.1
STRUCT_WIN = 30
COV_HIGH = 0.5


def mfe_track(seq):
    """MFE of the 30-nt window starting at each nucleotide; positions >= len(seq) - 29 are 0."""
    import RNA   # ViennaRNA; required (no silent fallback to zeros)
    n = len(seq); out = np.zeros(n, dtype=np.float32)
    for i in range(n - STRUCT_WIN + 1):
        out[i] = RNA.fold(seq[i:i + STRUCT_WIN])[1]
    return out


def _mfe_job(args):
    t, s = args; return t, mfe_track(s)


def kmer_pairs(seqs, group):
    """seqs: list[str]; group: list[int] (pairs within a group are not counted). Returns {(i, j): containment}, i < j."""
    enc = np.full(256, 255, np.uint8)
    for i, c in enumerate("ACGT"): enc[ord(c)] = i
    pw = (np.uint64(4) ** np.arange(K - 1, -1, -1, dtype=np.uint64))
    kms = []; txs = []; nk = np.zeros(len(seqs), np.int64)
    for ti, s in enumerate(seqs):
        a = enc[np.frombuffer(s.upper().encode(), np.uint8)]
        if len(a) < K: continue
        bad = np.lib.stride_tricks.sliding_window_view(a == 255, K).any(1)
        h = (np.lib.stride_tricks.sliding_window_view(a.astype(np.uint64), K) * pw).sum(1)
        h = np.unique(h[~bad]); nk[ti] = len(h); kms.append(h); txs.append(np.full(len(h), ti, np.int32))
    if not kms: return {}
    km = np.concatenate(kms); tx = np.concatenate(txs); o = np.argsort(km, kind="stable"); km = km[o]; tx = tx[o]
    b = np.flatnonzero(np.diff(km)) + 1; st = np.r_[0, b]; en = np.r_[b, len(km)]; sz = en - st
    g = np.asarray(group); pc = collections.Counter()
    for s, e in zip(st[(sz > 1) & (sz <= MAXOCC)], en[(sz > 1) & (sz <= MAXOCC)]):
        t = tx[s:e]
        if len(set(g[t])) < 2: continue
        for i in range(len(t)):
            for j in range(i + 1, len(t)):
                if g[t[i]] != g[t[j]]: pc[(min(t[i], t[j]), max(t[i], t[j]))] += 1
    return {k: n / min(nk[k[0]], nk[k[1]]) for k, n in pc.items()}


def components(nodes, edges):
    par = {x: x for x in nodes}
    def f(x):
        while par[x] != x: par[x] = par[par[x]]; x = par[x]
        return x
    for a, b in edges: par[f(a)] = f(b)
    comp = collections.defaultdict(list)
    for x in nodes: comp[f(x)].append(x)
    return {x: min(v) for v in comp.values() for x in v}      # unit id = smallest member id (deterministic)


def fold_and_val(units):
    us = sorted(set(units)); order = np.random.RandomState(SEED).permutation(len(us))
    uf = {us[i]: int(j % 5) for j, i in enumerate(order)}
    val = {}
    for f in range(5):
        tr = sorted(u for u in us if uf[u] != f); r = np.random.RandomState(VSEED); r.shuffle(tr)
        val[f] = set(tr[:int(VFRAC * len(tr))])
    return uf, val


def read_fasta(p):
    out, k, buf = {}, None, []
    for line in open(p):
        line = line.strip()
        if not line: continue
        if line.startswith(">"):
            if k is not None: out[k] = "".join(buf).upper()
            k = line[1:].split()[0]; buf = []
        else: buf.append(line)
    if k is not None: out[k] = "".join(buf).upper()
    return out


def read_counts(p, seqs):
    """Long-format TSV: transcript, codon, rep columns -> {tx: (R, L) float array}."""
    with open(p) as f:
        head = f.readline().rstrip("\n").split("\t"); reps = head[2:]
        assert head[0] == "transcript" and head[1] == "codon" and reps, "header must be: transcript, codon, rep1[, rep2 ...]"
        out = {}
        for line in f:
            x = line.rstrip("\n").split("\t"); t = x[0]
            if t not in seqs: continue
            if t not in out: out[t] = np.zeros((len(reps), len(seqs[t]) // 3), np.float64)
            out[t][:, int(x[1])] = [float(v) for v in x[2:]]
    return out, reps


def prepare(fasta, counts, out, tag, genes=None, workers=8, log=print):
    from multiprocessing import Pool
    seqs = read_fasta(fasta); cnt, reps = read_counts(counts, seqs)
    gene = {}
    if genes:
        for line in open(genes):
            x = line.rstrip("\n").split("\t")
            if len(x) >= 2 and x[0] in seqs: gene[x[0]] = x[1]
    ids = sorted(t for t in seqs if t in cnt and len(seqs[t]) % 3 == 0 and set(seqs[t]) <= set("ACGT")
                 and len(seqs[t]) >= 3 and cnt[t].sum() > 0)
    avg = {t: cnt[t].mean(0) for t in ids}
    high = [t for t in ids if avg[t].sum() / (3 * len(avg[t])) > COV_HIGH]; low = [t for t in ids if t not in set(high)]
    log(f"[prepare] {tag}: {len(seqs)} sequences, {len(ids)} admitted, {len(high)} high coverage, {len(low)} low coverage")
    gid = {t: gene.get(t, t) for t in ids}; gl = sorted(set(gid.values())); gi = {g: i for i, g in enumerate(gl)}
    pr = kmer_pairs([seqs[t] for t in ids], [gi[gid[t]] for t in ids])
    edges = [(gid[ids[i]], gid[ids[j]]) for (i, j), c in pr.items() if c >= TH]
    gunit = components(gl, edges); unit = {t: gunit[gid[t]] for t in ids}
    uf, val = fold_and_val(unit.values())
    d = os.path.join(out, tag); os.makedirs(d, exist_ok=True); os.makedirs(os.path.join(out, "mfe"), exist_ok=True)
    for nm, U in (("partI", high), ("partII", ids)):
        json.dump({"ids": U, "n": len(U), **({"strata": {"high": high, "low": low}} if nm == "partII" else {})},
                  open(f"{d}/universe_{nm}.json", "w"))
        json.dump({"folds": [{"fold": f, "test": sorted(t for t in U if uf[unit[t]] == f),
                              "val": sorted(t for t in U if unit[t] in val[f])} for f in range(5)],
                   "train_rule": "universe minus test minus val"}, open(f"{d}/folds_{nm}.json", "w"))
    json.dump({"tx_unit": unit}, open(f"{d}/units.json", "w"))
    off = np.concatenate([[0], np.cumsum([len(avg[t]) for t in ids])]).astype(np.int64)
    arr = dict(ids=np.array(ids), offsets=off, seq=np.frombuffer("".join(seqs[t] for t in ids).encode("ascii"), dtype=np.uint8))
    if len(reps) == 2 and all(np.allclose(cnt[t], np.round(cnt[t])) for t in ids):
        arr["reps"] = np.concatenate([cnt[t].astype(np.int32) for t in ids], axis=1)    # two replicates: loader uses their mean
    else:
        arr["avg_count"] = np.concatenate([avg[t] for t in ids]).astype(np.float64)
    np.savez_compressed(f"{d}/{tag}.npz", **arr)
    t0 = time.time()
    with Pool(workers) as pool: tracks = dict(pool.map(_mfe_job, [(t, seqs[t]) for t in ids], chunksize=8))
    log(f"[prepare] MFE tracks for {len(ids)} transcripts in {time.time() - t0:.0f}s")
    moff = np.concatenate([[0], np.cumsum([len(tracks[t]) for t in ids])]).astype(np.int64)
    np.savez_compressed(os.path.join(out, "mfe", f"{tag}_mfe.npz"), ids=np.array(ids), offsets=moff, mfe=np.concatenate([tracks[t] for t in ids]))
    json.dump({"n_valid": {t: len(seqs[t]) - 29 for t in ids}}, open(f"{d}/mfe_valid.json", "w"))
    norm = []
    for f in range(5):
        tr = [t for t in high if uf[unit[t]] != f and unit[t] not in val[f]]
        v = np.concatenate([tracks[t][:len(seqs[t]) - 29] for t in tr if len(seqs[t]) >= 30]).astype(np.float64)
        norm.append({"fold": f, "mean": float(v.mean()), "sd": float(v.std()), "n": int(len(v))})
    p = os.path.join(out, "MFE_NORM_FOLDWISE.json")
    fn = json.load(open(p)) if os.path.exists(p) else {"human": {}, "nonhuman": {}}
    fn["nonhuman"][tag] = norm; json.dump(fn, open(p, "w"), indent=1)
    log(f"[prepare] wrote {d}; load with ribopipe.data.load_dataset('{tag}', data='{out}')")
    return d


def register(sp):
    p = sp.add_parser("prepare", help="convert your own CDS FASTA + per-codon counts into the release format")
    p.add_argument("--fasta", required=True); p.add_argument("--counts", required=True); p.add_argument("--genes")
    p.add_argument("--out", required=True); p.add_argument("--tag", required=True); p.add_argument("--workers", type=int, default=8)
    p.set_defaults(func=lambda a: (prepare(a.fasta, a.counts, a.out, a.tag, a.genes, a.workers), 0)[1])
