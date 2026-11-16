"""Shared helpers for the figure-reproduction scripts: locate paper_results/, check input md5 against MANIFEST.json, and
write the numbers_drawn JSON next to the figure."""
import os, json, hashlib, argparse, sys

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DEFAULT_RESULTS = os.path.join(REPO, "paper_results")


def md5(p): return hashlib.md5(open(p, "rb").read()).hexdigest()


def parse_args(doc):
    ap = argparse.ArgumentParser(description=doc.strip().splitlines()[0])
    ap.add_argument("outdir", help="directory to write the figure (.pdf, .png) and its numbers_drawn JSON")
    ap.add_argument("--results", default=DEFAULT_RESULTS, help="paper_results directory (default: <repo>/paper_results)")
    a = ap.parse_args(); os.makedirs(a.outdir, exist_ok=True); return a


class Inputs:
    """Opens files under paper_results/ and checks each against the md5 recorded in MANIFEST.json."""
    def __init__(self, root):
        self.root = root; self.man = json.load(open(os.path.join(root, "MANIFEST.json")))["files"]; self.used = {}

    def path(self, rel):
        p = os.path.join(self.root, rel); h = md5(p)
        assert rel in self.man, f"{rel} is not listed in paper_results/MANIFEST.json"
        assert h == self.man[rel]["md5"], f"md5 mismatch for {rel}: {h} != {self.man[rel]['md5']}"
        self.used[rel] = h; return p

    def json(self, rel): return json.load(open(self.path(rel)))


def write_numbers(outdir, stem, numbers, inputs, extra=None):
    import matplotlib
    d = {"figure": stem, "python": sys.version.split()[0], "matplotlib": matplotlib.__version__,
         "inputs": inputs.used, "outputs": {f"{stem}.{e}": md5(os.path.join(outdir, f"{stem}.{e}")) for e in ("pdf", "png")}}
    if extra: d.update(extra)
    d["numbers_drawn"] = numbers
    p = os.path.join(outdir, f"{stem}_numbers_drawn.json")
    json.dump(d, open(p, "w"), indent=1, default=float, ensure_ascii=False)
    return p
