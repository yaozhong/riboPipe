# Baselines of the RiboPipe paper (Part I)

This directory holds the wrappers that produced the comparator rows of Part I (main Fig. 1, Supplementary Tables
"Evaluation setting 1/2" and the three-seed tables), with the configuration IDs used in the paper. Every wrapper
reads data only through `ribopipe.data.load_dataset` and writes runs only through `ribopipe.data.write_run`, in the
layout that `ribopipe score` reads:

```
<runs>/<config_id>/<dataset>/seed<k>/fold<f>/{pred.npz, prov.npz, run.json}     (deterministic methods: seed "det")
```

`pred.npz` maps transcript id to a float32 array of length L (A-site register). `prov.npz` maps it to int8 provenance
codes: 0 = model, 1 = adapter, 2 = undefined. The scored interior `[20, L-20)` is always 0. The wrappers never score
anything; scoring is done by `ribopipe score` / `ribopipe compare`.

## Quick start

```bash
# 0. dependencies: scikit-learn for RiboMIMO (the in-house methods and iXnos need nothing extra)
pip install ".[baselines]"
#    RiboExp and RiboGL use a second environment (torch_geometric, lightning; numpy 2): requirements/gnn-baselines.txt

# 1. official code of RiboMIMO, RiboExp and RiboGL (not redistributed, see "Licences"); verifies md5s, applies patches/
bash baselines/fetch_upstream.sh                # -> baselines/upstream (or set RIBOPIPE_BASELINE_UPSTREAM)

# 2. data (the repository's data/ or the full release)
export RIBOPIPE_DATA=/path/to/data RIBOPIPE_MFE=/path/to/data/mfe

# 3. run
python -m baselines.run --list
python -m baselines.run --config Paper_Ridge   --dataset GSE133393_WT --out runs/                 # 5 folds, seeddet
python -m baselines.run --config Paper_RiboMIMO --dataset Mohammad16  --out runs/ --seeds 123,124,125
python -m baselines.run --config Paper_iXnos_MFE_lr005_elig --dataset PRJNA_iPS --out runs/ --device cpu
python -m baselines.run --config Paper_RiboGL_heldout --dataset Mohammad16 --out runs/ --folds 0 --seeds 123

# smoke test (tiny subset; never for results)
python -m baselines.run --config Paper_RiboMIMO --dataset Mohammad16 --out /tmp/smoke --folds 0 --smoke 40 20 20 --max-epochs 2

# structural check of written runs, and comparison with reference runs
python baselines/tools/check_runs.py   --runs runs/ --config Paper_Ridge --dataset GSE133393_WT --seed det --folds 0,1,2,3,4
python baselines/tools/compare_runs.py --ref <paper runs> --new runs/ --config Paper_Ridge --datasets GSE133393_WT
```

Only the pre-registered seeds 123, 124 and 125 are accepted for seeded methods (`--allow-any-seed` for tests). The
paper's main tables use seed 123; the three-seed tables use 123, 124 and 125 for RiboMIMO, iXnos (MFE) and RiboGL.
Runs whose `run.json` exists are skipped unless `--overwrite` is given (a fixed-seed failure is never replaced).

### Environments used for the paper (machine with RTX 4090)

| Methods | Environment |
|---|---|
| in-house, iXnos (CPU), RiboMIMO (CUDA) | python 3.11.15, torch 2.1.2+cu121, numpy 1.26.4, scikit-learn 1.8.0 |
| RiboExp, RiboGL (CUDA) | python 3.10.22, torch 2.6.0+cu124, numpy 2.2.6, scipy 1.15.3, torch_geometric 2.8.0, lightning 2.6.5, ViennaRNA 2.7.2 |

All runs used `PYTHONNOUSERSITE=1`, which keeps packages from `~/.local` from shadowing the environment.

## Configurations used in the paper

| Paper row | Configuration ID | Wrapper | Seeds |
|---|---|---|---|
| Codon-mean | `Paper_CodonMean` | `inhouse.py` | det |
| Tri-codon | `Paper_TriCodon` | `inhouse.py` | det |
| Ridge | `Paper_Ridge` | `inhouse.py` | det |
| RUST | `Paper_RUST` | `inhouse.py` | det |
| iXnos | `Paper_iXnos_seq_lr005_elig` | `ixnos.py` | 123 |
| iXnos (MFE) (core comparator) | `Paper_iXnos_MFE_lr005_elig` | `ixnos.py` | 123, 124, 125 |
| iXnos (MFE), 500 densest genes (sensitivity) | `Paper_iXnos_MFE_lr005` | `ixnos.py` | 123 |
| RiboMIMO (core comparator) | `Paper_RiboMIMO` | `ribomimo.py` | 123, 124, 125 |
| RiboGL (core comparator) | `Paper_RiboGL_heldout` | `ribogl.py` | 123, 124, 125 |
| RiboExp | `Paper_RiboExp` | `riboexp.py` | 123 |
| Setting 2, RiboMIMO | `Paper_RiboMIMO_L2` | `ribomimo.py` | 123 |
| Setting 2, iXnos | `Paper_iXnos_nocut_lr005_elig` (setting-1 reference: `Paper_iXnos_seq_lr005_elig`) | `ixnos.py` | 123 |
| Setting 2, RiboGL | `Paper_RiboGL_L2_heldout` | `ribogl.py` | 123 |
| Components figure (RiboMIMO-adapt without / with MFE) | `Paper_RiboMIMO_adapt_noMFE`, `Paper_RiboMIMO_adapt_MFE` | `ribomimo.py` | 123 |

BiLSTM-97 and the RiboPipe variants are trained by `ribopipe/train.py` (`ribopipe train`). The CLI also lists
the configurations of the frozen protocol that are not in the paper's tables: `Paper_iXnos_seq` / `_MFE` / `_nocut`
(learning rate 0.01, 500 genes), `Paper_iXnos_seq_lr005`, `Paper_iXnos_nocut_lr005`, `Paper_RiboMIMO_impl` (identical
to `adapt_noMFE`, decision of 2026-11-13) and `Paper_RiboExp_noStruct_port` (non-default `--no_structure` variant).

## Common adaptations (all methods)

These follow from the frozen Part I protocol (v1.3), which applies to every method, RiboPipe included:

* **Data and splits.** PaperData_v1.0 through `ribopipe.data`. Five frozen folds that keep each sequence-similarity
  unit on one side. Every method trains on `train_ids(f)`, selects on `val_ids(f)` and predicts `test_ids(f)`. A
  method's own training filter acts on `train_ids(f)` only. The official random splits are not used.
* **Blind test.** Test-transcript counts never reach method code: they are deleted from memory (RiboMIMO), or only
  train and validation counts are passed (all others). RiboGL's graph cache holds sequence-derived data only.
* **Target register.** All outputs are in the A-site register of the release data. No offset is applied.
* **Seeds.** The seed is passed to `random`, `numpy`, `torch` and `torch.cuda`. Several official codes hard-coded a
  seed (RiboMIMO numpy 1234, RiboExp 1234/1111, RiboGL 42, iXnos none).
* **No length cap.** Older harnesses skipped transcripts longer than 1,000 codons.
* **Provenance.** A position whose prediction relies on padding the method does not define is written as adapter (1).
  This covers RiboMIMO's trimmed ends, the iXnos windows that run off the CDS, and the RiboExp structure model's end
  windows. The interior is always model (0).
* **Failures.** Non-finite training loss or all-flat predictions are written as `MODEL_FAILED` or left to the scorer.
  A machine error (for example CUDA out of memory) writes no `run.json`, so the run can be resumed with the same
  configuration and seed.
* **Count thresholds** of the official training filters are applied to the released A-site counts: the mean of the
  two replicates for human data, the non-integer released average for the other species. Official code applied
  them to raw read counts.

## Methods

### In-house references (`inhouse.py`; deterministic, trained once per fold)

This is our own code and is included directly. It is a line-by-line port of the paper's `lookup_ridge_paper.py` and
`rust_paper.py`.

* **Codon-mean**: a table of the mean `log1p(c/mean(c))` per A-site codon over training transcripts. The prediction
  is `expm1(table)`. An unseen key falls back to the unweighted mean of the table.
* **Tri-codon**: the same table keyed on codons (i−1, i, i+1) = (P, A, A+1). Positions past either end use the key −1.
* **Ridge**: one-hot codons at offsets −5…+5 (704 features), α = 10, with intercept, target `log1p(c/mean(c))`,
  prediction `expm1`. The model is solved in closed form from per-transcript sufficient statistics in float64. This
  equals sklearn `Ridge(alpha=10)` and bounds memory without a length cap.
* **RUST**: our re-implementation of the RUST enrichment ratio (O'Connor et al. 2016), not the official RUST code.
  Each transcript is binarised as `b = 1[c > mean(c)]`. For offsets −5…+5, `ρ_j(codon) = P(b=1 | codon) / P(b=1)`,
  floored at 1e-6, and the prediction is the product of ρ over the window. Offsets past the CDS are skipped.

Training uses `train_ids(f)`. Transcripts with mean(c) = 0 are skipped, and validation transcripts are not used.

### iXnos (`ixnos.py`; PyTorch re-implementation; the paper's iXnos runs were on CPU)

The official iXnos code (lareaulab/iXnos; Python 2, Theano and Lasagne) is not executed. `fetch_upstream.sh` fetches
it at commit 1ea2337 for reference only. `ixnos.py` re-implements the official recipe:

* **Inputs**: one-hot codons at −5…+4 and nucleotides at −15…+14 (760). The MFE model adds three raw RNAfold MFEs
  of the 30-nt windows that start at nt −17/−16/−15 (763).
* **Network**: Dense(200, tanh) → Dense(1, ReLU), Glorot-uniform weights and zero biases.
* **Training**: MSE loss; Nesterov SGD, momentum 0.9, `lr/(1+epoch/decay)`; batch 500 with the last incomplete
  batch dropped; the optimiser (and its momentum) is re-created each epoch, as Lasagne does. Training runs a fixed
  number of epochs and keeps the last one.
* **Target**: `c[20:L−20]/mean(c[20:L−20])` (20/20 end trim).

Adaptations:

| Item | Paper | Official | Why |
|---|---|---|---|
| Training genes | `_elig`: all training-fold genes passing the official quality thresholds (L > 40, interior Σc ≥ 200, ≥ 100 codons with footprints); the 500-densest selection kept as a sensitivity row (`Paper_iXnos_MFE_lr005`) | the 500 densest genes, split 333/167 into train/test | the official selection served its own train/test split; our test sets are the frozen folds (decision of 2026-11-14) |
| Learning rate | 0.005 | 0.005 declared in `reproduce_scripts/struc_series.py` but not passed to the trainer, so the code executes the default 0.01 | value written by the authors for the structure model; at 0.01 raw-MFE training collapsed to all-zero output in 5/8 datasets |
| Schedule (epochs/decay) | human 70/32, yeast 60/16, *E. coli* 60/16 | human 70/32 (Iwasaki), yeast 60/16 (Weinberg; 55/16 for the yeast structure model) | pre-registered; *E. coli* has no official schedule |
| Validation | val MSE logged per epoch, never used for selection | none | protocol logging |
| End positions | positions with zero-padded feature windows or undefined MFE windows -> adapter | not predicted | full-length deployment row |
| MFE track | frozen RNAfold tracks of the CDS (no UTR); windows reaching outside the CDS are undefined | `.fold` files including 13 nt 5′UTR / 10 nt 3′UTR | the release has CDS only; the 20/20 interior is unaffected |
| Setting 2 (`nocut`) | no trim, target c/mean(c) over the full CDS, all positions model | — | common training target |

Tie-break of the density ranking is by transcript id. The official ordering depends on dict order and is not
reproducible.

### RiboMIMO (`ribomimo.py`; official network, adapted driver)

Uses github.com/tiantz17/RiboMIMO @ 094c693.

* **Official code used** (via `patches/ribomimo_model.py.diff`): the network class and `weights_init` from the
  official `model.py`. The patch changes two lines and no numerics: `lengths.cpu()` for `pack_padded_sequence`
  (torch ≥ 1.7) and `dtype=object` for a ragged array (numpy ≥ 1.24).
* **Driver**: the training driver re-implements `runRiboMIMO.py` step by step. A one-epoch CPU run of the driver
  matched the official driver with max |Δ| = 0 on all parameters (paper audit).
* **Defaults**: official throughout.
  * Inputs: 97 one-hot features (64 codon + 12 nt + 21 aa, the recommended `--add_nt --add_aa`).
  * Network: 2-layer bidirectional GRU (256), Linear 512→256 + ReLU, a regression head and a 3-class head.
  * Loss: MSE + 3·CE. Class thresholds are mean + {0, 2}·SD of log10 training density.
  * Optimisation: Adam 1e-3 with weight decay 1e-6, StepLR(5, 0.95), batch 16, ≤ 100 epochs, early stop after 5
    epochs without improvement of the validation R², Xavier initialisation.
* **`Paper_RiboMIMO`**:
  * Target: the official target `1[c>0.5]·c/mean(c[c>0.5])`, with the first 5 and last 4 codons zeroed.
  * Loss and validation on density > 0.
  * Training filter: mean(c > 0.5) > 0.6, applied to the training fold only.
  * Trimmed ends are written as adapter, holding the raw network output.
* **`Paper_RiboMIMO_L2`** (setting 2): target `c/mean(c)` over the full CDS, no trimming, only padding masked; the
  training filter is kept.
* **Adaptations**:
  * Validation set: `val_ids(f)`. The official code uses the first 10% of a random 10-fold training split.
  * Class thresholds are computed on the untrimmed training density, at the official point; an older bridge
    computed them after trimming.
  * No test-fold evaluation during training.
  * A non-finite loss gives `MODEL_FAILED`.
  * Non-ACGT transcripts raise an error; the official code drops them silently, and none occur in the release.
  * Inputs are encoded once rather than once per epoch (same values).

### RiboExp (`riboexp.py`; official default with structure features)

Uses github.com/Liuxg16/Riboexp @ 8edf66b (Python 2.7 / PyTorch 0.3.1).

* **Ports**: `fetch_upstream.sh` applies `patches/riboexp_*.diff`, which produce the py3/torch2 ports used for the
  paper; md5s are checked.
  * `models.py`: API edits only. Both branches are kept, including the official default structure encoder.
  * `nntools.py`: the one-hot and repackage helpers in torch2 form.
  * `dataio.py`: py3 edits; codon vocabulary in the python-2 dict order A, C, T, G, as measured with python 2.7.18;
    adapter `paper_fold_track` / `paper_docs`.
  * `runRiboexp.py` → `riboexp_run_py3.py`: the training loop line by line.
* **Training defaults** (official): Adadelta lr 1.0, clip 1.0, fc max-norm 3; validation every 80 batches; save only
  when the correlation improves and the rationale length < 9; stop after 50 validations without improvement;
  ≤ 200 epochs; batch 600; window 10.
* **Ported-model check** (paper audit): loading the authors' trained weights into the ported model reproduces their
  logged test Pearson within 1e-5.
* **Adaptations**:
  * Training windows come from `train_ids(f)` through the official `process_structure`, including its Σc ≥ 200
    gate. Validation windows come from `val_ids(f)` without that gate.
  * Structure track: `fold[j] = MFE(window starting at j−14)` inside the CDS; elsewhere the official missing value −1.
  * Seed: passed to the window shuffle and the model constructor.
  * Full-length inference for the structure model mirrors the official `testSeq` padding; its end windows
    (i < 5 or i > L−5) are adapter.
  * Deterministic kernels are used at inference only.
  * The parameter count on the predictive path is measured with forward hooks: 6,509,059.

### RiboGL (`ribogl.py`; official code imported verbatim)

Uses github.com/vam-sin/ribogl @ f3722b6. No patch: `rna_struct_utils`, `graph_processing_utils` and
`gnn_utils.GCN_LSTM` are imported unchanged (md5-checked).

* **Official defaults** (notebook cell 1 / `trainGCN`): GCN layers [256, 128, 128, 64], dropout 0.1, AdamW 1e-3 with
  ReduceLROnPlateau, batch 1, SAGE convolutions, random-walk PE length 32, ≤ 200 epochs with early-stopping
  patience 10.
* **Loss and output**: MaskedPCCL1Loss with a softmax output. The native target is `log(1+c)`, with zero runs ≥ 30
  set to NaN and divided by its nansum.
* **`Paper_RiboGL_heldout`**: the checkpoint is chosen on the frozen validation set. The released `trainGCN` passes
  the test loader as validation (`gnn_utils.py:852-855`), which we do not do. The softmax output is written as is.
* **`Paper_RiboGL_L2_heldout`** (setting 2): `L_eff·softmax(z)` is trained against `c/mean(c)`; this is the only
  change to `_get_loss`.
* **Adaptations**:
  * The RiboGL study's training-gene filter (coverage > 0.3, NaN fraction ≤ 0.05, longest zero run ≤ 20) is
    restored and applied to training transcripts only.
  * The graph cache holds sequence-derived data only, and test graphs carry no target.
  * `seed_everything(seed)`; the training order is shuffled once, as `FileRiboDataset` does.
  * `ModelCheckpoint(save_top_k=1)`.
  * The print-only `verbose=True` is dropped from ReduceLROnPlateau, because torch 2.7 removed it.
  * Deterministic kernels are used at inference only.
  * The `transformers` package, imported but unused by `gnn_utils.py`, is replaced by a stub.

## Licences and how upstream code is handled

| Upstream | Commit | Licence file | Handling |
|---|---|---|---|
| tiantz17/RiboMIMO | 094c693d4df45f4b29f5a3209b31c452a39092a5 | none found | fetched by `fetch_upstream.sh`; 2-line patch |
| Liuxg16/Riboexp | 8edf66bd4db6a2ec14e9600cee58fb7ee8bff6c5 | none found | fetched; py3 port produced by patches |
| vam-sin/ribogl | f3722b60c33427484e9fbf483758e41a0736add5 | none found | fetched; imported verbatim |
| lareaulab/iXnos | 1ea2337dd24d1e6fa3859f08573e99f02512810f | none found | fetched for reference only; `ixnos.py` is our re-implementation |

No repository has a licence file, and no README states a licence. The code is therefore not redistributed: users
download it from the authors' repositories. `patches/` holds only unified diffs (one line of context) that turn the
official files into the versions we ran. The `+`/`-` lines of `riboexp_run_py3.py.diff` contain ported lines of the
official training loop, because the py3 port of that file is substantial.

`fetch_upstream.sh` exits non-zero unless every official file and every patched file has the md5 recorded for the
paper's runs.

## Files

| File | Content |
|---|---|
| `run.py` | CLI (`python -m baselines.run`) |
| `common.py` | data/write helpers, codon index, provenance codes |
| `inhouse.py` | codon-mean, tri-codon, ridge, RUST |
| `ixnos.py` | iXnos re-implementation and driver |
| `ribomimo.py`, `riboexp.py`, `ribogl.py` | drivers around the official code |
| `fetch_upstream.sh`, `patches/` | download at pinned commits, md5 checks, documented patches |
| `tools/check_runs.py` | structural check of written runs |
| `tools/compare_runs.py` | array-by-array comparison with reference runs |

Simplified relative to the paper's runners:

* The `--samecrit` same-checkpoint sensitivity output of RiboMIMO is not ported; it was not used in the tables.
* The iXnos variants with a retry rule (`_stabilized`, `_rt`) and with z-scored MFE (`_z`) are not ported; they were
  reported nowhere in the final paper.
* The blind-inference test switches and debug switches are not ported.
* The `config_md5` binding to the paper's audit files is replaced by `code_md5` of the files actually run.
* Smoke subsets take the first n sorted ids. The paper's RiboExp/RiboGL smoke used a RandomState(0) permutation;
  smoke runs are never used for results.
