# Guide

Reference documentation: results, setup, layout, and the decisions that are not
obvious from reading the code. See the [README](../README.md) for the overview.

**This is a research prototype trained on 183 genes on a 4 GB GPU. It is not a
clinical tool and no medical decision should rest on it.**

---

## Results

Held-out evaluation. Genes are split whole — a model that has seen other
variants in the same protein has a large and unrealistic advantage.

| Model | per-gene AUROC | pooled AUROC |
|---|---|---|
| BLOSUM62 | 0.7227 | 0.6879 |
| ESM-2 wt-marginal (zero-shot) | 0.8381 | 0.8408 |
| ESM-2 masked-marginal (zero-shot) | 0.8435 | 0.8422 |
| **ProteinNPT, 150M backbone** | **0.9271** | 0.9355 |
| ProteinNPT, 650M backbone | **0.9392** | 0.9254 |
| AlphaMissense | **0.9601** | 0.9306 |

31 of 34 held-out genes improved over zero-shot. Calibrated probabilities:
test ECE 0.097 → 0.052 (Platt scaling fitted on validation only).

**Per-gene AUROC is the headline, not pooled.** Pooled AUROC rewards a model
for knowing which *genes* are constrained, which is information it will not
have on a gene it has never seen. Per-gene asks the clinical question directly.

### AlphaMissense beats it

Scored on the identical 5,814 held-out variants across the same 34 genes,
100% coverage, no subsetting in anyone's favour:

| | per-gene | pooled |
|---|---|---|
| ESM-2 masked-marginal | 0.8435 | 0.8422 |
| ProteinNPT (this) | 0.9271 | **0.9355** |
| AlphaMissense | **0.9601** | 0.9306 |

A paired bootstrap over genes puts the gap at **-0.0330, 95% CI
[-0.0483, -0.0187]** - a real difference, not noise. AlphaMissense is ahead on
23 of the 34 genes.

That is the expected outcome and worth stating plainly. AlphaMissense is a much
larger model trained across the whole proteome with population-frequency
signal and structural context; this is a 150M-parameter backbone with a small
head trained on 183 genes and a 4 GB GPU. The point of comparison is not to
win, it is to say where the number actually sits.

Two things survive it. ProteinNPT is ahead on **pooled** AUROC (0.9355 vs
0.9306), the only metric where the ordering flips. And both supervised
approaches beat the language model alone by a wide margin - the zero-shot
baseline is 0.8435, so most of what either adds is real.

Reproduce with:

```bash
python -m vep.eval.alphamissense --alphamissense AlphaMissense_aa_substitutions.tsv.gz
```

The scores are CC BY-NC-SA 4.0 and are not redistributed here.

### The interesting result

ESM-2's likelihood answers *"would evolution tolerate this?"*, which is not the
same question as *"does this cause disease?"*. On gain-of-function and
aggregation mechanisms the two come apart, and zero-shot scoring fails
outright — five genes scored **below chance**, and using masked-marginals
instead of wt-marginals fixed none of them (PSEN2 moved 0.000).

Scaling the backbone does not fix it either. SMAD4's zero-shot score went from
0.358 to only 0.426 with a 4× larger model. It is not a capacity problem: those
variants are evolutionarily plausible.

Giving the model labelled neighbours at inference time does fix it. SMAD4
reaches ~0.86 under both backbones. That capability is what the non-parametric
architecture buys, and nothing else in the project reproduces it.

It is not the only route there, though: AlphaMissense reaches 0.951 on SMAD4
without any labelled neighbours, presumably from population-frequency signal
that carries information evolutionary likelihood does not.

### Where the 650M gain actually comes from

Zero-shot scoring improves far more than the end-to-end number does:

| | 150M | 650M |
|---|---|---|
| zero-shot, per-gene | 0.8435 | **0.8966** (+0.053) |
| with ProteinNPT | 0.9271 | 0.9392 (+0.012) |

The head had already recovered most of what the bigger backbone provides. They
partly substitute for each other.

### GRB2 fitness — a negative result

Spearman against measured binding fitness, on positions never seen in training
(n=1,220), with 95% CIs from 2,000 paired bootstrap resamples:

| model | rho | 95% CI | vs zero-shot |
|---|---|---|---|
| zero-shot mutant-marginal | 0.7292 | [0.701, 0.757] | — |
| multi-task ProteinNPT | 0.7427 | [0.713, 0.768] | +0.013, **not distinguishable** |
| fitness-only ProteinNPT | 0.7064 | [0.675, 0.736] | −0.022, **not distinguishable** |

**No supervised model beats zero-shot on genuinely novel positions.** Report the
zero-shot number for fitness.

An earlier version of this README claimed +0.045 for multi-task. That came from
position leakage. The split holds out whole positions, but a double mutant
spanning one training and one held-out position still lands in test, and 82% of
the test set is like that. On the 18% that shares nothing with training, the
gain evaporates.

The fitness-only specialist is the tell: it is the *best* model on the leaky
metric (0.7703) and the *worst* on the clean one (0.7064). That gap is what
fitting position identity looks like. Training it was meant to fix a multi-task
tradeoff and instead showed there was no fitness result to protect.

What survives: ProteinNPT clearly beats zero-shot on ClinVar pathogenicity
across 34 held-out genes, and the two questions are not equally amenable.
Pathogenicity has 183 genes of labels to generalise across; this DMS is one
56-residue domain, where "generalise to a new position" is a much harder ask
from far less data.

Multi-task training also cost pathogenicity accuracy (0.9271 → 0.9064), so the
served model is the single-task one and the tasks are kept apart.

---

## What is in the repository

Committed, so a clone can serve predictions without retraining anything:

| | size | |
|---|---|---|
| `data/processed/` | 1.4 MB | 29,088 parsed variants over 183 proteins |
| `artifacts/models/npt.pt` | 18 MB | the served ProteinNPT head (150M backbone) |
| `artifacts/cache/saturation/` | 9.5 MB | precomputed maps for the genes that have them |
| `artifacts/cache/zeroshot_all.parquet` | 1.4 MB | zero-shot scores; the predictor will not load without them |
| `artifacts/*.json` | small | every measured result behind the tables above |

Not committed, because it is either large or somebody else's:

| | size | how to get it |
|---|---|---|
| ESM-2 feature cache | 250 MB | `python -m vep.esm.cache`, ~2 min |
| ClinVar `variant_summary.txt.gz` | 423 MB | rebuild step below |
| GRB2 binding assay | 736 KB | fetch below — METL project data |
| 650M / multi-task / fitness checkpoints | 56 MB | retrain; they reproduce table rows, they do not serve |

## Setup

```bash
# PyPI's default torch wheel is CPU-only. With an NVIDIA GPU install the CUDA
# build first: the feature cache takes ~2 min on a GPU and roughly an hour on
# a CPU, and every other stage scales the same way.
pip install torch --index-url https://download.pytorch.org/whl/cu118
pip install transformers pandas numpy scikit-learn scipy h5py pyarrow pyyaml fastapi uvicorn requests pytest
```

Build the feature cache, then serve:

```bash
python -m vep.esm.cache
uvicorn api.main:app --port 8000
```

That is the whole path to a working API. Everything below is for rebuilding
from scratch rather than running what is already here.

### Rebuilding from scratch

Build the dataset (downloads ~440 MB from NCBI, resolves sequences from
UniProt, takes ~10 min):

```bash
python -c "from vep.config import Config; from vep.data.build import build; build(Config.load('configs/default.yaml'), 'data/raw/variant_summary.txt.gz')"
```

Then, in order:

```bash
python -m vep.esm.cache          # per-residue features  (~2 min)
python -m vep.eval.zeroshot      # zero-shot baseline    (~2 h)
python -m vep.pipeline           # train + evaluate      (~30 min)
```

### The GRB2 assay

Only needed to reproduce the fitness rows. It is the METL project's data, not
this repository's, so it is fetched rather than redistributed:

```bash
curl -L -o data/grb2-binding.tsv https://raw.githubusercontent.com/gitter-lab/metl-pub/main/data/dms_data/grb2-binding/grb2-binding.tsv
```

Run the API and the front end:

```bash
uvicorn api.main:app --port 8000
```

```bash
cd "front end/web" && npm install && npm run dev
```

---

## Layout

| path | what it is |
|---|---|
| `vep/data/` | ClinVar parsing, UniProt lookup, isoform reconciliation, DMS loading |
| `vep/esm/` | ESM-2 wrapper, long-protein windowing, feature and saturation caches |
| `vep/models/` | ProteinNPT, the serving predictor, calibration |
| `vep/train/` | datasets, trainer, multi-task trainer |
| `vep/eval/` | zero-shot baselines, metrics, report builder, AlphaMissense comparison, mechanism survey |
| `vep/pipeline.py` | end-to-end rebuild for one backbone |
| `vep/precompute.py` | saturation cache for the whole panel |
| `vep/gpu_guard.py` | thermal watchdog for long GPU jobs |
| `api/` | FastAPI service |
| `front end/web/` | Vue 3 front end |
| `archive/` | superseded scripts, kept for reference |
| `docs/` | this guide and the screenshots |
| `tools/` | screenshot capture for the docs |

---

## Things that are non-obvious

Each of these was a bug or a wrong result before it was a design decision.

**Numbering.** ClinVar submitters number variants against whichever transcript
they used, which is often not the UniProt canonical isoform. Dropping the
mismatches would have deleted ~90% of MECP2 and ~99% of WT1. `vep/data/isoform.py`
infers a piecewise-constant offset by Viterbi, which also recovers genes with
*internal* alternative exons — SCN5A's single-residue indel, CACNA1C's four
segments. Retention went 95.4% → 99.0%.

**Gene resolution.** `gene_exact:WAS` returns every reviewed human protein,
because "WAS" is an English stopword in UniProt's index. An earlier version
fell back to the longest hit and silently resolved WAS to a 1464-residue lipid
transfer protein. Unmatched symbols now fall through to HGNC rather than
guessing.

**Splitting.** By gene, never by variant. Neighbouring residues of one protein
share an embedding neighbourhood and often a label.

**Label leakage.** The query's own label is always replaced by `[MASK]`, and
the serving path additionally excludes the query from its own retrieved
neighbours — it is built as a fresh feature row, so without that check an
existing ClinVar variant could be sampled as its own neighbour and read the
answer straight out of the context.

**Cache invalidation.** Saturation matrices are namespaced by backbone;
probability matrices additionally by a hash of the checkpoint. A retrain
therefore misses the cache rather than serving stale predictions that look
perfectly valid.

**Masked-marginals are not worth it for interactive use.** They correlate with
wt-marginals at Spearman 0.988 and gain +0.005 per-gene AUROC for 27× the
compute. `/predict` defaults to `wt`.

**A cached artefact must not require the machinery that made it.** The
saturation route asked for the trained head before it looked in the cache, so
on a fresh clone - which ships 183 precomputed probability matrices but no
feature cache, and therefore cannot construct a predictor - every committed
matrix returned 503. The data was on disk and readable; the route just refused
to reach it. `tests/test_api.py` now pins the whole serving path against the
committed artefacts alone.

**Checkpoint compatibility is not automatic.** The continuous-label pathway
added for the fitness task introduced three parameters (`value_proj`,
`task_embed`) that the served pathogenicity head predates. `task_embed` is
zero-initialised precisely so an older checkpoint evaluates identically — but
`load_state_dict` is strict by default, so the checkpoint stopped loading at
all, and the API fell back to zero-shot while still reporting itself healthy.
`load_state_dict_compat` tolerates exactly those keys and raises on anything
else, because blanket `strict=False` would let a genuine rename serve randomly
initialised weights.

**Long proteins** are tiled into overlapping ≤1022-residue windows, and every
residue is scored from the window where it sits most centrally. Cost scales as
`length × min(length, 1022)`, so USH2A is 42× longer than KCNE2 but 351× more
expensive — which is why the precompute job reports honest cost-weighted
progress rather than gene counts.

---

## Tests

```bash
python -m pytest tests -q
```

50 tests, CPU-only, no network, ~45 s. They run on every push via
`.github/workflows/tests.yml`, on a CPU-only torch build - the suite
reaches no GPU, no large artefact and no remote service, so CI needs a
subset of the runtime dependencies. They pin the properties that would
otherwise break silently — leakage, isoform recovery, cache invalidation,
permutation invariance of multi-mutant pooling, scoring sign conventions.

---

## Data sources

This project is MIT licensed. That covers the code and the trained head; the
inputs it was built from carry their own terms, listed here because the
repository ships artefacts derived from several of them.

| source | used for | terms |
|---|---|---|
| ClinVar `variant_summary.txt.gz` (NCBI) | pathogenic/benign labels | public domain |
| UniProt REST | canonical sequences, accessions | CC BY 4.0 |
| HGNC REST | gene symbol fallback | CC0 |
| ESM-2 (Lin et al., 2023) via HuggingFace | backbone weights | MIT |
| AlphaFold DB | structures in the 3D view | CC BY 4.0 |
| 3Dmol.js | structure rendering | BSD-3-Clause |
| GRB2 binding DMS, METL project | fitness regression | fetched, not redistributed |

ProteinNPT (Notin et al., NeurIPS 2023) is the architecture and is implemented
here from the paper rather than vendored. Zero-shot scoring schemes follow
Meier et al., 2021.
