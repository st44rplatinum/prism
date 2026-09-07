# Prism - ESM-2 variant effect predictor

[![tests](https://github.com/st44rplatinum/prism/actions/workflows/tests.yml/badge.svg)](https://github.com/st44rplatinum/prism/actions/workflows/tests.yml)

Predicts whether a missense variant is pathogenic, using a protein language
model (ESM-2) with a ProteinNPT head trained on ClinVar labels. Ships a FastAPI
service and a Vue front end: per-gene saturation maps, an AlphaFold structure
view, and a held-out evaluation dashboard.

Held-out per-gene AUROC **0.927**, against 0.844 for zero-shot ESM-2 and 0.723
for BLOSUM62, on 34 genes the model never saw during training. AlphaMissense,
scored on the identical variants, gets **0.960** - see
[the comparison](docs/GUIDE.md#alphamissense-beats-it).

> **Research prototype.** Trained on 183 genes on a 4 GB GPU. Not a clinical
> tool — no medical decision should rest on it.

---

## Saturation map

Every possible substitution at every residue, coloured by predicted
pathogenicity. TP53 below: the disordered N-terminal domain is tolerant, the
DNA-binding core is not.

![Saturation heat map](docs/img/heatmap.png)

## Structure

The same per-residue scores painted onto the AlphaFold model, with a toggle for
AlphaFold's own pLDDT confidence.

![Structure view](docs/img/structure.png)

## Evaluation

Genes are split whole, never by variant, so a test gene contributes no labels of
its own.

![Metrics dashboard](docs/img/metrics.png)

## What no predictor here can do

Every model on that table, AlphaMissense included, outputs one number. One
number cannot say *which* disease.

In SCN5A, gain-of-function variants cause long QT syndrome and loss-of-function
variants cause Brugada syndrome. Opposite mechanisms, opposite treatment, same
gene, and both are "pathogenic". AlphaMissense tells them apart at **AUROC
0.503** - chance - while scoring 0.960 on pathogenicity itself. Sequence
position alone does better (0.708).

Across eight such within-gene contrasts, no pathogenicity score beat a plain
positional baseline. Pathogenicity prediction is in good shape; mechanism
prediction is untouched, and the single output axis is why.

---

## Quick start

```bash
# PyPI's default torch wheel is CPU-only. With an NVIDIA GPU, install the CUDA
# build first or everything below runs ~30x slower.
pip install torch --index-url https://download.pytorch.org/whl/cu118
pip install transformers pandas numpy scikit-learn scipy h5py pyarrow pyyaml fastapi uvicorn requests pytest
python -m vep.esm.cache          # per-residue features, ~2 min on a GPU
uvicorn api.main:app --port 8000
```

```bash
cd "front end/web" && npm install && npm run dev
```

The trained head, the parsed ClinVar data, the zero-shot scores and the
precomputed saturation maps are all committed, so nothing needs retraining and
the only setup step is the feature cache.

**[Full documentation → `docs/GUIDE.md`](docs/GUIDE.md)** — results, methods,
rebuild instructions, and the design decisions that are not obvious from the
code.

## License

[MIT](LICENSE). The committed model weights and parsed data derive from
ClinVar (public domain), UniProt (CC BY 4.0) and ESM-2 (MIT) — see
[`docs/GUIDE.md`](docs/GUIDE.md#data-sources) for attribution.
