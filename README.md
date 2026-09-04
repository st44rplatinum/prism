# ESM-2 variant effect predictor

Predicts whether a missense variant is pathogenic, using a protein language
model (ESM-2) with a ProteinNPT head trained on ClinVar labels. Ships a FastAPI
service and a Vue front end: per-gene saturation maps, an AlphaFold structure
view, and a held-out evaluation dashboard.

Held-out per-gene AUROC **0.927**, against 0.844 for zero-shot ESM-2 and 0.723
for BLOSUM62, on 34 genes the model never saw during training.

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

---

## Quick start

```bash
python -m pip install torch transformers pandas numpy scikit-learn scipy h5py pyarrow pyyaml fastapi uvicorn requests pytest
python -m vep.esm.cache
uvicorn api.main:app --port 8000
```

```bash
cd "front end/web" && npm install && npm run dev
```

The trained head, the parsed ClinVar data, and precomputed saturation maps are
committed, so nothing needs retraining.

**[Full documentation → `docs/GUIDE.md`](docs/GUIDE.md)** — results, methods,
rebuild instructions, and the design decisions that are not obvious from the
code.
