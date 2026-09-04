# Superseded scripts

The original exploratory scripts, kept for reference. All of them are broken as
written and none are imported by the project any more:

| file | superseded by | why it no longer runs |
|---|---|---|
| `esm2_model.py` | `vep/esm/backbone.py` | mixes the HuggingFace and `fair-esm` APIs; `esm` is never imported and `model` is overwritten |
| `main.py` | `vep/data/build.py`, `vep/data/dms.py` | reads `grb2_binding.tsv`; the file is `grb2-binding.tsv`. Also guards on 100 singles then samples 1000, and only 655 exist |
| `plt.py` | the Vue front end | imports `main` |
| `predict_prob.py` | `vep/esm/backbone.py` | uses `pd.DataFrame["sequence"]` as though the class were an instance |
| `score_variant.py` | `vep/eval/zeroshot.py` | depends on names `main.py` never defines (`sampled`, `tqdm`) |
| `calculate_embeddings.py` | `vep/esm/cache.py` | empty |
| `api.py` | the `api/` package | was shadowed by the package directory, so `import api` never loaded it |

Nothing here is on the import path. Delete the folder whenever you like.
