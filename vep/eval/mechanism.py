"""Can a pathogenicity score tell two syndromes in the same gene apart?

Every variant considered here is pathogenic; the question is which disease
results. Needs ClinVar's PhenotypeIDS column, which the main parser drops.
"""

from __future__ import annotations

import argparse
import gzip
import itertools
import json
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from vep.config import Config
from vep.constants import AA_TO_IDX

MIN_PER_CLASS = 15

# "not provided", "not specified", and other placeholders submitters use.
NOISE_CUIS = {"C3661900", "CN517202", "C5568766", "C0027672", "CN169374"}

# Mechanistically or clinically distinct.
REAL = {
    "SCN5A": "long QT (GoF) vs Brugada (LoF)",
    "RYR1": "central core myopathy vs malignant hyperthermia",
    "COL3A1": "aortic aneurysm vs EDS type 4",
    "FBN1": "aortic aneurysm vs Marfan",
    "MYO7A": "nonsyndromic deafness vs Usher",
    "USH2A": "retinal dystrophy vs Usher",
    "COL2A1": "achondrogenesis II vs SED congenita",
    "CRB1": "Leber congenital amaurosis vs retinal dystrophy",
}

# The same disease under two concept IDs. Whatever a score achieves here is
# submitter provenance rather than biology, so it sets the noise floor.
VOCAB = {
    "KCNQ2": "DEE vs DEE",
    "SCN8A": "DEE vs DEE",
    "LDLR": "FH vs FH",
    "BRCA1": "breast cancer vs BRCA1 predisposition",
    "MLH1": "Lynch vs HNPCC",
    "GJB1": "CMTX vs CMT-X",
}

SCORES = ["position", "esm2_masked", "npt_prob", "alphamissense"]


def _cuis(field: str) -> set[str]:
    out = set()
    for condition in field.replace(";", "|").split("|"):
        for token in condition.split(","):
            if token.startswith("MedGen:"):
                cui = token.split(":", 1)[1]
                if cui not in NOISE_CUIS:
                    out.add(cui)
    return out


def build_contrasts(cfg: Config, clinvar_path: Path) -> pd.DataFrame:
    """One binary phenotype contrast per gene, from unambiguous variants."""
    df = pd.read_parquet(cfg.paths.processed / "variants.parquet")
    row_of = {str(v): i for i, v in enumerate(df.variation_id)}
    ids = [""] * len(df)
    names: dict[str, str] = {}

    with gzip.open(clinvar_path, "rt", encoding="utf-8") as fh:
        header = fh.readline().rstrip("\n").lstrip("#").split("\t")
        iv, ip, ia, il = (header.index(c) for c in
                          ["VariationID", "PhenotypeIDS", "Assembly", "PhenotypeList"])
        for line in fh:
            f = line.rstrip("\n").split("\t")
            if len(f) <= il or f[ia] != "GRCh38" or f[iv] not in row_of:
                continue
            ids[row_of[f[iv]]] = f[ip]
            for condition, text in zip(f[ip].replace(";", "|").split("|"),
                                       f[il].replace(";", "|").split("|")):
                for token in condition.split(","):
                    if token.startswith("MedGen:"):
                        names.setdefault(token.split(":", 1)[1], text.strip())

    df["pheno_ids"] = ids
    pathogenic = df[df.label == "pathogenic"].copy()
    pathogenic["cuis"] = pathogenic["pheno_ids"].map(_cuis)

    picked = []
    for gene, sub in pathogenic.groupby("gene"):
        counts: Counter = Counter()
        for cs in sub["cuis"]:
            counts.update(cs)
        candidates = [c for c, n in counts.items() if n >= MIN_PER_CLASS]

        best = None
        for a, b in itertools.combinations(candidates, 2):
            has_a = sub["cuis"].map(lambda s, a=a: a in s)
            has_b = sub["cuis"].map(lambda s, b=b: b in s)
            only_one = has_a ^ has_b
            na, nb = int((has_a & only_one).sum()), int((has_b & only_one).sum())
            if min(na, nb) >= MIN_PER_CLASS and (best is None or min(na, nb) > best[0]):
                best = (min(na, nb), a, b, only_one, has_a)

        if best is None:
            continue
        _, a, b, only_one, has_a = best
        rows = sub[only_one].copy()
        rows["cls"] = has_a[only_one].astype(int)
        rows["contrast"] = f"{a}|{b}"
        rows["name_a"], rows["name_b"] = names.get(a, a), names.get(b, b)
        picked.append(rows)

    return pd.concat(picked, ignore_index=True)


def _score_columns(gene: str, sub: pd.DataFrame, cfg: Config,
                   proteins: dict, am_map: dict) -> dict[str, np.ndarray]:
    saturation = cfg.paths.cache / "saturation" / cfg.backbone_slug()
    prob_dir = next((p for p in saturation.iterdir()
                     if p.name.startswith("probability-")), None)

    out = {"position": (sub.pos0 / proteins[gene]["length"]).to_numpy()}
    paths = [("esm2_masked", saturation / "masked" / f"{gene}.npy")]
    if prob_dir is not None:
        paths.append(("npt_prob", prob_dir / f"{gene}.npy"))
    for key, path in paths:
        if path.exists():
            matrix = np.load(path)
            out[key] = np.array([matrix[p, AA_TO_IDX[a]]
                                 for p, a in zip(sub.pos0, sub.mut_aa)])

    accession = proteins[gene]["accession"]
    out["alphamissense"] = np.array(
        [am_map.get((accession, f"{w}{p + 1}{m}"), np.nan)
         for w, p, m in zip(sub.wt_aa, sub.pos0, sub.mut_aa)], dtype=float)
    return out


def _auroc(y: np.ndarray, score: np.ndarray) -> float:
    ok = ~np.isnan(score)
    if ok.sum() < 20 or len(set(y[ok])) < 2:
        return float("nan")
    a = roc_auc_score(y[ok], score[ok])
    # Which class a score should rank higher is not specified a priori. Taking
    # the better orientation biases small samples upward, which is exactly what
    # the vocabulary controls measure.
    return max(a, 1.0 - a)


def evaluate(contrasts: pd.DataFrame, cfg: Config, am_map: dict) -> dict:
    proteins = json.loads(
        (cfg.paths.processed / "proteins.json").read_text(encoding="utf-8"))
    report = {}
    for group_name, group in [("real", REAL), ("vocabulary_control", VOCAB)]:
        rows = []
        for gene, description in group.items():
            sub = contrasts[contrasts.gene == gene]
            if not len(sub):
                continue
            y = sub.cls.to_numpy()
            scored = _score_columns(gene, sub, cfg, proteins, am_map)
            rows.append({
                "gene": gene,
                "description": description,
                "n_a": int(y.sum()),
                "n_b": int(len(y) - y.sum()),
                **{k: round(_auroc(y, scored.get(k, np.full(len(sub), np.nan))), 4)
                   for k in SCORES},
            })
        means = {k: round(float(np.nanmean([r[k] for r in rows])), 4) for k in SCORES}
        report[group_name] = {"contrasts": rows, "mean_auroc": means}
    return report


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--clinvar", default="data/raw/variant_summary.txt.gz")
    ap.add_argument("--alphamissense", help="AlphaMissense_aa_substitutions.tsv.gz")
    ap.add_argument("--out", default="artifacts/mechanism_contrasts.json")
    args = ap.parse_args()

    cfg = Config.load(args.config)
    contrasts = build_contrasts(cfg, Path(args.clinvar))
    print(f"{contrasts.gene.nunique()} gene-contrasts, {len(contrasts):,} variants")

    am_map: dict = {}
    if args.alphamissense:
        from vep.eval.alphamissense import load_alphamissense

        proteins = json.loads(
            (cfg.paths.processed / "proteins.json").read_text(encoding="utf-8"))
        wanted = {proteins[g]["accession"] for g in contrasts.gene.unique()}
        am = load_alphamissense(Path(args.alphamissense), wanted)
        am_map = dict(zip(zip(am.uniprot_id, am.protein_variant), am.am_pathogenicity))

    report = evaluate(contrasts, cfg, am_map)
    Path(args.out).write_text(json.dumps(report, indent=1), encoding="utf-8")

    for group_name, block in report.items():
        print(f"\n{group_name}")
        print(f"  {'gene':8s} {'nA':>4s} {'nB':>4s} "
              + " ".join(f"{k:>13s}" for k in SCORES))
        for r in block["contrasts"]:
            print(f"  {r['gene']:8s} {r['n_a']:4d} {r['n_b']:4d} "
                  + " ".join(f"{r[k]:13.3f}" for k in SCORES)
                  + f"  {r['description']}")
        print(f"  {'mean':8s} {'':4s} {'':4s} "
              + " ".join(f"{block['mean_auroc'][k]:13.3f}" for k in SCORES))
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
