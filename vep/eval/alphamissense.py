"""Compare the trained head against AlphaMissense on the same held-out set.

Two things make this comparison honest and both are easy to get wrong.

JOIN KEY. ClinVar variants keep the submitter's transcript numbering in
`position`, which is often not the UniProt canonical numbering; `pos0` is the
reconciled index. Joining on `position` silently loses whichever genes ClinVar
numbered against a different isoform - here it dropped 95% of MUTYH and 100% of
WT1, and the survivors were a biased remnant that scored AlphaMissense at 0.27.
The join is built from pos0 and every key is verified against the sequence
before use.

SAME VARIANTS. Both models are scored on the intersection only, so neither is
credited for coverage the other lacks. In practice AlphaMissense covers all
5,814 held-out variants, which is itself a check on the isoform work: their
canonical sequences agree with ours residue for residue.

AlphaMissense scores are CC BY-NC-SA 4.0 (Cheng et al., Science 2023) and are
not redistributed here. Download them yourself:

  https://storage.googleapis.com/dm_alphamissense/AlphaMissense_aa_substitutions.tsv.gz
"""

from __future__ import annotations

import argparse
import gzip
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import roc_auc_score

from vep.config import Config, resolve_device
from vep.models.npt import ProteinNPT, load_state_dict_compat
from vep.train.datasets import build_features, split_indices

MIN_PER_CLASS = 3          # same rule the main report uses
N_BOOT = 10_000


def score_test_variants(cfg: Config) -> pd.DataFrame:
    """Per-variant held-out scores, keyed so an external predictor can join."""
    from vep.eval.report import _predict

    device = resolve_device(cfg.backbone.device)
    feats = build_features(cfg)
    idx = split_indices(feats)
    test = idx["test"]

    ckpt = torch.load(cfg.paths.models / "npt.pt", map_location=device, weights_only=False)
    model = ProteinNPT(
        d_emb=feats.d_emb, d_model=cfg.npt.d_model, n_layers=cfg.npt.n_layers,
        n_heads=cfg.npt.n_heads, dropout=cfg.npt.dropout,
    ).to(device)
    load_state_dict_compat(model, ckpt["state_dict"])
    model.eval()

    # Inductive only: the transductive protocol lets the test gene contribute
    # its own labels, which no external predictor is given.
    print(f"scoring {len(test)} held-out variants (inductive) ...", flush=True)
    npt = _predict(model, feats, test, idx["train"], device, same_gene=False)

    fr = feats.frame.iloc[test].reset_index(drop=True)
    proteins = json.loads(
        (cfg.paths.processed / "proteins.json").read_text(encoding="utf-8")
    )

    out = pd.DataFrame({
        "gene": fr["gene"].to_numpy(),
        "accession": [proteins[g]["accession"] for g in fr["gene"]],
        "protein_variant": (fr["wt_aa"].astype(str)
                            + (fr["pos0"] + 1).astype(str)
                            + fr["mut_aa"].astype(str)),
        "label": feats.labels[test].astype(int),
        "npt": npt,
        "zeroshot": feats.subst[test, 40].astype(np.float32),   # masked-marginal
    })

    wrong = [(g, v) for g, v, p0, w in
             zip(out["gene"], out["protein_variant"], fr["pos0"], fr["wt_aa"])
             if proteins[g]["sequence"][int(p0)] != w]
    if wrong:
        raise SystemExit(f"{len(wrong)} join keys do not match the sequence: {wrong[:3]}")
    return out


def load_alphamissense(path: Path, accessions: set[str]) -> pd.DataFrame:
    """Stream the 1.2 GB table, keeping only the proteins under test."""
    rows = []
    with gzip.open(path, "rt") as fh:
        for line in fh:
            if line.startswith("#"):
                continue
            acc = line[:line.index("\t")]
            if acc in accessions:
                rows.append(line.rstrip("\n").split("\t"))
    if rows and rows[0][0] == "uniprot_id":
        rows = rows[1:]
    df = pd.DataFrame(rows, columns=["uniprot_id", "protein_variant",
                                     "am_pathogenicity", "am_class"])
    df["am_pathogenicity"] = df["am_pathogenicity"].astype(float)
    return df


def compare(scores: pd.DataFrame, am: pd.DataFrame) -> dict:
    merged = scores.merge(
        am[["uniprot_id", "protein_variant", "am_pathogenicity"]],
        left_on=["accession", "protein_variant"],
        right_on=["uniprot_id", "protein_variant"], how="left",
    )
    covered = merged["am_pathogenicity"].notna()
    d = merged.loc[covered]

    models = {"zeroshot": "ESM-2 masked-marginal (zero-shot)",
              "npt": "ProteinNPT (inductive)",
              "am_pathogenicity": "AlphaMissense"}

    per_gene = []
    for gene, sub in d.groupby("gene"):
        y = sub["label"].to_numpy()
        if y.sum() < MIN_PER_CLASS or (len(y) - y.sum()) < MIN_PER_CLASS:
            continue
        per_gene.append({"gene": gene, "n": int(len(y)),
                         "base_rate": round(float(y.mean()), 3),
                         **{k: round(float(roc_auc_score(y, sub[k])), 4) for k in models}})
    pg = pd.DataFrame(per_gene)

    summary = {k: {"per_gene": round(float(pg[k].mean()), 4),
                   "pooled": round(float(roc_auc_score(d["label"], d[k])), 4)}
               for k in models}

    # Bootstrap over GENES: per-gene AUROC is a mean over genes, so resampling
    # variants would understate the between-gene variance that actually drives
    # the statistic.
    rng = np.random.default_rng(0)
    idx = np.arange(len(pg))
    pairs = {"npt_vs_alphamissense": ("npt", "am_pathogenicity"),
             "npt_vs_zeroshot": ("npt", "zeroshot"),
             "alphamissense_vs_zeroshot": ("am_pathogenicity", "zeroshot")}
    deltas = {}
    for name, (a, b) in pairs.items():
        va, vb = pg[a].to_numpy(), pg[b].to_numpy()
        draws = np.array([va[s].mean() - vb[s].mean()
                          for s in (rng.choice(idx, len(idx), replace=True)
                                    for _ in range(N_BOOT))])
        lo, hi = np.percentile(draws, [2.5, 97.5])
        deltas[name] = {"delta": round(float(draws.mean()), 4),
                        "ci95": [round(float(lo), 4), round(float(hi), 4)],
                        "distinguishable": bool(not (lo <= 0 <= hi))}

    pg["npt_minus_am"] = (pg["npt"] - pg["am_pathogenicity"]).round(4)
    return {
        "source": "AlphaMissense (Cheng et al., Science 2023), CC BY-NC-SA 4.0",
        "n_test": int(len(scores)),
        "n_covered": int(covered.sum()),
        "coverage": round(float(covered.mean()), 4),
        "n_genes": int(len(pg)),
        "labels": models,
        "summary": summary,
        "paired_bootstrap_over_genes": deltas,
        "n_genes_alphamissense_better": int((pg["npt_minus_am"] < 0).sum()),
        "per_gene": pg.sort_values("npt_minus_am").to_dict("records"),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--alphamissense", required=True,
                    help="AlphaMissense_aa_substitutions.tsv.gz")
    ap.add_argument("--out", default="artifacts/alphamissense_comparison.json")
    args = ap.parse_args()

    cfg = Config.load(args.config)
    scores = score_test_variants(cfg)
    print(f"loading AlphaMissense for {scores.accession.nunique()} proteins ...", flush=True)
    am = load_alphamissense(Path(args.alphamissense), set(scores["accession"]))
    report = compare(scores, am)

    Path(args.out).write_text(json.dumps(report, indent=1), encoding="utf-8")
    cov = report["coverage"]
    n_cov, n_test = report["n_covered"], report["n_test"]
    print()
    print(f"coverage {cov:.1%} ({n_cov:,}/{n_test:,}), "
          f"{report['n_genes']} genes")
    print()
    for key, label in report["labels"].items():
        st = report["summary"][key]
        print(f"  {label:36s} per-gene {st['per_gene']:.4f}   "
              f"pooled {st['pooled']:.4f}")
    print()
    for name, d in report["paired_bootstrap_over_genes"].items():
        verdict = "distinguishable" if d["distinguishable"] else "NOT distinguishable"
        print(f"  {name:28s} {d['delta']:+.4f}  95% CI "
              f"[{d['ci95'][0]:+.4f}, {d['ci95'][1]:+.4f}]  {verdict}")
    print()
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
