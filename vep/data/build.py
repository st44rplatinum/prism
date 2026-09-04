"""Build the modelling dataset: ClinVar labels x UniProt canonical sequences.

Produces
  data/processed/variants.parquet   one row per labelled missense substitution
  data/processed/proteins.json      canonical sequence + metadata per gene
  data/processed/build_report.json  provenance and QC counts
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from vep.config import Config
from vep.constants import GENE_PANEL
from vep.data.clinvar import load_dataframe
from vep.data.isoform import apply_piecewise, fit_all
from vep.data.uniprot import UniProtClient


def attach_sequences(
    df: pd.DataFrame, proteins: dict
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Keep variants whose wild-type residue matches the canonical isoform.

    A mismatch means the submitter numbered the variant against a different
    transcript. Silently keeping those rows would attach a correct label to the
    wrong residue, which is worse than dropping them - so we drop, and report
    the per-gene rate so a systematically mismatched gene is visible rather
    than quietly halved.
    """
    seqs = {g: rec["sequence"] for g, rec in proteins.items()}
    lengths = {g: len(s) for g, s in seqs.items()}

    df = df[df["gene"].isin(seqs)].copy()
    df["prot_len"] = df["gene"].map(lengths)
    if "pos0" not in df.columns:
        df["pos0"] = df["position"] - 1

    in_range = (df["pos0"] >= 0) & (df["pos0"] < df["prot_len"])
    observed = np.full(len(df), "", dtype=object)
    idx = np.flatnonzero(in_range.to_numpy())
    genes = df["gene"].to_numpy()
    pos0 = df["pos0"].to_numpy()
    for i in idx:
        observed[i] = seqs[genes[i]][pos0[i]]
    df["canonical_aa"] = observed
    df["wt_matches"] = df["canonical_aa"] == df["wt_aa"]

    report = (
        df.groupby("gene")
        .agg(n_total=("wt_matches", "size"), n_matched=("wt_matches", "sum"))
        .assign(match_rate=lambda d: d["n_matched"] / d["n_total"])
        .sort_values("match_rate")
        .reset_index()
    )
    return df[df["wt_matches"]].drop(columns=["canonical_aa", "wt_matches"]), report


def apply_isoform_offsets(
    df: pd.DataFrame, proteins: dict
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Shift each gene's residue numbering onto the canonical isoform."""
    df = df[df["gene"].isin(proteins)].copy()
    df["pos0"] = df["position"] - 1
    fits = fit_all(df, proteins)

    corrected = df["pos0"].to_numpy(dtype=np.int64).copy()
    for gene, fit in fits.items():
        if not fit.accepted:
            continue
        mask = (df["gene"] == gene).to_numpy()
        corrected[mask] = apply_piecewise(corrected[mask], fit.breakpoints)
    df["isoform_offset"] = corrected - df["pos0"].to_numpy(dtype=np.int64)
    df["pos0"] = corrected

    report = pd.DataFrame(
        [
            {
                "gene": f.gene,
                "segments": f.n_segments,
                "breakpoints": ";".join(f"{p}:{s:+d}" for p, s in f.breakpoints),
                "accepted": f.accepted,
                "match_before": round(f.match_rate_before, 4),
                "match_after": round(f.match_rate_after, 4),
                "n_variants": f.n_variants,
            }
            for f in fits.values()
        ]
    ).sort_values("match_before")
    return df, report


def split_by_gene(
    df: pd.DataFrame, test_fraction: float, val_fraction: float, seed: int
) -> pd.DataFrame:
    """Assign whole genes to train/val/test.

    Splitting by variant would leak: neighbouring residues of the same protein
    share an embedding neighbourhood and often a label, so a variant-level split
    reports an accuracy the model will not reproduce on a gene it has never
    seen. Genes are assigned greedily, largest first, each to whichever split is
    furthest below its quota - which keeps both the variant counts and the
    pathogenic/benign mix close to target without any split losing a class.
    """
    rng = np.random.default_rng(seed)
    per_gene = (
        df.groupby("gene")
        .agg(n=("label", "size"), n_path=("label", lambda s: int((s == "pathogenic").sum())))
        .reset_index()
    )
    # Shuffle first so equal-sized genes do not always land the same way.
    per_gene = per_gene.sample(frac=1.0, random_state=seed).sort_values("n", ascending=False)

    targets = {
        "train": 1.0 - test_fraction - val_fraction,
        "val": val_fraction,
        "test": test_fraction,
    }
    total = int(per_gene["n"].sum())
    quota = {k: max(v * total, 1.0) for k, v in targets.items()}
    got = {k: 0.0 for k in targets}
    got_path = {k: 0.0 for k in targets}
    assignment: dict[str, str] = {}
    target_share = float(df["label"].eq("pathogenic").mean())
    balance_weight = 3.0

    for row in per_gene.itertuples(index=False):
        def cost(split: str) -> float:
            new_n = got[split] + row.n
            new_path = got_path[split] + row.n_path
            # Resulting fill ratio. Assigning to whichever split would still be
            # least full packs the three splits proportionally. (Squared error
            # against the quota does not: it rewards whichever split comes
            # closest to being *filled* by this one gene, which hands the
            # largest genes to the smallest split and inverts the proportions.)
            size_err = new_n / quota[split]
            # ...and how far its pathogenic share would drift from the corpus
            # rate. Without this term the largest, most pathogenic-skewed genes
            # all land in train and the test set ends up with a different base
            # rate, which quietly shifts every threshold-dependent metric.
            share_err = (new_path / max(new_n, 1.0) - target_share) ** 2
            return size_err + balance_weight * share_err

        best = min(targets, key=cost)
        assignment[row.gene] = best
        got[best] += row.n
        got_path[best] += row.n_path

    df = df.copy()
    df["split"] = df["gene"].map(assignment)
    assert df["split"].notna().all(), "every gene must be assigned a split"
    _ = rng  # rng reserved for future stochastic tie-breaking
    return df


def build(cfg: Config, clinvar_path: str | Path) -> pd.DataFrame:
    cfg.paths.mkdirs()
    genes = cfg.data.genes or list(GENE_PANEL)

    print(f"[1/5] parsing ClinVar for {len(genes)} panel genes ...", flush=True)
    df = load_dataframe(clinvar_path, genes=genes)
    stats = {
        "clinvar_rows_parsed": int(df.attrs.get("n_rows_parsed", 0)),
        "conflicting_substitutions_dropped": int(df.attrs.get("n_conflicting_substitutions", 0)),
        "unique_substitutions": int(len(df)),
    }
    print(f"      {len(df):,} unique substitutions across {df['gene'].nunique()} genes")

    print("[2/5] resolving canonical sequences from UniProt ...", flush=True)
    client = UniProtClient(cfg.paths.cache / "uniprot.json")
    present = sorted(df["gene"].unique())
    records = client.fetch_many(present)
    proteins = {g: r.__dict__ for g, r in records.items()}
    stats["genes_resolved"] = len(proteins)
    stats["genes_unresolved"] = sorted(set(present) - set(proteins))

    print("[3/5] reconciling isoform numbering, then validating wild-type residues ...", flush=True)
    before = len(df)
    df, offset_report = apply_isoform_offsets(df, proteins)
    shifted = offset_report[offset_report["accepted"]]
    stats["genes_offset_corrected"] = int(len(shifted))
    print(f"      isoform offset applied to {len(shifted)} genes:")
    for r in shifted.sort_values("n_variants", ascending=False).head(12).itertuples(index=False):
        print(f"        {r.gene:10s} {r.segments} seg  {r.match_before:6.1%} -> {r.match_after:6.1%}  "
              f"(n={r.n_variants})  {r.breakpoints[:48]}")
    df, match_report = attach_sequences(df, proteins)
    stats["wt_mismatch_dropped"] = before - len(df)
    stats["wt_match_rate"] = round(len(df) / max(before, 1), 4)
    worst = match_report.head(8)
    print(f"      kept {len(df):,}/{before:,} ({stats['wt_match_rate']:.1%}); worst genes:")
    for r in worst.itertuples(index=False):
        print(f"        {r.gene:10s} {r.match_rate:6.1%}  ({int(r.n_matched)}/{int(r.n_total)})")

    print("[4/5] applying quality filters ...", flush=True)
    df = df[df["stars"] >= cfg.data.min_review_stars]
    counts = df.groupby("gene")["label"].agg(["size", "nunique"])
    keep = counts[(counts["size"] >= cfg.data.min_variants_per_gene)].index
    df = df[df["gene"].isin(keep)].reset_index(drop=True)
    stats["after_filters"] = int(len(df))
    stats["genes_after_filters"] = int(df["gene"].nunique())
    print(f"      {len(df):,} variants across {df['gene'].nunique()} genes "
          f"(>= {cfg.data.min_review_stars} star, >= {cfg.data.min_variants_per_gene}/gene)")

    print("[5/5] splitting by gene ...", flush=True)
    df = split_by_gene(df, cfg.data.test_fraction, cfg.data.val_fraction, cfg.data.seed)
    summary = (
        df.assign(is_path=df["label"].eq("pathogenic"))
        .groupby("split")
        .agg(variants=("label", "size"), genes=("gene", "nunique"), pathogenic_frac=("is_path", "mean"))
    )
    print(summary.to_string())
    stats["splits"] = json.loads(summary.to_json(orient="index"))

    out_dir = cfg.paths.processed
    df.to_parquet(out_dir / "variants.parquet", index=False)
    (out_dir / "proteins.json").write_text(json.dumps(proteins, indent=1), encoding="utf-8")
    (out_dir / "build_report.json").write_text(json.dumps(stats, indent=2), encoding="utf-8")
    match_report.to_csv(out_dir / "wt_match_report.csv", index=False)
    offset_report.to_csv(out_dir / "isoform_offsets.csv", index=False)
    print(f"\nwrote {out_dir/'variants.parquet'} and {out_dir/'proteins.json'}")
    return df
