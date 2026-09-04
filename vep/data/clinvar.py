"""Parse NCBI ClinVar `variant_summary.txt.gz` into labelled missense variants.

The file is ~1 GB uncompressed and every variant appears once per genome
assembly, so we stream it line by line, keep only GRCh38 rows, and pull the
protein consequence out of the HGVS `Name` field.
"""

from __future__ import annotations

import gzip
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Iterator

import pandas as pd

from vep.constants import THREE_TO_ONE, normalise_significance, review_stars

CLINVAR_URL = (
    "https://ftp.ncbi.nlm.nih.gov/pub/clinvar/tab_delimited/variant_summary.txt.gz"
)

# Matches the protein consequence in e.g.
#   NM_007294.4(BRCA1):c.5123C>A (p.Ala1708Glu)
# Deliberately anchored to 3-letter codes so that synonymous (p.Ala123=),
# nonsense (p.Arg123Ter) and frameshift (p.Arg123fs) rows fail to match.
_MISSENSE_RE = re.compile(r"\(p\.([A-Z][a-z]{2})(\d+)([A-Z][a-z]{2})\)")

# Column aliases: ClinVar renamed several headers when it split germline from
# somatic classifications in 2024, so accept either spelling.
_COLUMN_ALIASES: dict[str, tuple[str, ...]] = {
    "gene": ("GeneSymbol",),
    "name": ("Name",),
    "type": ("Type",),
    "significance": ("ClinicalSignificance", "Germline classification"),
    "review": ("ReviewStatus", "Germline review status"),
    "assembly": ("Assembly",),
    "variation_id": ("VariationID",),
    "phenotypes": ("PhenotypeList",),
    "chrom": ("Chromosome",),
    "pos": ("PositionVCF", "Start"),
    "ref": ("ReferenceAlleleVCF", "ReferenceAllele"),
    "alt": ("AlternateAlleleVCF", "AlternateAllele"),
    "submitters": ("NumberSubmitters",),
}


@dataclass(frozen=True)
class ParsedVariant:
    gene: str
    wt_aa: str
    position: int          # 1-based, as reported in HGVS
    mut_aa: str
    label: str             # "pathogenic" | "benign"
    stars: int
    variation_id: str
    phenotypes: str
    chrom: str
    pos_vcf: str
    ref: str
    alt: str
    n_submitters: int


def _resolve_columns(header: list[str]) -> dict[str, int]:
    index = {name: i for i, name in enumerate(header)}
    # The first column is '#AlleleID'; strip the comment marker if present.
    if header and header[0].startswith("#"):
        index[header[0].lstrip("#")] = 0
    resolved: dict[str, int] = {}
    for key, candidates in _COLUMN_ALIASES.items():
        for cand in candidates:
            if cand in index:
                resolved[key] = index[cand]
                break
    missing = {"gene", "name", "significance", "review", "assembly"} - resolved.keys()
    if missing:
        raise ValueError(f"ClinVar file is missing expected columns: {sorted(missing)}")
    return resolved


def parse_protein_change(name: str) -> tuple[str, int, str] | None:
    """Extract (wt_aa, 1-based position, mut_aa) for a missense change."""
    m = _MISSENSE_RE.search(name or "")
    if m is None:
        return None
    wt3, pos, mut3 = m.group(1), int(m.group(2)), m.group(3)
    wt, mut = THREE_TO_ONE.get(wt3), THREE_TO_ONE.get(mut3)
    if wt is None or mut is None or wt == mut:
        return None            # Sec/Pyl/Xaa or a synonymous call
    return wt, pos, mut


def iter_variants(
    path: str | Path,
    genes: Iterable[str] | None = None,
    assembly: str = "GRCh38",
) -> Iterator[ParsedVariant]:
    """Stream labelled missense variants for the requested genes."""
    wanted = {g.upper() for g in genes} if genes is not None else None
    path = Path(path)

    with gzip.open(path, "rt", encoding="utf-8", errors="replace") as fh:
        header = next(fh).rstrip("\n").split("\t")
        col = _resolve_columns(header)
        n_fields = len(header)

        for line in fh:
            row = line.rstrip("\n").split("\t")
            if len(row) < n_fields:
                continue

            if row[col["assembly"]] != assembly:
                continue                       # skip the duplicate GRCh37 row
            gene = row[col["gene"]].strip().upper()
            if wanted is not None and gene not in wanted:
                continue
            if "type" in col and row[col["type"]] != "single nucleotide variant":
                continue

            label = normalise_significance(row[col["significance"]])
            if label is None:
                continue
            change = parse_protein_change(row[col["name"]])
            if change is None:
                continue
            wt, pos, mut = change

            def get(key: str, default: str = "") -> str:
                i = col.get(key)
                return row[i] if i is not None else default

            try:
                n_sub = int(get("submitters", "0") or 0)
            except ValueError:
                n_sub = 0

            yield ParsedVariant(
                gene=gene,
                wt_aa=wt,
                position=pos,
                mut_aa=mut,
                label=label,
                stars=review_stars(row[col["review"]]),
                variation_id=get("variation_id"),
                phenotypes=get("phenotypes"),
                chrom=get("chrom"),
                pos_vcf=get("pos"),
                ref=get("ref"),
                alt=get("alt"),
                n_submitters=n_sub,
            )


def load_dataframe(
    path: str | Path,
    genes: Iterable[str] | None = None,
    assembly: str = "GRCh38",
) -> pd.DataFrame:
    """Parsed variants as a DataFrame, deduplicated on the protein change.

    The same amino-acid substitution can be reached by different nucleotide
    changes and appears as several ClinVar rows. We collapse to one row per
    (gene, wt, pos, mut), keeping the best-reviewed assertion; a genuine
    pathogenic/benign disagreement between rows is dropped as unreliable.
    """
    records = [v.__dict__ for v in iter_variants(path, genes, assembly)]
    df = pd.DataFrame.from_records(records)
    if df.empty:
        return df

    key = ["gene", "wt_aa", "position", "mut_aa"]
    n_rows = len(df)
    labels_per_key = df.groupby(key)["label"].transform("nunique")
    conflicted = labels_per_key > 1
    n_conflicted = int(df.loc[conflicted, key].drop_duplicates().shape[0])
    df = df[~conflicted]

    df = (
        df.sort_values(["stars", "n_submitters"], ascending=False)
        .drop_duplicates(subset=key, keep="first")
        .reset_index(drop=True)
    )
    # Provenance for the data-build report. Kept in .attrs rather than as a
    # broadcast column so it survives as metadata and not as fake per-row data.
    df.attrs["n_rows_parsed"] = n_rows
    df.attrs["n_conflicting_substitutions"] = n_conflicted
    return df
