"""How many genes in all of ClinVar could support a phenotype contrast?

Independent of the gene panel. This puts a ceiling on the mechanism question
for the whole database rather than describing one panel's worth of genes.
"""

from __future__ import annotations

import argparse
import gzip
import itertools
import json
import re
from collections import Counter
from pathlib import Path

from vep.eval.mechanism import _cuis

MISSENSE_RE = re.compile(r"\(p\.([A-Z][a-z]{2})(\d+)([A-Z][a-z]{2})\)")

STARS = {
    "practice guideline": 4,
    "reviewed by expert panel": 3,
    "criteria provided, multiple submitters, no conflicts": 2,
    "criteria provided, single submitter": 1,
}


def survey(clinvar_path: Path, min_per_class: int = 30) -> dict:
    by_gene: dict[str, list[set[str]]] = {}
    names: dict[str, str] = {}

    with gzip.open(clinvar_path, "rt", encoding="utf-8") as fh:
        header = fh.readline().rstrip("\n").lstrip("#").split("\t")
        ig, ina, ic, ir, ip, ia, il = (
            header.index(c) for c in
            ["GeneSymbol", "Name", "ClinicalSignificance", "ReviewStatus",
             "PhenotypeIDS", "Assembly", "PhenotypeList"]
        )
        for line in fh:
            f = line.rstrip("\n").split("\t")
            if len(f) <= il or f[ia] != "GRCh38":
                continue
            significance = f[ic].lower()
            if "conflict" in significance or not significance.startswith("pathogenic"):
                continue
            if STARS.get(f[ir].lower(), 0) < 1 or not MISSENSE_RE.search(f[ina]):
                continue
            gene = f[ig]
            if not gene or "," in gene or ";" in gene:
                continue

            by_gene.setdefault(gene, []).append(_cuis(f[ip]))
            for condition, text in zip(f[ip].replace(";", "|").split("|"),
                                       f[il].replace(";", "|").split("|")):
                for token in condition.split(","):
                    if token.startswith("MedGen:"):
                        names.setdefault(token.split(":", 1)[1], text.strip())

    found = []
    for gene, records in by_gene.items():
        counts: Counter = Counter()
        for concepts in records:
            counts.update(concepts)
        candidates = sorted(c for c, n in counts.items() if n >= min_per_class)

        best = None
        for a, b in itertools.combinations(candidates, 2):
            na = sum(1 for cs in records if a in cs and b not in cs)
            nb = sum(1 for cs in records if b in cs and a not in cs)
            if min(na, nb) < min_per_class:
                continue
            key = (-min(na, nb), a, b)
            if best is None or key < best[0]:
                best = (key, a, b, na, nb)

        if best:
            _, a, b, na, nb = best
            found.append({"gene": gene, "n_a": na, "n_b": nb,
                          "name_a": names.get(a, a), "name_b": names.get(b, b)})

    found.sort(key=lambda r: -min(r["n_a"], r["n_b"]))
    return {
        "min_per_class": min_per_class,
        "n_variants": sum(len(v) for v in by_gene.values()),
        "n_genes": len(by_gene),
        "n_genes_with_contrast": len(found),
        "contrasts": found,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--clinvar", default="data/raw/variant_summary.txt.gz")
    ap.add_argument("--min-per-class", type=int, default=30)
    ap.add_argument("--out", default="artifacts/mechanism_survey.json")
    args = ap.parse_args()

    result = survey(Path(args.clinvar), args.min_per_class)
    Path(args.out).write_text(json.dumps(result, indent=1), encoding="utf-8")

    print(f"{result['n_variants']:,} pathogenic missense variants, "
          f"{result['n_genes']:,} genes")
    print(f"{result['n_genes_with_contrast']} genes have two phenotypes with "
          f">={result['min_per_class']} variants each")
    print()
    for r in result["contrasts"]:
        print(f"  {r['gene']:9s} {r['n_a']:4d} {r['n_b']:4d}  "
              f"{r['name_a'][:38]:38s} | {r['name_b'][:38]}")
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
