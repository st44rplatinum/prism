"""Shared vocabulary, label maps and the curated gene panel."""

from __future__ import annotations

# The 20 canonical amino acids, in the conventional single-letter order used
# for every saturation-mutagenesis heatmap the UI renders. Order matters:
# cached score matrices are indexed by position in this tuple.
AA_ALPHABET: tuple[str, ...] = (
    "A", "C", "D", "E", "F", "G", "H", "I", "K", "L",
    "M", "N", "P", "Q", "R", "S", "T", "V", "W", "Y",
)
AA_TO_IDX: dict[str, int] = {aa: i for i, aa in enumerate(AA_ALPHABET)}

# Grouping used to order rows in the heatmap and to colour the axis labels.
AA_GROUPS: dict[str, str] = {
    "A": "hydrophobic", "V": "hydrophobic", "L": "hydrophobic", "I": "hydrophobic",
    "M": "hydrophobic", "F": "aromatic", "W": "aromatic", "Y": "aromatic",
    "S": "polar", "T": "polar", "N": "polar", "Q": "polar", "C": "special",
    "G": "special", "P": "special", "D": "acidic", "E": "acidic",
    "K": "basic", "R": "basic", "H": "basic",
}

# Three-letter -> one-letter, for parsing ClinVar HGVS protein strings such as
# "NP_000050.2:p.Val1736Ala".
THREE_TO_ONE: dict[str, str] = {
    "Ala": "A", "Arg": "R", "Asn": "N", "Asp": "D", "Cys": "C",
    "Gln": "Q", "Glu": "E", "Gly": "G", "His": "H", "Ile": "I",
    "Leu": "L", "Lys": "K", "Met": "M", "Phe": "F", "Pro": "P",
    "Ser": "S", "Thr": "T", "Trp": "W", "Tyr": "Y", "Val": "V",
}

# ClinVar clinical significance -> binary label. Anything not listed here
# (VUS, conflicting, drug response, ...) is held out as unlabelled and is what
# the deployed model is actually useful for.
PATHOGENIC_TERMS: frozenset[str] = frozenset({
    "pathogenic",
    "likely pathogenic",
    "pathogenic/likely pathogenic",
})
BENIGN_TERMS: frozenset[str] = frozenset({
    "benign",
    "likely benign",
    "benign/likely benign",
})

# Review status ranked by confidence; we keep the star rating as a sample
# weight so single-submitter assertions count for less during training.
REVIEW_STARS: dict[str, int] = {
    "practice guideline": 4,
    "reviewed by expert panel": 3,
    "criteria provided, multiple submitters, no conflicts": 2,
    "criteria provided, single submitter": 1,
    "criteria provided, conflicting classifications": 1,
    "criteria provided, conflicting interpretations": 1,
    "no assertion criteria provided": 0,
    "no classification provided": 0,
    "no assertion provided": 0,
}

# Curated panel of clinically dense genes: cancer predisposition, cardiac,
# metabolic, neuro/developmental and pharmacogenomic. Chosen for ClinVar
# missense density so that every gene contributes usable P/B labels.
_GENE_PANEL_RAW: tuple[str, ...] = (
    # Hereditary cancer
    "BRCA1", "BRCA2", "TP53", "PTEN", "STK11", "CDH1", "PALB2", "ATM", "CHEK2",
    "BARD1", "BRIP1", "RAD51C", "RAD51D", "NBN", "MLH1", "MSH2", "MSH6", "PMS2",
    "EPCAM", "APC", "MUTYH", "SMAD4", "BMPR1A", "VHL", "RET", "MEN1", "NF1",
    "NF2", "TSC1", "TSC2", "RB1", "SDHA", "SDHB", "SDHC", "SDHD", "FH", "FLCN",
    "MITF", "CDKN2A", "CDK4", "POLE", "POLD1", "AXIN2", "GREM1", "PTCH1",
    "SUFU", "DICER1", "WT1", "KIT", "PDGFRA", "ALK", "PHOX2B",
    # Cardiovascular
    "MYH7", "MYBPC3", "TNNT2", "TNNI3", "TPM1", "ACTC1", "MYL2", "MYL3",
    "PRKAG2", "GLA", "LMNA", "PKP2", "DSP", "DSG2", "DSC2", "JUP", "TMEM43",
    "RYR2", "CASQ2", "KCNQ1", "KCNH2", "SCN5A", "ANK2", "KCNE1", "KCNE2",
    "KCNJ2", "CACNA1C", "FBN1", "TGFBR1", "TGFBR2", "SMAD3", "ACTA2", "MYH11",
    "COL3A1", "MYLK", "LDLR", "APOB", "PCSK9",
    # Metabolic / mitochondrial / storage
    "CFTR", "PAH", "GAA", "GBA", "HFE", "SERPINA1", "ATP7B", "G6PD", "GALT",
    "OTC", "ASS1", "ACADM", "ACADVL", "CPT2", "HADHA", "MMACHC", "MTHFR",
    "IDUA", "GLB1", "HEXA", "NPC1", "SMPD1", "ABCD1", "PEX1", "POLG",
    # Neuro / developmental
    "SCN1A", "SCN2A", "SCN8A", "KCNQ2", "CDKL5", "MECP2", "STXBP1", "GRIN2A",
    "GRIN2B", "SYNGAP1", "ARID1B", "CHD7", "CHD8", "SHANK3", "FMR1", "UBE3A",
    "TSC2", "DMD", "SMN1", "SOD1", "TARDBP", "FUS", "C9orf72", "APP", "PSEN1",
    "PSEN2", "MAPT", "GRN", "LRRK2", "SNCA", "PARK7", "PINK1", "PRKN", "HTT",
    "ATXN1", "ATXN2", "ATXN3", "PMP22", "MPZ", "GJB1", "GDAP1",
    # Hearing / vision / other
    "GJB2", "USH2A", "MYO7A", "ABCA4", "RPGR", "RHO", "CRB1", "BEST1", "RS1",
    "OPA1", "PAX6", "COL4A5", "COL2A1", "COL1A1", "COL1A2", "FGFR2", "FGFR3",
    "RUNX2", "TCOF1",
    # Immune / haematologic / pharmacogenomic
    "F8", "F9", "VWF", "HBB", "HBA1", "SERPINC1", "PROC", "PROS1", "ITGA2B",
    "BTK", "WAS", "IL2RG", "ADA", "RAG1", "RAG2", "NLRP3", "MEFV", "TNFRSF1A",
    "DPYD", "TPMT", "NUDT15", "CYP2C19", "CYP2D6", "SLCO1B1", "RYR1", "CACNA1S",
)

# Deduplicated, order-preserving. Genes legitimately belong to more than one
# clinical grouping above (TSC2 is both a tumour-suppressor and a neuro gene),
# so the literal list is allowed to repeat and we collapse it here.
GENE_PANEL: tuple[str, ...] = tuple(dict.fromkeys(_GENE_PANEL_RAW))


def normalise_significance(raw: str) -> str | None:
    """Map a raw ClinVar significance string to 'pathogenic' / 'benign' / None.

    ClinVar packs several assertions into one field, e.g.
    "Pathogenic/Likely pathogenic|risk factor". We take the leading clause and
    refuse anything that mixes pathogenic and benign evidence.
    """
    if not raw:
        return None
    head = raw.split("|")[0].strip().lower().rstrip(".")
    if head in PATHOGENIC_TERMS:
        return "pathogenic"
    if head in BENIGN_TERMS:
        return "benign"
    return None


def review_stars(raw: str) -> int:
    return REVIEW_STARS.get((raw or "").strip().lower(), 0)
