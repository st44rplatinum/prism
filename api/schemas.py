"""Pydantic response models for the API.

Kept deliberately explicit about what is *not* yet available: until a
pathogenicity head is trained, `pathogenicity_prob` is None rather than a
placeholder number. A zero that reads as a confident "benign" is the most
dangerous default this service could return.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

Scheme = Literal["wt", "masked", "mutant"]
QueryKind = Literal["symbol", "accession", "hgvs", "protein_change", "text"]


class GeneSummary(BaseModel):
    symbol: str
    accession: str
    protein_name: str
    length: int
    n_variants: int
    n_pathogenic: int
    n_benign: int
    split: str | None = None
    has_saturation: bool = False
    saturation_schemes: list[str] = Field(default_factory=list)


class GeneDetail(GeneSummary):
    entry_name: str
    sequence: str


class VariantRecord(BaseModel):
    """A ClinVar variant we hold a label for."""

    variant: str
    wt_aa: str
    position: int
    mut_aa: str
    label: Literal["pathogenic", "benign"]
    review_stars: int
    n_submitters: int
    variation_id: str | None = None
    phenotypes: str | None = None
    split: str | None = None


class VariantPrediction(BaseModel):
    variant: str
    wt_aa: str
    position: int
    mut_aa: str
    # log-likelihood ratios keyed by scheme name; always populated
    llr: dict[str, float]
    # Calibrated probability from the ProteinNPT head. None only when no
    # checkpoint is loaded - never a placeholder number, because a 0.0 here
    # reads as a confident "benign".
    pathogenicity_prob: float | None = None
    npt_logit: float | None = None
    # Which neighbour protocol produced this prediction. Returned explicitly
    # because it changes the answer and the accuracy it should be trusted at:
    # transductive means the gene contributed labelled variants of its own.
    protocol: str | None = None
    n_context: int | None = None
    n_same_gene_context: int | None = None
    clinvar_label: str | None = None
    clinvar_stars: int | None = None
    error: str | None = None


class PredictRequest(BaseModel):
    gene: str
    variants: list[str] = Field(min_length=1, max_length=512)
    # Defaults to wt, not masked, on measured evidence: across all 29,088
    # labelled variants the two schemes correlate at Spearman 0.988 and masked
    # gains only +0.005 per-gene AUROC, for 27x the compute (112 min vs 4 min
    # over the panel). Masked remains available and is worth it for a one-off
    # saturation scan, but it is the wrong default for an interactive endpoint.
    scheme: Scheme = "wt"


class PredictResponse(BaseModel):
    gene: str
    scheme: Scheme
    model_available: bool
    n_requested: int
    n_scored: int
    results: list[VariantPrediction]


class HGVSRequest(BaseModel):
    hgvs: str = Field(min_length=1)
    scheme: Scheme = "masked"


class SearchHit(BaseModel):
    kind: QueryKind
    symbol: str | None = None
    accession: str | None = None
    protein_name: str | None = None
    variant: str | None = None
    label: str | None = None


class SearchResponse(BaseModel):
    query: str
    query_kind: QueryKind
    results: list[SearchHit]


class SaturationResponse(BaseModel):
    gene: str
    scheme: Scheme
    # "llr" = log-likelihood ratio (negative = damaging, diverging around 0)
    # "probability" = calibrated pathogenicity in [0,1]; wild-type cells null
    value: str = "llr"
    length: int
    alphabet: list[str]
    sequence: str
    # (L, 20) log-likelihood ratios, row-major. Wild-type entry of each row
    # is 0 by construction.
    matrix: list[list[float | None]]


class JobStatus(BaseModel):
    job_id: str
    gene: str
    scheme: Scheme
    status: Literal["queued", "running", "done", "error"]
    progress: float = 0.0
    detail: str | None = None


class HealthResponse(BaseModel):
    status: str
    backbone: str
    backbone_slug: str
    device: str
    backbone_loaded: bool
    n_genes: int
    n_variants: int
    pathogenicity_model: str | None = None
    pathogenicity_model_loaded: bool = False
    test_per_gene_auroc: float | None = None
    # cold | warming | ready | failed
    warm_state: str = "cold"
    warm_timings: dict[str, float | str] = Field(default_factory=dict)
    warm_detail: str | None = None
