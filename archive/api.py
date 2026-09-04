from fastapi import FastAPI, Query
from pathlib import Path

import yaml
CONFIG_PATH = Path(__file__).parent / "configs" / "650m.yaml"
with CONFIG_PATH.open("r", encoding="utf-8") as f:
    config = yaml.safe_load(f)

app = FastAPI()

def classify_query(q: str) -> Literal["symbol", "ascension", "hgvs", "text"]:
    q = q.strip()

    if ":" in q or "." in q and any(
        x in q.lower()
        for x in ("c.", "g.", "p.", "r.", "n.", "m.", "r")
    ):
        return "hgvs"

    if q.upper().startswith(
        ("NM_", "NR_", "NC_", "NG_", "NP_", "ENST", "ENSP")
    ):
        return "accession"

    if q.replace("-", "").isalnum() and q.upper() == q:
        return "symbol"

    return "text"

@app.get("/genes")
def get_genes():
    return {"genes": config["genes"]}

@app.get("/genes/{symbol}")
def get_genes(symbol: str):
    for gene in config["genes"]:
        if gene["symbol"].upper() == symbol.upper():
            return gene

    return {"error": "Gene not found"}

@app.get("/autocomplete")
def autocomplete(
    q: str = Query(..., min_length=1),
    limit: int = Query(10, ge=1, le=50),
):
    q = q.strip()
    query_type = classify_query(q)

    results = []

    if query_type == "symbol":
        results = [
            {
                "type": "gene",
                "symbol": gene["symbol"],
                "name": gene["name"],
            }
            for gene in GENES
            if gene["symbol"].upper().startswith(q.upper())
        ]

    elif query_type == "accession":
        results = [
            {
                "type": "accession",
                "accession": accession,
            }
            for accession in ACCESSIONS
            if accession.upper().startswith(q.upper())
        ]

    elif query_type == "hgvs":
        # In production, query an HGVS/variant database.
        results = [
            {
                "type": "variant",
                "input": q,
                "message": "HGVS variant lookup",
            }
        ]

    else:
        # General text search across symbols and names.
        q_lower = q.lower()

        results = [
            {
                "type": "gene",
                "symbol": gene["symbol"],
                "name": gene["name"],
            }
            for gene in GENES
            if q_lower in gene["symbol"].lower()
            or q_lower in gene["name"].lower()
        ]

    return {
        "query": q,
        "query_type": query_type,
        "results": results[:limit],
    }

@app.get("/search")
def search_genes(
    q: str = Query(..., min_length=1),
):
    return {
        "query": q,
        "message": f"Searching for genes with query: {q}",
    }

class PredictRequest(BaseModel):
    gene: str
    variants: list[str] = Field(min_length=1)
    scheme: str


class VariantPrediction(BaseModel):
    variant: str
    pathogenicity_prob: float
    calibrated_confidence: float
    llr: dict[str, float]
    clinvar_label: str | None = None


class PredictResponse(BaseModel):
    gene: str
    scheme: str
    results: list[VariantPrediction]


class HGVSRequest(BaseModel):
    hgvs: str


def predict_batch(
    gene: str,
    variants: list[str],
    scheme: str,
) -> list[VariantPrediction]:
    """
    Run the model once for the gene/protein, then score all variants.

    Replace this with your actual model implementation.
    """

    # Important: tile/load the protein ONCE here.
    # protein = load_and_tile_protein(gene)

    results = []

    for variant in variants:
        # prediction = model.predict(protein, variant, scheme)

        # Placeholder values:
        results.append(
            VariantPrediction(
                variant=variant,
                pathogenicity_prob=0.0,
                calibrated_confidence=0.0,
                llr={
                    "scheme_1": 0.0,
                    "scheme_2": 0.0,
                    "scheme_3": 0.0,
                },
                clinvar_label=None,
            )
        )

    return results


@app.post("/predict", response_model=PredictResponse)
def predict(request: PredictRequest):
    return PredictResponse(
        gene=request.gene,
        scheme=request.scheme,
        results=predict_batch(
            gene=request.gene,
            variants=request.variants,
            scheme=request.scheme,
        ),
    )


@app.post("/predict/hgvs")
def predict_hgvs(request: HGVSRequest):
    """
    Accept free-text HGVS such as:
        NM_000546.6:p.Arg175His
    """

    hgvs = request.hgvs.strip()

    if not hgvs:
        raise HTTPException(
            status_code=400,
            detail="HGVS string cannot be empty",
        )

    gene = "TP53"
    variant = "R175H"

    results = predict_batch(
        gene=gene,
        variants=[variant],
        scheme="default",
    )

    return {
        "input": hgvs,
        "gene": gene,
        "variant": variant,
        "result": results[0],
    }

