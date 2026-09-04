"""FastAPI app for the ESM-2 variant effect predictor.

Run with:
    uvicorn api.main:app --reload --port 8000
"""

from __future__ import annotations

import os
import threading
from contextlib import asynccontextmanager

import numpy as np
from fastapi import Depends, FastAPI, HTTPException, Query, Response
from fastapi.middleware.cors import CORSMiddleware

from api import search as search_mod
from api.jobs import JobRegistry
from api.schemas import (
    GeneDetail,
    GeneSummary,
    HealthResponse,
    HGVSRequest,
    JobStatus,
    PredictRequest,
    PredictResponse,
    SaturationResponse,
    SearchResponse,
    VariantPrediction,
    VariantRecord,
)
from api.store import Store, parse_protein_change
from vep.config import Config, resolve_device

CONFIG_PATH = os.environ.get("VEP_CONFIG", "configs/default.yaml")

state: dict = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    cfg = Config.load(CONFIG_PATH)
    store = Store(cfg)
    state["cfg"] = cfg
    state["store"] = store
    state["jobs"] = JobRegistry(store)
    print(
        f"[api] {len(store.genes)} genes, {store.n_variants:,} variants, "
        f"backbone={cfg.backbone.name}"
    )

    # Warm on a background thread rather than inline: a synchronous warm would
    # block startup for ~25s, which makes `uvicorn --reload` unusable during
    # front-end work and looks like a hang. Catalog and search endpoints work
    # immediately; /predict is simply slow until the thread finishes, exactly
    # as it was before, and /health reports the state.
    # Set VEP_NO_WARM=1 to skip entirely (no GPU touched at all).
    if os.environ.get("VEP_NO_WARM") != "1":
        threading.Thread(target=store.warm, name="vep-warm", daemon=True).start()
    else:
        print("[api] VEP_NO_WARM=1, skipping warm-up")

    yield
    state.clear()


app = FastAPI(
    title="ESM-2 Variant Effect Predictor",
    version="0.1.0",
    lifespan=lifespan,
)

# The Vue dev server runs on a different origin; without this every browser
# request fails CORS preflight before it reaches a route.
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "http://127.0.0.1:5173",
        "http://localhost:4173",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def get_store() -> Store:
    store = state.get("store")
    if store is None:
        raise HTTPException(503, "store not initialised")
    return store


def get_jobs() -> JobRegistry:
    jobs = state.get("jobs")
    if jobs is None:
        raise HTTPException(503, "job registry not initialised")
    return jobs


# ---------------------------------------------------------------------------
# health & model card
# ---------------------------------------------------------------------------
@app.get("/health", response_model=HealthResponse)
def health(store: Store = Depends(get_store)) -> HealthResponse:
    cfg: Config = state["cfg"]
    return HealthResponse(
        status="ok",
        backbone=cfg.backbone.name,
        # Reported explicitly: the feature cache is namespaced by backbone, and
        # serving 150M-cached features from a 650M-configured API is a silent
        # failure that is otherwise very hard to spot.
        backbone_slug=cfg.backbone_slug(),
        device=store.device,
        backbone_loaded=store.backbone_loaded,
        n_genes=len(store.genes),
        n_variants=store.n_variants,
        # Reads the cached flag; never calls store.predictor(), which would
        # make a health check block for seconds while it loads the model - and
        # would race the background warm thread for the same lock.
        pathogenicity_model="ProteinNPT" if store.predictor_loaded else None,
        pathogenicity_model_loaded=store.predictor_loaded,
        warm_state=store.warm_state,
        warm_timings=store.warm_timings,
        warm_detail=store.warm_detail,
        test_per_gene_auroc=0.9271 if store.predictor_loaded else None,
    )


@app.get("/metrics")
def metrics(store: Store = Depends(get_store)) -> dict:
    """Evaluation results for the dashboard.

    Served from the file written by `python -m vep.eval.report`, which needs the
    GPU to recompute per-gene ProteinNPT scores. Deliberately not computed on
    request: a metrics page should never be able to start a multi-minute job.
    """
    import json as _json

    path = state["cfg"].paths.figures.parent / "metrics.json"
    if not path.exists():
        raise HTTPException(
            503,
            f"no metrics report at {path}; run `python -m vep.eval.report`",
        )
    return _json.loads(path.read_text(encoding="utf-8"))


@app.get("/model")
def model_card(store: Store = Depends(get_store)) -> dict:
    cfg: Config = state["cfg"]
    return {
        "backbone": cfg.backbone.name,
        "backbone_slug": cfg.backbone_slug(),
        "embed_layer": cfg.backbone.embed_layer,
        "max_length": cfg.backbone.max_length,
        "pathogenicity_model": {
            "architecture": "ProteinNPT (Notin et al., NeurIPS 2023)",
            "loaded": store.predictor_loaded,
            "test_per_gene_auroc_inductive": 0.9271,
            "test_per_gene_auroc_transductive": 0.9196,
            "zero_shot_baseline_per_gene_auroc": 0.8435,
            "calibration": "Platt scaling fitted on validation; test ECE 0.097 -> 0.052",
        },
        "scoring_schemes": {
            "wt": "one forward pass on the wild-type sequence",
            "masked": "position masked before scoring (Meier et al. 2021); strongest",
            "mutant": "full mutant sequence scored; captures epistasis",
        },
        "saturation_cache": store.saturation_stats(),
        "dataset": store.build_report,
        "splits": {
            "policy": "held-out by gene, not by variant",
            "fractions": {
                "train": 1 - cfg.data.test_fraction - cfg.data.val_fraction,
                "val": cfg.data.val_fraction,
                "test": cfg.data.test_fraction,
            },
        },
    }


# ---------------------------------------------------------------------------
# catalog
# ---------------------------------------------------------------------------
@app.get("/genes", response_model=list[GeneSummary])
def list_genes(
    split: str | None = Query(None, pattern="^(train|val|test)$"),
    store: Store = Depends(get_store),
) -> list[GeneSummary]:
    out = []
    for symbol in store.genes:
        rec = store.gene_record(symbol)
        if split and rec["split"] != split:
            continue
        out.append(
            GeneSummary(
                **{
                    k: v
                    for k, v in rec.items()
                    if k not in ("sequence", "entry_name")
                }
            )
        )
    return out


@app.get("/genes/{symbol}", response_model=GeneDetail)
def get_gene(symbol: str, store: Store = Depends(get_store)) -> GeneDetail:
    rec = store.gene_record(symbol)
    if rec is None:
        # A real 404, not a 200 carrying an {"error": ...} body - clients
        # cannot branch on status codes that never vary.
        raise HTTPException(404, f"unknown gene {symbol!r}")
    return GeneDetail(**rec)


@app.get("/genes/{symbol}/variants", response_model=list[VariantRecord])
def gene_variants(
    symbol: str,
    label: str | None = Query(None, pattern="^(pathogenic|benign)$"),
    min_stars: int = Query(0, ge=0, le=4),
    store: Store = Depends(get_store),
) -> list[VariantRecord]:
    if not store.has_gene(symbol):
        raise HTTPException(404, f"unknown gene {symbol!r}")
    df = store.gene_variants(symbol)
    df = df[df["stars"] >= min_stars]
    if label:
        df = df[df["label"] == label]
    return [
        VariantRecord(
            variant=f"{r.wt_aa}{r.pos0 + 1}{r.mut_aa}",
            wt_aa=r.wt_aa,
            position=int(r.pos0) + 1,
            mut_aa=r.mut_aa,
            label=r.label,
            review_stars=int(r.stars),
            n_submitters=int(r.n_submitters),
            variation_id=str(r.variation_id) or None,
            phenotypes=(r.phenotypes or None),
            split=r.split,
        )
        for r in df.itertuples(index=False)
    ]


@app.get("/genes/{symbol}/structure")
def structure(symbol: str, store: Store = Depends(get_store)) -> dict:
    """AlphaFold model reference plus a per-residue colouring vector."""
    import requests as _requests

    try:
        payload = store.structure_payload(symbol)
    except KeyError:
        raise HTTPException(404, f"unknown gene {symbol!r}")

    # AlphaFold's file naming carries a version that changes (v4 -> v6 during
    # this project), so the URL is resolved through their API rather than
    # constructed. Resolution is cheap and cached per process.
    acc = payload["accession"]
    cached = state.setdefault("_af_urls", {})
    if acc not in cached:
        try:
            r = _requests.get(
                f"https://alphafold.ebi.ac.uk/api/prediction/{acc}", timeout=30
            )
            r.raise_for_status()
            cached[acc] = r.json()[0]["pdbUrl"]
        except Exception as exc:
            raise HTTPException(
                502, f"could not resolve an AlphaFold model for {acc}: {exc}"
            )
    payload["pdb_url"] = cached[acc]
    return payload


@app.get("/search", response_model=SearchResponse)
def search(
    q: str = Query(..., min_length=1),
    limit: int = Query(10, ge=1, le=50),
    store: Store = Depends(get_store),
) -> SearchResponse:
    kind, hits = search_mod.search(q, store, limit=limit)
    return SearchResponse(query=q.strip(), query_kind=kind, results=hits)


# ---------------------------------------------------------------------------
# prediction
# ---------------------------------------------------------------------------
def _score_variants(
    store: Store, symbol: str, variants: list[str], scheme: str
) -> list[VariantPrediction]:
    """Score a batch against one protein, tiling that protein exactly once."""
    sequence = store.sequence(symbol)
    if sequence is None:
        raise HTTPException(404, f"unknown gene {symbol!r}")

    parsed: list[tuple[str, object | None, str | None]] = []
    for raw in variants:
        change = search_mod.normalise_change(raw)
        p = parse_protein_change(change) if change else None
        if p is None:
            parsed.append((raw, None, "unparseable variant"))
        elif not 0 <= p.pos0 < len(sequence):
            parsed.append((raw, None, f"position {p.position} outside protein (length {len(sequence)})"))
        elif sequence[p.pos0] != p.wt_aa:
            parsed.append((
                raw, None,
                f"wild-type mismatch: canonical sequence has "
                f"{sequence[p.pos0]} at {p.position}, not {p.wt_aa}",
            ))
        else:
            parsed.append((raw, p, None))

    valid = [p for _, p, err in parsed if err is None]
    positions = sorted({p.pos0 for p in valid})

    from vep.esm.backbone import llr

    backbone = store.backbone()
    log_probs_by_scheme: dict[str, np.ndarray] = {}
    masked_lp = None
    if positions:
        # wt-marginals cost one pass per window and cover every position, so
        # they are always computed and always reported as a baseline.
        wt_lp, _ = backbone.wt_marginals(sequence)
        log_probs_by_scheme["wt"] = wt_lp
        # Masked-marginals are computed whenever the caller asked for them OR
        # the trained head is available, because the head was trained on the
        # masked score. Feeding it wt instead would be a train/serve mismatch
        # that degrades predictions silently.
        predictor = store.predictor()
        if scheme == "masked" or predictor is not None:
            masked_lp = backbone.masked_marginals(
                sequence, positions=positions, batch_size=16
            )
            if scheme == "masked":
                log_probs_by_scheme["masked"] = masked_lp

    # Run the trained head over all valid variants in one go.
    predictions_by_change: dict[tuple[int, str], object] = {}
    predictor = store.predictor()
    if predictor is not None and valid and masked_lp is not None:
        changes = [(p.wt_aa, p.pos0, p.mut_aa) for p in valid]
        masked_llr = np.array(
            [-llr(masked_lp, p.pos0, p.wt_aa, p.mut_aa) for p in valid],
            dtype=np.float32,
        )
        preds = predictor.predict(symbol, changes, masked_llr)
        for p, pred in zip(valid, preds):
            predictions_by_change[(p.pos0, p.mut_aa)] = pred

    results: list[VariantPrediction] = []
    for raw, p, err in parsed:
        if err is not None or p is None:
            results.append(
                VariantPrediction(
                    variant=raw, wt_aa="", position=0, mut_aa="", llr={}, error=err
                )
            )
            continue

        scores = {
            name: llr(matrix, p.pos0, p.wt_aa, p.mut_aa)
            for name, matrix in log_probs_by_scheme.items()
        }
        if scheme == "mutant":
            scores["mutant"] = backbone.score_substitution(
                sequence, p.pos0, p.wt_aa, p.mut_aa, scheme="mutant"
            )

        known = store.lookup_label(symbol, p)
        pred = predictions_by_change.get((p.pos0, p.mut_aa))
        results.append(
            VariantPrediction(
                variant=str(p),
                wt_aa=p.wt_aa,
                position=p.position,
                mut_aa=p.mut_aa,
                llr=scores,
                pathogenicity_prob=pred.pathogenicity_prob if pred else None,
                npt_logit=pred.logit if pred else None,
                protocol=pred.protocol if pred else None,
                n_context=pred.n_context if pred else None,
                n_same_gene_context=pred.n_same_gene_context if pred else None,
                clinvar_label=known[0] if known else None,
                clinvar_stars=known[1] if known else None,
            )
        )
    return results


@app.post("/predict", response_model=PredictResponse)
def predict(req: PredictRequest, store: Store = Depends(get_store)) -> PredictResponse:
    symbol = req.gene.upper()
    if not store.has_gene(symbol):
        raise HTTPException(404, f"unknown gene {req.gene!r}")
    results = _score_variants(store, symbol, req.variants, req.scheme)
    return PredictResponse(
        gene=symbol,
        scheme=req.scheme,
        model_available=store.predictor() is not None,
        n_requested=len(req.variants),
        n_scored=sum(1 for r in results if r.error is None),
        results=results,
    )


@app.post("/predict/hgvs", response_model=PredictResponse)
def predict_hgvs(req: HGVSRequest, store: Store = Depends(get_store)) -> PredictResponse:
    """Resolve a free-text HGVS or 'GENE CHANGE' string, then score it."""
    kind, hits = search_mod.search(req.hgvs, store, limit=1)
    if not hits or not hits[0].get("symbol") or not hits[0].get("variant"):
        raise HTTPException(
            422,
            f"could not resolve {req.hgvs!r} to a gene and protein change "
            f"(classified as {kind})",
        )
    symbol, variant = hits[0]["symbol"], hits[0]["variant"]
    results = _score_variants(store, symbol, [variant], req.scheme)
    return PredictResponse(
        gene=symbol,
        scheme=req.scheme,
        model_available=store.predictor() is not None,
        n_requested=1,
        n_scored=sum(1 for r in results if r.error is None),
        results=results,
    )


# ---------------------------------------------------------------------------
# saturation
# ---------------------------------------------------------------------------
@app.get(
    "/genes/{symbol}/saturation",
    # No single response_model: this route returns a matrix (200) or a job
    # handle (202), and declaring one of them makes FastAPI reject the other
    # during response validation.
    response_model=None,
    responses={
        200: {"model": SaturationResponse, "description": "Saturation matrix"},
        202: {"model": JobStatus, "description": "Masked scan started; poll /jobs/{id}"},
    },
)
def saturation(
    symbol: str,
    response: Response,
    scheme: str = Query("wt", pattern="^(wt|masked)$"),
    value: str = Query("llr", pattern="^(llr|probability)$"),
    store: Store = Depends(get_store),
    jobs: JobRegistry = Depends(get_jobs),
) -> SaturationResponse | JobStatus:
    if not store.has_gene(symbol):
        raise HTTPException(404, f"unknown gene {symbol!r}")
    symbol = symbol.upper()
    sequence = store.sequence(symbol)

    if value == "probability":
        # Probabilities are built on the masked matrix, so the masked scan has
        # to exist first. Once it does, scoring every cell takes a few seconds.
        if store.predictor() is None:
            raise HTTPException(503, "no pathogenicity model loaded")
        matrix = store.get_saturation(symbol, "probability")
        if matrix is None:
            if store.get_saturation(symbol, "masked") is None:
                job = jobs.submit(symbol, "masked")
                response.status_code = 202
                return JobStatus(**job.snapshot())
            matrix = store.compute_saturation_probability(symbol)
        return SaturationResponse(
            gene=symbol,
            scheme="masked",
            value="probability",
            length=len(sequence),
            alphabet=store.alphabet,
            sequence=sequence,
            matrix=[[None if np.isnan(v) else round(float(v), 4) for v in row] for row in matrix],
        )

    matrix = store.get_saturation(symbol, scheme)
    if matrix is None:
        if scheme == "wt":
            matrix = store.compute_saturation_wt(symbol)   # sub-second
        else:
            # One forward pass per residue: minutes to hours. Hand back the
            # job and let the client poll while it renders wt-marginals.
            job = jobs.submit(symbol, "masked")
            response.status_code = 202
            return JobStatus(**job.snapshot())

    return SaturationResponse(
        gene=symbol,
        scheme=scheme,
        value="llr",
        length=len(sequence),
        alphabet=store.alphabet,
        sequence=sequence,
        matrix=np.round(matrix, 4).tolist(),
    )


@app.post("/jobs/saturation", response_model=JobStatus)
def submit_saturation(
    gene: str = Query(...),
    scheme: str = Query("masked", pattern="^masked$"),
    store: Store = Depends(get_store),
    jobs: JobRegistry = Depends(get_jobs),
) -> JobStatus:
    if not store.has_gene(gene):
        raise HTTPException(404, f"unknown gene {gene!r}")
    return JobStatus(**jobs.submit(gene, scheme).snapshot())


@app.get("/jobs/{job_id}", response_model=JobStatus)
def job_status(job_id: str, jobs: JobRegistry = Depends(get_jobs)) -> JobStatus:
    job = jobs.get(job_id)
    if job is None:
        raise HTTPException(404, f"unknown job {job_id!r}")
    return JobStatus(**job.snapshot())
