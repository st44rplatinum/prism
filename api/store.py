"""In-process data store backing the API.

At 29k variants and 183 proteins everything fits comfortably in memory, so the
store is a pair of pandas frames plus a dict of sequences. A database here
would be ceremony, not engineering.

The ESM-2 backbone is loaded lazily on first use: importing the app should not
pull ~600 MB of weights onto the GPU, which keeps `--reload` usable during
front-end work and lets the catalog endpoints serve with no GPU at all.
"""

from __future__ import annotations

import json
import re
import threading
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from vep.config import Config, resolve_device
from vep.constants import AA_ALPHABET, AA_TO_IDX
from vep.esm.saturation import SaturationStore, fingerprint_file

# "R175H" and friends. Position is 1-based, matching how clinicians write it.
PROTEIN_CHANGE_RE = re.compile(r"^([A-Z])(\d+)([A-Z])$")


@dataclass
class ParsedChange:
    wt_aa: str
    position: int   # 1-based
    mut_aa: str

    @property
    def pos0(self) -> int:
        return self.position - 1

    def __str__(self) -> str:
        return f"{self.wt_aa}{self.position}{self.mut_aa}"


def parse_protein_change(text: str) -> ParsedChange | None:
    """Parse 'R175H' into its parts, rejecting non-standard amino acids."""
    m = PROTEIN_CHANGE_RE.match(text.strip().upper())
    if m is None:
        return None
    wt, pos, mut = m.group(1), int(m.group(2)), m.group(3)
    if wt not in AA_TO_IDX or mut not in AA_TO_IDX or pos < 1:
        return None
    return ParsedChange(wt, pos, mut)


class Store:
    def __init__(self, cfg: Config):
        self.cfg = cfg
        processed = cfg.paths.processed

        # Resolved once, at construction. torch.cuda.is_available() performs the
        # first CUDA context probe, which costs ~3s on this machine - fine once
        # at startup, but it was previously being paid on every /health call.
        self.device = resolve_device(cfg.backbone.device)

        self.proteins: dict[str, dict] = json.loads(
            (processed / "proteins.json").read_text(encoding="utf-8")
        )
        self.variants: pd.DataFrame = pd.read_parquet(processed / "variants.parquet")

        report_path = processed / "build_report.json"
        self.build_report: dict = (
            json.loads(report_path.read_text(encoding="utf-8"))
            if report_path.exists()
            else {}
        )

        # Per-gene aggregates, computed once. The catalog endpoint is hit on
        # every page load and must not re-group 29k rows each time.
        agg = (
            self.variants.assign(is_path=self.variants["label"].eq("pathogenic"))
            .groupby("gene")
            .agg(
                n_variants=("label", "size"),
                n_pathogenic=("is_path", "sum"),
                split=("split", "first"),
            )
        )
        agg["n_benign"] = agg["n_variants"] - agg["n_pathogenic"]
        self._gene_stats = agg.to_dict("index")

        # Only genes we have BOTH a sequence and labels for are servable.
        self.genes: list[str] = sorted(
            g for g in self.proteins if g in self._gene_stats
        )
        self._by_accession = {
            rec["accession"].upper(): g
            for g, rec in self.proteins.items()
            if g in self._gene_stats
        }

        self._backbone = None
        self._backbone_lock = threading.Lock()
        # Saturation matrices persist across restarts. The probability cache is
        # additionally keyed to the checkpoint, so a retrain misses rather than
        # serving stale predictions - the fingerprint is filled in when the
        # model loads, since hashing 18 MB is not worth doing if nobody asks
        # for a probability.
        # The fingerprint is computed here rather than when the predictor
        # loads. It keys the probability cache directory, so deferring it made
        # /genes report zero cached probabilities until the model finished
        # loading in the background - the same request returning different
        # answers depending on how soon after startup it arrived. Hashing an
        # 18 MB checkpoint costs ~50 ms, once.
        _ckpt = cfg.paths.models / "npt.pt"
        self._saturation = SaturationStore(
            root=cfg.paths.cache / "saturation",
            backbone_slug=cfg.backbone_slug(),
            model_fingerprint=fingerprint_file(_ckpt) if _ckpt.exists() else None,
        )
        self._predictor = None
        self._predictor_lock = threading.Lock()
        self._predictor_failed: str | None = None

        # Warm-up state, readable from /health while it runs in the background.
        self._warm_state = "cold"          # cold | warming | ready | failed
        self._warm_detail: str | None = None
        self._warm_timings: dict[str, float] = {}

    # -- catalog ------------------------------------------------------------
    def has_gene(self, symbol: str) -> bool:
        return symbol.upper() in self._gene_stats and symbol.upper() in self.proteins

    def gene_record(self, symbol: str) -> dict | None:
        key = symbol.upper()
        rec = self.proteins.get(key)
        if rec is None or key not in self._gene_stats:
            return None
        stats = self._gene_stats[key]
        return {
            "symbol": key,
            "accession": rec["accession"],
            "entry_name": rec["entry_name"],
            "protein_name": rec["protein_name"],
            "sequence": rec["sequence"],
            "length": rec["length"],
            "n_variants": int(stats["n_variants"]),
            "n_pathogenic": int(stats["n_pathogenic"]),
            "n_benign": int(stats["n_benign"]),
            "split": stats["split"],
            "has_saturation": bool(self._saturation.schemes_for(key)),
            "saturation_schemes": self._saturation.schemes_for(key),
        }

    def gene_by_accession(self, accession: str) -> str | None:
        return self._by_accession.get(accession.upper())

    def sequence(self, symbol: str) -> str | None:
        rec = self.proteins.get(symbol.upper())
        return rec["sequence"] if rec else None

    def gene_variants(self, symbol: str) -> pd.DataFrame:
        return self.variants[self.variants["gene"] == symbol.upper()]

    def lookup_label(self, symbol: str, change: ParsedChange) -> tuple[str, int] | None:
        """ClinVar label for a substitution, if we hold one.

        Matched on `pos0`, the isoform-reconciled 0-based position, not on the
        raw ClinVar position - the stored frame has already been shifted onto
        the canonical sequence.
        """
        rows = self.variants[
            (self.variants["gene"] == symbol.upper())
            & (self.variants["pos0"] == change.pos0)
            & (self.variants["mut_aa"] == change.mut_aa)
        ]
        if rows.empty:
            return None
        row = rows.iloc[0]
        return str(row["label"]), int(row["stars"])

    # -- backbone -----------------------------------------------------------
    @property
    def backbone_loaded(self) -> bool:
        return self._backbone is not None

    def backbone(self):
        """Load ESM-2 on first use. Thread-safe: uvicorn serves from a pool."""
        if self._backbone is None:
            with self._backbone_lock:
                if self._backbone is None:
                    from vep.esm.backbone import ESM2Backbone

                    self._backbone = ESM2Backbone(
                        model_name=self.cfg.backbone.name,
                        device=self.device,
                        fp16=self.cfg.backbone.fp16,
                        max_length=self.cfg.backbone.max_length,
                        embed_layer=self.cfg.backbone.embed_layer,
                    )
        return self._backbone

    # -- pathogenicity model ------------------------------------------------
    @property
    def predictor_loaded(self) -> bool:
        return self._predictor is not None

    def predictor(self):
        """Lazily load the trained ProteinNPT head.

        Returns None if no checkpoint exists, so the API still serves zero-shot
        LLRs on a fresh clone rather than refusing to start.
        """
        if self._predictor is None and self._predictor_failed is None:
            with self._predictor_lock:
                if self._predictor is None and self._predictor_failed is None:
                    from vep.models.predictor import VariantPredictor

                    ckpt = self.cfg.paths.models / "npt.pt"
                    if not ckpt.exists():
                        self._predictor_failed = f"no checkpoint at {ckpt}"
                        return None
                    try:
                        self._predictor = VariantPredictor(
                            self.cfg,
                            checkpoint_path=ckpt,
                            calibration_path=self.cfg.paths.models.parent / "calibration.json",
                        )
                        n = self._saturation.prune_stale_probabilities()
                        if n:
                            print(f"[api] dropped {n} probability matrices from a "
                                  f"previous checkpoint", flush=True)
                    except Exception as exc:
                        self._predictor_failed = f"{type(exc).__name__}: {exc}"
                        return None
        return self._predictor

    # -- warm-up ------------------------------------------------------------
    @property
    def warm_state(self) -> str:
        return self._warm_state

    @property
    def warm_timings(self) -> dict[str, float]:
        return dict(self._warm_timings)

    @property
    def warm_detail(self) -> str | None:
        return self._warm_detail

    def warm(self, verbose: bool = True) -> None:
        """Pay the cold-start cost up front instead of on a user's first call.

        Measured breakdown of a cold /predict: ESM-2 weight load and the move to
        GPU is 20.5s, building the 82 MB feature table plus the checkpoint is
        4.9s, and the forward passes themselves are negligible. So warming is
        almost entirely about loading those two artefacts; the dummy passes at
        the end cost nothing and confirm the whole path works rather than just
        that the objects constructed.

        Intended to run on a background thread: the catalog endpoints need
        neither the GPU nor the model and should stay available throughout.
        """
        import time

        self._warm_state = "warming"
        try:
            t0 = time.time()
            backbone = self.backbone()
            self._warm_timings["backbone_s"] = round(time.time() - t0, 2)

            t0 = time.time()
            predictor = self.predictor()
            self._warm_timings["predictor_s"] = round(time.time() - t0, 2)

            # Exercise the real path once so the first user request cannot be
            # the thing that discovers a broken cache or a shape mismatch.
            t0 = time.time()
            probe_gene = min(self.genes, key=lambda g: self.proteins[g]["length"])
            seq = self.sequence(probe_gene)
            backbone.wt_marginals(seq)
            backbone.masked_marginals(seq, positions=[0])
            if predictor is not None:
                import numpy as np

                wt = seq[0]
                mut = "A" if wt != "A" else "G"
                predictor.predict(
                    probe_gene, [(wt, 0, mut)], np.array([0.0], dtype=np.float32)
                )
            self._warm_timings["probe_s"] = round(time.time() - t0, 2)
            self._warm_timings["probe_gene"] = probe_gene

            self._warm_state = "ready"
            if verbose:
                total = sum(v for v in self._warm_timings.values() if isinstance(v, float))
                print(f"[api] warm-up complete in {total:.1f}s {self._warm_timings}", flush=True)
        except Exception as exc:
            self._warm_state = "failed"
            self._warm_detail = f"{type(exc).__name__}: {exc}"
            if verbose:
                print(f"[api] warm-up FAILED: {self._warm_detail}", flush=True)

    # -- saturation cache ---------------------------------------------------
    def get_saturation(self, symbol: str, scheme: str) -> np.ndarray | None:
        return self._saturation.get(symbol, scheme)

    def put_saturation(self, symbol: str, scheme: str, matrix: np.ndarray) -> None:
        self._saturation.put(symbol, scheme, matrix)

    def saturation_stats(self) -> dict:
        return self._saturation.stats()

    def structure_payload(self, symbol: str) -> dict:
        """Per-residue colouring for the 3D view, plus where to fetch the model.

        The structure itself is NOT proxied. AlphaFold serves
        `Access-Control-Allow-Origin: *`, so the browser can fetch the ~250 KB
        file directly and the API stays out of the way of a large static
        download it would only be relaying.

        Colour comes from the mean predicted pathogenicity over the 19
        substitutions at each residue - the per-position summary of the same
        matrix the heat map draws. Falls back to wild-type LLR when no
        probability matrix is cached, since that is computable in under a
        second and the alternative is refusing to render.
        """
        symbol = symbol.upper()
        rec = self.gene_record(symbol)
        if rec is None:
            raise KeyError(symbol)
        seq = rec["sequence"]

        matrix = self.get_saturation(symbol, "probability")
        if matrix is not None:
            source = "probability"
            per_residue = np.nanmean(matrix, axis=1)
        else:
            llr = self.get_saturation(symbol, "wt")
            if llr is None:
                llr = self.compute_saturation_wt(symbol)
            source = "llr"
            # Mask the wild-type column, which is 0 by construction and would
            # otherwise pull every position toward the neutral end.
            masked = llr.astype(np.float32).copy()
            for i, aa in enumerate(seq):
                if aa in AA_TO_IDX:
                    masked[i, AA_TO_IDX[aa]] = np.nan
            per_residue = np.nanmean(masked, axis=1)

        vals = np.nan_to_num(per_residue, nan=float(np.nanmin(per_residue)))
        return {
            "gene": symbol,
            "accession": rec["accession"],
            "length": rec["length"],
            "sequence": seq,
            "score_type": source,
            "scores": [round(float(v), 4) for v in vals],
        }

    def compute_saturation_probability(self, symbol: str, progress=None) -> np.ndarray:
        """(L, 20) calibrated pathogenicity probabilities for every substitution.

        Requires the masked-marginal saturation matrix, because the head was
        trained on the masked score.
        """
        from vep.models.predictor import saturation_probabilities

        symbol = symbol.upper()
        seq = self.sequence(symbol)
        if seq is None:
            raise KeyError(symbol)
        masked = self.get_saturation(symbol, "masked")
        if masked is None:
            raise ValueError(f"masked saturation for {symbol} not computed yet")
        predictor = self.predictor()
        if predictor is None:
            raise ValueError("no pathogenicity model loaded")

        out = saturation_probabilities(
            predictor, symbol, seq, masked, batch_size=16, progress=progress
        )
        self.put_saturation(symbol, "probability", out)
        return out

    def compute_saturation_wt(self, symbol: str) -> np.ndarray:
        """Full (L, 20) LLR matrix from wt-marginals - one pass per window."""
        from vep.esm.backbone import llr_matrix

        seq = self.sequence(symbol)
        if seq is None:
            raise KeyError(symbol)
        log_probs, _ = self.backbone().wt_marginals(seq)
        matrix = llr_matrix(log_probs, seq)
        self.put_saturation(symbol, "wt", matrix)
        return matrix

    # -- misc ---------------------------------------------------------------
    @property
    def alphabet(self) -> list[str]:
        return list(AA_ALPHABET)

    @property
    def n_variants(self) -> int:
        return int(len(self.variants))
