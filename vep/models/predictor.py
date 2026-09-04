"""Serve ProteinNPT predictions for arbitrary variants.

Two things make inference different from training and are worth stating plainly.

1. A query variant is usually NOT in ClinVar - that is the entire point, since
   the useful case is a VUS. Its feature blocks therefore have to be built from
   scratch: embeddings and wild-type log-probabilities come from the frozen
   cache, but the masked-marginal score does not exist for an arbitrary
   position and has to be computed with a live forward pass. Substituting the
   cheap wt-marginal instead would be a train/serve feature mismatch; the two
   correlate at 0.988 so the damage would be small, silent, and impossible to
   notice from the outside, which is exactly the kind of bug worth avoiding.

2. The model needs labelled neighbours at inference time. Which neighbours it
   gets changes the answer, so the protocol used is returned alongside every
   prediction rather than left implicit.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

from vep.config import Config, resolve_device
from vep.constants import AA_TO_IDX
from vep.esm.cache import FeatureCache, cache_path
from vep.models.npt import LABEL_MASK, ProteinNPT, load_state_dict_compat
from vep.train.datasets import build_features, label_tokens, split_indices, to_tensors


@dataclass
class Calibration:
    """Platt scaling: p = sigmoid(a * logit + b)."""

    a: float = 1.0
    b: float = 0.0
    fitted_on: str = "none"
    ece_before: float | None = None
    ece_after: float | None = None

    def apply(self, logits: np.ndarray) -> np.ndarray:
        return 1.0 / (1.0 + np.exp(-(self.a * logits + self.b)))


def expected_calibration_error(probs: np.ndarray, y: np.ndarray, n_bins: int = 10) -> float:
    """Mean |confidence - accuracy| over equal-width probability bins."""
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    total = 0.0
    for i in range(n_bins):
        m = (probs >= edges[i]) & (probs < edges[i + 1] if i < n_bins - 1 else probs <= 1.0)
        if not m.any():
            continue
        total += m.mean() * abs(probs[m].mean() - y[m].mean())
    return float(total)


@dataclass
class Prediction:
    pathogenicity_prob: float
    logit: float
    n_context: int
    n_same_gene_context: int
    protocol: str


class VariantPredictor:
    """Loads a trained NPT and scores arbitrary substitutions."""

    def __init__(
        self,
        cfg: Config,
        checkpoint_path: str | Path = "artifacts/models/npt.pt",
        calibration_path: str | Path = "artifacts/calibration.json",
        device: str | None = None,
    ):
        self.cfg = cfg
        self.device = device or resolve_device(cfg.backbone.device)
        ckpt = torch.load(checkpoint_path, map_location=self.device, weights_only=False)

        if ckpt.get("backbone") != cfg.backbone.name:
            # Features are namespaced by backbone; serving a 150M-trained head
            # from a 650M cache would silently produce nonsense.
            raise ValueError(
                f"checkpoint was trained on {ckpt.get('backbone')!r} but the "
                f"config asks for {cfg.backbone.name!r}"
            )

        self.feats = build_features(cfg)
        self.idx = split_indices(self.feats)
        self.genes = sorted(self.feats.frame["gene"].unique())
        self.gene_to_idx = {g: i for i, g in enumerate(self.genes)}

        self.model = ProteinNPT(
            d_emb=self.feats.d_emb,
            d_model=cfg.npt.d_model,
            n_layers=cfg.npt.n_layers,
            n_heads=cfg.npt.n_heads,
            dropout=cfg.npt.dropout,
        ).to(self.device)
        load_state_dict_compat(self.model, ckpt["state_dict"])
        self.model.eval()

        self.cache = FeatureCache(cache_path(cfg))
        self.meta = ckpt.get("meta", {})

        self.calibration = Calibration()
        p = Path(calibration_path)
        if p.exists():
            self.calibration = Calibration(**json.loads(p.read_text(encoding="utf-8")))

        # Labelled pool used as neighbours. Training-split variants only: the
        # val and test genes are held out and must not leak into a served
        # prediction, or the reported held-out metrics stop describing the
        # thing that is actually deployed.
        self.context_pool = self.idx["train"]
        self._pool_by_gene: dict[int, np.ndarray] = {}
        for g in np.unique(self.feats.gene_idx[self.context_pool]):
            self._pool_by_gene[int(g)] = self.context_pool[
                self.feats.gene_idx[self.context_pool] == g
            ]
        self._rng = np.random.default_rng(0)

        # Row lookup so a query can be excluded from its own context. The query
        # is built as a fresh feature row rather than referenced by index, so
        # without this an existing ClinVar variant could be sampled as one of
        # its own labelled neighbours and read its answer straight out of the
        # context - the precise leak the masked label column exists to prevent.
        self._row_of: dict[tuple[str, int, str], int] = {}
        fr = self.feats.frame
        for row, (g, pos0, mut) in enumerate(
            zip(fr["gene"], fr["pos0"], fr["mut_aa"])
        ):
            self._row_of[(g, int(pos0), mut)] = row

    # -- features for a novel variant ---------------------------------------
    def _gene_context(self, gene: str) -> np.ndarray:
        if gene in self.gene_to_idx:
            return self.feats.gene_context[self.gene_to_idx[gene]]
        return self.cache.embeddings(gene).astype(np.float32).mean(axis=0)

    def build_query_features(
        self,
        gene: str,
        changes: list[tuple[str, int, str]],   # (wt, pos0, mut)
        masked_llr: np.ndarray,
    ) -> dict[str, np.ndarray]:
        from vep.eval.zeroshot import BLOSUM62

        pos0 = np.array([c[1] for c in changes], dtype=np.int64)
        wt_idx = np.array([AA_TO_IDX[c[0]] for c in changes], dtype=np.int64)
        mut_idx = np.array([AA_TO_IDX[c[2]] for c in changes], dtype=np.int64)

        subst = np.zeros((len(changes), 42), dtype=np.float32)
        subst[np.arange(len(changes)), wt_idx] = 1.0
        subst[np.arange(len(changes)), 20 + mut_idx] = 1.0
        subst[:, 40] = masked_llr / 10.0
        subst[:, 41] = BLOSUM62[wt_idx, mut_idx] / 10.0

        ctx = self._gene_context(gene)
        return {
            "emb": self.cache.embeddings(gene, pos0).astype(np.float32),
            "logprobs": self.cache.log_probs(gene, pos0).astype(np.float32),
            "subst": subst,
            "context": np.tile(ctx, (len(changes), 1)).astype(np.float32),
        }

    # -- prediction ---------------------------------------------------------
    @torch.no_grad()
    def predict(
        self,
        gene: str,
        changes: list[tuple[str, int, str]],
        masked_llr: np.ndarray,
        n_context: int = 31,
        use_same_gene_context: bool = True,
    ) -> list[Prediction]:
        """Score substitutions in one gene. `masked_llr` is oriented higher = more pathogenic."""
        if not changes:
            return []
        q = self.build_query_features(gene, changes, masked_llr)

        gidx = self.gene_to_idx.get(gene)
        same_all = (
            self._pool_by_gene.get(gidx, np.array([], dtype=np.int64))
            if gidx is not None
            else np.array([], dtype=np.int64)
        )

        out: list[Prediction] = []
        for i, (wt_aa, pos0, mut_aa) in enumerate(changes):
            # Drop this exact substitution from the candidate neighbours.
            self_row = self._row_of.get((gene, int(pos0), mut_aa))
            same = same_all if self_row is None else same_all[same_all != self_row]
            pool = (
                self.context_pool
                if self_row is None
                else self.context_pool[self.context_pool != self_row]
            )

            n_same = 0
            if use_same_gene_context and len(same) > 0:
                take = min(n_context, len(same))
                ctx = self._rng.choice(same, size=take, replace=False)
                n_same = take
                if take < n_context:
                    ctx = np.concatenate(
                        [ctx, self._rng.choice(pool, n_context - take, replace=False)]
                    )
            else:
                ctx = self._rng.choice(pool, size=n_context, replace=False)

            ctx_batch = to_tensors(self.feats, ctx, self.device)
            query_batch = {
                k: torch.as_tensor(v[i : i + 1], device=self.device) for k, v in q.items()
            }
            batch = {k: torch.cat([query_batch[k], ctx_batch[k]]) for k in query_batch}

            labels = torch.cat(
                [
                    torch.tensor([LABEL_MASK], device=self.device, dtype=torch.long),
                    label_tokens(self.feats.labels[ctx], self.device),
                ]
            )
            logit = float(self.model(**batch, labels=labels).pathogenicity_logit[0])
            out.append(
                Prediction(
                    pathogenicity_prob=float(self.calibration.apply(np.array([logit]))[0]),
                    logit=logit,
                    n_context=len(ctx),
                    n_same_gene_context=n_same,
                    protocol="transductive" if n_same > 0 else "inductive",
                )
            )
        return out

    @torch.no_grad()
    def predict_many(
        self,
        gene: str,
        changes: list[tuple[str, int, str]],
        masked_llr: np.ndarray,
        n_context: int = 31,
        batch_size: int = 16,
        use_same_gene_context: bool = True,
        progress=None,
    ) -> np.ndarray:
        """Batched prediction: many masked queries share one labelled context.

        Returns calibrated probabilities, one per change. Used for saturation
        scans, where predicting 7,000+ cells one at a time would take a minute
        instead of a few seconds.

        `batch_size` is capped deliberately. Masked queries cannot leak labels
        to each other (that is what the mask is for), but they do change the
        proportion of masked rows the model sees, and the model was trained at
        roughly 15% masked. Measured drift against one-at-a-time prediction:
        mean |delta logit| is 0.05 at K=4, 0.13 at K=16, and 0.25 at K=64,
        while Spearman correlation stays above 0.998 throughout. Ranking is
        therefore safe at any K, but a heat map colours by value, so K=16 keeps
        the error near 0.015 in probability while still being ~50x faster than
        scoring one at a time.
        """
        if not changes:
            return np.zeros(0, dtype=np.float32)

        gidx = self.gene_to_idx.get(gene)
        same_all = (
            self._pool_by_gene.get(gidx, np.array([], dtype=np.int64))
            if gidx is not None
            else np.array([], dtype=np.int64)
        )
        pool = same_all if (use_same_gene_context and len(same_all) >= n_context) else self.context_pool

        logits = np.zeros(len(changes), dtype=np.float32)
        for start in range(0, len(changes), batch_size):
            chunk = changes[start : start + batch_size]
            k = len(chunk)
            # Exclude any query in this chunk from its own context.
            self_rows = {
                self._row_of.get((gene, int(pos0), mut))
                for _, pos0, mut in chunk
            } - {None}
            ctx_pool = pool[~np.isin(pool, list(self_rows))] if self_rows else pool
            ctx = self._rng.choice(
                ctx_pool, size=min(n_context, len(ctx_pool)), replace=False
            )

            q = self.build_query_features(gene, chunk, masked_llr[start : start + k])
            qb = {kk: torch.as_tensor(v, device=self.device) for kk, v in q.items()}
            cb = to_tensors(self.feats, ctx, self.device)
            batch = {kk: torch.cat([qb[kk], cb[kk]]) for kk in qb}
            labels = torch.cat(
                [
                    torch.full((k,), LABEL_MASK, device=self.device, dtype=torch.long),
                    label_tokens(self.feats.labels[ctx], self.device),
                ]
            )
            logits[start : start + k] = (
                self.model(**batch, labels=labels).pathogenicity_logit[:k].cpu().numpy()
            )
            if progress is not None:
                progress(min(start + k, len(changes)), len(changes))

        return self.calibration.apply(logits).astype(np.float32)

    def close(self) -> None:
        self.cache.close()


def saturation_probabilities(
    predictor: VariantPredictor,
    gene: str,
    sequence: str,
    masked_llr: np.ndarray,
    batch_size: int = 16,
    progress=None,
) -> np.ndarray:
    """(L, 20) calibrated probabilities for every possible substitution.

    `masked_llr` is the masked-marginal LLR matrix in its natural orientation
    (log p(mut) - log p(wt), so negative means damaging); it is negated here to
    match the pathogenic orientation the model was trained on.

    Wild-type cells come back as NaN. "The probability that this residue is
    pathogenic when unchanged" is not a question the model answers, and filling
    those cells with 0 would render them as confidently benign.

    Shared by the API and the offline precompute job so the two can never
    disagree about how a cached matrix was produced.
    """
    from vep.constants import AA_ALPHABET

    changes: list[tuple[str, int, str]] = []
    cells: list[tuple[int, int]] = []
    for pos0, wt in enumerate(sequence):
        if wt not in AA_ALPHABET:
            continue
        for j, aa in enumerate(AA_ALPHABET):
            if aa == wt:
                continue
            changes.append((wt, pos0, aa))
            cells.append((pos0, j))

    llr = np.array([-masked_llr[p, j] for p, j in cells], dtype=np.float32)
    probs = predictor.predict_many(
        gene, changes, llr, batch_size=batch_size, progress=progress
    )

    out = np.full((len(sequence), 20), np.nan, dtype=np.float32)
    for (p, j), v in zip(cells, probs):
        out[p, j] = v
    return out
