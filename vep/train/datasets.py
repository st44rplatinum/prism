"""Assemble NPT batches from the frozen feature cache and the ClinVar labels.

Two evaluation protocols are supported, and the difference between them matters
more than any hyperparameter:

  inductive     the held-out gene contributes NO labelled variants. Neighbours
                are drawn from training genes only. This is the protocol that
                is directly comparable to the zero-shot baseline (0.8435
                per-gene AUROC), because zero-shot also sees no labels.

  transductive  the held-out gene contributes k labelled variants as context,
                and the rest are queries. This is NOT comparable to zero-shot -
                it has strictly more information - but it is the realistic
                clinical setting: a gene typically has some classified variants
                and a long tail of VUS, and it is the setting in which the
                non-parametric mechanism can actually help the gain-of-function
                genes that no likelihood score can rank.

Reporting only the transductive number against a zero-shot baseline would be a
straightforward apples-to-oranges comparison, so both are always produced.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from vep.config import Config
from vep.constants import AA_TO_IDX
from vep.esm.cache import FeatureCache, cache_path
from vep.models.npt import LABEL_BENIGN, LABEL_PATHOGENIC, LABEL_PAD


@dataclass
class VariantFeatures:
    """Precomputed per-variant feature blocks, held in RAM.

    At 29k variants and 640-dim embeddings this is ~80 MB in float32, so there
    is no reason to stream it from disk during training.
    """

    emb: np.ndarray            # (N, d_emb) embedding at the mutated residue
    logprobs: np.ndarray       # (N, 20)    ESM distribution at that residue
    subst: np.ndarray          # (N, 42)    wt/mut one-hots + LLR + BLOSUM
    gene_context: np.ndarray   # (G, d_emb) mean-pooled protein embedding
    gene_idx: np.ndarray       # (N,)       row -> gene
    labels: np.ndarray         # (N,)       1 = pathogenic
    weights: np.ndarray        # (N,)       review-star sample weights
    frame: pd.DataFrame        # the source rows, aligned by position

    @property
    def d_emb(self) -> int:
        return self.emb.shape[1]

    def __len__(self) -> int:
        return len(self.labels)


def build_features(
    cfg: Config,
    scores_path: str | Path = "artifacts/cache/zeroshot_all.parquet",
) -> VariantFeatures:
    """Read the HDF5 cache and assemble every feature block once."""
    from vep.eval.zeroshot import BLOSUM62

    df = pd.read_parquet(scores_path).reset_index(drop=True)
    proteins = json.loads(
        (cfg.paths.processed / "proteins.json").read_text(encoding="utf-8")
    )
    fc = FeatureCache(cache_path(cfg))

    genes = sorted(df["gene"].unique())
    gene_to_idx = {g: i for i, g in enumerate(genes)}
    d_emb = fc.hidden_size

    emb = np.zeros((len(df), d_emb), dtype=np.float32)
    logprobs = np.zeros((len(df), 20), dtype=np.float32)
    gene_context = np.zeros((len(genes), d_emb), dtype=np.float32)

    for gene, grp in df.groupby("gene"):
        rows = grp.index.to_numpy()
        pos0 = grp["pos0"].to_numpy()
        emb[rows] = fc.embeddings(gene, pos0).astype(np.float32)
        logprobs[rows] = fc.log_probs(gene, pos0).astype(np.float32)
        # Whole-protein context, so the model can tell which gene it is looking
        # at without being handed the gene identity as a categorical.
        gene_context[gene_to_idx[gene]] = (
            fc.embeddings(gene).astype(np.float32).mean(axis=0)
        )
    fc.close()

    wt_idx = np.array([AA_TO_IDX[a] for a in df["wt_aa"]], dtype=np.int64)
    mut_idx = np.array([AA_TO_IDX[a] for a in df["mut_aa"]], dtype=np.int64)
    subst = np.zeros((len(df), 42), dtype=np.float32)
    subst[np.arange(len(df)), wt_idx] = 1.0
    subst[np.arange(len(df)), 20 + mut_idx] = 1.0
    # The zero-shot score itself is a feature: it is strong on its own, and
    # making the head learn it from scratch would waste capacity. Column 40 is
    # in the project's usual orientation (higher = more pathogenic).
    subst[:, 40] = df["esm_masked"].to_numpy(dtype=np.float32) / 10.0
    # Column 41 is RAW BLOSUM62, i.e. higher = more tolerated - the opposite of
    # the orientation used for scores in vep.eval. That is deliberate and
    # harmless as a model input (the projection learns the sign), but compute
    # an AUROC on this column directly and you will get 1 - AUROC. Kept raw so
    # that trained checkpoints stay consistent with this feature builder.
    subst[:, 41] = BLOSUM62[wt_idx, mut_idx] / 10.0

    labels = (df["label"] == "pathogenic").to_numpy().astype(np.int64)
    # Review stars as sample weights: a 1-star single-submitter assertion is
    # much weaker evidence than a 3-star expert-panel review.
    stars = df["stars"].to_numpy(dtype=np.float32)
    weights = 0.5 + 0.25 * np.clip(stars, 0, 4)

    return VariantFeatures(
        emb=emb,
        logprobs=logprobs,
        subst=subst,
        gene_context=gene_context,
        gene_idx=np.array([gene_to_idx[g] for g in df["gene"]], dtype=np.int64),
        labels=labels,
        weights=weights.astype(np.float32),
        frame=df,
    )


class BatchSampler:
    """Builds NPT batches. A batch is a *set* of variants, not one variant.

    Batches are drawn from a single gene with probability `same_gene_prob`, and
    otherwise mixed across genes. Training purely on single-gene batches teaches
    the model to lean entirely on same-gene neighbours, which then collapses in
    the inductive protocol where no such neighbour exists; training purely on
    mixed batches never exercises the mechanism that makes the architecture
    worth having. The mixture is what makes it work in both settings.
    """

    def __init__(
        self,
        feats: VariantFeatures,
        indices: np.ndarray,
        batch_size: int = 32,
        same_gene_prob: float = 0.6,
        seed: int = 42,
    ):
        self.f = feats
        self.indices = indices
        self.batch_size = batch_size
        self.same_gene_prob = same_gene_prob
        self.rng = np.random.default_rng(seed)
        self.by_gene: dict[int, np.ndarray] = {}
        for g in np.unique(feats.gene_idx[indices]):
            self.by_gene[int(g)] = indices[feats.gene_idx[indices] == g]
        self.gene_ids = list(self.by_gene)

    def sample(self) -> np.ndarray:
        if self.rng.random() < self.same_gene_prob:
            g = self.gene_ids[self.rng.integers(len(self.gene_ids))]
            pool = self.by_gene[g]
            n = min(self.batch_size, len(pool))
            return self.rng.choice(pool, size=n, replace=False)
        return self.rng.choice(
            self.indices, size=min(self.batch_size, len(self.indices)), replace=False
        )

    def epoch(self, n_batches: int):
        for _ in range(n_batches):
            yield self.sample()


def to_tensors(
    feats: VariantFeatures, rows: np.ndarray, device: str
) -> dict[str, torch.Tensor]:
    t = lambda a: torch.as_tensor(a, device=device)
    return {
        "emb": t(feats.emb[rows]),
        "logprobs": t(feats.logprobs[rows]),
        "subst": t(feats.subst[rows]),
        "context": t(feats.gene_context[feats.gene_idx[rows]]),
    }


def label_tokens(labels: np.ndarray, device: str) -> torch.Tensor:
    arr = np.where(labels == 1, LABEL_PATHOGENIC, LABEL_BENIGN)
    return torch.as_tensor(arr, device=device, dtype=torch.long)


def pad_to(
    batch: dict[str, torch.Tensor],
    labels: torch.Tensor,
    target_n: int,
) -> tuple[dict[str, torch.Tensor], torch.Tensor, torch.Tensor]:
    """Pad a short batch up to `target_n` rows, returning a padding mask."""
    n = labels.shape[0]
    if n >= target_n:
        return batch, labels, torch.zeros(n, dtype=torch.bool, device=labels.device)
    pad = target_n - n
    out = {
        k: torch.cat([v, torch.zeros(pad, v.shape[1], device=v.device, dtype=v.dtype)])
        for k, v in batch.items()
    }
    labels = torch.cat(
        [labels, torch.full((pad,), LABEL_PAD, device=labels.device, dtype=torch.long)]
    )
    mask = torch.zeros(target_n, dtype=torch.bool, device=labels.device)
    mask[n:] = True
    return out, labels, mask


def split_indices(feats: VariantFeatures) -> dict[str, np.ndarray]:
    return {
        name: np.flatnonzero((feats.frame["split"] == name).to_numpy())
        for name in ("train", "val", "test")
    }
