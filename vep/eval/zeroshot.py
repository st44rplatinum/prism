"""Zero-shot variant scoring: no training, no labels, just ESM-2 likelihoods.

This is the number every supervised model in the project has to beat. It is
common for a head trained on frozen features to only just match zero-shot
scoring on held-out *genes*, so establishing this baseline before any training
is what makes the later comparison meaningful.

Scores produced (all oriented so that HIGHER means MORE pathogenic):

  esm_wt        -LLR from the wild-type-marginal distribution. One forward pass
                per window, so effectively free.
  esm_masked    -LLR with the position masked (Meier et al. 2021). One forward
                pass per residue - hours, not minutes.
  blosum62      negated BLOSUM62 substitution score. A 1992 substitution matrix
                with no notion of position or context. Included deliberately:
                if a 650M-parameter language model cannot clear this, something
                is wrong with the pipeline rather than with the model.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

from vep.config import Config, resolve_device
from vep.constants import AA_ALPHABET, AA_TO_IDX

# BLOSUM62, in AA_ALPHABET order (ACDEFGHIKLMNPQRSTVWY).
_BLOSUM62_ROWS = """
 4  0 -2 -1 -2  0 -2 -1 -1 -1 -1 -2 -1 -1 -1  1  0  0 -3 -2
 0  9 -3 -4 -2 -3 -3 -1 -3 -1 -1 -3 -3 -3 -3 -1 -1 -1 -2 -2
-2 -3  6  2 -3 -1 -1 -3 -1 -4 -3  1 -1  0 -2  0 -1 -3 -4 -3
-1 -4  2  5 -3 -2  0 -3  1 -3 -2  0 -1  2  0  0 -1 -2 -3 -2
-2 -2 -3 -3  6 -3 -1  0 -3  0  0 -3 -4 -3 -3 -2 -2 -1  1  3
 0 -3 -1 -2 -3  6 -2 -4 -2 -4 -3  0 -2 -2 -2  0 -2 -3 -2 -3
-2 -3 -1  0 -1 -2  8 -3 -1 -3 -2  1 -2  0  0 -1 -2 -3 -2  2
-1 -1 -3 -3  0 -4 -3  4 -3  2  1 -3 -3 -3 -3 -2 -1  3 -3 -1
-1 -3 -1  1 -3 -2 -1 -3  5 -2 -1  0 -1  1  2  0 -1 -2 -3 -2
-1 -1 -4 -3  0 -4 -3  2 -2  4  2 -3 -3 -2 -2 -2 -1  1 -2 -1
-1 -1 -3 -2  0 -3 -2  1 -1  2  5 -2 -2  0 -1 -1 -1  1 -1 -1
-2 -3  1  0 -3  0  1 -3  0 -3 -2  6 -2  0  0  1  0 -3 -4 -2
-1 -3 -1 -1 -4 -2 -2 -3 -1 -3 -2 -2  7 -1 -2 -1 -1 -2 -4 -3
-1 -3  0  2 -3 -2  0 -3  1 -2  0  0 -1  5  1  0 -1 -2 -2 -1
-1 -3 -2  0 -3 -2  0 -3  2 -2 -1  0 -2  1  5 -1 -1 -3 -3 -2
 1 -1  0  0 -2  0 -1 -2  0 -2 -1  1 -1  0 -1  4  1 -2 -3 -2
 0 -1 -1 -1 -2 -2 -2 -1 -1 -1 -1  0 -1 -1 -1  1  5  0 -2 -2
 0 -1 -3 -2 -1 -3 -3  3 -2  1  1 -3 -2 -2 -3 -2  0  4 -3 -1
-3 -2 -4 -3  1 -2 -2 -3 -3 -2 -1 -4 -4 -2 -3 -3 -2 -3 11  2
-2 -2 -3 -2  3 -3  2 -1 -2 -1 -1 -2 -3 -1 -2 -2 -2 -1  2  7
"""
BLOSUM62 = np.array(
    [[int(x) for x in row.split()] for row in _BLOSUM62_ROWS.strip().splitlines()],
    dtype=np.float32,
)


def blosum_score(wt_aa: str, mut_aa: str) -> float:
    return float(BLOSUM62[AA_TO_IDX[wt_aa], AA_TO_IDX[mut_aa]])


def score_dataset(
    cfg: Config,
    scheme: str = "wt",
    genes: list[str] | None = None,
    batch_size: int = 16,
    verbose: bool = True,
    checkpoint_dir: Path | None = None,
) -> pd.DataFrame:
    """Score every labelled variant under one scheme.

    Proteins are processed one at a time and each is tiled exactly once, so the
    cost is one pass per window (wt) or one pass per distinct variant position
    (masked) - never one pass per variant.

    If `checkpoint_dir` is given, each gene's scores are written as soon as
    that gene finishes and completed genes are skipped on a later call. The
    masked scheme takes about four hours across the panel, which is long enough
    that a closed laptop or an ended session would otherwise throw all of it
    away - as happened once already.
    """
    from vep.esm.backbone import ESM2Backbone

    processed = cfg.paths.processed
    variants = pd.read_parquet(processed / "variants.parquet")
    proteins = json.loads((processed / "proteins.json").read_text(encoding="utf-8"))

    if genes is not None:
        variants = variants[variants["gene"].isin(genes)]

    backbone = ESM2Backbone(
        model_name=cfg.backbone.name,
        device=resolve_device(cfg.backbone.device),
        fp16=cfg.backbone.fp16,
        max_length=cfg.backbone.max_length,
        embed_layer=cfg.backbone.embed_layer,
    )

    # Thermal gate: honour a pause requested by vep.gpu_guard between batches.
    # Without this the guard can only protect the card by suspending the whole
    # process from outside, which is the mode that can strand a job if the
    # guard itself is killed.
    from vep.gpu_guard import wait_if_paused

    paused_total = 0.0

    def thermal_gate(done: int, total: int) -> None:
        nonlocal paused_total
        paused_total += wait_if_paused()

    col = f"esm_{scheme}"
    scores = np.full(len(variants), np.nan, dtype=np.float32)
    gene_list = sorted(variants["gene"].unique())
    positions_of = variants.groupby("gene").indices

    done: set[str] = set()
    if checkpoint_dir is not None:
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        for part in checkpoint_dir.glob("*.parquet"):
            cached = pd.read_parquet(part)
            gene = part.stem
            if gene not in positions_of:
                continue
            idx = positions_of[gene]
            # Re-align on the variant key rather than trusting row order, so a
            # checkpoint stays valid even if the dataset is rebuilt.
            key = ["pos0", "wt_aa", "mut_aa"]
            merged = variants.iloc[idx].merge(cached[key + [col]], on=key, how="left")
            scores[idx] = merged[col].to_numpy(dtype=np.float32)
            done.add(gene)
        if done and verbose:
            print(f"  resuming: {len(done)} genes already scored, "
                  f"{len(gene_list) - len(done)} remaining", flush=True)

    t_start = time.time()
    n_todo = len([g for g in gene_list if g not in done])
    n_done_now = 0
    for i, gene in enumerate(gene_list, 1):
        if gene in done:
            continue
        idx = positions_of[gene]
        sub = variants.iloc[idx]
        sequence = proteins[gene]["sequence"]

        wait_if_paused()
        if scheme == "wt":
            log_probs, _ = backbone.wt_marginals(sequence)
        elif scheme == "masked":
            wanted = sorted(set(sub["pos0"].tolist()))
            log_probs = backbone.masked_marginals(
                sequence, positions=wanted, batch_size=batch_size,
                progress=thermal_gate,
            )
        else:
            raise ValueError(f"unsupported scheme {scheme!r}")

        pos0 = sub["pos0"].to_numpy()
        wt_idx = np.array([AA_TO_IDX[a] for a in sub["wt_aa"]], dtype=np.int64)
        mut_idx = np.array([AA_TO_IDX[a] for a in sub["mut_aa"]], dtype=np.int64)
        llr = log_probs[pos0, mut_idx] - log_probs[pos0, wt_idx]
        # Negated so that higher = more pathogenic, matching the label
        # orientation every metric in this project assumes.
        scores[idx] = -llr

        if checkpoint_dir is not None:
            part = sub[["gene", "pos0", "wt_aa", "mut_aa"]].copy()
            part[col] = -llr
            part.to_parquet(checkpoint_dir / f"{gene}.parquet", index=False)

        n_done_now += 1
        if verbose:
            elapsed = time.time() - t_start
            rate = n_done_now / max(elapsed, 1e-9)
            eta = (n_todo - n_done_now) / max(rate, 1e-9)
            print(
                f"  [{n_done_now:3d}/{n_todo}] {gene:10s} L={len(sequence):5d} "
                f"n={len(idx):4d}  elapsed={elapsed/60:5.1f}m  eta={eta/60:5.1f}m"
                + (f"  (+{paused_total:.0f}s thermal)" if paused_total else ""),
                flush=True,
            )

    out = variants.copy()
    out[col] = scores
    return out


def add_blosum(df: pd.DataFrame) -> pd.DataFrame:
    """Negated BLOSUM62 score, oriented higher = more pathogenic."""
    df = df.copy()
    wt_idx = np.array([AA_TO_IDX[a] for a in df["wt_aa"]], dtype=np.int64)
    mut_idx = np.array([AA_TO_IDX[a] for a in df["mut_aa"]], dtype=np.int64)
    df["blosum62"] = -BLOSUM62[wt_idx, mut_idx]
    return df


def merge_scores(base: pd.DataFrame, other: pd.DataFrame, col: str) -> pd.DataFrame:
    """Attach a score column from another run, joined on the variant key."""
    key = ["gene", "pos0", "wt_aa", "mut_aa"]
    return base.merge(other[key + [col]], on=key, how="left")


def save(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path, index=False)
