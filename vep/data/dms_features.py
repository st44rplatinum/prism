"""Feature assembly for deep mutational scanning variants.

Two things differ from the ClinVar path and both matter.

1. MULTI-MUTANTS. 98% of the GRB2 assay is double mutants, but the ClinVar
   feature builder encodes exactly one substitution. Here the per-substitution
   blocks are mean-pooled, which is permutation-invariant - a double mutant
   "A,B" must be represented identically to "B,A", since the assay does not
   distinguish them - and works for any number of substitutions.

2. EPISTASIS. Mean-pooling per-substitution features is additive by
   construction: it cannot express that two substitutions interact, which is
   precisely what a pairwise DMS measures. The mutant-marginal score supplies
   that, because every position is scored with the *other* substitutions
   already applied.

Scoring the mutant sequences is batched here rather than reusing
ESM2Backbone.mutant_marginals, which re-runs the forward pass once per
substitution. For a 217-residue protein the window is the whole sequence, so a
double mutant paid for the identical pass twice - 90 minutes across the assay
where batching does it in about one.
"""

from __future__ import annotations

import numpy as np
import torch

from vep.constants import AA_TO_IDX


@torch.no_grad()
def mutant_marginal_scores(
    backbone,
    wt_sequence: str,
    variants: list[list[tuple[int, str]]],
    batch_size: int = 32,
    progress=None,
) -> np.ndarray:
    """Summed log-likelihood ratio per variant, scored on the mutant sequence.

    `variants` is a list of substitution lists, each [(pos0, mutant_aa), ...].
    Returns (n_variants,), oriented so HIGHER means more damaging, matching the
    orientation used everywhere else in this project.

    Every substitution in a variant is applied before scoring, so each position
    is read in the context of the others - that context is the epistasis.
    """
    scores = np.zeros(len(variants), dtype=np.float32)
    for start in range(0, len(variants), batch_size):
        chunk = variants[start : start + batch_size]
        seqs = []
        for subs in chunk:
            mutant = list(wt_sequence)
            for pos, aa in subs:
                mutant[pos] = aa
            seqs.append("".join(mutant))

        batch = backbone._encode(seqs)
        logits = backbone.model(**batch).logits[:, 1:, :]
        log_probs = backbone._aa_log_probs(logits)          # (B, T, 20)

        for row, subs in enumerate(chunk):
            total = 0.0
            for pos, mut_aa in subs:
                wt_aa = wt_sequence[pos]
                lp = log_probs[row, pos]
                total += float(lp[AA_TO_IDX[mut_aa]] - lp[AA_TO_IDX[wt_aa]])
            scores[start + row] = -total                    # higher = more damaging
        if progress is not None:
            progress(min(start + len(chunk), len(variants)), len(variants))
    return scores


def build_dms_features(
    cache,
    gene: str,
    wt_sequence: str,
    variants: list[list[tuple[int, str]]],
    masked_log_probs: np.ndarray,
    mutant_llr: np.ndarray,
    blosum: np.ndarray,
    d_emb: int,
) -> dict[str, np.ndarray]:
    """Per-variant feature blocks, shaped exactly like the ClinVar ones.

    Keeping the shapes identical is what lets one model train on both tasks:
    the tokeniser cannot tell a pooled double mutant from a single substitution,
    so no architectural branch is needed.
    """
    n = len(variants)
    emb = np.zeros((n, d_emb), dtype=np.float32)
    logprobs = np.zeros((n, 20), dtype=np.float32)
    subst = np.zeros((n, 42), dtype=np.float32)

    all_emb = cache.embeddings(gene).astype(np.float32)
    for i, subs in enumerate(variants):
        positions = np.array([p for p, _ in subs], dtype=np.int64)
        # Mean, not sum: a double mutant must not look twice as extreme as a
        # single simply for having more substitutions. Count is passed
        # separately so the model can use it if it is informative.
        emb[i] = all_emb[positions].mean(axis=0)
        logprobs[i] = masked_log_probs[positions].mean(axis=0)

        for pos, mut_aa in subs:
            subst[i, AA_TO_IDX[wt_sequence[pos]]] += 1.0 / len(subs)
            subst[i, 20 + AA_TO_IDX[mut_aa]] += 1.0 / len(subs)
        subst[i, 40] = mutant_llr[i] / 10.0
        subst[i, 41] = blosum[i] / 10.0

    return {"emb": emb, "logprobs": logprobs, "subst": subst}


def split_by_position(
    variants: list[list[tuple[int, str]]],
    val_frac: float = 0.1,
    test_frac: float = 0.1,
    seed: int = 42,
    return_held_out: bool = False,
):
    """Assign variants to train/val/test by holding out whole POSITIONS.

    A random split over variants would be far too easy: with 98% double
    mutants, almost every held-out variant shares a position with something in
    training, and the model can memorise per-position effects rather than learn
    anything transferable. Holding out positions asks the harder and more
    honest question. A variant is held out if ANY of its positions is.
    """
    rng = np.random.default_rng(seed)
    all_positions = sorted({p for subs in variants for p, _ in subs})
    shuffled = rng.permutation(all_positions)
    n_test = max(1, int(round(test_frac * len(shuffled))))
    n_val = max(1, int(round(val_frac * len(shuffled))))
    test_pos = set(shuffled[:n_test].tolist())
    val_pos = set(shuffled[n_test : n_test + n_val].tolist())

    out = np.empty(len(variants), dtype=object)
    for i, subs in enumerate(variants):
        positions = {p for p, _ in subs}
        if positions & test_pos:
            out[i] = "test"
        elif positions & val_pos:
            out[i] = "val"
        else:
            out[i] = "train"

    # The chosen position sets are returnable so the guarantee can actually be
    # checked. It is one-directional and worth stating precisely: no TRAINING
    # variant contains a held-out position, so those residues are genuinely
    # unseen. The converse does not hold - a double mutant spanning one
    # training and one held-out position lands in test, and therefore shares
    # its partner position with training.
    if return_held_out:
        return out, {"val": val_pos, "test": test_pos}
    return out
