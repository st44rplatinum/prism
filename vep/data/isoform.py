"""Reconcile ClinVar residue numbering with the UniProt canonical isoform.

ClinVar submitters number a protein change against whichever RefSeq transcript
they used. When that transcript is a different isoform from the UniProt
canonical entry, every variant in the gene is displaced by the same constant -
MECP2 is the textbook case: the canonical UniProt entry is isoform e2 while
clinical reporting overwhelmingly uses e1, which has a different N-terminus.

Naively dropping wild-type mismatches would therefore delete ~90% of the
variants in exactly the genes clinicians care most about. Instead we search for
the single integer shift that best reconciles the gene's reported wild-type
residues with the canonical sequence, and accept it only when the evidence is
overwhelming.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class OffsetFit:
    gene: str
    offset: int
    match_rate_before: float
    match_rate_after: float
    n_variants: int
    accepted: bool

    @property
    def gain(self) -> float:
        return self.match_rate_after - self.match_rate_before


def _match_rate(seq: str, positions: np.ndarray, wt: np.ndarray, shift: int) -> float:
    """Fraction of variants whose reported WT residue matches seq at pos+shift."""
    idx = positions + shift            # positions are already 0-based
    valid = (idx >= 0) & (idx < len(seq))
    if not valid.any():
        return 0.0
    arr = np.frombuffer(seq.encode("ascii"), dtype="S1")
    hits = arr[idx[valid]] == wt[valid]
    # Denominator is all variants, not just in-range ones: a shift that pushes
    # half the gene off the end of the sequence must not look good.
    return float(hits.sum()) / float(len(positions))


def fit_offset(
    gene: str,
    seq: str,
    positions0: np.ndarray,
    wt_aa: np.ndarray,
    max_shift: int = 250,
    min_variants: int = 5,
    min_rate: float = 0.80,
    min_gain: float = 0.25,
) -> OffsetFit:
    """Find the best constant numbering shift for one gene.

    Accepted only if the shift lifts the match rate above `min_rate` and gains
    at least `min_gain` over no shift - a weak improvement is far more likely to
    be coincidence than a real isoform difference.
    """
    wt_bytes = np.frombuffer("".join(wt_aa.tolist()).encode("ascii"), dtype="S1")
    base = _match_rate(seq, positions0, wt_bytes, 0)

    if len(positions0) < min_variants:
        return OffsetFit(gene, 0, base, base, len(positions0), False)

    best_shift, best_rate = 0, base
    for shift in range(-max_shift, max_shift + 1):
        if shift == 0:
            continue
        rate = _match_rate(seq, positions0, wt_bytes, shift)
        if rate > best_rate:
            best_shift, best_rate = shift, rate

    accepted = (
        best_shift != 0
        and best_rate >= min_rate
        and (best_rate - base) >= min_gain
    )
    return OffsetFit(gene, best_shift if accepted else 0, base, best_rate, len(positions0), accepted)


def fit_all(df, proteins: dict, **kwargs) -> dict[str, "PiecewiseFit"]:
    """Fit a piecewise numbering offset per gene.

    `df` needs columns gene, pos0, wt_aa. Genes that need no correction come
    back with a single zero-shift segment and accepted=False.
    """
    fits: dict[str, PiecewiseFit] = {}
    for gene, grp in df.groupby("gene", sort=False):
        rec = proteins.get(gene)
        if rec is None:
            continue
        fits[gene] = fit_piecewise(
            gene,
            rec["sequence"],
            grp["pos0"].to_numpy(dtype=np.int64),
            grp["wt_aa"].to_numpy(dtype=object),
            **kwargs,
        )
    return fits


# --------------------------------------------------------------------------
# Piecewise offsets
#
# A single constant shift only models isoforms that differ at one terminus.
# Genes whose alternative transcript skips or includes an *internal* exon
# (SCN5A, ARID1B, OPA1, PAX6) need the offset to change partway along the
# sequence. We model the offset as a piecewise-constant function of residue
# position and infer it with a Viterbi pass: each variant either matches under
# the current shift or it does not, and switching shift costs a fixed penalty.
# The penalty is what keeps this honest - a switch has to explain several
# mismatches at once to be worth taking, so noise cannot fragment a gene into
# a patchwork of shifts that each "explain" one variant by luck.
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class PiecewiseFit:
    gene: str
    breakpoints: list[tuple[int, int]]   # (first pos0 of segment, shift)
    match_rate_before: float
    match_rate_after: float
    n_variants: int
    n_segments: int
    accepted: bool


def _candidate_shifts(
    seq: str, positions0: np.ndarray, wt_bytes: np.ndarray, max_shift: int, top_k: int
) -> list[int]:
    """Shifts worth considering, ranked by how many variants each explains."""
    scored: list[tuple[int, int]] = []
    arr = np.frombuffer(seq.encode("ascii"), dtype="S1")
    for shift in range(-max_shift, max_shift + 1):
        idx = positions0 + shift
        valid = (idx >= 0) & (idx < len(arr))
        if not valid.any():
            continue
        hits = int((arr[idx[valid]] == wt_bytes[valid]).sum())
        scored.append((hits, shift))
    scored.sort(key=lambda t: (-t[0], abs(t[1])))
    cands = [s for _, s in scored[:top_k]]
    if 0 not in cands:
        cands.append(0)
    return cands


def fit_piecewise(
    gene: str,
    seq: str,
    positions0: np.ndarray,
    wt_aa: np.ndarray,
    max_shift: int = 250,
    top_k: int = 8,
    switch_penalty: float = 6.0,
    min_variants: int = 10,
    min_rate: float = 0.80,
    min_gain: float = 0.15,
) -> PiecewiseFit:
    """Infer a piecewise-constant numbering offset for one gene."""
    order = np.argsort(positions0, kind="stable")
    pos = positions0[order]
    wt_bytes = np.frombuffer("".join(wt_aa[order].tolist()).encode("ascii"), dtype="S1")
    arr = np.frombuffer(seq.encode("ascii"), dtype="S1")
    n = len(pos)

    base = _match_rate(seq, positions0, np.frombuffer("".join(wt_aa.tolist()).encode("ascii"), dtype="S1"), 0)
    if n < min_variants:
        return PiecewiseFit(gene, [(0, 0)], base, base, n, 1, False)

    cands = _candidate_shifts(seq, pos, wt_bytes, max_shift, top_k)
    k = len(cands)

    # emission[j, i] = 0 if candidate j explains variant i, else 1
    emission = np.ones((k, n), dtype=np.float64)
    for j, shift in enumerate(cands):
        idx = pos + shift
        valid = (idx >= 0) & (idx < len(arr))
        ok = np.zeros(n, dtype=bool)
        ok[valid] = arr[idx[valid]] == wt_bytes[valid]
        emission[j, ok] = 0.0

    # Viterbi over shift states along the position-ordered variant sequence.
    # State = which candidate shift is in force. Staying is free; switching
    # costs `switch_penalty`, so a new segment must explain that many extra
    # mismatches before it is worth opening.
    cost = emission[:, 0].copy()
    back = np.zeros((k, n), dtype=np.int64)
    all_states = np.arange(k)

    for i in range(1, n):
        prev = cost
        # Cheapest predecessor that is *not* state j, via the two smallest costs.
        order2 = np.argsort(prev, kind="stable")
        best_j, best_v = int(order2[0]), float(prev[order2[0]])
        if k > 1:
            second_j, second_v = int(order2[1]), float(prev[order2[1]])
        else:
            second_j, second_v = best_j, np.inf

        other_val = np.where(all_states == best_j, second_v, best_v)
        other_arg = np.where(all_states == best_j, second_j, best_j)

        switch_cost = other_val + switch_penalty
        stay_cost = prev
        take_switch = switch_cost < stay_cost

        back[:, i] = np.where(take_switch, other_arg, all_states)
        cost = np.where(take_switch, switch_cost, stay_cost) + emission[:, i]

    # Backtrace
    path = np.zeros(n, dtype=np.int64)
    path[-1] = int(np.argmin(cost))
    for i in range(n - 1, 0, -1):
        path[i - 1] = back[path[i], i]

    shifts_per_variant = np.array([cands[j] for j in path], dtype=np.int64)

    # Collapse to breakpoints
    breakpoints: list[tuple[int, int]] = [(int(pos[0]), int(shifts_per_variant[0]))]
    for i in range(1, n):
        if shifts_per_variant[i] != shifts_per_variant[i - 1]:
            breakpoints.append((int(pos[i]), int(shifts_per_variant[i])))

    idx_after = pos + shifts_per_variant
    valid = (idx_after >= 0) & (idx_after < len(arr))
    hits = np.zeros(n, dtype=bool)
    hits[valid] = arr[idx_after[valid]] == wt_bytes[valid]
    after = float(hits.sum()) / n

    accepted = after >= min_rate and (after - base) >= min_gain
    return PiecewiseFit(
        gene=gene,
        breakpoints=breakpoints if accepted else [(0, 0)],
        match_rate_before=base,
        match_rate_after=after,
        n_variants=n,
        n_segments=len(breakpoints) if accepted else 1,
        accepted=accepted,
    )


def apply_piecewise(positions0: np.ndarray, breakpoints: list[tuple[int, int]]) -> np.ndarray:
    """Map 0-based positions through a piecewise-constant offset function."""
    starts = np.array([b[0] for b in breakpoints], dtype=np.int64)
    shifts = np.array([b[1] for b in breakpoints], dtype=np.int64)
    seg = np.searchsorted(starts, positions0, side="right") - 1
    seg = np.clip(seg, 0, len(shifts) - 1)
    return positions0 + shifts[seg]
