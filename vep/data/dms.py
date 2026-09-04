"""Deep mutational scanning data: the GRB2 binding assay.

Source: Gelman et al., the METL paper's `grb2-binding.tsv`. The file carries
only `variant`, `num_mutations` and `score` - no sequence column at all, which
is what stalled the original `score_variant.py`.

The sequence does not need to be fetched, because it is recoverable from the
data. Every single mutant names its own wild-type residue ("T0A" says position
0 is T), so collecting those across all 655 singles reconstructs the assayed
region directly, and the doubles fill any position the singles miss. Doing it
this way also validates the file: 33,441 variants agreed on the wild-type
residue at every position with zero conflicts, which would not happen if the
numbering convention were anything other than what we assumed.

The reconstructed 56-mer aligns exactly at offset 158 of UniProt P62993 - the
C-terminal SH3 domain of human GRB2. DMS position p is therefore canonical
position p + 158 (0-based).
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from vep.constants import AA_TO_IDX

VARIANT_RE = re.compile(r"^([A-Z])(\d+)([A-Z])$")

GRB2_ACCESSION = "P62993"
# Offset of the assayed SH3 domain within the canonical GRB2 sequence,
# determined by exact substring match of the reconstructed wild-type.
GRB2_DOMAIN_OFFSET = 158


@dataclass
class DMSDataset:
    name: str
    frame: pd.DataFrame          # variant, num_mutations, score, subs, ...
    domain_sequence: str         # reconstructed, as assayed
    domain_offset: int           # 0-based offset into the full protein
    full_sequence: str | None = None
    accession: str | None = None

    @property
    def n_singles(self) -> int:
        return int((self.frame["num_mutations"] == 1).sum())

    @property
    def n_doubles(self) -> int:
        return int((self.frame["num_mutations"] == 2).sum())


def parse_variant(variant: str) -> list[tuple[str, int, str]] | None:
    """Parse 'T0A' or 'T0A,Y5F' into [(wt, pos0, mut), ...]."""
    subs = []
    for part in str(variant).split(","):
        m = VARIANT_RE.match(part.strip())
        if m is None:
            return None
        wt, pos, mut = m.group(1), int(m.group(2)), m.group(3)
        if wt not in AA_TO_IDX or mut not in AA_TO_IDX:
            return None
        subs.append((wt, pos, mut))
    return subs


def reconstruct_wildtype(frame: pd.DataFrame) -> tuple[str, int, int]:
    """Rebuild the assayed sequence from the variant strings.

    Returns (sequence, first_position, n_conflicts). Singles are consulted
    first because they are unambiguous; doubles then fill any gaps. A non-zero
    conflict count means two variants disagree about the wild-type residue at
    one position, which would invalidate the whole reconstruction.
    """
    wt_at: dict[int, str] = {}
    conflicts = 0

    ordered = frame.sort_values("num_mutations")   # singles first
    for variant in ordered["variant"]:
        subs = parse_variant(variant)
        if subs is None:
            continue
        for wt, pos, _ in subs:
            if pos in wt_at and wt_at[pos] != wt:
                conflicts += 1
            else:
                wt_at[pos] = wt

    if not wt_at:
        raise ValueError("no parseable variants; cannot reconstruct wild-type")

    lo, hi = min(wt_at), max(wt_at)
    gaps = [p for p in range(lo, hi + 1) if p not in wt_at]
    if gaps:
        raise ValueError(f"wild-type reconstruction has {len(gaps)} gaps at {gaps[:10]}")
    return "".join(wt_at[p] for p in range(lo, hi + 1)), lo, conflicts


def load_grb2(
    path: str | Path = "data/grb2-binding.tsv",
    full_sequence: str | None = None,
    verify_offset: bool = True,
) -> DMSDataset:
    """Load the GRB2 binding DMS, reconstructing and validating its sequence."""
    frame = pd.read_csv(path, sep="\t")
    required = {"variant", "num_mutations", "score"}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"{path} is missing columns: {sorted(missing)}")

    domain_seq, first_pos, conflicts = reconstruct_wildtype(frame)
    if conflicts:
        raise ValueError(
            f"{conflicts} wild-type conflicts in {path}; the numbering "
            f"convention is not what this loader assumes"
        )
    if first_pos != 0:
        raise ValueError(f"expected 0-based positions, first position is {first_pos}")

    offset = GRB2_DOMAIN_OFFSET
    if full_sequence is not None and verify_offset:
        found = full_sequence.find(domain_seq)
        if found < 0:
            raise ValueError(
                "reconstructed domain does not occur in the supplied full "
                "sequence; wrong protein or wrong isoform"
            )
        offset = found

    parsed = frame["variant"].map(parse_variant)
    keep = parsed.notna()
    if (~keep).any():
        print(f"  dropped {(~keep).sum()} unparseable variants")
    frame = frame[keep].copy()
    frame["subs"] = parsed[keep]
    frame["positions"] = frame["subs"].map(lambda s: [p for _, p, _ in s])
    frame["canonical_positions"] = frame["positions"].map(
        lambda ps: [p + offset for p in ps]
    )
    frame["mutation_type"] = np.where(frame["num_mutations"] == 1, "single", "double")

    return DMSDataset(
        name="grb2-binding",
        frame=frame.reset_index(drop=True),
        domain_sequence=domain_seq,
        domain_offset=offset,
        full_sequence=full_sequence,
        accession=GRB2_ACCESSION,
    )


def sample_balanced(
    ds: DMSDataset, n_per_type: int | None = None, seed: int = 42
) -> pd.DataFrame:
    """Equal-sized single and double mutant samples.

    `n_per_type=None` uses the size of the smaller class, which for GRB2 means
    655 - the number of single mutants actually assayed. The original script
    guarded on 100 of each but then asked for 1000, so on this very file it
    raised from inside pandas rather than at its own check. Defaulting to the
    limiting class removes the trap entirely; pass an explicit count only when
    you want fewer.
    """
    counts = {
        kind: int((ds.frame["mutation_type"] == kind).sum())
        for kind in ("single", "double")
    }
    available = min(counts.values())
    if available == 0:
        raise ValueError(f"one mutation class is empty: {counts}")

    if n_per_type is None:
        n_per_type = available
    elif n_per_type > available:
        raise ValueError(
            f"asked for {n_per_type} of each class but only {available} "
            f"available (counts: {counts}); pass n_per_type<={available} "
            f"or None to use all of the limiting class"
        )

    out = [
        ds.frame[ds.frame["mutation_type"] == kind].sample(
            n=n_per_type, random_state=seed
        )
        for kind in ("single", "double")
    ]
    return pd.concat(out, ignore_index=True)
