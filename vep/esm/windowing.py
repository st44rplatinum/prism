"""Tiling long proteins into ESM-2-sized windows.

ESM-2 is trained with a 1024-token context (1022 residues once BOS/EOS are
accounted for) and attention memory grows quadratically, which matters a lot on
a 4 GB card. Many clinically important genes are far longer than that - BRCA2 is
3418 residues, RYR1 is 5038 - so each protein is tiled into overlapping windows
and every residue is scored from the window in which it sits most centrally.
Centring matters: a residue at the edge of a window sees context on one side
only, which measurably degrades both its embedding and its masked-token
distribution.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Window:
    start: int   # 0-based, inclusive
    end: int     # 0-based, exclusive

    @property
    def length(self) -> int:
        return self.end - self.start

    @property
    def centre(self) -> float:
        return (self.start + self.end - 1) / 2.0

    def contains(self, pos0: int) -> bool:
        return self.start <= pos0 < self.end


def plan_windows(length: int, max_len: int = 1022, overlap_frac: float = 0.5) -> list[Window]:
    """Tile [0, length) into windows of at most `max_len` residues.

    Windows advance by `max_len * (1 - overlap_frac)`; the final window is
    pinned to the C-terminus so the last residues are covered without a short,
    context-starved tile.
    """
    if length <= 0:
        raise ValueError("length must be positive")
    if max_len <= 0:
        raise ValueError("max_len must be positive")
    if not 0.0 <= overlap_frac < 1.0:
        raise ValueError("overlap_frac must be in [0, 1)")

    if length <= max_len:
        return [Window(0, length)]

    stride = max(1, int(round(max_len * (1.0 - overlap_frac))))
    windows: list[Window] = []
    start = 0
    while True:
        end = min(start + max_len, length)
        windows.append(Window(start, end))
        if end >= length:
            break
        start += stride

    # Pin the last window to the C-terminus and drop any window it subsumes.
    tail = Window(length - max_len, length)
    windows = [w for w in windows if not (w.start >= tail.start and w.end <= tail.end)]
    windows.append(tail)
    return sorted(set(windows), key=lambda w: w.start)


def assign_windows(length: int, windows: list[Window]) -> list[int]:
    """For each 0-based residue, the index of the window that scores it.

    Ties and edges resolve to the window whose centre is nearest, so a residue
    is always read from its most context-rich tile.
    """
    if not windows:
        raise ValueError("no windows supplied")
    assignment: list[int] = []
    for pos in range(length):
        best_idx, best_dist = -1, float("inf")
        for idx, w in enumerate(windows):
            if not w.contains(pos):
                continue
            dist = abs(pos - w.centre)
            if dist < best_dist:
                best_idx, best_dist = idx, dist
        if best_idx < 0:
            raise ValueError(f"residue {pos} is not covered by any window")
        assignment.append(best_idx)
    return assignment


def window_for_position(length: int, pos0: int, max_len: int = 1022) -> Window:
    """The single best-centred window containing `pos0`.

    Used when scoring one variant on demand, where tiling the whole protein
    would be wasted work.
    """
    if not 0 <= pos0 < length:
        raise ValueError(f"position {pos0} outside protein of length {length}")
    if length <= max_len:
        return Window(0, length)
    half = max_len // 2
    start = min(max(0, pos0 - half), length - max_len)
    return Window(start, start + max_len)
