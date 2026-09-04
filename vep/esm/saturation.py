"""On-disk cache for saturation matrices.

A masked-marginal scan costs one forward pass per residue - about 55 seconds for
TP53 and considerably longer for a 5,000-residue protein - so losing it on every
API restart is expensive. These matrices are also small (a 400-residue protein
is 31 KB as float32), which makes keeping them on disk an easy trade.

Correctness here is entirely about *invalidation*, because a stale matrix is
indistinguishable from a fresh one once it is loaded:

  wt, masked      depend only on the backbone, so they are namespaced by the
                  backbone slug. Features from esm2_t30_150M and esm2_t33_650M
                  mean different things and must never share a directory.

  probability     additionally depends on the trained head. Retrain the model
                  and every cached probability is wrong while still looking
                  perfectly valid, so those are namespaced by a fingerprint of
                  the checkpoint file as well. Swapping in a new checkpoint
                  simply misses the cache instead of serving old predictions.

Writes go to a temporary file and are renamed into place, so a process killed
mid-write leaves no truncated array behind.
"""

from __future__ import annotations

import hashlib
import os
from pathlib import Path

import numpy as np

VALID_SCHEMES = ("wt", "masked", "probability")


def fingerprint_file(path: str | Path, n_bytes: int = 12) -> str:
    """Short content hash of a file, used to key model-dependent caches."""
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()[:n_bytes]


class SaturationStore:
    """Two-level cache: in-memory dict in front of a directory of .npy files."""

    def __init__(
        self,
        root: Path,
        backbone_slug: str,
        model_fingerprint: str | None = None,
    ):
        self.root = Path(root)
        self.backbone_slug = backbone_slug
        self.model_fingerprint = model_fingerprint
        self._mem: dict[tuple[str, str], np.ndarray] = {}

    # -- paths --------------------------------------------------------------
    def _subdir(self, scheme: str) -> str:
        if scheme == "probability":
            # Unfingerprinted probabilities would silently survive a retrain.
            fp = self.model_fingerprint or "unknown"
            return f"probability-{fp}"
        return scheme

    def directory(self, scheme: str) -> Path:
        return self.root / self.backbone_slug / self._subdir(scheme)

    def path(self, gene: str, scheme: str) -> Path:
        return self.directory(scheme) / f"{gene.upper()}.npy"

    # -- access -------------------------------------------------------------
    def get(self, gene: str, scheme: str) -> np.ndarray | None:
        key = (gene.upper(), scheme)
        if key in self._mem:
            return self._mem[key]

        p = self.path(gene, scheme)
        if not p.exists():
            return None
        try:
            matrix = np.load(p)
        except (ValueError, OSError):
            # A corrupt file should cost one recomputation, not a 500.
            try:
                p.unlink()
            except OSError:
                pass
            return None
        self._mem[key] = matrix
        return matrix

    def put(self, gene: str, scheme: str, matrix: np.ndarray) -> None:
        if scheme not in VALID_SCHEMES:
            raise ValueError(f"unknown scheme {scheme!r}")
        key = (gene.upper(), scheme)
        self._mem[key] = matrix

        p = self.path(gene, scheme)
        p.parent.mkdir(parents=True, exist_ok=True)
        tmp = p.with_name(p.name + ".tmp")
        # Written through a file handle: np.save() appends ".npy" to any path
        # that does not already end in it, so passing "TP53.npy.tmp" would
        # silently produce "TP53.npy.tmp.npy" and the rename below would fail.
        with open(tmp, "wb") as fh:
            np.save(fh, matrix)
        os.replace(tmp, p)          # atomic within a filesystem

    def has(self, gene: str, scheme: str) -> bool:
        return (gene.upper(), scheme) in self._mem or self.path(gene, scheme).exists()

    def schemes_for(self, gene: str) -> list[str]:
        return [s for s in VALID_SCHEMES if self.has(gene, s)]

    # -- maintenance --------------------------------------------------------
    def stats(self) -> dict:
        out: dict[str, object] = {
            "root": str(self.root),
            "backbone": self.backbone_slug,
            "model_fingerprint": self.model_fingerprint,
            "in_memory": len(self._mem),
        }
        per_scheme: dict[str, int] = {}
        total_bytes = 0
        for scheme in VALID_SCHEMES:
            d = self.directory(scheme)
            files = list(d.glob("*.npy")) if d.exists() else []
            per_scheme[scheme] = len(files)
            total_bytes += sum(f.stat().st_size for f in files)
        out["on_disk"] = per_scheme
        out["disk_mb"] = round(total_bytes / 1e6, 2)
        return out

    def prune_stale_probabilities(self) -> int:
        """Delete probability caches keyed to a different checkpoint."""
        base = self.root / self.backbone_slug
        if not base.exists() or self.model_fingerprint is None:
            return 0
        keep = f"probability-{self.model_fingerprint}"
        removed = 0
        for d in base.glob("probability-*"):
            if d.name == keep:
                continue
            for f in d.glob("*.npy"):
                f.unlink()
                removed += 1
            try:
                d.rmdir()
            except OSError:
                pass
        return removed
