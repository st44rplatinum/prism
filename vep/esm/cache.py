"""Frozen-feature cache: per-residue ESM-2 embeddings and log-probabilities.

ProteinNPT trains on frozen backbone features, so the expensive forward passes
happen exactly once and every subsequent training run reads from disk. That is
what makes iterating on the head practical on a 4 GB card - the GPU is needed
for hours to build this, and then barely at all to train.

Layout (HDF5, one file per backbone):

    artifacts/cache/features_<backbone_slug>.h5
      /<GENE>/embeddings     (L, d)  float16   last hidden state per residue
      /<GENE>/log_probs_wt   (L, 20) float32   wt-marginal log-probabilities
      attrs: backbone, embed_layer, max_length, created

Namespacing by backbone slug matters: features from esm2_t30_150M and
esm2_t33_650M have different dimensionalities and different meanings, and
silently mixing them would produce a model that trains and then fails in ways
that look like a modelling problem rather than a plumbing one.

Writes are per gene and resumable - the file is opened in append mode and a
gene already present is skipped.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import h5py
import numpy as np

from vep.config import Config, resolve_device


def cache_path(cfg: Config) -> Path:
    return cfg.paths.cache / f"features_{cfg.backbone_slug()}.h5"


def completed_genes(path: Path) -> set[str]:
    """Genes already fully written, so a resumed run can skip them."""
    if not path.exists():
        return set()
    done: set[str] = set()
    with h5py.File(path, "r") as fh:
        for gene, grp in fh.items():
            if "embeddings" in grp and "log_probs_wt" in grp:
                done.add(gene)
    return done


def build_cache(
    cfg: Config,
    genes: list[str] | None = None,
    verbose: bool = True,
    overwrite: bool = False,
) -> Path:
    """Extract and store per-residue features for every panel protein."""
    from vep.esm.backbone import ESM2Backbone

    cfg.paths.mkdirs()
    out_path = cache_path(cfg)
    if overwrite and out_path.exists():
        out_path.unlink()

    proteins = json.loads(
        (cfg.paths.processed / "proteins.json").read_text(encoding="utf-8")
    )
    targets = sorted(genes if genes is not None else proteins.keys())

    done = completed_genes(out_path)
    todo = [g for g in targets if g not in done]
    if verbose:
        print(f"feature cache -> {out_path}")
        print(f"  {len(done)} genes already cached, {len(todo)} to extract")
    if not todo:
        return out_path

    backbone = ESM2Backbone(
        model_name=cfg.backbone.name,
        device=resolve_device(cfg.backbone.device),
        fp16=cfg.backbone.fp16,
        max_length=cfg.backbone.max_length,
        embed_layer=cfg.backbone.embed_layer,
    )

    t0 = time.time()
    with h5py.File(out_path, "a") as fh:
        fh.attrs["backbone"] = cfg.backbone.name
        fh.attrs["embed_layer"] = cfg.backbone.embed_layer
        fh.attrs["max_length"] = cfg.backbone.max_length
        fh.attrs["hidden_size"] = backbone.hidden_size

        for i, gene in enumerate(todo, 1):
            sequence = proteins[gene]["sequence"]
            log_probs, embeddings = backbone.wt_marginals(sequence)

            grp = fh.require_group(gene)
            for name in ("embeddings", "log_probs_wt"):
                if name in grp:
                    del grp[name]
            # gzip level 4: roughly 2x smaller with negligible read cost, and
            # the cache is read repeatedly during training.
            grp.create_dataset("embeddings", data=embeddings, compression="gzip",
                               compression_opts=4)
            grp.create_dataset("log_probs_wt", data=log_probs, compression="gzip",
                               compression_opts=4)
            grp.attrs["length"] = len(sequence)
            grp.attrs["accession"] = proteins[gene]["accession"]
            fh.flush()

            if verbose:
                elapsed = time.time() - t0
                eta = (len(todo) - i) / max(i / max(elapsed, 1e-9), 1e-9)
                print(
                    f"  [{i:3d}/{len(todo)}] {gene:10s} L={len(sequence):5d} "
                    f"elapsed={elapsed/60:5.1f}m eta={eta/60:5.1f}m",
                    flush=True,
                )

    if verbose:
        size_mb = out_path.stat().st_size / 1e6
        print(f"done in {(time.time()-t0)/60:.1f} min, {size_mb:.0f} MB")
    return out_path


class FeatureCache:
    """Read-only accessor. Opens the file once and reads lazily per gene."""

    def __init__(self, path: str | Path):
        self.path = Path(path)
        if not self.path.exists():
            raise FileNotFoundError(
                f"no feature cache at {self.path}; run build_cache first"
            )
        self._fh = h5py.File(self.path, "r")
        self.backbone = self._fh.attrs.get("backbone", "unknown")
        self.hidden_size = int(self._fh.attrs.get("hidden_size", 0))

    def __contains__(self, gene: str) -> bool:
        return gene in self._fh

    @property
    def genes(self) -> list[str]:
        return sorted(self._fh.keys())

    def embeddings(self, gene: str, positions: np.ndarray | None = None) -> np.ndarray:
        ds = self._fh[gene]["embeddings"]
        if positions is None:
            return ds[:]
        # h5py fancy indexing requires sorted, unique indices; restore the
        # caller's order afterwards so rows line up with their variants.
        order = np.argsort(positions)
        uniq, inverse = np.unique(positions[order], return_inverse=True)
        rows = ds[uniq]
        out = np.empty((len(positions), rows.shape[1]), dtype=rows.dtype)
        out[order] = rows[inverse]
        return out

    def log_probs(self, gene: str, positions: np.ndarray | None = None) -> np.ndarray:
        ds = self._fh[gene]["log_probs_wt"]
        if positions is None:
            return ds[:]
        order = np.argsort(positions)
        uniq, inverse = np.unique(positions[order], return_inverse=True)
        rows = ds[uniq]
        out = np.empty((len(positions), rows.shape[1]), dtype=rows.dtype)
        out[order] = rows[inverse]
        return out

    def close(self) -> None:
        self._fh.close()

    def __enter__(self) -> "FeatureCache":
        return self

    def __exit__(self, *exc) -> None:
        self.close()


def main() -> None:
    import argparse

    ap = argparse.ArgumentParser(
        description="Extract per-residue ESM-2 features for the gene panel."
    )
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--genes", help="comma-separated subset")
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()

    cfg = Config.load(args.config)
    genes = args.genes.split(",") if args.genes else None
    path = build_cache(cfg, genes=genes, overwrite=args.overwrite)
    print(f"feature cache at {path} ({path.stat().st_size / 1e6:.0f} MB)")


if __name__ == "__main__":
    main()
