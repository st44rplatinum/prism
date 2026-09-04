"""Typed configuration with YAML overrides.

Every stage of the pipeline (data build, embedding extraction, training,
serving) reads from a single Config so the API and the training scripts can
never disagree about where the cache lives or which backbone produced it.
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

REPO_ROOT = Path(__file__).resolve().parent.parent


@dataclass
class PathConfig:
    raw: Path = REPO_ROOT / "data" / "raw"
    processed: Path = REPO_ROOT / "data" / "processed"
    cache: Path = REPO_ROOT / "artifacts" / "cache"
    models: Path = REPO_ROOT / "artifacts" / "models"
    figures: Path = REPO_ROOT / "artifacts" / "figures"

    def mkdirs(self) -> None:
        for p in (self.raw, self.processed, self.cache, self.models, self.figures):
            p.mkdir(parents=True, exist_ok=True)


@dataclass
class BackboneConfig:
    # esm2_t33_650M_UR50D is the accuracy sweet spot but needs ~1.4 GB in fp16
    # for inference; t30_150M is the safe default on a 4 GB card and is what the
    # shipped cache is built with. Both are inference-only here - ProteinNPT
    # trains on frozen features, so the backbone never needs gradients.
    name: str = "facebook/esm2_t30_150M_UR50D"
    embed_layer: int = -1          # which hidden layer to cache (-1 = last)
    fp16: bool = True
    max_length: int = 1022         # ESM-2 positional limit minus BOS/EOS
    batch_tokens: int = 8192       # token budget per forward batch
    device: str = "auto"


@dataclass
class DataConfig:
    genes: list[str] = field(default_factory=list)   # empty = use GENE_PANEL
    min_review_stars: int = 1        # drop 0-star "no assertion" submissions
    min_variants_per_gene: int = 10
    # Held-out split is by GENE, not by variant: a model that has seen other
    # variants in the same protein has a large and unrealistic advantage.
    split_by: str = "gene"
    test_fraction: float = 0.2
    val_fraction: float = 0.1
    seed: int = 42


@dataclass
class NPTConfig:
    """ProteinNPT-style non-parametric head (Notin et al., NeurIPS 2023)."""
    d_model: int = 256
    n_layers: int = 4
    n_heads: int = 8
    dropout: float = 0.1
    # Number of labelled neighbours retrieved into each attention batch.
    n_neighbours: int = 32
    # Fraction of in-batch labels masked during training so the model learns to
    # predict from neighbours rather than copying its own target label.
    label_mask_prob: float = 0.15


@dataclass
class TrainConfig:
    epochs: int = 60
    batch_size: int = 32
    lr: float = 3e-4
    weight_decay: float = 0.01
    grad_clip: float = 1.0
    warmup_frac: float = 0.06
    patience: int = 10
    # Multi-task weighting: DMS fitness regression is auxiliary to the
    # pathogenicity objective.
    w_pathogenicity: float = 1.0
    w_fitness: float = 0.3
    use_review_star_weights: bool = True
    seed: int = 42


@dataclass
class Config:
    paths: PathConfig = field(default_factory=PathConfig)
    backbone: BackboneConfig = field(default_factory=BackboneConfig)
    data: DataConfig = field(default_factory=DataConfig)
    npt: NPTConfig = field(default_factory=NPTConfig)
    train: TrainConfig = field(default_factory=TrainConfig)

    @classmethod
    def load(cls, path: str | Path | None = None) -> "Config":
        cfg = cls()
        if path is None:
            return cfg
        blob = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
        return cfg.merged(blob)

    def merged(self, overrides: dict[str, Any]) -> "Config":
        """Return a copy with a nested dict of overrides applied."""
        out = dataclasses.replace(self)
        for section, values in overrides.items():
            if not hasattr(out, section):
                raise KeyError(f"unknown config section: {section!r}")
            target = getattr(out, section)
            if not dataclasses.is_dataclass(target):
                setattr(out, section, values)
                continue
            known = {f.name for f in dataclasses.fields(target)}
            for key, value in (values or {}).items():
                if key not in known:
                    raise KeyError(f"unknown config key: {section}.{key}")
                if isinstance(getattr(target, key), Path):
                    value = Path(value)
                setattr(target, key, value)
        return out

    def backbone_slug(self) -> str:
        """Filesystem-safe backbone id, used to namespace the feature cache."""
        return self.backbone.name.split("/")[-1]


def resolve_device(spec: str = "auto") -> str:
    import torch
    if spec != "auto":
        return spec
    return "cuda" if torch.cuda.is_available() else "cpu"
