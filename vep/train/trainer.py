"""Train ProteinNPT on cached ESM-2 features and ClinVar labels.

Selection is on validation *per-gene* AUROC, not pooled AUROC and not loss.
Pooled AUROC rewards a model for learning which genes are constrained, which is
free information it will not have on a new gene; per-gene AUROC asks the
clinical question directly. Loss is not used because the review-star weighting
and label masking make it only loosely coupled to ranking quality.
"""

from __future__ import annotations

import json
import math
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import roc_auc_score

from vep.config import Config, resolve_device
from vep.models.npt import (
    LABEL_MASK,
    ProteinNPT,
    mask_labels,
    npt_loss,
)
from vep.train.datasets import (
    BatchSampler,
    VariantFeatures,
    label_tokens,
    pad_to,
    split_indices,
    to_tensors,
)


@dataclass
class EvalResult:
    protocol: str
    pooled_auroc: float
    per_gene_auroc: float
    per_gene_std: float
    n_genes: int
    n_variants: int


def _per_gene_auroc(
    genes: np.ndarray, y: np.ndarray, s: np.ndarray, min_per_class: int = 3
) -> tuple[float, float, int]:
    scores = []
    for g in np.unique(genes):
        m = genes == g
        yy = y[m]
        if yy.sum() < min_per_class or (len(yy) - yy.sum()) < min_per_class:
            continue
        scores.append(roc_auc_score(yy, s[m]))
    if not scores:
        return float("nan"), float("nan"), 0
    return float(np.mean(scores)), float(np.std(scores)), len(scores)


@torch.no_grad()
def evaluate(
    model: ProteinNPT,
    feats: VariantFeatures,
    query_idx: np.ndarray,
    context_pool: np.ndarray,
    device: str,
    protocol: str,
    n_context: int = 31,
    seed: int = 0,
    same_gene_context: bool = False,
) -> EvalResult:
    """Score every query variant with its label masked.

    `context_pool` supplies the labelled neighbours. Under the inductive
    protocol that pool contains only training-gene variants; under the
    transductive protocol it also contains other variants of the query's own
    gene.
    """
    model.eval()
    rng = np.random.default_rng(seed)
    preds = np.zeros(len(query_idx), dtype=np.float32)

    gene_of = feats.gene_idx
    pool_by_gene: dict[int, np.ndarray] = {}
    if same_gene_context:
        for g in np.unique(gene_of[context_pool]):
            pool_by_gene[int(g)] = context_pool[gene_of[context_pool] == g]

    for i, q in enumerate(query_idx):
        if same_gene_context:
            same = pool_by_gene.get(int(gene_of[q]), np.array([], dtype=np.int64))
            same = same[same != q]
            if len(same) >= n_context:
                ctx = rng.choice(same, size=n_context, replace=False)
            else:
                # Top up from the global pool when the gene is sparsely labelled.
                extra = rng.choice(
                    context_pool, size=n_context - len(same), replace=False
                )
                ctx = np.concatenate([same, extra])
        else:
            ctx = rng.choice(context_pool, size=n_context, replace=False)
            ctx = ctx[ctx != q]

        rows = np.concatenate([[q], ctx])
        batch = to_tensors(feats, rows, device)
        labels = label_tokens(feats.labels[rows], device)
        labels[0] = LABEL_MASK          # the query never sees its own label
        out = model(**batch, labels=labels)
        preds[i] = out.pathogenicity_logit[0].item()

    y = feats.labels[query_idx]
    genes = gene_of[query_idx]
    mean_auc, std_auc, n_g = _per_gene_auroc(genes, y, preds)
    return EvalResult(
        protocol=protocol,
        pooled_auroc=float(roc_auc_score(y, preds)),
        per_gene_auroc=mean_auc,
        per_gene_std=std_auc,
        n_genes=n_g,
        n_variants=len(query_idx),
    )


def train(
    cfg: Config,
    feats: VariantFeatures,
    steps_per_epoch: int = 300,
    epochs: int | None = None,
    n_context: int = 31,
    val_subsample: int | None = 1200,
    verbose: bool = True,
) -> tuple[ProteinNPT, dict]:
    device = resolve_device(cfg.backbone.device)
    torch.manual_seed(cfg.train.seed)
    epochs = epochs or cfg.train.epochs

    idx = split_indices(feats)
    # Per-epoch validation scores one query at a time, so the full 2,905-variant
    # split costs more than the training steps it is meant to monitor. A fixed
    # subsample is enough to rank epochs; the selected model is re-evaluated on
    # the full split afterwards.
    val_idx = idx["val"]
    if val_subsample is not None and len(val_idx) > val_subsample:
        val_idx = np.random.default_rng(cfg.train.seed).choice(
            val_idx, size=val_subsample, replace=False
        )
    sampler = BatchSampler(
        feats, idx["train"], batch_size=cfg.train.batch_size, seed=cfg.train.seed
    )

    model = ProteinNPT(
        d_emb=feats.d_emb,
        d_model=cfg.npt.d_model,
        n_layers=cfg.npt.n_layers,
        n_heads=cfg.npt.n_heads,
        dropout=cfg.npt.dropout,
    ).to(device)

    opt = torch.optim.AdamW(
        model.parameters(), lr=cfg.train.lr, weight_decay=cfg.train.weight_decay
    )
    total_steps = epochs * steps_per_epoch
    warmup = int(cfg.train.warmup_frac * total_steps)

    def lr_at(step: int) -> float:
        if step < warmup:
            return step / max(warmup, 1)
        p = (step - warmup) / max(total_steps - warmup, 1)
        return 0.5 * (1 + math.cos(math.pi * p))

    sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_at)

    history: list[dict] = []
    best = {"per_gene_auroc": -1.0, "epoch": -1, "state": None}
    patience_left = cfg.train.patience
    step = 0
    t0 = time.time()

    for epoch in range(epochs):
        model.train()
        losses = []
        for rows in sampler.epoch(steps_per_epoch):
            batch = to_tensors(feats, rows, device)
            labels = label_tokens(feats.labels[rows], device)
            masked, is_masked = mask_labels(labels, cfg.npt.label_mask_prob)
            if not is_masked.any():
                is_masked[0] = True
                masked = masked.clone()
                masked[0] = LABEL_MASK

            out = model(**batch, labels=masked)
            target = torch.as_tensor(
                feats.labels[rows], device=device, dtype=torch.float32
            )
            weight = (
                torch.as_tensor(feats.weights[rows], device=device)
                if cfg.train.use_review_star_weights
                else None
            )
            loss, stats = npt_loss(
                out,
                target,
                is_masked,
                sample_weight=weight,
                w_pathogenicity=cfg.train.w_pathogenicity,
            )
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.train.grad_clip)
            opt.step()
            sched.step()
            losses.append(stats["loss"])
            step += 1

        # Validation uses the inductive protocol: context from training genes
        # only. Selecting on the transductive number would tune the model for
        # the easier setting and quietly inflate the headline comparison.
        val = evaluate(
            model,
            feats,
            query_idx=val_idx,
            context_pool=idx["train"],
            device=device,
            protocol="inductive",
            n_context=n_context,
            same_gene_context=False,
        )
        history.append(
            {
                "epoch": epoch,
                "loss": float(np.mean(losses)),
                "val_per_gene_auroc": val.per_gene_auroc,
                "val_pooled_auroc": val.pooled_auroc,
            }
        )
        if verbose:
            print(
                f"  epoch {epoch:3d}  loss={np.mean(losses):.4f}  "
                f"val per-gene AUROC={val.per_gene_auroc:.4f}  "
                f"pooled={val.pooled_auroc:.4f}  ({time.time()-t0:.0f}s)",
                flush=True,
            )

        if val.per_gene_auroc > best["per_gene_auroc"]:
            best = {
                "per_gene_auroc": val.per_gene_auroc,
                "epoch": epoch,
                "state": {k: v.detach().cpu().clone() for k, v in model.state_dict().items()},
            }
            patience_left = cfg.train.patience
        else:
            patience_left -= 1
            if patience_left <= 0:
                if verbose:
                    print(f"  early stop at epoch {epoch} "
                          f"(best {best['per_gene_auroc']:.4f} @ {best['epoch']})")
                break

    if best["state"] is not None:
        model.load_state_dict(best["state"])
    return model, {"history": history, "best_epoch": best["epoch"],
                   "best_val_per_gene_auroc": best["per_gene_auroc"]}


def save_checkpoint(model: ProteinNPT, cfg: Config, meta: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "state_dict": model.state_dict(),
            "npt_config": asdict(cfg.npt),
            "backbone": cfg.backbone.name,
            "meta": meta,
        },
        path,
    )
