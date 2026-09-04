"""Multi-task training: ClinVar pathogenicity + GRB2 deep-mutational-scan fitness.

    python -m vep.train.multitask

The two tasks share one ProteinNPT body and differ only in the label column and
the output head. Batches alternate between tasks rather than mixing them:
neighbours are only useful if they are comparable, and a pathogenic BRCA1
variant tells the model nothing about the binding affinity of a GRB2 double
mutant.

Fitness targets are standardised before the MSE. The raw assay scores sit in
[-1.44, 0.56] with a mean near -0.44, so an unstandardised regression term would
be numerically tiny next to the cross-entropy and the weighting in the config
would not mean what it says. Spearman is scale-invariant, so reported numbers
are unaffected.

Evaluation holds out whole POSITIONS, not random variants. With 98% double
mutants a random split leaves almost every test variant sharing a position with
something in training, which measures memorisation rather than generalisation.
"""

from __future__ import annotations

import json
import math
import time
from pathlib import Path

import numpy as np
import requests
import torch
from scipy.stats import spearmanr

from vep.config import Config, resolve_device
from vep.data.dms import load_grb2
from vep.data.dms_features import build_dms_features, split_by_position
from vep.esm.cache import FeatureCache, cache_path
from vep.eval.zeroshot import BLOSUM62
from vep.constants import AA_TO_IDX
from vep.models.npt import (
    LABEL_BENIGN,
    LABEL_MASK,
    LABEL_PATHOGENIC,
    TASK_FITNESS,
    TASK_PATHOGENICITY,
    ProteinNPT,
    mask_labels,
)
from vep.train.datasets import (
    BatchSampler,
    build_features,
    label_tokens,
    split_indices,
    to_tensors,
)
from vep.train.trainer import _per_gene_auroc, evaluate


class GRB2Fitness:
    """GRB2 binding assay, packaged like the ClinVar feature blocks."""

    def __init__(self, cfg: Config, verbose: bool = True):
        seq_url = "https://rest.uniprot.org/uniprotkb/P62993.json"
        full = requests.get(seq_url, timeout=30).json()["sequence"]["value"]
        ds = load_grb2(full_sequence=full)
        self.sequence = full
        self.frame = ds.frame

        self.variants = [
            [(p, m) for (_, _, m), p in zip(r.subs, r.canonical_positions)]
            for r in ds.frame.itertuples(index=False)
        ]

        cache = FeatureCache(cache_path(cfg))
        if "GRB2" not in cache:
            cache.close()
            raise RuntimeError(
                "GRB2 is not in the feature cache; it is not part of the ClinVar "
                "panel and must be added before the fitness task can train"
            )

        masked_path = cfg.paths.cache / "grb2_masked_logprobs.npy"
        if masked_path.exists():
            masked_lp = np.load(masked_path)
        else:
            from vep.esm.backbone import ESM2Backbone

            if verbose:
                print("  computing masked-marginals over the assayed domain ...", flush=True)
            bb = ESM2Backbone(
                model_name=cfg.backbone.name,
                device=resolve_device(cfg.backbone.device),
                fp16=cfg.backbone.fp16,
                max_length=cfg.backbone.max_length,
            )
            positions = sorted({p for subs in self.variants for p, _ in subs})
            masked_lp = bb.masked_marginals(full, positions=positions, batch_size=16)
            masked_lp = np.nan_to_num(masked_lp)
            np.save(masked_path, masked_lp)
            del bb
            torch.cuda.empty_cache()

        llr = np.load(cfg.paths.cache / "grb2_mutant_llr.npy")
        blosum = np.array(
            [
                np.mean([BLOSUM62[AA_TO_IDX[full[p]], AA_TO_IDX[m]] for p, m in subs])
                for subs in self.variants
            ],
            dtype=np.float32,
        )
        self.blocks = build_dms_features(
            cache, "GRB2", full, self.variants, masked_lp, llr, blosum,
            d_emb=cache.hidden_size,
        )
        self.context = cache.embeddings("GRB2").astype(np.float32).mean(axis=0)
        cache.close()

        raw = ds.frame["score"].to_numpy(dtype=np.float32)
        self.raw_values = raw
        self.mean, self.std = float(raw.mean()), float(raw.std())
        self.values = (raw - self.mean) / self.std
        self.split = split_by_position(self.variants)
        self.zero_shot = llr

        if verbose:
            counts = {s: int((self.split == s).sum()) for s in ("train", "val", "test")}
            print(f"  GRB2: {len(raw):,} variants  split(by position) {counts}")

    def indices(self, split: str) -> np.ndarray:
        return np.flatnonzero(self.split == split)

    def tensors(self, rows: np.ndarray, device: str) -> dict[str, torch.Tensor]:
        t = lambda a: torch.as_tensor(a, device=device)
        return {
            "emb": t(self.blocks["emb"][rows]),
            "logprobs": t(self.blocks["logprobs"][rows]),
            "subst": t(self.blocks["subst"][rows]),
            "context": t(np.tile(self.context, (len(rows), 1))),
        }


def train_multitask(
    cfg: Config,
    steps_per_epoch: int = 300,
    epochs: int = 40,
    fitness_batch_frac: float = 0.5,
    tasks: tuple[str, ...] = ("pathogenicity", "fitness"),
    verbose: bool = True,
):
    """Train one body on the requested tasks.

    `tasks=("fitness",)` gives a fitness specialist - which is the whole point
    of the separate-heads arrangement. Sharing one body between both objectives
    cost 0.021 AUROC on pathogenicity, so the two are better trained apart and
    served as a pair than fused into a model that is second-best at both.
    """
    want_path = "pathogenicity" in tasks
    want_fit = "fitness" in tasks
    if not (want_path or want_fit):
        raise ValueError("at least one task is required")
    if not want_path:
        fitness_batch_frac = 1.0
    elif not want_fit:
        fitness_batch_frac = 0.0
    device = resolve_device(cfg.backbone.device)
    torch.manual_seed(cfg.train.seed)
    rng = np.random.default_rng(cfg.train.seed)

    if verbose:
        print("loading features ...", flush=True)
    feats = build_features(cfg)
    idx = split_indices(feats)
    grb2 = GRB2Fitness(cfg, verbose=verbose)

    sampler = BatchSampler(feats, idx["train"], batch_size=cfg.train.batch_size,
                           seed=cfg.train.seed)
    grb2_train = grb2.indices("train")
    grb2_val = grb2.indices("val")

    model = ProteinNPT(
        d_emb=feats.d_emb, d_model=cfg.npt.d_model, n_layers=cfg.npt.n_layers,
        n_heads=cfg.npt.n_heads, dropout=cfg.npt.dropout,
    ).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.train.lr,
                            weight_decay=cfg.train.weight_decay)
    total = epochs * steps_per_epoch
    warmup = int(cfg.train.warmup_frac * total)
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: s / max(warmup, 1) if s < warmup
        else 0.5 * (1 + math.cos(math.pi * (s - warmup) / max(total - warmup, 1)))
    )

    best = {"score": -1e9, "epoch": -1, "state": None}
    patience = cfg.train.patience
    history = []

    for epoch in range(epochs):
        model.train()
        losses_p, losses_f = [], []
        for _ in range(steps_per_epoch):
            if rng.random() < fitness_batch_frac:
                rows = rng.choice(grb2_train, size=cfg.train.batch_size, replace=False)
                batch = grb2.tensors(rows, device)
                labels = torch.full((len(rows),), LABEL_BENIGN, device=device, dtype=torch.long)
                masked, is_masked = mask_labels(labels, cfg.npt.label_mask_prob)
                if not is_masked.any():
                    is_masked[0] = True
                    masked = masked.clone(); masked[0] = LABEL_MASK
                values = torch.as_tensor(grb2.values[rows], device=device)
                task = torch.full((len(rows),), TASK_FITNESS, device=device, dtype=torch.long)
                out = model(**batch, labels=masked, values=values, task=task)
                loss = torch.nn.functional.mse_loss(out.fitness[is_masked], values[is_masked])
                loss = cfg.train.w_fitness * loss
                losses_f.append(float(loss.detach()))
            else:
                rows = sampler.sample()
                batch = to_tensors(feats, rows, device)
                labels = label_tokens(feats.labels[rows], device)
                masked, is_masked = mask_labels(labels, cfg.npt.label_mask_prob)
                if not is_masked.any():
                    is_masked[0] = True
                    masked = masked.clone(); masked[0] = LABEL_MASK
                task = torch.full((len(rows),), TASK_PATHOGENICITY, device=device, dtype=torch.long)
                out = model(**batch, labels=masked, task=task)
                target = torch.as_tensor(feats.labels[rows], device=device, dtype=torch.float32)
                w = torch.as_tensor(feats.weights[rows], device=device)
                per = torch.nn.functional.binary_cross_entropy_with_logits(
                    out.pathogenicity_logit[is_masked], target[is_masked], reduction="none")
                loss = cfg.train.w_pathogenicity * ((per * w[is_masked]).sum()
                                                    / w[is_masked].sum().clamp_min(1e-8))
                losses_p.append(float(loss.detach()))

            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.train.grad_clip)
            opt.step(); sched.step()

        auroc = float("nan")
        rho = float("nan")
        score = 0.0
        if want_path:
            val_path = evaluate(
                model, feats,
                np.random.default_rng(0).choice(idx["val"], 800, replace=False),
                idx["train"], device, "inductive", same_gene_context=False)
            auroc = val_path.per_gene_auroc
            score += auroc
        if want_fit:
            rho = evaluate_fitness(model, grb2, grb2_val, grb2_train, device)
            score += rho
        # With both tasks, selection has to balance them: picking on
        # pathogenicity alone would happily choose an epoch where the fitness
        # head had collapsed. With one task it is simply that task's metric.
        history.append({"epoch": epoch, "loss_path": float(np.mean(losses_p or [0])),
                        "loss_fit": float(np.mean(losses_f or [0])),
                        "val_per_gene_auroc": auroc, "val_fitness_rho": rho})
        if verbose:
            bits = []
            if want_path:
                bits.append(f"path={np.mean(losses_p or [0]):.4f} val AUROC={auroc:.4f}")
            if want_fit:
                bits.append(f"fit={np.mean(losses_f or [0]):.4f} val rho={rho:.4f}")
            print(f"  epoch {epoch:3d}  " + "  ".join(bits), flush=True)

        if score > best["score"]:
            best = {"score": score, "epoch": epoch,
                    "state": {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}}
            patience = cfg.train.patience
        else:
            patience -= 1
            if patience <= 0:
                if verbose:
                    print(f"  early stop at {epoch} (best epoch {best['epoch']})")
                break

    if best["state"] is not None:
        model.load_state_dict(best["state"])
    return model, feats, idx, grb2, {"history": history, "best_epoch": best["epoch"]}


@torch.no_grad()
def evaluate_fitness(model, grb2, query, context_pool, device, n_context: int = 31,
                     seed: int = 0) -> float:
    """Spearman between predicted and measured fitness on held-out positions."""
    model.eval()
    rng = np.random.default_rng(seed)
    preds = np.zeros(len(query), dtype=np.float32)
    for i, q in enumerate(query):
        ctx = rng.choice(context_pool, n_context, replace=False)
        ctx = ctx[ctx != q]
        rows = np.concatenate([[q], ctx])
        batch = grb2.tensors(rows, device)
        labels = torch.full((len(rows),), LABEL_BENIGN, device=device, dtype=torch.long)
        labels[0] = LABEL_MASK
        values = torch.as_tensor(grb2.values[rows], device=device)
        task = torch.full((len(rows),), TASK_FITNESS, device=device, dtype=torch.long)
        preds[i] = model(**batch, labels=labels, values=values, task=task).fitness[0].item()
    model.train()
    return float(spearmanr(preds, grb2.values[query]).statistic)


def main() -> int:
    import argparse

    ap = argparse.ArgumentParser(description="Train ProteinNPT on one or both tasks.")
    ap.add_argument("--tasks", default="pathogenicity,fitness",
                    help="comma-separated: pathogenicity, fitness (default: both)")
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--out", default=None, help="checkpoint path override")
    args = ap.parse_args()
    tasks = tuple(t.strip() for t in args.tasks.split(",") if t.strip())

    cfg = Config.load("configs/default.yaml")
    device = resolve_device(cfg.backbone.device)
    t0 = time.time()
    model, feats, idx, grb2, meta = train_multitask(cfg, tasks=tasks, epochs=args.epochs)

    suffix = "multitask" if len(tasks) > 1 else tasks[0]
    results: dict = {"tasks": list(tasks), "meta": meta}

    if "fitness" in tasks:
        print("\n=== GRB2 fitness (held-out positions) ===")
        train = grb2.indices("train")
        strict_path = cfg.paths.cache / "grb2_strict_test.npy"
        subsets = [("all_test", grb2.indices("test"))]
        if strict_path.exists():
            # Most of the headline test set still shares a partner position
            # with training; this subset shares none, so it is the honest one.
            subsets.append(("strict_test", np.load(strict_path)))
        for name, ix in subsets:
            rho = evaluate_fitness(model, grb2, ix, train, device)
            zs = float(spearmanr(-grb2.zero_shot[ix], grb2.raw_values[ix]).statistic)
            results[name] = {"n": int(len(ix)), "spearman": rho, "zero_shot": zs}
            print(f"  {name:12s} n={len(ix):5d}  zero-shot {zs:.4f}  model {rho:.4f}  "
                  f"({rho - zs:+.4f})")

    if "pathogenicity" in tasks:
        r = evaluate(model, feats, idx["test"], idx["train"], device, "inductive",
                     same_gene_context=False)
        results["pathogenicity"] = r.__dict__
        prev = json.loads(Path("artifacts/npt_results.json").read_text())["inductive"]
        print(f"\n=== ClinVar pathogenicity (held-out genes) ===")
        print(f"  single-task reference  {prev['per_gene_auroc']:.4f}")
        print(f"  this model             {r.per_gene_auroc:.4f}   "
              f"({r.per_gene_auroc - prev['per_gene_auroc']:+.4f})")

    out_json = Path(f"artifacts/{suffix}_results.json")
    out_json.write_text(json.dumps(results, indent=2, default=float))
    ckpt = Path(args.out or f"artifacts/models/npt_{suffix}.pt")
    torch.save({"state_dict": model.state_dict(), "backbone": cfg.backbone.name,
                "tasks": list(tasks), "meta": meta}, ckpt)
    print(f"\n{(time.time()-t0)/60:.1f} min; wrote {out_json.name} and {ckpt.name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
