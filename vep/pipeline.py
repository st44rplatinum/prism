"""End-to-end rebuild for a given backbone: features -> scores -> train -> evaluate.

    python -m vep.pipeline --config configs/650m.yaml

Every artefact is namespaced by the backbone slug, so a 650M run cannot disturb
the 150M results already on disk - it produces a parallel set that can be
compared against them directly.

Each stage skips itself if its output already exists, so the whole thing is
resumable: kill it at any point, run the same command, and it continues from
the last completed stage. The long stage (masked-marginal scoring) is
additionally checkpointed per gene.

Designed to be left running overnight, so it logs a timestamped line per stage
and never prompts.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression

from vep.config import Config, resolve_device


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def run(cfg: Config, force: bool = False) -> int:
    slug = cfg.backbone_slug()
    cache = cfg.paths.cache
    cfg.paths.mkdirs()

    scores_path = cache / f"zeroshot_all_{slug}.parquet"
    ckpt_path = cfg.paths.models / f"npt_{slug}.pt"
    calib_path = cfg.paths.models.parent / f"calibration_{slug}.json"
    results_path = cfg.paths.models.parent / f"npt_results_{slug}.json"

    log(f"backbone {cfg.backbone.name}")
    log(f"artefacts namespaced as *_{slug}")
    t_start = time.time()

    # -- 1. per-residue feature cache ---------------------------------------
    from vep.esm.cache import build_cache, cache_path

    if force or not cache_path(cfg).exists():
        log("stage 1/5: extracting embeddings + wt log-probs")
        build_cache(cfg, verbose=True)
    else:
        log(f"stage 1/5: feature cache present ({cache_path(cfg).name}), skipping")

    # -- 2. masked-marginal scores at variant positions ---------------------
    if force or not scores_path.exists():
        log("stage 2/5: masked-marginal scoring (the long one)")
        from vep.eval.zeroshot import add_blosum, score_dataset

        df = score_dataset(
            cfg,
            scheme="masked",
            batch_size=8,
            verbose=True,
            checkpoint_dir=cache / f"masked_parts_{slug}",
        )
        df = add_blosum(df)
        df.to_parquet(scores_path, index=False)
        log(f"        wrote {scores_path.name}: {df['esm_masked'].notna().sum():,} scored")
    else:
        log(f"stage 2/5: scores present ({scores_path.name}), skipping")

    # -- 3. train ------------------------------------------------------------
    from vep.train.datasets import build_features, split_indices
    from vep.train.trainer import evaluate, save_checkpoint, train

    feats = build_features(cfg, scores_path=scores_path)
    idx = split_indices(feats)
    log(f"features: {len(feats):,} variants, d_emb={feats.d_emb}")

    device = resolve_device(cfg.backbone.device)
    if force or not ckpt_path.exists():
        log("stage 3/5: training ProteinNPT")
        model, meta = train(cfg, feats, steps_per_epoch=300, epochs=40, verbose=True)
        save_checkpoint(model, cfg, meta, ckpt_path)
        log(f"        best epoch {meta['best_epoch']} "
            f"(val per-gene AUROC {meta['best_val_per_gene_auroc']:.4f})")
    else:
        log(f"stage 3/5: checkpoint present ({ckpt_path.name}), loading")
        from vep.models.npt import ProteinNPT, load_state_dict_compat

        blob = torch.load(ckpt_path, map_location=device, weights_only=False)
        model = ProteinNPT(
            d_emb=feats.d_emb, d_model=cfg.npt.d_model,
            n_layers=cfg.npt.n_layers, n_heads=cfg.npt.n_heads, dropout=cfg.npt.dropout,
        ).to(device)
        load_state_dict_compat(model, blob["state_dict"])
        meta = blob.get("meta", {})
    model.eval()

    # -- 4. calibration (fitted on validation, never on test) ---------------
    from vep.models.npt import LABEL_MASK
    from vep.models.predictor import expected_calibration_error
    from vep.train.datasets import label_tokens, to_tensors

    @torch.no_grad()
    def logits_for(query, pool):
        rng = np.random.default_rng(0)
        out = np.zeros(len(query), dtype=np.float32)
        for i, q in enumerate(query):
            ctx = rng.choice(pool, 31, replace=False)
            ctx = ctx[ctx != q]
            rows = np.concatenate([[q], ctx])
            lab = label_tokens(feats.labels[rows], device)
            lab[0] = LABEL_MASK
            out[i] = model(**to_tensors(feats, rows, device), labels=lab).pathogenicity_logit[0].item()
        return out

    if force or not calib_path.exists():
        log("stage 4/5: fitting calibration on the validation split")
        v_logit = logits_for(idx["val"], idx["train"])
        v_y = feats.labels[idx["val"]]
        lr = LogisticRegression(C=1e6).fit(v_logit.reshape(-1, 1), v_y)
        a, b = float(lr.coef_[0][0]), float(lr.intercept_[0])
        sig = lambda z: 1 / (1 + np.exp(-z))
        calib = {
            "a": a, "b": b, "fitted_on": "val_inductive",
            "ece_before": expected_calibration_error(sig(v_logit), v_y),
            "ece_after": expected_calibration_error(sig(a * v_logit + b), v_y),
        }
        calib_path.write_text(json.dumps(calib, indent=2), encoding="utf-8")
        log(f"        val ECE {calib['ece_before']:.4f} -> {calib['ece_after']:.4f}")
    else:
        log(f"stage 4/5: calibration present ({calib_path.name}), skipping")

    # -- 5. held-out evaluation ---------------------------------------------
    log("stage 5/5: evaluating on held-out test genes")
    r_ind = evaluate(model, feats, idx["test"], idx["train"], device, "inductive",
                     same_gene_context=False)
    r_trn = evaluate(model, feats, idx["test"],
                     np.concatenate([idx["train"], idx["test"]]), device,
                     "transductive", same_gene_context=True)
    results = {
        "backbone": cfg.backbone.name,
        "d_emb": int(feats.d_emb),
        "inductive": r_ind.__dict__,
        "transductive": r_trn.__dict__,
        "meta": meta,
    }
    results_path.write_text(json.dumps(results, indent=2, default=float), encoding="utf-8")

    log("")
    log(f"inductive     per-gene AUROC {r_ind.per_gene_auroc:.4f} "
        f"+/- {r_ind.per_gene_std:.4f} | pooled {r_ind.pooled_auroc:.4f}")
    log(f"transductive  per-gene AUROC {r_trn.per_gene_auroc:.4f} "
        f"+/- {r_trn.per_gene_std:.4f} | pooled {r_trn.pooled_auroc:.4f}")

    baseline = cfg.paths.models.parent / "npt_results.json"
    if baseline.exists():
        prev = json.loads(baseline.read_text(encoding="utf-8"))
        delta = r_ind.per_gene_auroc - prev["inductive"]["per_gene_auroc"]
        log(f"vs the 150M run: {prev['inductive']['per_gene_auroc']:.4f} -> "
            f"{r_ind.per_gene_auroc:.4f}  ({delta:+.4f})")
    log(f"total {(time.time()-t_start)/3600:.2f} h; wrote {results_path.name}")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description="Rebuild everything for one backbone.")
    ap.add_argument("--config", default="configs/650m.yaml")
    ap.add_argument("--force", action="store_true", help="redo stages even if outputs exist")
    args = ap.parse_args()
    return run(Config.load(args.config), force=args.force)


if __name__ == "__main__":
    raise SystemExit(main())
