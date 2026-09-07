"""Assemble every evaluation result into one JSON for the metrics dashboard.

The per-gene ProteinNPT scores are recomputed here rather than read from
`npt_results.json`, which only stores panel-level aggregates. The per-gene view
is the one worth looking at: the headline "+0.084 AUROC" is an average that
hides the fact that most of the gain comes from a handful of genes where
likelihood-based scoring fails outright.

Also produces a reliability curve, so the calibration claim on the front page
is something a reader can check rather than take on trust.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import roc_auc_score, roc_curve

from vep.config import Config, resolve_device
from vep.models.npt import LABEL_MASK, ProteinNPT, load_state_dict_compat
from vep.train.datasets import build_features, label_tokens, split_indices, to_tensors


@torch.no_grad()
def _predict(model, feats, query, pool, device, same_gene: bool, n_context: int = 31):
    rng = np.random.default_rng(0)
    out = np.zeros(len(query), dtype=np.float32)
    by_gene = (
        {int(g): pool[feats.gene_idx[pool] == g] for g in np.unique(feats.gene_idx[pool])}
        if same_gene
        else {}
    )
    for i, q in enumerate(query):
        if same_gene:
            same = by_gene.get(int(feats.gene_idx[q]), np.array([], dtype=np.int64))
            same = same[same != q]
            ctx = (
                rng.choice(same, n_context, replace=False)
                if len(same) >= n_context
                else np.concatenate(
                    [same, rng.choice(pool, n_context - len(same), replace=False)]
                )
            )
        else:
            ctx = rng.choice(pool, n_context, replace=False)
            ctx = ctx[ctx != q]
        rows = np.concatenate([[q], ctx])
        labels = label_tokens(feats.labels[rows], device)
        labels[0] = LABEL_MASK
        out[i] = model(**to_tensors(feats, rows, device), labels=labels).pathogenicity_logit[0].item()
    return out


def _alphamissense_block() -> dict | None:
    """Comparison against AlphaMissense, if it has been run.

    Optional because it needs a 1.2 GB third-party download that this repo does
    not redistribute. Absent, the dashboard simply omits the row - it does not
    fall back to a stale number.
    """
    try:
        return json.loads(
            Path("artifacts/alphamissense_comparison.json").read_text(encoding="utf-8")
        )
    except FileNotFoundError:
        return None


def _backbone_scale_block() -> dict | None:
    """What the 4x larger backbone bought, and what multi-task training cost.

    Both are measured but were only ever written down in the guide.
    """
    try:
        big = json.loads(
            Path("artifacts/npt_results_esm2_t33_650M_UR50D.json").read_text())
        multi = json.loads(Path("artifacts/multitask_results.json").read_text())
        zs = pd.read_parquet(
            "artifacts/cache/zeroshot_all_esm2_t33_650M_UR50D.parquet")
    except FileNotFoundError:
        return None

    test = zs[zs.split == "test"]
    aurocs = []
    for _, sub in test.groupby("gene"):
        y = (sub.label == "pathogenic").astype(int)
        if y.sum() < 3 or (len(y) - y.sum()) < 3:
            continue
        aurocs.append(roc_auc_score(y, sub.esm_masked))

    return {
        "rows": [
            {"metric": "zero-shot masked-marginal",
             "small": 0.8435, "large": round(float(np.mean(aurocs)), 4)},
            {"metric": "with ProteinNPT",
             "small": 0.9271,
             "large": round(float(big["inductive"]["per_gene_auroc"]), 4)},
        ],
        "multitask": {
            "single_task": 0.9271,
            "multi_task": round(float(multi["pathogenicity"]["per_gene_auroc"]), 4),
        },
    }


def _fitness_block() -> dict | None:
    """The GRB2 result, paired across the leaky and the clean split.

    Both splits are reported deliberately. The leaky one is what a normal
    held-out-positions protocol produces and it says supervised training wins;
    the clean one says it does not. Showing only the honest number would hide
    the more useful fact, which is how far apart the two protocols land on the
    same models.
    """
    try:
        boot = json.loads(Path("artifacts/fitness_bootstrap.json").read_text())
        multi = json.loads(Path("artifacts/multitask_results.json").read_text())
        only = json.loads(Path("artifacts/fitness_results.json").read_text())
    except FileNotFoundError:
        return None

    sp = boot["spearman"]
    delta = boot["paired_delta_vs_zero_shot"]
    leaky = {
        "zero_shot": round(float(multi["grb2_fitness"]["zero_shot_spearman"]), 4),
        "multi_task": round(float(multi["grb2_fitness"]["test_spearman"]), 4),
        "fitness_only": round(float(only["all_test"]["spearman"]), 4),
    }

    models = []
    for key, label in [("zero_shot", "zero-shot mutant-marginal"),
                       ("multi_task", "multi-task ProteinNPT"),
                       ("fitness_only", "fitness-only ProteinNPT")]:
        row = {
            "key": key,
            "label": label,
            "leaky": leaky[key],
            "strict": sp[key]["rho"],
            "ci95": sp[key]["ci95"],
        }
        if key in delta:
            row["delta"] = delta[key]["delta"]
            row["delta_ci95"] = delta[key]["ci95"]
            row["distinguishable"] = delta[key]["distinguishable"]
        models.append(row)

    return {
        "assay": "GRB2 binding DMS (METL project)",
        "metric": "Spearman rho against measured binding fitness",
        "splits": {
            "leaky": {
                "n": int(multi["grb2_fitness"]["n_test"]),
                "label": "held-out positions",
                "note": "82% of these variants are double mutants sharing one "
                        "position with training",
            },
            "strict": {
                "n": int(boot["test_set"]["n"]),
                "label": "no training position",
                "note": boot["test_set"]["definition"],
            },
        },
        "models": models,
        "conclusion": boot["conclusion"],
        "method": boot["method"],
    }


def build_report(cfg: Config, out_path: Path) -> dict:
    device = resolve_device(cfg.backbone.device)
    feats = build_features(cfg)
    idx = split_indices(feats)

    ckpt_path = cfg.paths.models / "npt.pt"
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    model = ProteinNPT(
        d_emb=feats.d_emb,
        d_model=cfg.npt.d_model,
        n_layers=cfg.npt.n_layers,
        n_heads=cfg.npt.n_heads,
        dropout=cfg.npt.dropout,
    ).to(device)
    load_state_dict_compat(model, ckpt["state_dict"])
    model.eval()

    test = idx["test"]
    print(f"scoring {len(test)} held-out variants ...", flush=True)
    p_ind = _predict(model, feats, test, idx["train"], device, same_gene=False)
    p_trn = _predict(
        model, feats, test, np.concatenate([idx["train"], test]), device, same_gene=True
    )

    gene_names = sorted(feats.frame["gene"].unique())
    y_all = feats.labels[test]
    zs = feats.subst[test, 40]          # esm_masked, pathogenic orientation

    per_gene = []
    for g in np.unique(feats.gene_idx[test]):
        m = feats.gene_idx[test] == g
        y = y_all[m]
        if y.sum() < 3 or (len(y) - y.sum()) < 3:
            continue
        per_gene.append(
            {
                "gene": gene_names[g],
                "n": int(m.sum()),
                "base_rate": round(float(y.mean()), 3),
                "zeroshot": round(float(roc_auc_score(y, zs[m])), 4),
                "npt_inductive": round(float(roc_auc_score(y, p_ind[m])), 4),
                "npt_transductive": round(float(roc_auc_score(y, p_trn[m])), 4),
            }
        )
    for row in per_gene:
        row["gain"] = round(row["npt_inductive"] - row["zeroshot"], 4)
    per_gene.sort(key=lambda r: -r["gain"])

    # Reliability curve from the calibrated probabilities.
    cal = json.loads((cfg.paths.models.parent / "calibration.json").read_text())
    probs = 1.0 / (1.0 + np.exp(-(cal["a"] * p_ind + cal["b"])))
    bins = []
    edges = np.linspace(0, 1, 11)
    for i in range(10):
        lo, hi = edges[i], edges[i + 1]
        m = (probs >= lo) & (probs < hi if i < 9 else probs <= 1.0)
        if m.sum() == 0:
            continue
        bins.append(
            {
                "bin": f"{lo:.1f}-{hi:.1f}",
                "n": int(m.sum()),
                "mean_predicted": round(float(probs[m].mean()), 4),
                "observed": round(float(y_all[m].mean()), 4),
            }
        )

    fpr, tpr, _ = roc_curve(y_all, p_ind)
    step = max(1, len(fpr) // 200)      # thin the curve for transport
    fpr_z, tpr_z, _ = roc_curve(y_all, zs)
    step_z = max(1, len(fpr_z) // 200)

    zs_report = json.loads(Path("artifacts/zeroshot_benchmark.json").read_text())
    npt_report = json.loads(Path("artifacts/npt_results.json").read_text())

    report = {
        "dataset": zs_report["dataset"],
        "splits": {
            k: {"variants": int(len(v)), "genes": int(len(np.unique(feats.gene_idx[v])))}
            for k, v in idx.items()
        },
        "headline": {
            "test_per_gene_auroc": {
                "blosum62": 0.7227,
                "esm2_wt": 0.8381,
                "esm2_masked": 0.8435,
                "protein_npt_inductive": round(float(npt_report["inductive"]["per_gene_auroc"]), 4),
                "protein_npt_transductive": round(float(npt_report["transductive"]["per_gene_auroc"]), 4),
            },
            "test_pooled_auroc": {
                "blosum62": 0.6879,
                "esm2_wt": 0.8408,
                "esm2_masked": 0.8422,
                "protein_npt_inductive": round(float(npt_report["inductive"]["pooled_auroc"]), 4),
                "protein_npt_transductive": round(float(npt_report["transductive"]["pooled_auroc"]), 4),
            },
            "n_genes_evaluable": len(per_gene),
            "n_improved": sum(1 for r in per_gene if r["gain"] > 0),
            "n_below_chance_zeroshot": sum(1 for r in per_gene if r["zeroshot"] < 0.5),
            "n_below_chance_npt": sum(1 for r in per_gene if r["npt_inductive"] < 0.5),
        },
        "per_gene": per_gene,
        "training": npt_report["meta"]["history"],
        "best_epoch": npt_report["meta"]["best_epoch"],
        "calibration": {
            **cal,
            "reliability": bins,
        },
        "roc": {
            "npt": [[round(float(a), 4), round(float(b), 4)] for a, b in zip(fpr[::step], tpr[::step])],
            "zeroshot": [[round(float(a), 4), round(float(b), 4)] for a, b in zip(fpr_z[::step_z], tpr_z[::step_z])],
        },
        "fitness": _fitness_block(),
        "alphamissense": _alphamissense_block(),
        "backbone_scale": _backbone_scale_block(),
        "compute_minutes": zs_report["compute_minutes"],
        "spearman_wt_vs_masked": zs_report["spearman_wt_vs_masked"],
        "backbone": cfg.backbone.name,
    }

    am = report["alphamissense"]
    if am:
        # Same held-out variants, same metric, so it belongs in the same table.
        report["headline"]["test_per_gene_auroc"]["alphamissense"] =             am["summary"]["am_pathogenicity"]["per_gene"]
        report["headline"]["test_pooled_auroc"]["alphamissense"] =             am["summary"]["am_pathogenicity"]["pooled"]

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, indent=1), encoding="utf-8")
    print(f"wrote {out_path}")
    return report


if __name__ == "__main__":
    cfg = Config.load("configs/default.yaml")
    r = build_report(cfg, Path("artifacts/metrics.json"))
    h = r["headline"]
    print(f"\nper-gene AUROC (held-out test genes):")
    for k, v in h["test_per_gene_auroc"].items():
        print(f"  {k:28s} {v:.4f}")
    print(f"\ngenes improved: {h['n_improved']}/{h['n_genes_evaluable']}")
    print(f"below chance: zero-shot {h['n_below_chance_zeroshot']} -> NPT {h['n_below_chance_npt']}")
