"""Evaluation metrics for variant effect prediction.

The headline choice here is reporting *two* AUROCs, because on this dataset
they measure different things and only one of them is the number a clinician
cares about.

  pooled AUROC     computed over every variant at once. Inflated, because the
                   panel's genes have wildly different pathogenic base rates -
                   ABCA4 is 95% pathogenic, others are near 50%. A predictor
                   that learned nothing except "variants in constrained genes
                   are usually pathogenic" scores well on this.

  per-gene AUROC   computed within each gene, then averaged. This is the
                   question actually being asked in the clinic: given a variant
                   in *this* gene, is it damaging? Between-gene base rates
                   cannot help here, so it is the honest measure.

Pooled AUROC is reported anyway, since it is what most papers quote and you
need it to compare against them - but the gap between the two is itself a
diagnostic, and a large gap means the pooled figure is mostly base rate.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    balanced_accuracy_score,
    roc_auc_score,
    roc_curve,
)


def _binary(labels: pd.Series) -> np.ndarray:
    return (labels == "pathogenic").astype(int).to_numpy()


def pooled_metrics(df: pd.DataFrame, score_col: str) -> dict:
    """Metrics over all variants at once."""
    mask = df[score_col].notna()
    sub = df[mask]
    if sub.empty or sub["label"].nunique() < 2:
        return {"n": int(len(sub)), "auroc": np.nan, "auprc": np.nan}

    y = _binary(sub["label"])
    s = sub[score_col].to_numpy()
    base_rate = float(y.mean())
    return {
        "n": int(len(sub)),
        "n_genes": int(sub["gene"].nunique()),
        "base_rate": round(base_rate, 4),
        "auroc": round(float(roc_auc_score(y, s)), 4),
        "auprc": round(float(average_precision_score(y, s)), 4),
        # AUPRC is only interpretable against the base rate; a model scoring
        # 0.80 on a corpus that is 70% positive has barely moved.
        "auprc_lift": round(float(average_precision_score(y, s)) / base_rate, 3),
        "balanced_acc_at_best_j": round(_best_youden(y, s)[1], 4),
        "threshold_at_best_j": round(_best_youden(y, s)[0], 4),
    }


def _best_youden(y: np.ndarray, s: np.ndarray) -> tuple[float, float]:
    """Threshold maximising Youden's J, and the balanced accuracy there."""
    fpr, tpr, thresholds = roc_curve(y, s)
    j = tpr - fpr
    best = int(np.argmax(j))
    thr = float(thresholds[best])
    return thr, float(balanced_accuracy_score(y, (s >= thr).astype(int)))


def per_gene_metrics(df: pd.DataFrame, score_col: str, min_per_class: int = 3) -> pd.DataFrame:
    """AUROC/AUPRC within each gene.

    Genes without at least `min_per_class` of both labels are excluded: AUROC
    on two benign variants is noise, and averaging it in would mostly measure
    which genes happen to be sparsely annotated.
    """
    rows = []
    for gene, grp in df.groupby("gene"):
        grp = grp[grp[score_col].notna()]
        y = _binary(grp["label"])
        if y.sum() < min_per_class or (len(y) - y.sum()) < min_per_class:
            continue
        s = grp[score_col].to_numpy()
        rows.append({
            "gene": gene,
            "n": int(len(grp)),
            "n_pathogenic": int(y.sum()),
            "base_rate": round(float(y.mean()), 3),
            "auroc": round(float(roc_auc_score(y, s)), 4),
            "auprc": round(float(average_precision_score(y, s)), 4),
        })
    return pd.DataFrame(rows).sort_values("auroc", ascending=False).reset_index(drop=True)


def summarise(df: pd.DataFrame, score_col: str, name: str = "") -> dict:
    """Pooled + per-gene summary for one score column."""
    pooled = pooled_metrics(df, score_col)
    by_gene = per_gene_metrics(df, score_col)
    return {
        "score": name or score_col,
        "pooled": pooled,
        "per_gene": {
            "n_genes_evaluable": int(len(by_gene)),
            "mean_auroc": round(float(by_gene["auroc"].mean()), 4) if len(by_gene) else np.nan,
            "median_auroc": round(float(by_gene["auroc"].median()), 4) if len(by_gene) else np.nan,
            "std_auroc": round(float(by_gene["auroc"].std()), 4) if len(by_gene) else np.nan,
            "mean_auprc": round(float(by_gene["auprc"].mean()), 4) if len(by_gene) else np.nan,
            "worst_gene": by_gene.iloc[-1]["gene"] if len(by_gene) else None,
            "worst_auroc": float(by_gene.iloc[-1]["auroc"]) if len(by_gene) else np.nan,
            "best_gene": by_gene.iloc[0]["gene"] if len(by_gene) else None,
            "best_auroc": float(by_gene.iloc[0]["auroc"]) if len(by_gene) else np.nan,
        },
        "_by_gene_frame": by_gene,
    }


def comparison_table(df: pd.DataFrame, score_cols: dict[str, str], split: str | None = None) -> pd.DataFrame:
    """One row per scoring scheme: pooled and per-gene metrics side by side."""
    sub = df if split is None else df[df["split"] == split]
    rows = []
    for name, col in score_cols.items():
        if col not in sub.columns:
            continue
        s = summarise(sub, col, name)
        rows.append({
            "score": name,
            "n": s["pooled"]["n"],
            "pooled_auroc": s["pooled"]["auroc"],
            "pooled_auprc": s["pooled"]["auprc"],
            "auprc_lift": s["pooled"].get("auprc_lift"),
            "per_gene_auroc": s["per_gene"]["mean_auroc"],
            "per_gene_std": s["per_gene"]["std_auroc"],
            "n_genes": s["per_gene"]["n_genes_evaluable"],
        })
    return pd.DataFrame(rows).sort_values("per_gene_auroc", ascending=False).reset_index(drop=True)
