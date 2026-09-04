"""Precompute saturation matrices for the whole panel.

    python -m vep.precompute --dry-run          # plan and time estimate
    python -m vep.precompute                    # run it

Masked-marginal scanning costs one forward pass per residue, so covering all
~200,000 residues in the panel is an overnight job rather than a coffee break.
Everything here is built around that fact:

  resumable      each gene's matrices are written as soon as that gene finishes,
                 and an already-cached gene is skipped. Stop with Ctrl+C and
                 restart whenever; nothing is lost and nothing is redone.

  shortest-first the default order. If the job is interrupted at any point, the
                 largest possible number of genes is finished - RYR1 alone
                 (5,038 residues) costs about as much as the sixty shortest
                 genes put together, so doing it first would be a poor bet.

  dry run        prints the plan and an estimate before committing hours to it.

The probability pass is cheap by comparison (a few seconds per gene) but
depends on the masked matrix, so it always follows it for a given gene.
"""

from __future__ import annotations

import argparse
import json
import signal
import sys
import time
from pathlib import Path

import numpy as np

from vep.config import Config, resolve_device
from vep.esm.saturation import SaturationStore, fingerprint_file
from vep.gpu_guard import wait_if_paused

# Calibrated from a measured run: TP53 (393 residues, single window) took 58s
# for a full masked scan. Cost scales as positions x window length, since every
# masked position is one forward pass over its window.
COST_UNITS_PER_SECOND = 393 * 393 / 58.0

_stop = False


def _handle_sigint(signum, frame):
    global _stop
    if _stop:
        print("\nsecond interrupt - exiting immediately", flush=True)
        sys.exit(130)
    _stop = True
    print("\ninterrupt received; finishing the current gene then stopping "
          "(press Ctrl+C again to abort now)", flush=True)


def cost_units(length: int, max_len: int) -> int:
    """Relative cost of a full masked scan: positions x window length."""
    return length * min(length, max_len)


def fmt_duration(seconds: float) -> str:
    if seconds < 90:
        return f"{seconds:.0f}s"
    if seconds < 5400:
        return f"{seconds/60:.1f}m"
    return f"{seconds/3600:.1f}h"


def plan(
    proteins: dict,
    store: SaturationStore,
    schemes: list[str],
    order: str,
    max_length: int | None,
    genes: list[str] | None,
    max_len_window: int,
) -> list[tuple[str, int]]:
    """Genes still needing work, in execution order, with their lengths."""
    todo: list[tuple[str, int]] = []
    for gene, rec in proteins.items():
        if genes is not None and gene not in genes:
            continue
        length = rec["length"]
        if max_length is not None and length > max_length:
            continue
        # A gene is done only if every requested scheme is present.
        if all(store.has(gene, s) for s in schemes):
            continue
        todo.append((gene, length))

    if order == "short":
        todo.sort(key=lambda t: t[1])
    elif order == "long":
        todo.sort(key=lambda t: -t[1])
    elif order == "alpha":
        todo.sort(key=lambda t: t[0])
    else:
        raise ValueError(f"unknown order {order!r}")
    return todo


def run(args) -> int:
    cfg = Config.load(args.config)
    cfg.paths.mkdirs()
    proteins = json.loads(
        (cfg.paths.processed / "proteins.json").read_text(encoding="utf-8")
    )

    # proteins.json holds every gene we resolved a sequence for (198), but the
    # API only serves those that survived the ClinVar label filters (183) and
    # 404s on the rest. Precomputing the other 15 burns GPU hours on matrices
    # nothing can display, so restrict to the servable set by default.
    if not args.all_proteins:
        import pandas as pd

        labelled = set(
            pd.read_parquet(cfg.paths.processed / "variants.parquet")["gene"].unique()
        )
        skipped = sorted(set(proteins) - labelled)
        proteins = {g: r for g, r in proteins.items() if g in labelled}
        if skipped:
            print(f"note       skipping {len(skipped)} genes with no servable "
                  f"labels ({', '.join(skipped[:6])}"
                  f"{', ...' if len(skipped) > 6 else ''})")

    schemes = [s.strip() for s in args.schemes.split(",") if s.strip()]
    for s in schemes:
        if s not in ("wt", "masked", "probability"):
            raise SystemExit(f"unknown scheme {s!r}")

    ckpt = cfg.paths.models / "npt.pt"
    fingerprint = fingerprint_file(ckpt) if ckpt.exists() else None
    if "probability" in schemes and fingerprint is None:
        raise SystemExit(
            f"probability requested but no checkpoint at {ckpt}; "
            f"train a model first or pass --schemes masked"
        )

    store = SaturationStore(
        root=cfg.paths.cache / "saturation",
        backbone_slug=cfg.backbone_slug(),
        model_fingerprint=fingerprint,
    )

    genes = [g.strip().upper() for g in args.genes.split(",")] if args.genes else None
    todo = plan(
        proteins, store, schemes, args.order, args.max_length, genes,
        cfg.backbone.max_length,
    )
    if args.limit:
        todo = todo[: args.limit]

    total_units = sum(cost_units(L, cfg.backbone.max_length) for _, L in todo)
    est = total_units / COST_UNITS_PER_SECOND if "masked" in schemes else 0.0

    print(f"backbone   {cfg.backbone.name}")
    print(f"cache      {store.root}")
    print(f"schemes    {', '.join(schemes)}")
    if fingerprint:
        print(f"checkpoint {ckpt.name} ({fingerprint})")
    print(f"genes      {len(todo)} to do, "
          f"{sum(1 for g in proteins if all(store.has(g, s) for s in schemes))} already cached")
    print(f"residues   {sum(L for _, L in todo):,}")
    print(f"estimate   {fmt_duration(est)} for the masked scans "
          f"(+~5s/gene for probabilities)\n")

    if args.dry_run:
        print("first 10 in order:")
        for gene, L in todo[:10]:
            print(f"  {gene:10s} L={L:5d}  ~{fmt_duration(cost_units(L, cfg.backbone.max_length)/COST_UNITS_PER_SECOND)}")
        if len(todo) > 10:
            longest = max(todo, key=lambda t: t[1])
            print(f"  ... and {len(todo)-10} more; longest is {longest[0]} "
                  f"(L={longest[1]}, ~{fmt_duration(cost_units(longest[1], cfg.backbone.max_length)/COST_UNITS_PER_SECOND)})")
        return 0

    if not todo:
        print("nothing to do")
        return 0

    from vep.esm.backbone import ESM2Backbone, llr_matrix

    backbone = ESM2Backbone(
        model_name=cfg.backbone.name,
        device=resolve_device(cfg.backbone.device),
        fp16=cfg.backbone.fp16,
        max_length=cfg.backbone.max_length,
        embed_layer=cfg.backbone.embed_layer,
    )
    predictor = None
    if "probability" in schemes:
        from vep.models.predictor import VariantPredictor

        predictor = VariantPredictor(cfg, checkpoint_path=ckpt)

    signal.signal(signal.SIGINT, _handle_sigint)

    import torch

    def batch_for(length: int) -> int:
        """Token-budget batching, so peak activation memory stays flat.

        A fixed batch size makes memory scale with sequence length: 16 x 781
        tokens fits on a 4 GB card, 16 x 1022 does not, and the panel is sorted
        shortest-first so the largest windows arrive last - i.e. hours in, on an
        unattended run. Holding batch_size x window_length roughly constant
        keeps every gene at about the same peak instead.
        """
        window = min(length, cfg.backbone.max_length)
        return max(1, min(32, cfg.backbone.batch_tokens // max(window, 1)))

    done_units = 0
    t_start = time.time()
    for i, (gene, length) in enumerate(todo, 1):
        if _stop:
            print("stopped by request")
            break
        seq = proteins[gene]["sequence"]
        t_gene = time.time()

        # Honour a thermal pause between batches rather than only between genes:
        # USH2A alone runs for half an hour, so gene granularity would let the
        # card stay hot for far too long before the guard could take effect.
        paused_this_gene = 0.0

        def thermal_gate(done: int, total: int) -> None:
            nonlocal paused_this_gene
            paused_this_gene += wait_if_paused(
                on_wait=lambda left: print(
                    f"\n      thermal pause: waiting {left:.0f}s", flush=True
                )
            )

        wait_if_paused()

        if "wt" in schemes and not store.has(gene, "wt"):
            log_probs, _ = backbone.wt_marginals(seq)
            store.put(gene, "wt", llr_matrix(log_probs, seq))

        if ("masked" in schemes or "probability" in schemes) and not store.has(gene, "masked"):
            log_probs = backbone.masked_marginals(
                seq, positions=None, batch_size=batch_for(length),
                progress=thermal_gate,
            )
            store.put(gene, "masked", llr_matrix(log_probs, seq))

        if "probability" in schemes and not store.has(gene, "probability"):
            from vep.models.predictor import saturation_probabilities

            masked = store.get(gene, "masked")
            store.put(
                gene,
                "probability",
                saturation_probabilities(
                    predictor, gene, seq, masked, batch_size=16, progress=thermal_gate
                ),
            )

        # Hand cached-but-unused blocks back to the driver between genes. The
        # allocator otherwise holds its high-water mark, which on a 4 GB card
        # left under 100 MB free before the long proteins were even reached.
        torch.cuda.empty_cache()

        done_units += cost_units(length, cfg.backbone.max_length)
        elapsed = time.time() - t_start
        rate = done_units / max(elapsed, 1e-9)
        remaining = sum(
            cost_units(L, cfg.backbone.max_length) for _, L in todo[i:]
        ) / max(rate, 1e-9)
        print(
            f"  [{i:3d}/{len(todo)}] {gene:10s} L={length:5d} "
            f"bs={batch_for(length):<2d} took {fmt_duration(time.time()-t_gene):>6s}"
            + (f" (+{paused_this_gene:.0f}s paused)" if paused_this_gene else "") + "  "
            f"elapsed {fmt_duration(elapsed):>6s}  eta {fmt_duration(remaining):>6s}",
            flush=True,
        )

    print(f"\n{fmt_duration(time.time()-t_start)} total")
    print("cache:", store.stats())
    if predictor is not None:
        predictor.close()
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Precompute saturation matrices for the gene panel."
    )
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument(
        "--schemes",
        default="masked,probability",
        help="comma-separated: wt, masked, probability (default: masked,probability)",
    )
    ap.add_argument(
        "--order",
        default="short",
        choices=("short", "long", "alpha"),
        help="shortest-first finishes the most genes if interrupted (default)",
    )
    ap.add_argument("--genes", default=None, help="comma-separated subset")
    ap.add_argument(
        "--max-length",
        type=int,
        default=None,
        help="skip proteins longer than this; the handful of very long ones "
             "dominate total runtime",
    )
    ap.add_argument("--limit", type=int, default=None, help="stop after N genes")
    ap.add_argument(
        "--all-proteins",
        action="store_true",
        help="include genes the API cannot serve (no ClinVar labels after "
             "filtering); off by default",
    )
    ap.add_argument("--dry-run", action="store_true", help="print the plan and exit")
    return run(ap.parse_args())


if __name__ == "__main__":
    raise SystemExit(main())
