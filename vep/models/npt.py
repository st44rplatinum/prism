"""ProteinNPT: a non-parametric transformer head over frozen ESM-2 features.

After Notin et al., "ProteinNPT: Improving Protein Property Prediction and
Design with Non-Parametric Transformers" (NeurIPS 2023).

The idea that makes this different from an MLP on embeddings: a conventional
model maps one variant to one prediction, and the training labels are visible
only through the weights. An NPT instead takes a *whole batch* as its input -
the variant being predicted plus a set of retrieved labelled neighbours - and
attends across the batch, with the labels themselves as an input column. The
label of the variant being predicted is replaced by a learned [MASK] embedding,
so the model has to reconstruct it from its neighbours.

Why that matters here specifically. The zero-shot benchmark found five genes
scoring *below chance* - PSEN1, PSEN2, MEFV, FUS, SMAD4 - and masked-marginal
scoring did not fix any of them (PSEN2 moved 0.000). These are gain-of-function
and aggregation mechanisms: the pathogenic variants are evolutionarily
plausible, so no likelihood-based score will ever rank them correctly. But
given a handful of labelled variants from the same gene, a model that can
*look at those labels at inference time* can learn that this gene's pathogenic
variants sit exactly where ESM-2 says "fine". That is a thing only the
non-parametric mechanism can do.

Attention is axial over a (datapoints x tokens) grid, which is what keeps it
affordable:

  within-datapoint   each variant's few feature tokens attend to each other
  between-datapoint  each token slot attends across the batch of variants

Full ProteinNPT runs the between-datapoint axis over every residue of every
sequence. That is not affordable on a 4 GB card and is not necessary here: a
single amino-acid substitution is well summarised by a handful of tokens drawn
from the frozen features at the mutated position, so the token axis is 5 long
rather than 1022.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

# Label column vocabulary. MASK is what the model sees in place of the label it
# has to predict; PAD covers unused neighbour slots.
LABEL_BENIGN = 0
LABEL_PATHOGENIC = 1
LABEL_MASK = 2
LABEL_PAD = 3
N_LABEL_TOKENS = 4

# Tasks sharing one model. Pathogenicity is a class label; DMS fitness is a
# continuous measurement, so the label column has to carry both kinds.
TASK_PATHOGENICITY = 0
TASK_FITNESS = 1
N_TASKS = 2


@dataclass
class NPTOutput:
    pathogenicity_logit: torch.Tensor   # (N,)
    fitness: torch.Tensor               # (N,)
    hidden: torch.Tensor                # (N, T, d_model)


class FeatureTokeniser(nn.Module):
    """Turn per-variant frozen features into a short sequence of tokens.

    Each block gets its own projection rather than being concatenated into one
    long vector, so between-datapoint attention can compare variants on one
    aspect at a time - two variants can be neighbours because they sit in
    similar structural context even if their substitutions differ.
    """

    def __init__(self, d_emb: int, d_model: int, dropout: float = 0.1):
        super().__init__()
        self.emb_proj = nn.Linear(d_emb, d_model)        # embedding at the mutated residue
        self.logprob_proj = nn.Linear(20, d_model)       # ESM distribution at that residue
        self.subst_proj = nn.Linear(42, d_model)         # wt/mut identity + LLR + BLOSUM
        self.context_proj = nn.Linear(d_emb, d_model)    # mean-pooled protein embedding

        self.token_type = nn.Parameter(torch.zeros(4, d_model))
        nn.init.normal_(self.token_type, std=0.02)
        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        emb: torch.Tensor,        # (N, d_emb)
        logprobs: torch.Tensor,   # (N, 20)
        subst: torch.Tensor,      # (N, 42)
        context: torch.Tensor,    # (N, d_emb)
    ) -> torch.Tensor:
        tokens = torch.stack(
            [
                self.emb_proj(emb),
                self.logprob_proj(logprobs),
                self.subst_proj(subst),
                self.context_proj(context),
            ],
            dim=1,
        )                                          # (N, 4, d_model)
        tokens = tokens + self.token_type.unsqueeze(0)
        return self.dropout(self.norm(tokens))


class AxialBlock(nn.Module):
    """One NPT layer: within-datapoint attention, then between-datapoint.

    Pre-norm residual throughout. The between-datapoint pass is the whole point
    of the architecture - it is the only path by which a neighbour's label can
    reach the prediction.
    """

    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.1):
        super().__init__()
        self.norm_within = nn.LayerNorm(d_model)
        self.attn_within = nn.MultiheadAttention(
            d_model, n_heads, dropout=dropout, batch_first=True
        )
        self.norm_between = nn.LayerNorm(d_model)
        self.attn_between = nn.MultiheadAttention(
            d_model, n_heads, dropout=dropout, batch_first=True
        )
        self.norm_ffn = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, 4 * d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(4 * d_model, d_model),
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, pad_mask: torch.Tensor | None = None) -> torch.Tensor:
        """x: (N, T, d_model); pad_mask: (N,) True where the datapoint is padding."""
        n, t, d = x.shape

        # --- within-datapoint: each variant's tokens attend to each other ---
        h = self.norm_within(x)
        attn, _ = self.attn_within(h, h, h, need_weights=False)
        x = x + self.dropout(attn)

        # --- between-datapoint: each token slot attends across the batch ---
        # Transpose so the token axis becomes the batch axis and the datapoint
        # axis becomes the sequence being attended over.
        h = self.norm_between(x).transpose(0, 1)          # (T, N, d)
        key_padding_mask = None
        if pad_mask is not None:
            # MultiheadAttention wants (batch, seq) = (T, N).
            key_padding_mask = pad_mask.unsqueeze(0).expand(t, n)
        attn, _ = self.attn_between(
            h, h, h, key_padding_mask=key_padding_mask, need_weights=False
        )
        x = x + self.dropout(attn.transpose(0, 1))

        x = x + self.dropout(self.ffn(self.norm_ffn(x)))
        return x


class ProteinNPT(nn.Module):
    """Non-parametric transformer over a batch of variants.

    A forward pass consumes one *set* of variants jointly. Every datapoint whose
    label is masked is a prediction target; the rest act as labelled context.
    """

    def __init__(
        self,
        d_emb: int = 640,
        d_model: int = 256,
        n_layers: int = 4,
        n_heads: int = 8,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.d_model = d_model
        self.tokeniser = FeatureTokeniser(d_emb, d_model, dropout)
        # The label column. This embedding is the only route by which a
        # neighbour's label enters the computation.
        self.label_embed = nn.Embedding(N_LABEL_TOKENS, d_model)
        # Continuous labels get their own projection: a fitness measurement
        # cannot be one of four categories, and rounding it into bins would
        # throw away exactly the resolution the regression is judged on.
        self.value_proj = nn.Linear(1, d_model)
        # Zero-initialised on purpose. It is added to every label token, so
        # starting at zero means a checkpoint trained before this existed loads
        # with strict=False and behaves bit-identically - the 150M model the API
        # serves must not shift because a second task was added.
        self.task_embed = nn.Embedding(N_TASKS, d_model)
        nn.init.zeros_(self.task_embed.weight)
        self.label_type = nn.Parameter(torch.zeros(1, d_model))
        nn.init.normal_(self.label_type, std=0.02)

        self.blocks = nn.ModuleList(
            [AxialBlock(d_model, n_heads, dropout) for _ in range(n_layers)]
        )
        self.norm_out = nn.LayerNorm(d_model)
        self.head_pathogenicity = nn.Linear(d_model, 1)
        self.head_fitness = nn.Linear(d_model, 1)

    def forward(
        self,
        emb: torch.Tensor,
        logprobs: torch.Tensor,
        subst: torch.Tensor,
        context: torch.Tensor,
        labels: torch.Tensor,             # (N,) values in the label vocabulary
        pad_mask: torch.Tensor | None = None,
        values: torch.Tensor | None = None,   # (N,) continuous targets
        task: torch.Tensor | None = None,     # (N,) TASK_* ids
    ) -> NPTOutput:
        tokens = self.tokeniser(emb, logprobs, subst, context)     # (N, 4, d)
        label_tok = self.label_embed(labels)                        # (N, d)

        if values is not None and task is not None:
            # Continuous rows carry their measured value, EXCEPT where masked -
            # a masked row must keep the [MASK] embedding, or the target leaks
            # straight into the input it is supposed to be predicted from.
            is_reg = (task == TASK_FITNESS) & (labels != LABEL_MASK) & (labels != LABEL_PAD)
            if is_reg.any():
                projected = self.value_proj(values.unsqueeze(-1).float())
                label_tok = torch.where(is_reg.unsqueeze(-1), projected, label_tok)
        if task is not None:
            label_tok = label_tok + self.task_embed(task)
        label_tok = label_tok + self.label_type
        x = torch.cat([tokens, label_tok.unsqueeze(1)], dim=1)      # (N, 5, d)

        for block in self.blocks:
            x = block(x, pad_mask=pad_mask)

        # Read out from the label slot: it is the position that aggregates
        # label information from the rest of the batch.
        h = self.norm_out(x[:, -1])
        return NPTOutput(
            pathogenicity_logit=self.head_pathogenicity(h).squeeze(-1),
            fitness=self.head_fitness(h).squeeze(-1),
            hidden=x,
        )


def mask_labels(
    labels: torch.Tensor,
    mask_prob: float,
    always_mask: torch.Tensor | None = None,
    generator: torch.Generator | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Replace a random subset of labels with [MASK].

    Returns (masked_labels, is_masked). `always_mask` forces specific rows to be
    masked - at inference the query variant must always be, or the model would
    simply read its own answer out of the input.
    """
    is_masked = torch.rand(labels.shape, device=labels.device, generator=generator) < mask_prob
    if always_mask is not None:
        is_masked = is_masked | always_mask
    masked = labels.clone()
    masked[is_masked] = LABEL_MASK
    return masked, is_masked


def npt_loss(
    out: NPTOutput,
    target_pathogenicity: torch.Tensor,
    is_masked: torch.Tensor,
    target_fitness: torch.Tensor | None = None,
    fitness_mask: torch.Tensor | None = None,
    sample_weight: torch.Tensor | None = None,
    w_pathogenicity: float = 1.0,
    w_fitness: float = 0.3,
) -> tuple[torch.Tensor, dict[str, float]]:
    """Multi-task loss, computed only on masked datapoints.

    Scoring unmasked rows would be free marks - their labels were handed to the
    model as input.
    """
    stats: dict[str, float] = {}
    total = torch.zeros((), device=out.pathogenicity_logit.device)

    if is_masked.any():
        logit = out.pathogenicity_logit[is_masked]
        target = target_pathogenicity[is_masked].float()
        losses = F.binary_cross_entropy_with_logits(logit, target, reduction="none")
        if sample_weight is not None:
            w = sample_weight[is_masked]
            loss_path = (losses * w).sum() / w.sum().clamp_min(1e-8)
        else:
            loss_path = losses.mean()
        total = total + w_pathogenicity * loss_path
        stats["loss_pathogenicity"] = float(loss_path.detach())

    if target_fitness is not None and fitness_mask is not None:
        sel = fitness_mask & is_masked
        if sel.any():
            loss_fit = F.mse_loss(out.fitness[sel], target_fitness[sel])
            total = total + w_fitness * loss_fit
            stats["loss_fitness"] = float(loss_fit.detach())

    stats["loss"] = float(total.detach())
    return total, stats


# Parameters added when the continuous-label pathway was introduced for the
# fitness task. Checkpoints trained before that - including the single-task
# pathogenicity head that is actually served - do not contain them.
_CONTINUOUS_LABEL_PARAMS = frozenset(
    {"value_proj.weight", "value_proj.bias", "task_embed.weight"}
)


def load_state_dict_compat(model: "ProteinNPT", state_dict: dict) -> list[str]:
    """Load a checkpoint, tolerating only the continuous-label parameters.

    `task_embed` is zero-initialised and `value_proj` is reached only when a
    fitness label is present, so a pathogenicity-only checkpoint that omits
    them still evaluates bit-identically. That was the intent when they were
    added - but load_state_dict is strict by default, so the served checkpoint
    stopped loading entirely and the API silently fell back to zero-shot.

    Anything else missing is real architecture drift and must still raise:
    strict=False everywhere would turn a shape or naming change into a model
    quietly serving randomly-initialised weights.
    """
    result = model.load_state_dict(state_dict, strict=False)
    unexpected = list(result.unexpected_keys)
    missing = [k for k in result.missing_keys if k not in _CONTINUOUS_LABEL_PARAMS]
    if missing or unexpected:
        raise RuntimeError(
            "checkpoint does not match the model: "
            f"missing={missing} unexpected={unexpected}"
        )
    return list(result.missing_keys)
