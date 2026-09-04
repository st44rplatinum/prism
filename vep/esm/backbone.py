"""ESM-2 wrapper: per-residue embeddings and substitution log-likelihood ratios.

Implements the three zero-shot scoring schemes from Meier et al. (2021),
"Language models enable zero-shot prediction of the effects of mutations on
protein function":

  wt-marginal      one forward pass on the wild-type sequence; read the
                   distribution at the mutated position. Cheap - one pass per
                   window covers every position at once.
  masked-marginal  the position is replaced by <mask> before the pass, so the
                   model cannot simply copy the wild-type residue out of its
                   own input. Reliably the strongest of the three, and the
                   reason it is worth one forward pass per residue.
  mutant-marginal  the full mutant sequence is scored, which is what captures
                   epistasis between substitutions in a multi-mutant.

All three reduce to a log-likelihood ratio

    LLR = log p(mutant residue) - log p(wild-type residue)

at the mutated position, which is the single quantity every downstream model
in this project consumes.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from transformers import AutoTokenizer, EsmForMaskedLM

from vep.constants import AA_ALPHABET, AA_TO_IDX
from vep.esm.windowing import Window, assign_windows, plan_windows, window_for_position


@dataclass
class ProteinFeatures:
    """Per-residue outputs for one protein, indexed by 0-based residue."""

    embeddings: np.ndarray                        # (L, d)  float16
    log_probs_wt: np.ndarray                      # (L, 20) float32
    log_probs_masked: np.ndarray | None = None    # (L, 20) float32, NaN where unscored

    @property
    def length(self) -> int:
        return self.embeddings.shape[0]


class ESM2Backbone:
    """Frozen ESM-2, used only for inference.

    ProteinNPT trains on cached features, so the backbone never needs
    gradients. Everything here runs under inference mode with parameters
    detached, which is what keeps peak memory low enough for a 4 GB card.
    """

    def __init__(
        self,
        model_name: str = "facebook/esm2_t30_150M_UR50D",
        device: str = "cuda",
        fp16: bool = True,
        max_length: int = 1022,
        embed_layer: int = -1,
    ):
        self.model_name = model_name
        self.device = torch.device(device)
        self.max_length = max_length
        self.embed_layer = embed_layer

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        dtype = torch.float16 if (fp16 and self.device.type == "cuda") else torch.float32
        self.model = EsmForMaskedLM.from_pretrained(model_name, dtype=dtype)
        self.model.eval().to(self.device)
        for param in self.model.parameters():
            param.requires_grad_(False)

        # Column indices of the 20 standard amino acids within ESM's 33-token
        # vocabulary. Every log-prob matrix we cache is therefore (L, 20) in
        # AA_ALPHABET order, and no downstream code has to know about ESM's
        # token ids or its special tokens.
        self.aa_token_ids = torch.tensor(
            [self.tokenizer.convert_tokens_to_ids(aa) for aa in AA_ALPHABET],
            device=self.device,
            dtype=torch.long,
        )
        self.mask_id = int(self.tokenizer.mask_token_id)
        self.hidden_size = int(self.model.config.hidden_size)

    # ------------------------------------------------------------------
    # plumbing
    # ------------------------------------------------------------------
    def _encode(self, seqs: list[str]) -> dict[str, torch.Tensor]:
        batch = self.tokenizer(
            seqs, return_tensors="pt", padding=True, add_special_tokens=True
        )
        return {k: v.to(self.device) for k, v in batch.items()}

    def _aa_log_probs(self, logits: torch.Tensor) -> torch.Tensor:
        """(B, T, vocab) logits -> (B, T, 20) log-probabilities.

        The log-softmax is taken over the *full* vocabulary and only then
        restricted to the 20 standard amino acids. Renormalising over the 20
        instead would redistribute the mass ESM assigns to rare and special
        tokens (X, B, U, Z, <mask>, ...) across the standard ones and shift
        every log-ratio - small on confident positions, large on the
        low-confidence ones where the ratio matters most.
        """
        return torch.log_softmax(logits.float(), dim=-1)[..., self.aa_token_ids]

    @torch.inference_mode()
    def _forward_window(
        self, seqs: list[str], want_hidden: bool
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """Forward pass returning (log_probs, hidden) with BOS already stripped.

        After the slice, index i corresponds to residue i of the window. The
        trailing EOS column is left in place; callers slice to the true window
        length, which drops it.
        """
        batch = self._encode(seqs)
        out = self.model(**batch, output_hidden_states=want_hidden)
        log_probs = self._aa_log_probs(out.logits[:, 1:, :])
        hidden = None
        if want_hidden:
            hidden = out.hidden_states[self.embed_layer][:, 1:, :]
        return log_probs, hidden

    # ------------------------------------------------------------------
    # scoring schemes
    # ------------------------------------------------------------------
    @torch.inference_mode()
    def wt_marginals(self, sequence: str) -> tuple[np.ndarray, np.ndarray]:
        """Wild-type-marginal log-probs and embeddings for a whole protein.

        Returns (log_probs (L, 20) float32, embeddings (L, d) float16). Costs
        one forward pass per window, so it is cheap even for RYR1.
        """
        length = len(sequence)
        windows = plan_windows(length, self.max_length)
        owner = assign_windows(length, windows)

        log_probs = np.zeros((length, 20), dtype=np.float32)
        embeddings = np.zeros((length, self.hidden_size), dtype=np.float16)

        for w_idx, window in enumerate(windows):
            sub = sequence[window.start : window.end]
            lp, hidden = self._forward_window([sub], want_hidden=True)
            lp = lp[0, : window.length].cpu().numpy()
            hs = hidden[0, : window.length].to(torch.float16).cpu().numpy()

            # Write back only the residues this window owns - the tiling
            # overlaps, and each residue is taken from the window in which it
            # sits most centrally.
            rows = [p for p in range(window.start, window.end) if owner[p] == w_idx]
            if not rows:
                continue
            local = np.asarray(rows, dtype=np.int64) - window.start
            log_probs[rows] = lp[local]
            embeddings[rows] = hs[local]

        return log_probs, embeddings

    @torch.inference_mode()
    def masked_marginals(
        self,
        sequence: str,
        positions: list[int] | None = None,
        batch_size: int = 8,
        progress=None,
    ) -> np.ndarray:
        """Masked-marginal log-probs, (L, 20), NaN at positions not requested.

        Each requested residue is masked inside its own best-centred window, so
        this costs one forward pass per position. Positions are grouped by
        window so that every batch shares a single sequence and stacks without
        ragged padding.
        """
        length = len(sequence)
        wanted = sorted(set(range(length) if positions is None else positions))
        out = np.full((length, 20), np.nan, dtype=np.float32)
        if not wanted:
            return out

        by_window: dict[Window, list[int]] = {}
        for pos in wanted:
            window = window_for_position(length, pos, self.max_length)
            by_window.setdefault(window, []).append(pos)

        done = 0
        for window, positions_in_window in by_window.items():
            sub = sequence[window.start : window.end]
            for start in range(0, len(positions_in_window), batch_size):
                chunk = positions_in_window[start : start + batch_size]
                batch = self._encode([sub] * len(chunk))
                for row, pos in enumerate(chunk):
                    batch["input_ids"][row, pos - window.start + 1] = self.mask_id  # +1 for BOS
                logits = self.model(**batch).logits[:, 1:, :]
                lp = self._aa_log_probs(logits)
                for row, pos in enumerate(chunk):
                    out[pos] = lp[row, pos - window.start].cpu().numpy()
                done += len(chunk)
                if progress is not None:
                    progress(done, len(wanted))
        return out

    @torch.inference_mode()
    def mutant_marginals(
        self, sequence: str, substitutions: list[tuple[int, str]]
    ) -> np.ndarray:
        """Log-probs read from the *mutant* sequence, (n_substitutions, 20).

        `substitutions` is a list of (pos0, mutant_aa). Every substitution is
        applied before scoring, so each position is read in the context of the
        others - this is the scheme that sees epistasis in a multi-mutant, and
        it is what the GRB2 double mutants need.
        """
        if not substitutions:
            return np.zeros((0, 20), dtype=np.float32)

        mutant = list(sequence)
        for pos, aa in substitutions:
            if not 0 <= pos < len(mutant):
                raise ValueError(f"position {pos} outside protein of length {len(mutant)}")
            mutant[pos] = aa
        mutant_seq = "".join(mutant)

        rows = []
        for pos, _ in substitutions:
            window = window_for_position(len(mutant_seq), pos, self.max_length)
            lp, _ = self._forward_window(
                [mutant_seq[window.start : window.end]], want_hidden=False
            )
            rows.append(lp[0, pos - window.start].cpu().numpy())
        return np.stack(rows).astype(np.float32)

    # ------------------------------------------------------------------
    # convenience
    # ------------------------------------------------------------------
    def features(
        self,
        sequence: str,
        masked_positions: list[int] | None = None,
        batch_size: int = 8,
        progress=None,
    ) -> ProteinFeatures:
        """Everything the feature cache stores for one protein."""
        log_probs_wt, embeddings = self.wt_marginals(sequence)
        log_probs_masked = None
        if masked_positions is not None:
            log_probs_masked = self.masked_marginals(
                sequence, masked_positions, batch_size=batch_size, progress=progress
            )
        return ProteinFeatures(embeddings, log_probs_wt, log_probs_masked)

    def score_substitution(
        self,
        sequence: str,
        pos0: int,
        wt_aa: str,
        mut_aa: str,
        scheme: str = "masked",
    ) -> float:
        """Single-variant LLR under the requested scheme.

        Used by the API for one-off lookups, where tiling and caching the whole
        protein would be wasted work.
        """
        if sequence[pos0] != wt_aa:
            raise ValueError(
                f"wild-type mismatch at {pos0}: sequence has {sequence[pos0]!r}, "
                f"variant claims {wt_aa!r}"
            )
        if scheme == "masked":
            log_probs = self.masked_marginals(sequence, [pos0])
        elif scheme == "wt":
            log_probs, _ = self.wt_marginals(sequence)
        elif scheme == "mutant":
            row = self.mutant_marginals(sequence, [(pos0, mut_aa)])[0]
            return float(row[AA_TO_IDX[mut_aa]] - row[AA_TO_IDX[wt_aa]])
        else:
            raise ValueError(f"unknown scheme {scheme!r}; expected wt/masked/mutant")
        return llr(log_probs, pos0, wt_aa, mut_aa)


def llr(log_probs: np.ndarray, pos0: int, wt_aa: str, mut_aa: str) -> float:
    """Log-likelihood ratio log p(mut) - log p(wt) at one residue."""
    row = log_probs[pos0]
    return float(row[AA_TO_IDX[mut_aa]] - row[AA_TO_IDX[wt_aa]])


def llr_matrix(log_probs: np.ndarray, sequence: str) -> np.ndarray:
    """Full saturation LLR matrix, (L, 20).

    Entry [i, a] is the score for substituting residue i with amino acid a;
    the wild-type column of each row is 0 by construction. This is exactly the
    matrix the front-end heatmap renders.
    """
    wt_idx = np.array([AA_TO_IDX[aa] for aa in sequence], dtype=np.int64)
    wt_log_probs = log_probs[np.arange(len(sequence)), wt_idx][:, None]
    return log_probs - wt_log_probs
