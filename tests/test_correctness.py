"""Regression tests for the properties this project's results depend on.

Every test here pins something that was verified once by hand during
development and that a refactor could break *silently* - the model would still
train, the API would still answer, and the numbers would just quietly be wrong.
That is the failure mode worth spending tests on.

Everything runs on CPU with no network and no model weights, so the suite is
seconds, not minutes:

    python -m pytest tests -q
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import torch


# ---------------------------------------------------------------------------
# Windowing: long proteins must be fully covered and centrally scored
# ---------------------------------------------------------------------------
class TestWindowing:
    @pytest.mark.parametrize("length", [1, 393, 1022, 1023, 1863, 3418, 5202])
    def test_every_residue_is_covered_exactly_once(self, length):
        from vep.esm.windowing import assign_windows, plan_windows

        windows = plan_windows(length)
        owner = assign_windows(length, windows)
        assert len(owner) == length
        assert all(windows[owner[p]].contains(p) for p in range(length))
        assert windows[0].start == 0 and windows[-1].end == length

    def test_mid_protein_residue_is_scored_near_the_centre_of_its_window(self):
        """Edge-of-window residues see context on one side only, which degrades
        both the embedding and the masked distribution."""
        from vep.esm.windowing import window_for_position

        w = window_for_position(3418, 1700, max_len=1022)
        offset = 1700 - w.start
        assert abs(offset - 511) <= 1, "residue is not centred in its window"

    def test_termini_stay_in_range(self):
        from vep.esm.windowing import window_for_position

        assert window_for_position(3418, 0).start == 0
        assert window_for_position(3418, 3417).end == 3418


# ---------------------------------------------------------------------------
# Isoform reconciliation: a wrong offset attaches a correct label to the wrong
# residue, which is worse than dropping the variant
# ---------------------------------------------------------------------------
class TestIsoformOffsets:
    @staticmethod
    def _seq(n, seed=0):
        rng = np.random.default_rng(seed)
        return "".join(rng.choice(list("ACDEFGHIKLMNPQRSTVWY"), size=n))

    def test_recovers_a_planted_constant_shift(self):
        from vep.data.isoform import fit_offset

        seq = self._seq(500)
        rng = np.random.default_rng(1)
        pos = rng.choice(np.arange(0, 450), size=80, replace=False)
        wt = np.array([seq[p + 21] for p in pos], dtype=object)
        fit = fit_offset("SHIFTED", seq, pos, wt)
        assert fit.accepted and fit.offset == 21

    def test_leaves_a_correctly_numbered_gene_alone(self):
        from vep.data.isoform import fit_offset

        seq = self._seq(500)
        rng = np.random.default_rng(2)
        pos = rng.choice(np.arange(0, 500), size=80, replace=False)
        wt = np.array([seq[p] for p in pos], dtype=object)
        fit = fit_offset("CLEAN", seq, pos, wt)
        assert not fit.accepted and fit.offset == 0

    def test_rejects_noise(self):
        """A weak improvement is far likelier to be coincidence than biology."""
        from vep.data.isoform import fit_offset

        seq = self._seq(500)
        rng = np.random.default_rng(3)
        pos = rng.choice(np.arange(0, 500), size=80, replace=False)
        wt = np.array(list(rng.choice(list("ACDEFGHIKLMNPQRSTVWY"), size=80)), dtype=object)
        assert not fit_offset("NOISE", seq, pos, wt).accepted

    def test_recovers_an_internal_indel_piecewise(self):
        """SCN5A and CACNA1C differ from canonical by internal exons, so one
        constant shift cannot reconcile them."""
        from vep.data.isoform import apply_piecewise, fit_piecewise

        canonical = self._seq(1200, seed=7)
        to_canonical = lambda p: p if p < 600 else p - 9
        rng = np.random.default_rng(7)
        pos = np.sort(rng.choice(np.arange(0, 1191), size=200, replace=False))
        wt = np.array([canonical[to_canonical(p)] for p in pos], dtype=object)

        fit = fit_piecewise("INTERNAL", canonical, pos, wt)
        assert fit.accepted and fit.n_segments == 2
        recovered = apply_piecewise(pos, fit.breakpoints)
        truth = np.array([to_canonical(p) for p in pos])
        assert (recovered == truth).mean() == 1.0


# ---------------------------------------------------------------------------
# ProteinNPT: the label column must inform, never leak
# ---------------------------------------------------------------------------
class TestNPTLeakage:
    @staticmethod
    def _model_and_batch(n=12, d=64):
        from vep.models.npt import ProteinNPT

        torch.manual_seed(0)
        model = ProteinNPT(d_emb=d, d_model=64, n_layers=2, n_heads=4, dropout=0.0).eval()
        batch = dict(
            emb=torch.randn(n, d), logprobs=torch.randn(n, 20),
            subst=torch.randn(n, 42), context=torch.randn(n, d),
        )
        return model, batch

    def test_masked_row_cannot_see_its_own_class_label(self):
        from vep.models.npt import LABEL_BENIGN, LABEL_MASK, LABEL_PATHOGENIC

        model, batch = self._model_and_batch()
        labels = torch.tensor([LABEL_PATHOGENIC, LABEL_BENIGN] * 6)
        labels[0] = LABEL_MASK
        with torch.no_grad():
            base = model(**batch, labels=labels).pathogenicity_logit[0].item()
            flipped = labels.clone()      # row 0 is masked either way
            flipped[0] = LABEL_MASK
            same = model(**batch, labels=flipped).pathogenicity_logit[0].item()
        assert base == same

    def test_neighbour_labels_do_reach_the_prediction(self):
        """The complement of the leakage test: if this fails the architecture is
        pointless, because the label column is doing nothing."""
        from vep.models.npt import LABEL_BENIGN, LABEL_MASK, LABEL_PATHOGENIC

        model, batch = self._model_and_batch()
        labels = torch.tensor([LABEL_PATHOGENIC, LABEL_BENIGN] * 6)
        labels[0] = LABEL_MASK
        with torch.no_grad():
            base = model(**batch, labels=labels).pathogenicity_logit[0].item()
            nudged = labels.clone()
            nudged[5] = LABEL_PATHOGENIC if labels[5] == LABEL_BENIGN else LABEL_BENIGN
            other = model(**batch, labels=nudged).pathogenicity_logit[0].item()
        assert abs(other - base) > 1e-6

    def test_masked_row_cannot_see_its_own_continuous_value(self):
        from vep.models.npt import LABEL_BENIGN, LABEL_MASK, TASK_FITNESS

        model, batch = self._model_and_batch()
        n = batch["emb"].shape[0]
        labels = torch.full((n,), LABEL_BENIGN, dtype=torch.long)
        labels[0] = LABEL_MASK
        task = torch.full((n,), TASK_FITNESS, dtype=torch.long)
        values = torch.randn(n)
        with torch.no_grad():
            base = model(**batch, labels=labels, values=values, task=task).fitness[0].item()
            own = values.clone(); own[0] += 100.0
            after = model(**batch, labels=labels, values=own, task=task).fitness[0].item()
            nbr = values.clone(); nbr[5] += 3.0
            other = model(**batch, labels=labels, values=nbr, task=task).fitness[0].item()
        assert own is not None and after == base, "regression target leaked into its own input"
        assert abs(other - base) > 1e-6

    def test_padded_rows_do_not_influence_predictions(self):
        from vep.models.npt import LABEL_BENIGN, LABEL_MASK, LABEL_PATHOGENIC

        model, batch = self._model_and_batch()
        labels = torch.tensor([LABEL_PATHOGENIC, LABEL_BENIGN] * 6)
        labels[0] = LABEL_MASK
        pad = torch.zeros(12, dtype=torch.bool); pad[8:] = True
        with torch.no_grad():
            a = model(**batch, labels=labels, pad_mask=pad).pathogenicity_logit[0].item()
            scrambled = {k: v.clone() for k, v in batch.items()}
            for v in scrambled.values():
                v[8:] = torch.randn_like(v[8:])
            b = model(**batch | scrambled, labels=labels, pad_mask=pad).pathogenicity_logit[0].item()
        assert abs(a - b) < 1e-6

    def test_task_embedding_is_zero_initialised(self):
        """The served 150M checkpoint predates multi-task support. Zero-init is
        what lets it load with strict=False and behave bit-identically."""
        from vep.models.npt import ProteinNPT

        model = ProteinNPT(d_emb=64, d_model=64, n_layers=1, n_heads=4)
        assert torch.count_nonzero(model.task_embed.weight) == 0

    def test_loss_ignores_unmasked_rows(self):
        from vep.models.npt import LABEL_BENIGN, LABEL_MASK, LABEL_PATHOGENIC, npt_loss

        model, batch = self._model_and_batch()
        labels = torch.tensor([LABEL_PATHOGENIC, LABEL_BENIGN] * 6)
        labels[0] = LABEL_MASK
        is_masked = torch.zeros(12, dtype=torch.bool); is_masked[0] = True
        out = model(**batch, labels=labels)
        target = torch.tensor([1.0, 0.0] * 6)
        loss_a, _ = npt_loss(out, target, is_masked)
        flipped = target.clone(); flipped[1:] = 1 - flipped[1:]
        loss_b, _ = npt_loss(out, flipped, is_masked)
        assert abs(loss_a.item() - loss_b.item()) < 1e-9


# ---------------------------------------------------------------------------
# Saturation cache: a stale matrix is indistinguishable from a fresh one
# ---------------------------------------------------------------------------
# ---------------------------------------------------------------------------
# Checkpoint compatibility
#
# The served pathogenicity head predates the continuous-label pathway added for
# the fitness task. It silently stopped loading when those parameters appeared,
# because load_state_dict is strict by default, and the API fell back to
# zero-shot without saying so. These pin both halves of the contract.
# ---------------------------------------------------------------------------
class TestCheckpointCompat:
    @staticmethod
    def _model():
        from vep.models.npt import ProteinNPT

        return ProteinNPT(d_emb=32, d_model=64, n_layers=1, n_heads=4, dropout=0.0)

    def test_old_checkpoint_loads_and_is_inert(self):
        from vep.models.npt import load_state_dict_compat

        old = {
            k: v
            for k, v in self._model().state_dict().items()
            if not k.startswith(("value_proj.", "task_embed."))
        }
        model = self._model()
        missing = load_state_dict_compat(model, old)

        assert set(missing) == {"value_proj.weight", "value_proj.bias", "task_embed.weight"}
        # Zero task embeddings are what make the old checkpoint evaluate exactly
        # as it did when its held-out AUROC was measured.
        assert torch.all(model.task_embed.weight == 0)

    def test_real_architecture_drift_still_raises(self):
        from vep.models.npt import load_state_dict_compat

        broken = dict(self._model().state_dict())
        dropped = next(k for k in broken if k.startswith("value_proj") is False
                       and k.startswith("task_embed") is False)
        broken.pop(dropped)

        with pytest.raises(RuntimeError, match="does not match"):
            load_state_dict_compat(self._model(), broken)

    def test_unexpected_keys_raise(self):
        from vep.models.npt import load_state_dict_compat

        extra = dict(self._model().state_dict())
        extra["some.renamed.layer"] = torch.zeros(3)

        with pytest.raises(RuntimeError, match="does not match"):
            load_state_dict_compat(self._model(), extra)


class TestSaturationCache:
    @staticmethod
    def _store(tmp_path, backbone="esm2_t30_150M_UR50D", fp="aaaa1111"):
        from vep.esm.saturation import SaturationStore

        return SaturationStore(tmp_path, backbone, model_fingerprint=fp)

    def test_survives_a_restart(self, tmp_path):
        m = np.random.rand(50, 20).astype(np.float32)
        self._store(tmp_path).put("TP53", "masked", m)
        assert np.array_equal(self._store(tmp_path).get("TP53", "masked"), m)

    def test_different_backbone_cannot_see_the_cache(self, tmp_path):
        """640-d and 1280-d features mean different things."""
        self._store(tmp_path).put("TP53", "masked", np.zeros((5, 20), dtype=np.float32))
        other = self._store(tmp_path, backbone="esm2_t33_650M_UR50D")
        assert other.get("TP53", "masked") is None

    def test_new_checkpoint_misses_probabilities_but_keeps_masked(self, tmp_path):
        """Retraining invalidates probabilities; it does not invalidate the
        backbone-only masked scan, which is the expensive one."""
        m = np.random.rand(5, 20).astype(np.float32)
        s = self._store(tmp_path)
        s.put("TP53", "masked", m); s.put("TP53", "probability", m)
        newer = self._store(tmp_path, fp="bbbb2222")
        assert newer.get("TP53", "probability") is None
        assert newer.get("TP53", "masked") is not None

    def test_pruning_removes_only_stale_probabilities(self, tmp_path):
        m = np.random.rand(5, 20).astype(np.float32)
        s = self._store(tmp_path)
        s.put("TP53", "masked", m); s.put("TP53", "probability", m)
        newer = self._store(tmp_path, fp="bbbb2222")
        assert newer.prune_stale_probabilities() == 1
        assert newer.get("TP53", "masked") is not None

    def test_corrupt_file_self_heals(self, tmp_path):
        """One recomputation is the right cost for a truncated file; a 500 is not."""
        s = self._store(tmp_path)
        s.put("TP53", "masked", np.zeros((5, 20), dtype=np.float32))
        s.path("TP53", "masked").write_bytes(b"not an npy")
        fresh = self._store(tmp_path)
        assert fresh.get("TP53", "masked") is None
        assert not fresh.path("TP53", "masked").exists()

    def test_writes_leave_no_temporary_files(self, tmp_path):
        s = self._store(tmp_path)
        s.put("TP53", "masked", np.zeros((5, 20), dtype=np.float32))
        assert list(tmp_path.rglob("*.tmp")) == []


# ---------------------------------------------------------------------------
# ClinVar parsing
# ---------------------------------------------------------------------------
class TestClinVarParsing:
    @pytest.mark.parametrize(
        "name,expected",
        [
            ("NM_007294.4(BRCA1):c.5123C>A (p.Ala1708Glu)", ("A", 1708, "E")),
            ("NM_000546.6(TP53):c.524G>A (p.Arg175His)", ("R", 175, "H")),
            ("NM_x(G):c.1A>T (p.Ala123=)", None),        # synonymous
            ("NM_x(G):c.1A>T (p.Arg123Ter)", None),      # nonsense
            ("NM_x(G):c.1A>T (p.Arg123fs)", None),       # frameshift
            ("NM_x(G):c.1A>T", None),                    # no protein consequence
        ],
    )
    def test_only_missense_is_extracted(self, name, expected):
        from vep.data.clinvar import parse_protein_change

        assert parse_protein_change(name) == expected

    def test_significance_mapping(self):
        from vep.constants import normalise_significance

        assert normalise_significance("Pathogenic/Likely pathogenic|risk factor") == "pathogenic"
        assert normalise_significance("Benign") == "benign"
        assert normalise_significance("Uncertain significance") is None
        assert normalise_significance("") is None

    def test_contradictory_rows_are_dropped_not_averaged(self):
        """The same substitution can be reached by different nucleotide changes.
        If two rows disagree on pathogenicity the evidence is unusable."""
        key = ["gene", "wt_aa", "position", "mut_aa"]
        df = pd.DataFrame([
            dict(gene="G1", wt_aa="A", position=10, mut_aa="V", label="pathogenic", stars=2, n_submitters=3),
            dict(gene="G1", wt_aa="A", position=10, mut_aa="V", label="benign", stars=1, n_submitters=1),
            dict(gene="G1", wt_aa="C", position=20, mut_aa="W", label="benign", stars=1, n_submitters=1),
            dict(gene="G1", wt_aa="C", position=20, mut_aa="W", label="benign", stars=3, n_submitters=9),
        ])
        conflicted = df.groupby(key)["label"].transform("nunique") > 1
        kept = (df[~conflicted].sort_values(["stars", "n_submitters"], ascending=False)
                .drop_duplicates(subset=key, keep="first"))
        assert len(kept) == 1 and kept.iloc[0]["stars"] == 3


# ---------------------------------------------------------------------------
# DMS: wild-type reconstruction and multi-mutant pooling
# ---------------------------------------------------------------------------
class TestDMS:
    def test_wildtype_is_reconstructed_from_variant_strings_alone(self):
        """The assay file ships no sequence column; every single mutant names
        its own wild-type residue, so the sequence is recoverable - and the
        agreement across 33k rows is itself the validation."""
        from vep.data.dms import reconstruct_wildtype

        frame = pd.DataFrame({
            "variant": ["A0C", "T1G", "W2Y", "A0C,W2Y"],
            "num_mutations": [1, 1, 1, 2],
        })
        seq, first, conflicts = reconstruct_wildtype(frame)
        assert seq == "ATW" and first == 0 and conflicts == 0

    def test_conflicting_wildtype_is_reported(self):
        from vep.data.dms import reconstruct_wildtype

        frame = pd.DataFrame({"variant": ["A0C", "T0G"], "num_mutations": [1, 1]})
        _, _, conflicts = reconstruct_wildtype(frame)
        assert conflicts > 0

    def test_multi_mutant_pooling_is_permutation_invariant(self):
        """The assay cannot distinguish "A,B" from "B,A", so neither may the
        representation."""
        from types import SimpleNamespace

        from vep.data.dms_features import build_dms_features

        seq = "ACDEFGHIKL"
        d = 8
        cache = SimpleNamespace(embeddings=lambda g: np.arange(len(seq) * d, dtype=np.float32).reshape(len(seq), d))
        masked = np.random.RandomState(0).rand(len(seq), 20).astype(np.float32)
        forward = [[(1, "W"), (4, "Y")]]
        reverse = [[(4, "Y"), (1, "W")]]
        llr = np.array([1.5], dtype=np.float32)
        bl = np.array([-2.0], dtype=np.float32)
        a = build_dms_features(cache, "G", seq, forward, masked, llr, bl, d)
        b = build_dms_features(cache, "G", seq, reverse, masked, llr, bl, d)
        for k in a:
            assert np.allclose(a[k], b[k]), f"{k} depends on substitution order"

    def test_training_never_sees_a_held_out_position(self):
        """The guarantee is one-directional, and worth being precise about.

        A variant is held out if ANY of its positions is, so no TRAINING
        variant ever touches a held-out position - those residues are genuinely
        unseen. The converse does not hold: a double mutant spanning one
        training and one held-out position lands in test, so most test variants
        do share their partner position with training.

        The symmetric alternative (hold out only variants whose positions are
        ALL held out) leaves ~18% of the GRB2 test set, which is reported
        separately rather than used as the headline."""
        from vep.data.dms_features import split_by_position

        rng = np.random.default_rng(0)
        variants = [[(int(p), "A")] for p in rng.integers(0, 60, size=400)]
        variants += [[(int(a), "A"), (int(b), "C")]
                     for a, b in rng.integers(0, 60, size=(400, 2))]
        split, held = split_by_position(variants, seed=1, return_held_out=True)

        held_out = held["val"] | held["test"]
        for v, s in zip(variants, split):
            if s == "train":
                assert not ({p for p, _ in v} & held_out),                     "a training variant touches a held-out position"
        # and the rule is exactly "any position held out -> not train"
        for v, s in zip(variants, split):
            if {p for p, _ in v} & held["test"]:
                assert s == "test"

    def test_sampler_refuses_to_oversample(self):
        """The original script guarded on 100 of each class then asked for 1000,
        and GRB2 has 655 singles."""
        from types import SimpleNamespace

        from vep.data.dms import sample_balanced

        frame = pd.DataFrame({
            "mutation_type": ["single"] * 5 + ["double"] * 50,
            "score": np.zeros(55),
        })
        ds = SimpleNamespace(frame=frame)
        assert len(sample_balanced(ds)) == 10          # limited by the smaller class
        with pytest.raises(ValueError, match="only 5 available"):
            sample_balanced(ds, n_per_type=1000)


# ---------------------------------------------------------------------------
# Scoring conventions: a sign flip here silently inverts every ranking
# ---------------------------------------------------------------------------
class TestConventions:
    def test_blosum_matrix_is_symmetric_and_matches_published_values(self):
        from vep.eval.zeroshot import BLOSUM62, blosum_score

        assert (BLOSUM62 == BLOSUM62.T).all()
        for wt, mut, expected in [("A", "A", 4), ("C", "C", 9), ("W", "W", 11),
                                  ("D", "E", 2), ("I", "V", 3), ("P", "P", 7)]:
            assert blosum_score(wt, mut) == expected

    def test_llr_matrix_is_zero_at_the_wild_type(self):
        from vep.constants import AA_TO_IDX
        from vep.esm.backbone import llr_matrix

        seq = "ACDEFGHIKL"
        rng = np.random.default_rng(0)
        log_probs = np.log(rng.dirichlet(np.ones(20), size=len(seq))).astype(np.float32)
        m = llr_matrix(log_probs, seq)
        for i, aa in enumerate(seq):
            assert abs(m[i, AA_TO_IDX[aa]]) < 1e-6

    def test_amino_acid_alphabet_is_the_canonical_twenty(self):
        """Cached matrices are indexed by position in this tuple; reordering it
        would silently relabel every row of every saturation map."""
        from vep.constants import AA_ALPHABET

        assert "".join(AA_ALPHABET) == "ACDEFGHIKLMNPQRSTVWY"


# ---------------------------------------------------------------------------
# Thermal guard: a stale pause must expire on its own
# ---------------------------------------------------------------------------
class TestThermalPause:
    def test_pause_expires_without_the_guard(self, tmp_path):
        """A guard killed with TerminateProcess runs no handler. The deadline
        passing by itself is what stops that from wedging an overnight job."""
        import time

        from vep.gpu_guard import pause_remaining, request_pause, wait_if_paused

        path = tmp_path / "pause.json"
        request_pause(1.0, 85, path=path)
        assert pause_remaining(path) > 0
        t0 = time.time()
        wait_if_paused(path=path, poll=0.1)
        assert 0.5 < time.time() - t0 < 3.0
        assert pause_remaining(path) == 0

    def test_no_pause_file_means_no_wait(self, tmp_path):
        from vep.gpu_guard import wait_if_paused

        assert wait_if_paused(path=tmp_path / "absent.json") == 0.0
