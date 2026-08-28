"""
Migrating a pre-Aug-2026 reconstruction config, and the hint that points at the migrator.

``reconstruct_mesh`` used to take a ``**kwargs`` that read one key and swallowed the rest,
so configs accumulated keys that named nothing (``docs/KNOWN_ISSUES.md`` History 20). Those
configs now raise. Every key this module removes was inert before and after, so a migrated
config cannot produce a different result -- which is the property that makes the migration
safe to apply unattended, and the thing these tests exist to hold.
"""

import inspect

import pytest

from NSM.reconstruct import reconstruct_latent
from NSM.reconstruct._config_migration import migrate_reconstruct_config, migration_hint
from NSM.reconstruct.main import reconstruct_mesh

#: The optimization block of a real sweep config from Aug 2025, trimmed to the keys that
#: matter here. Kept verbatim rather than minimised: the point is that a config someone
#: actually ran migrates, and every key below was in it.
HISTORICAL_CONFIG = {
    "num_iterations": 2000,
    "lr": 0.01,
    "batch_size": 1000000,
    "loss_type": "l1",
    "convergence": "recon_loss",
    "convergence_patience": 5,
    "latent_optimizer_name": "lbfgs",
    "hybrid_optimizer": True,
    "adam_iterations": 10,
    "lbfgs_iterations": 50,
    "lbfgs_lr": 1.0,
    "lbfgs_max_iter": 10,
    "lbfgs_history_size": 50,
    "n_samples_latent_recon": 1000000,
    "latent_norm": 10.0,
    "norm_penalty_weight": 100,
    "log_wandb": True,
    "log_wandb_step": 1,
    "min_rel_improve": 0.001,
    "grad_tol": 1e-05,
    "param_change_tol": 0.001,
}


class TestMigratingAHistoricalConfig:
    def test_every_remaining_key_is_a_real_parameter(self):
        """The property that matters: the migrated config gets past the refusal."""
        cleaned, _ = migrate_reconstruct_config(HISTORICAL_CONFIG)
        named = set(inspect.signature(reconstruct_mesh).parameters)
        assert sorted(set(cleaned) - named) == []

    def test_the_five_removed_keys_are_the_inert_ones(self):
        cleaned, _ = migrate_reconstruct_config(HISTORICAL_CONFIG)
        removed = set(HISTORICAL_CONFIG) - set(cleaned)
        assert removed == {
            "min_rel_improve",
            "grad_tol",
            "param_change_tol",
            "log_wandb_step",
            "latent_optimizer_name",
        }

    def test_nothing_that_changes_a_result_is_touched(self):
        """
        The safety property. Every key that reaches the optimizer -- the optimizer's own
        settings, the loss, the sampling, the norm constraint -- survives untouched, so
        migrating cannot move a number.
        """
        cleaned, _ = migrate_reconstruct_config(HISTORICAL_CONFIG)
        for key in (
            "num_iterations",
            "lr",
            "loss_type",
            "convergence",
            "convergence_patience",
            "hybrid_optimizer",
            "adam_iterations",
            "lbfgs_iterations",
            "lbfgs_lr",
            "lbfgs_max_iter",
            "lbfgs_history_size",
            "n_samples_latent_recon",
            "latent_norm",
            "norm_penalty_weight",
        ):
            assert cleaned[key] == HISTORICAL_CONFIG[key], key

    def test_every_removal_is_explained(self):
        cleaned, notes = migrate_reconstruct_config(HISTORICAL_CONFIG)
        removed = set(HISTORICAL_CONFIG) - set(cleaned)
        for key in removed:
            assert any(f"removed {key!r}" in note for note in notes), key

    def test_batch_size_is_kept_and_flagged(self):
        """
        It works, so removing it would change a result. It is flagged because a
        ``batch_size`` in an optimization block reads as a fit knob and is the
        marching-cubes decode batch.
        """
        cleaned, notes = migrate_reconstruct_config(HISTORICAL_CONFIG)
        assert cleaned["batch_size"] == HISTORICAL_CONFIG["batch_size"]
        assert any("marching-cubes" in note for note in notes)

    def test_a_current_config_is_returned_unchanged(self):
        current = {"num_iterations": 100, "lr": 0.01, "loss_type": "l1"}
        cleaned, notes = migrate_reconstruct_config(current)
        assert cleaned == current
        assert notes == []

    def test_the_input_is_not_mutated(self):
        before = dict(HISTORICAL_CONFIG)
        migrate_reconstruct_config(HISTORICAL_CONFIG)
        assert HISTORICAL_CONFIG == before

    def test_latent_optimizer_name_survives_without_hybrid(self):
        """It is only inert under hybrid; on its own it selects the optimizer."""
        config = dict(HISTORICAL_CONFIG, hybrid_optimizer=False)
        cleaned, _ = migrate_reconstruct_config(config)
        assert cleaned["latent_optimizer_name"] == "lbfgs"


class TestTheRefusalPointsAtTheMigrator:
    def test_a_known_stale_key_earns_the_hint(self):
        with pytest.raises(TypeError, match="migrate_reconstruct_config"):
            reconstruct_latent(
                decoders=None,
                num_iterations=1,
                latent_size=8,
                xyz=None,
                sdf_gt=None,
                grad_tol=1e-5,
            )

    def test_an_unrecognised_key_gets_no_hint(self):
        """
        A typo is not a stale config, and pointing a typo at a migration helper would send
        the reader somewhere with no answer for them.
        """
        with pytest.raises(TypeError) as excinfo:
            reconstruct_latent(
                decoders=None,
                num_iterations=1,
                latent_size=8,
                xyz=None,
                sdf_gt=None,
                num_iteration=1,
            )
        assert "migrate_reconstruct_config" not in str(excinfo.value)

    def test_the_hint_is_empty_for_unknown_keys(self):
        assert migration_hint(["definitely_not_a_key"]) == ""
