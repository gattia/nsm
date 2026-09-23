"""
Migrating a pre-Aug-2026 reconstruction config, and the hint that points at the migrator.
Delete this file with ``NSM/reconstruct/_config_migration.py``.

Old configs can carry keys that name nothing, and those now raise (``docs/KNOWN_ISSUES.md``
History 20). Every key the migrator removes was inert before and after, so a migrated config
gives the same result. That property makes the migration safe to apply unattended, and these
tests hold it.
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
    "recon_tol": 0.001,
}


NAMED = set(inspect.signature(reconstruct_mesh).parameters)

#: A real harness passed these on every run, whatever its config said. Five name nothing.
HARNESS_KEYWORDS = """
latent_norm use_soft_norm_constraint norm_penalty_weight norm_penalty_type hybrid_optimizer
adam_iterations lbfgs_iterations lbfgs_lr lbfgs_max_iter lbfgs_history_size min_rel_improve
grad_tol param_change_tol recon_tol log_wandb_step return_registration_params
max_n_samples_latent_recon n_steps_sample_ramp_latent_recon
""".split()


def test_a_historical_config_loses_exactly_the_inert_keys():
    """
    Fails if ``migrate_reconstruct_config`` removes a key other than the six inert ones,
    changes a kept value, mutates its input, omits the note for a removal or for
    ``batch_size``, or keeps a key ``reconstruct_mesh`` does not name.

    ``batch_size`` is kept and flagged: in an optimization block it reads as a fit knob and
    is the marching-cubes decode batch.
    """
    before = dict(HISTORICAL_CONFIG)
    cleaned, notes = migrate_reconstruct_config(HISTORICAL_CONFIG)
    assert HISTORICAL_CONFIG == before, "the input was mutated"

    removed = set(HISTORICAL_CONFIG) - set(cleaned)
    assert removed == {
        "min_rel_improve",
        "grad_tol",
        "param_change_tol",
        "recon_tol",
        "log_wandb_step",
        "latent_optimizer_name",
    }
    assert set(cleaned) <= NAMED
    assert all(cleaned[key] == HISTORICAL_CONFIG[key] for key in cleaned)
    assert all(any(f"removed {key!r}" in note for note in notes) for key in removed)
    assert any("marching-cubes" in note for note in notes)

    harness, _ = migrate_reconstruct_config({k: None for k in HARNESS_KEYWORDS})
    assert set(harness) <= NAMED and len(harness) == 13


def test_a_current_config_and_a_non_hybrid_optimizer_name_are_left_alone():
    """
    Fails if ``migrate_reconstruct_config`` changes a current config or notes anything on it,
    or removes ``latent_optimizer_name`` when ``hybrid_optimizer`` is False.

    ``latent_optimizer_name`` is inert only under hybrid; alone it picks the optimizer.
    """
    current = {"num_iterations": 100, "lr": 0.01, "loss_type": "l1"}
    assert migrate_reconstruct_config(current) == (current, [])
    cleaned, _ = migrate_reconstruct_config(dict(HISTORICAL_CONFIG, hybrid_optimizer=False))
    assert cleaned["latent_optimizer_name"] == "lbfgs"


def test_only_a_known_stale_key_earns_the_migration_hint():
    """
    Fails if ``refuse_unknown_kwargs`` leaves the migration hint off a known stale key
    (``grad_tol``), adds it to a typo, or ``migration_hint`` returns text for an unknown key.

    A typo is not a stale config, and the migrator has no answer for it.
    """
    call = dict(
        decoders=None, num_iterations=1, latent_size=8, xyz=None, sdf_gt=None, pts_surface=[0]
    )
    with pytest.raises(TypeError, match="migrate_reconstruct_config"):
        reconstruct_latent(grad_tol=1e-5, **call)
    with pytest.raises(TypeError) as excinfo:
        reconstruct_latent(num_iteration=1, **call)
    assert "migrate_reconstruct_config" not in str(excinfo.value)
    assert migration_hint(["definitely_not_a_key"]) == ""
