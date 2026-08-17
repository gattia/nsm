"""
Fixtures for the numerical regression harness. The machinery is in ``_harness.py``.

Everything expensive is session-scoped and seeds itself, so what a test sees does not
depend on which tests ran before it.
"""

import os

import pytest
from _harness import (
    BASELINE_DIR,
    BaselineStore,
    build_dataset,
    build_model,
    load_reconstruction_decoder,
    regenerating_decoder,
    run_reconstruction,
    run_training,
    save_reconstruction_decoder,
    train_reconstruction_decoder,
    training_config,
    write_synthetic_meshes,
)


def pytest_terminal_summary(terminalreporter, exitstatus, config):
    """
    Say what the xfails mean, rather than leaving "N xfailed" to be interpreted.

    Every xfail in this directory asserts behaviour NSM *should* have and is marked
    ``strict=True``, so the day one is fixed it XPASSes and the suite goes red -- which is
    what stops "known defect" from decaying into "permanently ignored".
    """
    xfailed = terminalreporter.stats.get("xfailed", [])
    ours = [r for r in xfailed if "NSM/regression/" in str(getattr(r, "nodeid", ""))]
    if not ours:
        return

    terminalreporter.write_sep("-", "known defects (regression harness)")
    terminalreporter.write_line(
        f"{len(ours)} assertion(s) describe behaviour NSM should have and does not. "
        f"They are not failures and not passes."
    )
    terminalreporter.write_line(
        "  Tracked in docs/KNOWN_ISSUES.md (Open). Run with -rx to list them with reasons."
    )
    terminalreporter.write_line(
        "  Each is strict: fixing one turns this suite RED until its xfail mark is removed."
    )


def _baseline(name):
    store = BaselineStore(os.path.join(BASELINE_DIR, f"{name}.json"))
    yield store
    store.flush()


@pytest.fixture(scope="session")
def training_baseline():
    yield from _baseline("training")


@pytest.fixture(scope="session")
def reconstruction_baseline():
    yield from _baseline("reconstruction")


@pytest.fixture(scope="session")
def synthetic_meshes(tmp_path_factory):
    """``[[bone, cart], ...]`` paths, one pair per synthetic subject."""
    return write_synthetic_meshes(tmp_path_factory.mktemp("meshes"))


@pytest.fixture(scope="session")
def training_dataset(synthetic_meshes, tmp_path_factory):
    return build_dataset(synthetic_meshes, tmp_path_factory.mktemp("train_cache"))


@pytest.fixture(scope="session")
def training_run(training_dataset, tmp_path_factory):
    """The reference 8-epoch run: its config, the trained model, and the epoch records."""
    config = training_config(tmp_path_factory.mktemp("train_run"))
    model = build_model(config)
    records, returned = run_training(config, model, training_dataset)
    return {"config": config, "model": model, "records": records, "returned": returned}


@pytest.fixture(scope="session")
def reconstruction_model(request, tmp_path_factory):
    """
    A decoder trained far enough that its zero level set exists -- LOADED from a committed
    asset, not retrained.

    It used to be retrained here every session, which is what made the reconstruction
    baselines pin a 60-epoch gradient-descent trajectory instead of ``reconstruct_mesh``.
    ``_harness``'s asset section has the measurements; the short version is that a torch
    bump moved the geometry baselines 763x their tolerance through the training, and 0.005x
    through reconstruction on fixed weights.

    ``training_dataset`` is requested rather than declared, because only the regeneration
    branch needs training data and building a dataset the load path never touches would be
    a dependency that is not one.
    """
    if regenerating_decoder():
        model = train_reconstruction_decoder(
            request.getfixturevalue("training_dataset"), tmp_path_factory.mktemp("recon_train")
        )
        save_reconstruction_decoder(model)
        return model
    return load_reconstruction_decoder()


@pytest.fixture(scope="session")
def reconstruction(synthetic_meshes, reconstruction_model):
    """One reconstruction of subject 0 through the full ``reconstruct_mesh`` path."""
    return run_reconstruction(synthetic_meshes[0], reconstruction_model)
