"""
One-time migration help for reconstruction configs written before Aug 2026.

Nothing here is permanent API. It exists only to hand someone holding an older
``reconstruct_mesh`` config a corrected copy of it, and to say what each removed key was
actually doing -- which in every case below is nothing.

Until Aug 2026 ``reconstruct_mesh`` had a ``**kwargs`` that read one key and swallowed the
rest, so a config could carry keys that named nothing at all and run without complaint
(``docs/KNOWN_ISSUES.md`` section History 20). Those configs now raise ``TypeError``. The
keys were inert before and are inert after, so migrating one cannot change a result.

Apply it to a config on disk with::

    import json
    from NSM.reconstruct._config_migration import migrate_reconstruct_config

    raw = json.loads(path.read_text(encoding="utf-8"))
    clean, notes = migrate_reconstruct_config(raw)
    print("\\n".join(notes))
    path.write_text(json.dumps(clean, indent=4), encoding="utf-8")

DELETE THIS FILE once no reconstruction config still in use predates the refusals. The
only caller inside NSM is ``reconstruct.utils.refuse_unknown_kwargs``, which imports it
lazily for its error text and needs the plain message in its place.
"""

#: Keys that have never named anything in NSM -- zero occurrences in ``NSM/`` at any
#: commit, checked with ``git log -S``. They read like tolerances an optimizer might take;
#: nothing ever read them. They turn up in configs *and* in harness scripts that pass a
#: fixed keyword set, which is the case that raises on every run rather than on some.
_NEVER_EXISTED = {
    "min_rel_improve": "no such parameter; use convergence/convergence_patience instead",
    "grad_tol": "no such parameter; torch's LBFGS has tolerance_grad, not exposed here",
    "param_change_tol": "no such parameter; torch's LBFGS has tolerance_change, not exposed",
    "recon_tol": "no such parameter; use convergence/convergence_patience instead",
}

#: Keys naming a real ``reconstruct_latent`` parameter that ``reconstruct_mesh`` does not
#: forward, so setting them on ``reconstruct_mesh`` never reached the fit.
_NOT_FORWARDED = {
    "log_wandb_step": (
        "reconstruct_mesh does not forward it; the latent fit logs every 10 steps and "
        "that is not configurable from here"
    ),
}

#: Keys that are read on one code path and ignored on another. The value is the condition
#: under which the key is inert, as a predicate over the whole config.
_INERT_UNDER = {
    "latent_optimizer_name": (
        lambda config: bool(config.get("hybrid_optimizer")),
        "hybrid_optimizer=True derives the optimizer from the step number, so "
        "latent_optimizer_name is never read; the two together now raise",
    ),
}

#: Not removed -- these work. They are listed because their name in a config block reads
#: as something they are not, which is its own way of wasting an afternoon.
_MISLEADING = {
    "batch_size": (
        "is the marching-cubes decode batch (default 32**3), not a latent-fit batch. "
        "The latent fit's memory knob is n_samples_per_chunk"
    ),
}


def migrate_reconstruct_config(config):
    """Return ``(cleaned_config, notes)`` for a ``reconstruct_mesh`` keyword mapping.

    ``cleaned_config`` is a copy with every inert key removed. ``notes`` is a list of
    human-readable lines saying what was removed and why, plus advisories for keys that
    are kept but do not mean what their name suggests. An already-current config comes
    back unchanged with an empty note list.
    """
    cleaned = dict(config)
    notes = []

    for key, reason in sorted(_NEVER_EXISTED.items()):
        if key in cleaned:
            del cleaned[key]
            notes.append(f"removed {key!r}: {reason}")

    for key, reason in sorted(_NOT_FORWARDED.items()):
        if key in cleaned:
            del cleaned[key]
            notes.append(f"removed {key!r}: {reason}")

    for key, (is_inert, reason) in sorted(_INERT_UNDER.items()):
        if key in cleaned and is_inert(config):
            del cleaned[key]
            notes.append(f"removed {key!r}: {reason}")

    for key, reason in sorted(_MISLEADING.items()):
        if key in cleaned:
            notes.append(f"kept {key!r}, but note it {reason}")

    return cleaned, notes


def migration_hint(unknown_keys):
    """One line naming the migration helper, or ``""`` if it has nothing to offer.

    Appended to the ``TypeError`` that refuses unknown keywords, so a caller holding a
    pre-Aug-2026 config is told where the corrected copy comes from instead of deleting
    keys by trial and error.
    """
    known = sorted(set(unknown_keys) & (set(_NEVER_EXISTED) | set(_NOT_FORWARDED)))
    if not known:
        return ""
    return (
        f". {', '.join(repr(k) for k in known)} never reached the fit even when accepted; "
        "NSM.reconstruct._config_migration.migrate_reconstruct_config() returns a "
        "corrected copy of the config and says why."
    )
