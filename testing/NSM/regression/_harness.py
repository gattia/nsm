"""
Shared machinery for the numerical regression harness: baselines, the synthetic config,
and the three entry points (build a dataset, train, reconstruct).

Fixtures live in ``conftest.py``; everything importable lives here so test modules never
have to import ``conftest`` itself.

Determinism
-----------
``SDFSamples(random_seed=...)`` seeds every draw, on both sampling paths, so the fixtures
run on the near-surface path production uses. ``build_dataset`` passes its ``seed`` there.

It also still calls ``np.random.seed``, because ``random_seed=None`` deliberately leaves
sampling on the legacy global stream -- that is what keeps an unseeded call drawing the
numbers it always did, and ``test_dataset_cache.TestSeeding`` pins it.

``torch`` is seeded globally at each entry point for the model and optimizer.
"""

import contextlib
import io
import json
import os

import numpy as np
import torch

# ---------------------------------------------------------------------------
# Baseline storage
# ---------------------------------------------------------------------------

BASELINE_DIR = os.path.join(os.path.dirname(__file__), "baselines")

#: Bumped when the *meaning* of a stored key changes -- not when a number moves.
SCHEMA_VERSION = 1

#: Set this to record observed values instead of asserting against them.
REGENERATE_ENV = "NSM_REGENERATE_BASELINES"

REGENERATE_CMD = f"{REGENERATE_ENV}=1 pytest testing/NSM/regression/"

#: Set this to retrain and rewrite the committed reconstruction decoder,
#: :data:`RECON_DECODER_ASSET`.
#:
#: A SECOND switch rather than a mode of :data:`REGENERATE_ENV`, because the two do
#: opposite things. Regenerating a baseline records what the code now produces.
#: Regenerating the decoder changes what the code is asked to produce -- every
#: reconstruction baseline is fitted to these weights, so they all have to be regenerated
#: after it, in a separate run. One variable driving both would hide that second step.
REGENERATE_DECODER_ENV = "NSM_REGENERATE_RECON_DECODER"

REGENERATE_DECODER_CMD = f"{REGENERATE_DECODER_ENV}=1 pytest testing/NSM/regression/"


def _enabled(variable):
    return os.environ.get(variable, "") not in ("", "0")


def regenerating():
    return _enabled(REGENERATE_ENV)


def regenerating_decoder():
    return _enabled(REGENERATE_DECODER_ENV)


def provenance():
    """The stack a baseline was generated on. Recorded in every baseline file."""
    import platform

    return {
        "platform": f"{platform.system()}-{platform.machine()}",
        "python": platform.python_version(),
        "torch": torch.__version__,
        "numpy": np.__version__,
    }


def platform_matches(generated_on):
    """
    Whether the numeric baselines apply on the machine running them.

    Only ``platform`` gates, and the asymmetry is deliberate:

    * **A different OS/architecture skips.** These baselines are pinned to Linux-x86_64,
      which is where the work happens. The CI matrix also runs ``macos-latest``; there is
      no macOS baseline, and inventing one by loosening tolerances until both platforms fit
      would leave a harness that detects nothing. Skipping says so out loud.
    * **A different torch or numpy goes RED.** A dependency bump that moves training output
      is exactly what this harness exists to report, so it is never skipped -- the failure
      message names the version difference (see ``BaselineStore._provenance_note``).

    Structural assertions -- learning rates, result keys, mesh ordering, cache keys,
    checkpoint round-trip -- are exact arithmetic or identity, so they run everywhere and
    are unaffected by this.
    """
    return not generated_on or generated_on.get("platform") == provenance()["platform"]


def _jsonable(value):
    if isinstance(value, torch.Tensor):
        value = value.detach().cpu().numpy()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    return value


class BaselineStore:
    """
    One JSON file of recorded numbers, compared with per-key tolerances.

    A missing key is a failure, never a silent pass: a harness that quietly accepts
    whatever it is given is not a regression harness.
    """

    def __init__(self, path):
        self.path = path
        self.regenerate = regenerating()
        self.recorded = {}
        if os.path.exists(path):
            with open(path) as f:
                stored = json.load(f)
            if stored.get("schema_version") != SCHEMA_VERSION:
                raise AssertionError(
                    f"{path} has schema_version {stored.get('schema_version')!r}; the "
                    f"harness expects {SCHEMA_VERSION}. Regenerate with: {REGENERATE_CMD}"
                )
            self.values = stored["values"]
            self.generated_on = stored.get("generated_on", {})
        else:
            self.values = {}
            self.generated_on = {}
            if not self.regenerate:
                raise AssertionError(
                    f"No baseline file at {path}. Create it with: {REGENERATE_CMD}"
                )

    def check(self, key, value, rtol=0.0, atol=0.0, portable=False):
        """
        Assert ``value`` matches the stored baseline for ``key``; return the baseline.

        ``portable=True`` marks a value produced by exact arithmetic -- integer or Python
        float, no accumulation -- which is identical on any machine and is therefore
        checked even where the numeric baselines do not apply. See :func:`platform_matches`.
        """
        value = _jsonable(value)
        if self.regenerate:
            self.recorded[key] = value
            return value

        if key not in self.values:
            raise AssertionError(
                f"No baseline recorded for {key!r} in {os.path.basename(self.path)}. "
                f"Regenerate with: {REGENERATE_CMD}"
            )

        if not portable and not platform_matches(self.generated_on):
            import pytest

            pytest.skip(
                f"{key!r} is a numeric baseline pinned to "
                f"{self.generated_on.get('platform')}; this is {provenance()['platform']}. "
                f"Structural and exact-arithmetic assertions still ran. Supporting a second "
                f"platform means adding a per-platform baseline file, not regenerating this "
                f"one -- see testing/NSM/regression/README.md."
            )

        expected = self.values[key]
        assert_matches(key, expected, value, rtol=rtol, atol=atol, context=self._provenance_note())
        return expected

    def _provenance_note(self):
        here = provenance()
        if not self.generated_on or self.generated_on == here:
            return ""
        differing = [
            f"{field}: baseline {self.generated_on.get(field)!r} vs here {here[field]!r}"
            for field in here
            if self.generated_on.get(field) != here[field]
        ]
        return (
            "\n  NOTE: this baseline was generated on a different stack -- "
            + "; ".join(differing)
            + "\n  A numeric difference of this size may be arithmetic, not a regression."
        )

    def flush(self):
        if not self.regenerate or not self.recorded:
            return
        if not platform_matches(self.generated_on):
            raise AssertionError(
                f"Refusing to overwrite {os.path.basename(self.path)}: it was generated on "
                f"{self.generated_on.get('platform')} and this is "
                f"{provenance()['platform']}. Regenerating here would silently replace the "
                f"pinned baseline with one from a different machine. Delete the file first "
                f"if that is really what you want."
            )
        merged = dict(self.values)
        merged.update(self.recorded)
        os.makedirs(os.path.dirname(self.path), exist_ok=True)
        with open(self.path, "w") as f:
            json.dump(
                {
                    "schema_version": SCHEMA_VERSION,
                    "generated_on": provenance(),
                    "values": merged,
                },
                f,
                indent=2,
                sort_keys=True,
            )
            f.write("\n")


def assert_matches(key, expected, actual, rtol=0.0, atol=0.0, context=""):
    if isinstance(expected, dict):
        assert set(expected) == set(
            actual
        ), f"{key}: baseline keys {sorted(expected)} != observed {sorted(actual)}"
        for sub in expected:
            assert_matches(f"{key}.{sub}", expected[sub], actual[sub], rtol, atol, context)
        return

    if expected is None or isinstance(expected, (str, bool)):
        assert expected == actual, f"{key}: baseline {expected!r}, observed {actual!r}"
        return

    exp = np.asarray(expected, dtype=float)
    got = np.asarray(actual, dtype=float)
    assert exp.shape == got.shape, f"{key}: baseline shape {exp.shape}, observed {got.shape}"
    if np.allclose(exp, got, rtol=rtol, atol=atol, equal_nan=True):
        return

    delta = np.abs(got - exp)
    worst = int(np.nanargmax(delta)) if delta.size else 0
    raise AssertionError(
        f"{key}: differs from baseline (rtol={rtol}, atol={atol}).\n"
        f"  worst element [{worst}]: baseline {exp.ravel()[worst]!r} "
        f"observed {got.ravel()[worst]!r} (abs diff {delta.ravel()[worst]:.3e})\n"
        f"  If this change is intended, regenerate with: {REGENERATE_CMD}" + context
    )


# ---------------------------------------------------------------------------
# Tolerances
# ---------------------------------------------------------------------------
#
# Every numeric tolerance the harness compares against, in one block, because they are
# cross-referenced and copies drift: ``test_gpu`` compares GPU divergence against the
# reconstruction tolerances and used to carry its own 1e-4 copies of both, which had never
# matched the real 5e-4 and 3e-4.
#
# Each is sized from a deliberate break rather than by taste, and that margin is now
# asserted on every run rather than transcribed into a table -- see :data:`MIN_HEADROOM`.
# The two latent tolerances are separate constants because they are separate quantities: a
# training latent NORM and a component of a fitted latent VECTOR.

#: Training: the loss trajectory and its components.
LOSS_RTOL = 1e-3

#: Training: the per-object latent norms.
LATENT_NORM_ATOL = 1e-4

#: Reconstruction: the fitted latent vector. Above its noise floor -- the latent is 25 Adam
#: steps from a seeded init.
FITTED_LATENT_ATOL = 5e-4

#: Reconstruction: vertex-position deciles, bounding boxes, centroids, registration centre.
#: Above the float32 floor -- marching cubes on a float32 SDF grid, then a VTK float32 save
#: (~5e-9 per point).
GEOMETRY_ATOL = 3e-4

#: Reconstruction: ASSD and the registration scale. Not a break detector.
METRIC_RTOL = 2e-3

#: Reconstruction: mesh point counts, which marching cubes can move by a vertex or two.
COUNT_RTOL = 0.03

#: The floor on :func:`headroom`: how many times its tolerance a deliberate break must move
#: a baseline for that tolerance to count as sized rather than coincidental. Asserted by
#: both ``TestDeliberateBreak`` classes, so a fixture change that weakens a break goes red
#: on the run that weakens it.
MIN_HEADROOM = 10


def _leaf_pairs(expected, actual, key):
    """``(baseline, observed)`` float arrays, walking the nesting ``assert_matches`` allows."""
    if isinstance(expected, dict):
        assert set(expected) == set(
            actual
        ), f"{key}: baseline keys {sorted(expected)} != observed {sorted(actual)}"
        for sub in expected:
            yield from _leaf_pairs(expected[sub], actual[sub], f"{key}.{sub}")
        return
    exp = np.asarray(expected, dtype=float).ravel()
    got = np.asarray(actual, dtype=float).ravel()
    assert exp.shape == got.shape, f"{key}: baseline shape {exp.shape}, observed {got.shape}"
    yield exp, got


def headroom(store, key, observed, rtol=0.0, atol=0.0):
    """
    How many times the tolerance ``observed`` actually deviates from the baseline.

    Same arguments as :meth:`BaselineStore.check`, and meant to be read beside it: ``check``
    asserts the deviation is under 1x the tolerance, ``headroom`` says what it is. The
    baseline is taken from the store rather than from ``check``'s return value because the
    callers are the deliberate-break tests, where ``check`` raises instead of returning.

    * ``atol``: ``max|observed - baseline| / atol``
    * ``rtol``: ``max(|observed - baseline| / |baseline|) / rtol``

    Exactly one of the two, because a margin against ``np.allclose``'s combined
    ``atol + rtol * |baseline|`` has no single meaning.
    """
    if (atol > 0) == (rtol > 0):
        raise ValueError("headroom takes exactly one of rtol= or atol=")

    worst = 0.0
    for expected, actual in _leaf_pairs(store.values[key], _jsonable(observed), key):
        delta = np.abs(actual - expected)
        scale = np.full(delta.shape, atol) if atol else rtol * np.abs(expected)
        # A zero baseline under rtol can never be matched by a nonzero observation.
        ratio = np.divide(delta, scale, out=np.full(delta.shape, np.inf), where=scale > 0)
        worst = max(worst, float(np.max(np.where(delta == 0, 0.0, ratio))))
    return worst


# ---------------------------------------------------------------------------
# Synthetic anatomy
# ---------------------------------------------------------------------------

#: Three "subjects", each a bone plus a cartilage surface. The two surfaces are disjoint
#: solids -- a sphere, and a small oblate ellipsoid sitting above it -- rather than nested
#: shells, because ``MultiSurfaceSDFSamples.remove_overlapping_points`` drops every point
#: interior to two objects. Nesting them leaves the inner surface with no negative samples
#: and ``sdf_pos_neg_idx`` then divides by zero.
#:
#: The offset also makes the surfaces individually identifiable by centroid, which is what
#: lets ``test_reconstruction_regression`` assert the result ``mesh`` list ORDER rather
#: than merely its length.
SUBJECTS = (
    {"bone_radius": 1.00, "cart_radius": 0.70, "cart_z": 1.45},
    {"bone_radius": 0.90, "cart_radius": 0.65, "cart_z": 1.35},
    {"bone_radius": 1.10, "cart_radius": 0.75, "cart_z": 1.60},
)


def write_synthetic_meshes(directory, subjects=SUBJECTS):
    """Write ``[[bone, cart], ...]``. Analytic: no sampling, no meshfix, no randomness."""
    import pyvista as pv

    paths = []
    for idx, subject in enumerate(subjects):
        bone = pv.Sphere(
            radius=subject["bone_radius"], theta_resolution=24, phi_resolution=24
        ).triangulate()
        cart = pv.ParametricEllipsoid(
            subject["cart_radius"],
            subject["cart_radius"],
            subject["cart_radius"] * 0.55,
            u_res=20,
            v_res=20,
        ).triangulate()
        cart.translate((0.0, 0.0, subject["cart_z"]), inplace=True)

        bone_path = os.path.join(str(directory), f"subject{idx}_bone.vtk")
        cart_path = os.path.join(str(directory), f"subject{idx}_cart.vtk")
        bone.save(bone_path)
        cart.save(cart_path)
        paths.append([bone_path, cart_path])
    return paths


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

LATENT_SIZE = 8
N_PTS_PER_SURFACE = 2000
SUBSAMPLE = 256
N_EPOCHS = 8

#: The architecture keys ``load_model``'s triplanar branch reads. Tiny, but the same shape
#: as the shipped models: ``conv_norm_type="layer"`` is what both 647 and 551 use, and it
#: is NOT the constructor default (ARCHITECTURE.md section 7.1 -- the two behave
#: differently, and only the "layer" variant is nonlinear).
ARCHITECTURE = {
    "latent_size": LATENT_SIZE,
    "objects_per_decoder": 2,
    "mesh_names": ["bone", "cart"],
    "conv_hidden_dims": [16, 16],
    "conv_deep_image_size": 2,
    "conv_norm": True,
    "conv_norm_type": "layer",
    "conv_start_with_mlp": True,
    "sdf_latent_size": 16,
    "sdf_hidden_dims": [32, 32],
    "weight_norm": True,
    "final_activation": "tanh",
    "activation": "relu",
    "dropout_prob": 0.0,
    "sum_conv_output_features": True,
    "conv_pred_sdf": False,
    "padding": 0.1,
}

#: The two entries differ in Interval AND Factor, not just Initial, so transposing their
#: targets inverts the run rather than perturbing it. Both decay inside the 8 epochs.
LR_SCHEDULE = [
    {"Target": "model", "Type": "Step", "Initial": 0.005, "Interval": 3, "Factor": 0.5},
    {"Target": "latent", "Type": "Step", "Initial": 0.001, "Interval": 2, "Factor": 0.9},
]


def training_config(experiment_directory):
    """The full config ``train_deep_sdf`` consumes. CPU, 8 epochs, no wandb, no profiler."""
    config = dict(ARCHITECTURE)
    config.update(
        {
            "LearningRateSchedule": [dict(entry) for entry in LR_SCHEDULE],
            "optimizer": "Adam",
            "weight_decay": 1e-4,
            "n_epochs": N_EPOCHS,
            "checkpoint_epochs": N_EPOCHS,
            "additional_checkpoints": [],
            "save_frequency": 4,
            "device": "cpu",
            "objects_per_batch": 2,
            "num_data_loader_threads": 0,
            "prefetch_factor": None,
            "batch_split": 1,
            "samples_per_object_per_batch": SUBSAMPLE,
            "enforce_minmax": True,
            # 1.0 is what both shipped ShapeMedKnee configs (647, 551) use; the shipped
            # default_config.json's 0.1 is the DeepSDF value. The choice is not neutral:
            # `enforce_minmax` clamps the PREDICTION as well as the target, so every
            # sample predicted outside +/-clamp_dist contributes exactly zero gradient.
            # See test_training_regression.TestClampedPredictionGradients.
            "clamp_dist": 1.0,
            "surface_accuracy_e": None,
            "surface_accuracy_schedule": "linear",
            "surface_accuracy_cooldown": None,
            "sample_difficulty_weight": None,
            "sample_difficulty_weight_schedule": "linear",
            "sample_difficulty_cooldown": None,
            "code_regularization": True,
            "code_regularization_type_prior": "identity",
            "code_regularization_weight": 1e-4,
            "code_regularization_warmup": 2,
            "code_cyclic_anneal": False,
            "grad_clip": None,
            "verbose": False,
            "profiler": False,
            "latent_bound": 10,
            "latent_init_std": 0.01,
            "latent_init_normal": True,
            "variational": False,
            "experiment_directory": str(experiment_directory),
        }
    )
    return config


# ---------------------------------------------------------------------------
# Entry points
# ---------------------------------------------------------------------------


@contextlib.contextmanager
def quiet():
    """NSM prints heavily on every code path; keep failure output readable."""
    with contextlib.redirect_stdout(io.StringIO()):
        yield


#: Near- and far-surface perturbation widths, in the harness's normalized coordinates.
#:
#: Scaled from the shipped ShapeMedKnee configs rather than picked: ``647_nsm_femur_v0.0.1``
#: and ``551_nsm_femur_bone_v0.0.1`` use ``sigma_near`` ~= 0.743 and ``sigma_far`` = 2.35 in
#: millimetres, with ``scale_jointly: True``, against a femur roughly 80 mm across. The
#: harness normalizes each subject to ``max_rad`` 1 (``scale_jointly=False``), so the same
#: widths relative to the object are ~0.009 and ~0.029.
SIGMA_NEAR = 0.01
SIGMA_FAR = 0.03


def build_dataset(mesh_paths, cache_dir, seed=0, **overrides):
    """
    A ``MultiSurfaceSDFSamples`` on the near-surface sampling path production uses.

    ``seed`` reaches sampling two ways, and both are deliberate: as ``random_seed``, which
    seeds every draw, and as ``np.random.seed``, which still governs the legacy global
    stream an unseeded (``random_seed=None``) call draws from.

    ``loc_save`` is always an explicit temporary directory. The constructor's default is
    read from ``LOC_SDF_CACHE`` *at import time*, so setting that env var inside a test
    would come too late and the test would write into the developer's real
    ``~/.cache/nsm_sdf_cache``.
    """
    from NSM.datasets.sdf_dataset import MultiSurfaceSDFSamples

    n_surfaces = len(mesh_paths[0])
    kwargs = dict(
        list_mesh_paths=mesh_paths,
        subsample=SUBSAMPLE,
        n_pts=[N_PTS_PER_SURFACE] * n_surfaces,
        p_near_surface=[0.4] * n_surfaces,
        p_further_from_surface=[0.4] * n_surfaces,
        sigma_near=[SIGMA_NEAR] * n_surfaces,
        sigma_far=[SIGMA_FAR] * n_surfaces,
        center_pts=True,
        norm_pts=True,
        scale_method="max_rad",
        loc_save=str(cache_dir),
        multiprocessing=False,
        store_data_in_memory=False,
        save_cache=True,
        load_cache=False,
        random_seed=seed,
        fix_mesh=False,
        mesh_to_scale=0,
        scale_all_meshes=True,
        equal_pos_neg=True,
    )
    kwargs.update(overrides)

    np.random.seed(seed)
    with quiet():
        return MultiSurfaceSDFSamples(**kwargs)


def build_single_surface_dataset(mesh_paths, cache_dir, seed=0, **overrides):
    """
    An ``SDFSamples`` -- the single-surface PARENT class -- on the same near-surface path.

    Same conventions as :func:`build_dataset`, for the same reasons: explicit ``loc_save``,
    ``multiprocessing=False``, and ``seed`` used both as ``random_seed`` and through
    ``np.random.seed``.

    What differs is the shape of the arguments, not their meaning. ``SDFSamples`` takes
    SCALARS where the subclass takes one entry per surface -- ``n_pts`` is an int, the two
    sigmas and the two probabilities are floats -- and ``mesh_paths`` is a list of single
    paths rather than a list of lists. The multi-surface arguments (``mesh_to_scale``,
    ``scale_all_meshes``) do not exist on the parent at all.

    ``SDFSamples.get_sample_data_dict``, ``get_pt_sample_combos`` and ``__getitem__`` are
    each separate code from the subclass's overrides, so nothing the subclass's tests
    establish carries over to them.
    """
    from NSM.datasets.sdf_dataset import SDFSamples

    kwargs = dict(
        list_mesh_paths=mesh_paths,
        subsample=SUBSAMPLE,
        n_pts=N_PTS_PER_SURFACE,
        p_near_surface=0.4,
        p_further_from_surface=0.4,
        sigma_near=SIGMA_NEAR,
        sigma_far=SIGMA_FAR,
        center_pts=True,
        norm_pts=True,
        scale_method="max_rad",
        loc_save=str(cache_dir),
        multiprocessing=False,
        store_data_in_memory=False,
        save_cache=True,
        load_cache=False,
        random_seed=seed,
        fix_mesh=False,
        equal_pos_neg=True,
    )
    kwargs.update(overrides)

    np.random.seed(seed)
    with quiet():
        return SDFSamples(**kwargs)


def build_model(config, seed=42):
    """
    Construct the decoder ``config`` describes, exactly as ``load_model`` would.

    NSM offers no public "build the model this config describes" call: ``load_model``
    needs a checkpoint, which a fresh model does not have, and the downstream consumer
    works around that by hand-rolling the mapping and dropping ``padding`` (SCOPE.md
    section 3.1). Going through ``loader._get_triplanar_params`` keeps the model this
    harness trains identical to the model ``load_model`` builds. When the decoder registry
    of plan section 8.1 lands, this import is what should fail loudly.
    """
    from NSM.models.loader import _get_triplanar_params

    model_class, params = _get_triplanar_params(config)
    torch.manual_seed(seed)
    return model_class(**params)


def run_training(config, model, dataset, seed=42):
    """
    Run ``train_deep_sdf`` and return ``(records, return_value)``.

    One record per epoch: loss, its components, every param group's learning rate, and the
    latent norms. ``train_deep_sdf`` returns ``None``, so none of that is observable from
    the public entry point; the harness wraps ``train_epoch`` rather than re-implementing
    the loop, so what is recorded is what the real trainer did.
    """
    import NSM.train.train_deep_sdf as module

    records = []
    original = module.train_epoch

    def recording_train_epoch(*args, **kwargs):
        # train_deep_sdf passes optimizer/config/epoch by keyword; take them from the call
        # rather than restating the signature, so a signature change is not silently
        # absorbed here.
        log = original(*args, **kwargs)
        latent_vecs = args[2] if len(args) > 2 else kwargs["latent_vecs"]
        optimizer = kwargs["optimizer"]
        epoch = kwargs["epoch"]
        records.append(
            {
                "epoch": epoch,
                "loss": log["loss"],
                "l1_loss": log["l1_loss"],
                "code_reg_loss": log["latent_code_regularization_loss"],
                # Read AFTER the epoch: adjust_learning_rate() runs at the top of
                # train_epoch, so these are the rates the epoch actually ran with.
                "lrs": {group["name"]: group["lr"] for group in optimizer.param_groups},
                "targets": {group["name"]: group["target"] for group in optimizer.param_groups},
                "latent_norms": torch.norm(latent_vecs.weight.data, dim=1).tolist(),
            }
        )
        return log

    module.train_epoch = recording_train_epoch
    try:
        torch.manual_seed(seed)
        np.random.seed(seed)
        with quiet():
            returned = module.train_deep_sdf(config, model, dataset, use_wandb=False)
    finally:
        module.train_epoch = original
    return records, returned


# ---------------------------------------------------------------------------
# The reconstruction decoder, as a committed asset
# ---------------------------------------------------------------------------
#
# Every reconstruction test runs on ONE decoder, and until Aug 2026 the harness retrained
# it in-session on every run. That made ``baselines/reconstruction.json`` a pin on a
# 60-epoch gradient-descent trajectory rather than on ``reconstruct_mesh``, and gradient
# descent amplifies a last-bit arithmetic difference exponentially. Measured between torch
# 2.8.0+cu128 and 2.7.1+cu126 on identical inputs: weights diverge 6.3e-07 by epoch 10,
# 1.7e-05 by 20, 1.4e-02 by 30, saturating near 3.9e-02 -- past epoch 30 the two stacks
# hold different models. The geometry baselines moved 763x GEOMETRY_ATOL across that bump;
# ``reconstruct_mesh`` run on FIXED weights moved 0.005x. Absorbing the difference would
# have meant a tolerance 12x wider than the deliberate break it exists to detect.
#
# Training output is pinned directly, and better, by ``baselines/training.json``. So the
# decoder is generated once and committed, and what the reconstruction baselines pin is
# reconstruction. README.md has the full decomposition and the regeneration procedure.

RECON_DECODER_ASSET = os.path.join(os.path.dirname(__file__), "assets", "reconstruction_decoder.pt")

#: Epochs the committed decoder was trained for. A decoder that has not learnt a sign
#: change has no zero level set, and every reconstruction returns ``mesh=[None, None]``.
#: See ``test_reconstruction_regression.TestDecoderWithNoZeroLevelSet``.
RECON_TRAINING_EPOCHS = 60


def train_reconstruction_decoder(dataset, experiment_directory):
    """
    Train the decoder :data:`RECON_DECODER_ASSET` holds. The only producer of those weights.

    Also run on every suite invocation by
    ``test_reconstruction_regression.TestAFreshlyTrainedDecoder``, so the regeneration path
    cannot rot between the rare occasions anyone needs it.
    """
    config = training_config(experiment_directory)
    config.update(
        {
            "n_epochs": RECON_TRAINING_EPOCHS,
            "checkpoint_epochs": RECON_TRAINING_EPOCHS,
            "save_frequency": RECON_TRAINING_EPOCHS,
            "code_regularization_warmup": 20,
            "LearningRateSchedule": [
                {"Target": "model", "Type": "Step", "Initial": 0.01, "Interval": 40, "Factor": 0.5},
                {
                    "Target": "latent",
                    "Type": "Step",
                    "Initial": 0.005,
                    "Interval": 40,
                    "Factor": 0.5,
                },
            ],
        }
    )
    model = build_model(config)
    run_training(config, model, dataset)
    model.eval()
    return model


def save_reconstruction_decoder(model, path=RECON_DECODER_ASSET):
    """
    Write the asset, provenance included.

    ``generated_on`` goes INSIDE the checkpoint rather than in a sidecar file, for the same
    reason ``baselines/*.json`` carry theirs inside: it cannot then be separated from, or
    left stale against, the weights it describes. Its values are coerced to ``str`` because
    ``torch.__version__`` is a ``TorchVersion``, which ``weights_only=True`` refuses to
    unpickle -- so an uncoerced dict would write an asset :func:`load_reconstruction_decoder`
    cannot read.

    Refuses to overwrite an asset from another platform, mirroring ``BaselineStore.flush``
    and for a stronger reason: the committed reconstruction baselines are fitted to these
    exact weights, so replacing them from a different machine moves every one of them.
    """
    if os.path.exists(path):
        existing = torch.load(path, weights_only=True).get("generated_on", {})
        if not platform_matches(existing):
            raise AssertionError(
                f"Refusing to overwrite {os.path.basename(path)}: it was generated on "
                f"{existing.get('platform')} and this is {provenance()['platform']}. Every "
                f"committed reconstruction baseline is fitted to that decoder and would "
                f"move. Delete the file first if that is really what you want."
            )

    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(
        {
            "generated_on": {field: str(value) for field, value in provenance().items()},
            "state_dict": model.state_dict(),
        },
        path,
    )


def load_reconstruction_decoder(path=RECON_DECODER_ASSET):
    """
    The committed decoder, in eval mode.

    A missing or unloadable asset is an ERROR that says how to rebuild it, never a skip: a
    reconstruction suite that quietly stops running is indistinguishable from one that
    passes.

    The model is built through :func:`build_model`, so the architecture stays stated in one
    place, and loaded with ``strict=True`` on purpose -- an architecture change must fail
    here, loudly, rather than half-load a checkpoint into a model it no longer fits.
    """
    if not os.path.exists(path):
        raise AssertionError(
            f"No reconstruction decoder at {path}. It is a committed test asset, not "
            f"something a run rebuilds on its own. Regenerate with: {REGENERATE_DECODER_CMD}"
        )

    model = build_model(dict(ARCHITECTURE))
    try:
        model.load_state_dict(torch.load(path, weights_only=True)["state_dict"], strict=True)
    except Exception as error:
        raise AssertionError(
            f"{os.path.basename(path)} did not load into the model ARCHITECTURE describes "
            f"-- an architecture change, or a damaged file: {error}\n"
            f"  Regenerate with: {REGENERATE_DECODER_CMD}\n"
            f"  Then regenerate the reconstruction baselines with {REGENERATE_CMD}, "
            f"because a different decoder reconstructs different numbers."
        ) from error
    model.eval()
    return model


#: Reconstruction settings. Small enough for CPU, large enough to resolve both surfaces.
#: ``create_mesh_adaptive``'s coarse pass is fixed at 64^3 and is not reachable through
#: ``reconstruct_mesh``'s signature, so ``n_pts_per_axis`` is not the whole cost.
RECON_KWARGS = dict(
    latent_size=LATENT_SIZE,
    num_iterations=25,
    lr=0.005,
    l2reg=False,
    latent_reg_weight=1e-4,
    loss_type="l1",
    n_lr_updates=2,
    lr_update_factor=10,
    return_latent=True,
    register_similarity=True,
    scale_jointly=False,
    scale_all_meshes=True,
    objects_per_decoder=2,
    get_rand_pts=False,
    n_pts_random=1000,
    sigma_rand_pts=0.01,
    n_samples_latent_recon=2000,
    calc_assd=True,
    convergence="num_iterations",
    convergence_patience=5,
    clamp_dist=0.1,
    fix_mesh=False,
    verbose=False,
    return_registration_params=True,
    n_pts_per_axis=48,
    n_pts_per_axis_mean_mesh=32,
    device="cpu",
)


def run_reconstruction(mesh_paths, model, seed=42, sample_seed=None, **overrides):
    """
    Call ``reconstruct_mesh`` the way ``kneepipeline/steps/run_nsm.py:170`` does: a *list*
    of mesh paths, every argument by name.

    Two different seeds meet here and they are not interchangeable:

    * ``seed`` is this harness's own. It seeds ``torch`` and ``numpy`` globally before the
      call, which covers the latent initialization and the latent optimizer.
    * ``sample_seed`` is ``reconstruct_mesh``'s ``seed`` argument, which seeds the POINT
      DRAW. It does nothing at all unless ``get_rand_pts=True``, and ``RECON_KWARGS``
      leaves that False.

    They need separate names because a parameter this function declares can never reach
    ``overrides``: ``run_reconstruction(..., seed=7)`` reseeds torch and numpy and leaves
    ``reconstruct_mesh`` on its default ``seed=None``. That shadowing is silent -- the
    reconstruction still runs and still looks seeded -- so it is stated here rather than
    left to be rediscovered.

    ``sample_seed=None`` is the default and is what ``reconstruct_mesh`` would have used
    anyway, so passing it explicitly leaves every existing caller, and every committed
    baseline, bit-for-bit unchanged.
    """
    from NSM.reconstruct import reconstruct_mesh

    kwargs = dict(RECON_KWARGS)
    kwargs.update(overrides)
    torch.manual_seed(seed)
    np.random.seed(seed)
    with quiet():
        return reconstruct_mesh(path=list(mesh_paths), decoders=model, seed=sample_seed, **kwargs)


#: Deciles of each coordinate axis. Compared instead of the raw vertex array because
#: marching cubes can add or drop a vertex near the level set from a last-bit difference
#: in the SDF, and a raw array pinned to an exact length is not a portable baseline. These
#: are still vertex positions -- an order-independent summary of all of them -- and they
#: move far more than float noise when the surface genuinely changes.
DECILES = [i / 10 for i in range(11)]


def mesh_summary(mesh):
    """Topology-tolerant geometric fingerprint of a reconstructed surface."""
    points = np.asarray(mesh.point_coords, dtype=float)
    centroid = points.mean(axis=0)
    return {
        "centroid": centroid.tolist(),
        "bbox_min": points.min(axis=0).tolist(),
        "bbox_max": points.max(axis=0).tolist(),
        "mean_radius": float(np.linalg.norm(points - centroid, axis=1).mean()),
        "x_deciles": np.quantile(points[:, 0], DECILES).tolist(),
        "y_deciles": np.quantile(points[:, 1], DECILES).tolist(),
        "z_deciles": np.quantile(points[:, 2], DECILES).tolist(),
    }
