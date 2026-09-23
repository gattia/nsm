"""
The regression harness's machinery: baselines, tolerances, the synthetic subjects and
config, and the three entry points (build a dataset, train, reconstruct). Fixtures are in
``conftest.py``. ``README.md`` explains the design.
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

#: Set this to retrain and rewrite :data:`RECON_DECODER_ASSET`. A separate switch, because
#: every reconstruction baseline is fitted to those weights and must be regenerated after.
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
    Whether the numeric baselines apply here. A different OS or architecture skips them; a
    different torch or numpy does not, because that is what the harness exists to report.
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
    """One JSON file of recorded numbers, compared with per-key tolerances. A missing key fails."""

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
        Assert ``value`` matches the baseline for ``key``. ``portable=True`` marks exact
        arithmetic, identical on any machine, so it is checked on every platform.
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
# Every tolerance, in one block, because copies drift. Each is sized from a deliberate break,
# and the margin is asserted on every run (:data:`MIN_HEADROOM`).

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

#: How many times its tolerance a deliberate break must move a baseline. Asserted by both
#: ``TestDeliberateBreak`` classes.
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
    How many times its tolerance ``observed`` deviates from the baseline:
    ``max|observed - baseline| / atol``, or the relative deviation over ``rtol``. Exactly one
    of the two, since ``np.allclose``'s combined bound has no single meaning.
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

#: Three subjects, each a bone sphere and a small cartilage ellipsoid above it. Disjoint,
#: because a nested surface loses its interior to ``remove_overlapping_points``. The offset
#: makes the surfaces identifiable by centroid, which is how the ``mesh`` ORDER is asserted.
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

#: Tiny, but the shape of the shipped models: ``conv_norm_type="layer"``, as 647 and 551 use.
ARCHITECTURE = {
    "latent_size": LATENT_SIZE,
    "objects_per_decoder": 2,
    "mesh_names": ["bone", "cart"],
    "conv_hidden_dims": [16, 16],
    "conv_deep_image_size": 2,
    "conv_norm": True,
    "conv_norm_type": "layer",
    "conv_start_with_mlp": True,
    #: The historical architecture, which every shipped model and baseline uses.
    "conv_activation": None,
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

#: The entries differ in Interval and Factor, so swapping targets inverts the run. Both
#: decay within the 8 epochs.
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
            # The shipped ShapeMedKnee value. Not neutral: see
            # test_training_regression.TestClampedPredictionGradients.
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


#: The shipped widths (0.743 mm and 2.35 mm on an ~80 mm femur), in max-radius-1 units.
SIGMA_NEAR = 0.01
SIGMA_FAR = 0.03


def build_dataset(mesh_paths, cache_dir, seed=0, **overrides):
    """
    A ``MultiSurfaceSDFSamples`` on the near-surface path production uses. ``seed`` is
    passed as ``random_seed`` and also to ``np.random.seed``, which an unseeded
    (``random_seed=None``) call still draws from. ``loc_save`` is always explicit, so no
    test writes into the developer's real cache.
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
    An ``SDFSamples``, the single-surface parent class, under :func:`build_dataset`'s
    conventions. It takes scalars where the subclass takes one entry per surface.
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
    The decoder ``config`` describes, built as ``load_model`` builds it. NSM has no public
    call for this, so it uses ``loader._get_triplanar_params``.
    """
    from NSM.models.loader import _get_triplanar_params

    model_class, params = _get_triplanar_params(config)
    torch.manual_seed(seed)
    return model_class(**params)


def run_training(config, model, dataset, seed=42):
    """
    Run ``train_deep_sdf``. Returns ``(records, history)``: one record per epoch in the
    shape the baselines pin, and the history the trainer returned (#28).
    """
    from NSM.train.train_deep_sdf import train_deep_sdf

    torch.manual_seed(seed)
    np.random.seed(seed)
    with quiet():
        history = train_deep_sdf(config, model, dataset, use_wandb=False)
    records = [
        {
            "epoch": entry["epoch"],
            "loss": entry["loss"],
            "l1_loss": entry["l1_loss"],
            "code_reg_loss": entry["latent_code_regularization_loss"],
            "lrs": entry["lrs"],
            "targets": entry["targets"],
            "latent_norms": entry["latent_norms"],
        }
        for entry in history
    ]
    return records, history


# ---------------------------------------------------------------------------
# The reconstruction decoder, as a committed asset
# ---------------------------------------------------------------------------
#
# Every reconstruction test runs on one committed decoder. Retraining it each run pinned a
# 60-epoch gradient-descent trajectory instead of ``reconstruct_mesh``, and a torch bump
# moved the geometry baselines 763x their tolerance. README.md has the measurements.

RECON_DECODER_ASSET = os.path.join(os.path.dirname(__file__), "assets", "reconstruction_decoder.pt")

#: Epochs the committed decoder was trained for. Fewer, and it may have no zero level set.
RECON_TRAINING_EPOCHS = 60


def train_reconstruction_decoder(dataset, experiment_directory):
    """
    Train the decoder :data:`RECON_DECODER_ASSET` holds. ``TestAFreshlyTrainedDecoder`` runs
    it every time, so the regeneration path cannot rot.
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
    Write the asset with its provenance inside, so the two cannot separate. Values are
    strings because ``weights_only=True`` refuses to unpickle a ``TorchVersion``. Refuses to
    overwrite an asset from another platform.
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
    The committed decoder, in eval mode, loaded strictly. A missing or unloadable asset is
    an error naming the regeneration command, never a skip.
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


#: Small enough for CPU, large enough to resolve both surfaces.
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
    return_registration_params=True,
    n_pts_per_axis=48,
    n_pts_per_axis_mean_mesh=32,
    device="cpu",
)


def run_reconstruction(mesh_paths, model, seed=42, sample_seed=None, **overrides):
    """
    Call ``reconstruct_mesh`` as kneepipeline does: a list of paths, every argument by name.

    ``seed`` seeds torch and numpy globally, for the latent's initialization and optimizer.
    ``sample_seed`` is ``reconstruct_mesh``'s own ``seed``, for the point draw, which only
    happens with ``get_rand_pts=True``. It has its own name because ``seed=`` here can never
    reach ``reconstruct_mesh``, and that shadowing would be silent.
    """
    from NSM.reconstruct import reconstruct_mesh

    kwargs = dict(RECON_KWARGS)
    kwargs.update(overrides)
    torch.manual_seed(seed)
    np.random.seed(seed)
    with quiet():
        return reconstruct_mesh(path=list(mesh_paths), decoders=model, seed=sample_seed, **kwargs)


#: Deciles of each axis stand in for the vertex array: marching cubes can add or drop a
#: vertex on a last-bit difference, so an exact-length array is not a portable baseline.
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
