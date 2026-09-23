"""
``reconstruct_mesh`` and the batch driver ``get_mean_errors``: what they accept, the stages
they run, what they report, and what they hand their collaborators.

The end-to-end runs use the sampled branch, ``get_rand_pts=True``, on one sphere path.
``TestASubjectMissingASurface`` passes two paths, one of them ``None``.
"""

import inspect
import json
import logging
import re
from pathlib import Path

import numpy as np
import pytest
import torch
import wandb

import NSM
import NSM.reconstruct.main as recon_main
from NSM.mesh import create_mesh_adaptive
from NSM.reconstruct.predictive_validation_class import Regress
from NSM.reconstruct.recon_evaluation import compute_recon_loss, get_mean_errors
from NSM.reconstruct.utils import compute_chamfer, refuse_unknown_kwargs
from NSM.reconstruct.wandb_logging import _process_meshes_for_wandb, prepare_results_for_wandb


class SphereDecoder(torch.nn.Module):
    """A radius-0.5 sphere SDF that counts the points it evaluates."""

    def __init__(self, objects=1):
        super().__init__()
        self.objects = objects
        self.n_points_evaluated = 0

    def forward(self, x=None, latent=None, xyz=None, epoch=None):
        pts = xyz if xyz is not None else x[:, -3:]
        self.n_points_evaluated += pts.shape[0]
        sdf = torch.norm(pts, dim=1, keepdim=True) - 0.5
        if latent is not None:
            sdf = sdf + 0.0 * latent.sum()  # keeps the output on the latent's graph
        return sdf.repeat(1, self.objects)


@pytest.fixture(scope="module")
def sphere_path(tmp_path_factory):
    import pyvista as pv

    path = str(tmp_path_factory.mktemp("recon") / "sphere.vtk")
    pv.Sphere(radius=0.5, theta_resolution=18, phi_resolution=18).triangulate().save(path)
    return path


def run(path, decoder=None, **overrides):
    kwargs = dict(
        latent_size=8,
        num_iterations=2,
        get_rand_pts=True,
        n_pts_random=50,
        sigma_rand_pts=0.05,
        seed=3,
        n_pts_per_axis=32,
        fix_mesh=False,
        device="cpu",
    )
    kwargs.update(overrides)
    return recon_main.reconstruct_mesh(path=path, decoders=decoder or SphereDecoder(), **kwargs)


class TestUnknownKeywordsAreRefused:
    def test_a_misspelled_or_deleted_keyword_raises_by_name(self, sphere_path):
        """
        Fails if ``reconstruct_mesh`` accepts a misspelled keyword or the deleted ``verbose``
        instead of raising ``TypeError`` naming it (KNOWN_ISSUES History 20).

        The ``log_wandb_step`` check calls ``refuse_unknown_kwargs`` with a hand-built
        deprecated set, not through ``reconstruct_mesh``. It would still pass if
        ``reconstruct_mesh`` named ``log_wandb_step``.
        """
        for wrong in (
            "n_pts_per_axes",
            "num_iteration",
            "calc_assd_",
            "latent_reg_wieght",
            "clamp_distance",
            "verbose",
        ):
            with pytest.raises(TypeError, match=wrong):
                run(sphere_path, **{wrong: 1})
        with pytest.raises(TypeError, match="log_wandb_step"):
            refuse_unknown_kwargs(
                {"log_wandb_step": 5},
                function_name="reconstruct_mesh",
                deprecated=frozenset({"batch_size_latent_recon"}),
            )

    def test_the_consumers_keyword_set_is_accepted(self):
        """
        Fails if ``reconstruct_mesh``'s signature renames or removes a keyword that
        ``kneepipeline/steps/run_nsm.py`` passes.

        The list is copied by hand. Its two unnamed entries must stay unnamed:
        ``batch_size_latent_recon`` is accepted as deprecated, and ``verbose`` is deleted.
        ``run_nsm.py`` no longer passes ``verbose``.
        """
        consumer = set(
            """
            path decoders latent_size num_iterations l2reg latent_reg_weight loss_type lr
            lr_update_factor n_lr_updates return_latent register_similarity scale_jointly
            scale_all_meshes objects_per_decoder batch_size_latent_recon get_rand_pts
            n_pts_random sigma_rand_pts n_samples_latent_recon calc_assd convergence
            convergence_patience clamp_dist fix_mesh verbose return_registration_params
            """.split()
        )
        named = set(inspect.signature(recon_main.reconstruct_mesh).parameters)
        assert consumer - named == {"batch_size_latent_recon", "verbose"}


def test_any_truthy_register_similarity_takes_the_registered_path(sphere_path):
    """
    Fails if ``reconstruct_mesh`` registers only when ``register_similarity is True``, so ``1``
    or ``"similarity"`` skips the mean-mesh build or the registration.
    """
    for flag in (True, 1, "similarity"):
        result = run(
            sphere_path,
            register_similarity=flag,
            n_pts_per_axis_mean_mesh=24,
            return_registration_params=True,
        )
        assert result["icp_transform"] is not None, flag


class TestTheReferenceMeshIsBuiltWhenItIsUsed:
    """
    Only registration reads the mean mesh. Building it for ``scale_jointly`` costs 876,269
    decoder evaluations at the default grid, and raises ``NoZeroLevelSetError`` when the mean
    shape has no surface.
    """

    def test_scale_jointly_alone_builds_no_mean_mesh(self, sphere_path):
        """
        Fails if ``reconstruct_mesh(scale_jointly=True)`` builds the mean mesh while
        ``register_similarity`` is False.

        The decoder counts the points it evaluates, so a build shows as a higher count.
        """
        counts = []
        for scale_jointly in (False, True):
            decoder = SphereDecoder()
            run(sphere_path, decoder, scale_jointly=scale_jointly, register_similarity=False)
            counts.append(decoder.n_points_evaluated)
        assert counts[0] == counts[1]

    def test_building_a_mesh_consumes_no_randomness(self):
        """
        Fails if ``create_mesh_adaptive`` draws from the global torch or numpy generator.

        ``reconstruct_mesh`` builds the mean mesh before it draws the initial latent from the
        global torch generator.
        """
        torch.manual_seed(0)
        np.random.seed(0)
        torch_before, numpy_before = torch.get_rng_state().clone(), np.random.get_state()
        create_mesh_adaptive(
            decoder=SphereDecoder(),
            latent_vector=torch.zeros(1, 8),
            n_pts_per_axis=32,
            objects=1,
            batch_size=32**3,
            device="cpu",
        )
        assert torch.equal(torch_before, torch.get_rng_state())
        assert numpy_before[1].tolist() == np.random.get_state()[1].tolist()


def test_the_result_is_complete_deterministic_and_timed(sphere_path):
    """
    Fails if two identically seeded ``reconstruct_mesh`` runs differ, a registration or latent
    key is missing, or a ``timings.stage(...)`` in ``main.py`` is not returned as
    ``time_<name>`` under ``return_timing``.

    Stage names are read from the source with a regex, so a new stage is checked too.
    """

    def full_run():
        torch.manual_seed(11)
        np.random.seed(11)
        return run(
            sphere_path,
            register_similarity=True,
            n_pts_per_axis_mean_mesh=24,
            calc_symmetric_chamfer=True,
            calc_assd=True,
            return_latent=True,
            return_registration_params=True,
            return_timing=True,
        )

    first, second = full_run(), full_run()
    assert {"mesh", "orig_mesh", "latent", "icp_transform", "center", "scale"} <= set(first)
    assert torch.equal(first["latent"], second["latent"])
    np.testing.assert_array_equal(first["mesh"][0].point_coords, second["mesh"][0].point_coords)
    for key, value in first.items():
        if isinstance(value, float) and not key.startswith("time_"):
            assert value == second[key], key

    source = open(recon_main.__file__, encoding="utf-8").read()
    stages = {f"time_{name}" for name in re.findall(r'timings\.stage\(\s*"(\w+)"', source)}
    timings = {key: value for key, value in first.items() if key.startswith("time_")}
    assert stages and stages <= set(timings)
    assert all(0.0 <= value < 600.0 for value in timings.values())


def test_a_host_at_debug_sees_the_stage_records(sphere_path, caplog):
    """
    Fails if ``reconstruct_mesh`` stops logging its "Loaded mesh in" and "Created mesh in"
    stage records at DEBUG on the ``NSM`` logger.
    """
    with caplog.at_level(logging.DEBUG, logger="NSM"):
        run(sphere_path)
    assert "Loaded mesh in" in caplog.text and "Created mesh in" in caplog.text


class TestASubjectMissingASurface:
    """
    ``SCOPE`` §2.5b: fitting from a subset of surfaces is supported. The shipped config asks
    for chamfer and ASSD on every fit.
    """

    def test_the_fit_decodes_every_surface_and_scores_the_missing_one_nan(self, sphere_path):
        """
        Fails if ``reconstruct_mesh`` with a ``None`` path crashes, leaves that surface out of
        ``mesh``, or scores its chamfer or ASSD other than NaN.
        """
        kwargs = dict(decoder=SphereDecoder(objects=2), objects_per_decoder=2)
        result = run([sphere_path, None], return_latent=True, **kwargs)
        assert result["orig_mesh"][1] is None and result["latent"] is not None
        assert len(result["mesh"]) == 2 and all(mesh is not None for mesh in result["mesh"])

        for flag, key in (("calc_symmetric_chamfer", "chamfer"), ("calc_assd", "assd")):
            result = run([sphere_path, None], **{flag: True}, **kwargs)
            assert not np.isnan(result[f"{key}_0"]) and np.isnan(result[f"{key}_1"])


class TestTheKnobsThatDifferByLayer:
    """
    ``chamfer_norm`` is a power, so 1 and 2 report chamfer in different units. The defaults are
    the values every shipped run used: no config carries ``chamfer_norm``, and the trainer
    does not pass it (``test_default_config_sync``).
    """

    def test_each_knob_has_one_default_and_it_is_the_shipped_value(self):
        """
        Fails if ``chamfer_norm`` or ``sigma_rand_pts`` has a different default in
        ``reconstruct_mesh``, ``get_mean_errors``, ``compute_recon_loss`` or
        ``default_config.json`` (#56, KNOWN_ISSUES History 29).

        It also fails if ``compute_chamfer`` ignores ``power`` or stops defaulting it to 1.
        """

        def default(func, name):
            return inspect.signature(func).parameters[name].default

        assert {
            default(f, "chamfer_norm")
            for f in (recon_main.reconstruct_mesh, get_mean_errors, compute_recon_loss)
        } == {2}
        assert {
            default(f, "sigma_rand_pts") for f in (recon_main.reconstruct_mesh, get_mean_errors)
        } == {0.01}
        config = json.loads(
            (Path(NSM.__file__).parent / "configs" / "default_config.json").read_text(
                encoding="utf-8"
            )
        )
        assert config["sigma_rand_pts_recon"] == 0.01

        # The generic helper keeps the textbook exponent; its one NSM caller passes it.
        assert default(compute_chamfer, "power") == 1
        rng = np.random.default_rng(0)
        a, b = rng.normal(size=(64, 3)), rng.normal(size=(64, 3)) + 0.3
        at_1, at_2 = compute_chamfer(a, b, power=1), compute_chamfer(a, b, power=2)
        assert at_2 != pytest.approx(at_1) and at_2 != pytest.approx(at_1**2)


class _Abort(Exception):
    pass


class TestNPtsRandomReachesTheReaders:
    def test_the_request_arrives_as_n_pts(self, monkeypatch):
        """
        Fails if ``reconstruct_mesh`` passes ``n_pts_random`` to either sampled-points reader
        under any name but ``n_pts``, or ``get_mean_errors`` forwards the deprecated
        ``batch_size_latent_recon`` (KNOWN_ISSUES History 9).

        The readers are stubbed, so the test checks keyword names, not the points drawn.
        """
        captured = {}

        def recorder(*args, **kwargs):
            captured.update(kwargs)
            raise _Abort

        for reader, path, expected in (
            ("read_meshes_get_sampled_pts", ["bone.vtk", "cart.vtk"], [200, 200]),
            ("read_mesh_get_sampled_pts", "bone.vtk", 200),
        ):
            captured.clear()
            monkeypatch.setattr(recon_main, reader, recorder)
            with pytest.raises(_Abort):
                recon_main.reconstruct_mesh(
                    path=path,
                    decoders=torch.nn.Linear(1, 1),
                    latent_size=8,
                    get_rand_pts=True,
                    n_pts_random=200,
                )
            assert captured.get("n_pts") == expected and "n_pts_random" not in captured

        captured.clear()
        monkeypatch.setattr(
            recon_main,
            "reconstruct_mesh",
            lambda path=None, **kwargs: captured.update(kwargs) or {"mesh": []},
        )
        recon_main.get_mean_errors(mesh_paths=["a.vtk"], decoders=None, latent_size=4)
        assert "batch_size_latent_recon" not in captured


class TestGetMeanErrorsSurvivesADegenerateModel:
    """
    ``reconstruct_mesh`` raises ``NoZeroLevelSetError`` when the mean shape has no surface.
    ``get_mean_errors`` scores that subject NaN, so a training run survives its early
    validation epochs.
    """

    def test_nan_scores_and_no_crash(self, monkeypatch):
        """
        Fails if ``get_mean_errors`` lets ``NoZeroLevelSetError`` escape, or scores chamfer,
        ASSD or ``val_prediction_*`` other than NaN (KNOWN_ISSUES History 10).

        ``reconstruct_mesh`` is stubbed to raise for every subject.
        """

        def degenerate(path=None, **kwargs):
            raise recon_main.NoZeroLevelSetError("no zero level set (stub)")

        monkeypatch.setattr(recon_main, "reconstruct_mesh", degenerate)
        results = recon_main.get_mean_errors(
            mesh_paths=[f"subj{i}_age_{a}-mesh.vtk" for i, a in enumerate((45, 60))],
            decoders=None,
            latent_size=4,
            calc_symmetric_chamfer=True,
            calc_assd=True,
            predict_val_variables=["age"],
        )
        assert np.isnan(results["chamfer_0"]) and np.isnan(results["assd_0"])
        assert np.isnan(results["val_prediction_age"])


def test_predictive_validation_regresses_the_fitted_latents(monkeypatch):
    """
    Fails if ``get_mean_errors`` hands ``Regress.add_latent`` anything but the flattened
    fitted latent, such as the whole result dict (#48).

    The stub returns a ``(1, L)`` grad-tracking latent encoding the age, so r2 == 1 means the
    vector arrived intact. The first half checks ``Regress`` alone, so a failure there is not
    ``get_mean_errors``'s.
    """
    ages = (45, 60, 50, 55)
    paths = [f"subj{i}_age_{a}-mesh.vtk" for i, a in enumerate(ages)]

    regress = Regress(list_factors=["age"], list_paths=paths[:3])
    rng = np.random.default_rng(0)
    for age in ages[:3]:
        regress.add_latent(np.r_[age / 100.0, rng.normal(size=7)])
    assert regress.calc_r2()["val_prediction_age"] == pytest.approx(1.0)

    def fitted(path=None, **kwargs):
        age = float(path.split("age_")[1].split("-")[0])
        return {"mesh": [], "latent": torch.full((1, 4), age / 100.0, requires_grad=True)}

    monkeypatch.setattr(recon_main, "reconstruct_mesh", fitted)
    results = recon_main.get_mean_errors(
        mesh_paths=paths, decoders=None, latent_size=4, predict_val_variables=["age"]
    )
    assert results["val_prediction_age"] == pytest.approx(1.0)


class TestPrepareResultsForWandb:
    """``wandb.Object3D`` is built locally, so the filtering is checked offline."""

    class FakeMesh:
        """The two attributes the wandb path reads."""

        def __init__(self, n_points=10):
            self.point_coords = np.random.default_rng(0).normal(size=(n_points, 3))
            self.faces = None

    def test_only_small_serializable_values_are_kept(self):
        """
        Fails if ``prepare_results_for_wandb`` keeps arrays or tensors over 10 elements or
        unserializable objects, stops converting numpy scalars and small arrays or tensors,
        or mutates its input.
        """
        original = {
            "kept_int": 1,
            "kept_none": None,
            "kept_tuple": (1, 2),
            "np_scalar": np.float64(2.5),
            "small_array": np.arange(5),
            "large_array": np.zeros(100),
            "small_tensor": torch.ones(3),
            "large_tensor": torch.zeros(50),
            "unserializable": object(),
        }
        result = prepare_results_for_wandb(original)
        assert result == {
            "kept_int": 1,
            "kept_none": None,
            "kept_tuple": (1, 2),
            "np_scalar": 2.5,
            "small_array": [0, 1, 2, 3, 4],
            "small_tensor": [1.0, 1.0, 1.0],
        }
        assert isinstance(result["np_scalar"], float)
        assert len(original) == 9 and isinstance(original["large_array"], np.ndarray)

    def test_meshes_become_object3d_entries(self):
        """
        Fails if ``prepare_results_for_wandb`` stops turning ``mesh`` and ``orig_mesh`` into
        ``recon_mesh_i`` and ``orig_mesh_i`` ``Object3D`` entries or keeps the lists, or
        ``_process_meshes_for_wandb`` fails on a ``None`` mesh.

        ``_n_points`` is the mesh's own count, 10, not the subsample's 4. The subsample
        itself is not checked.
        """
        result = prepare_results_for_wandb(
            {"mesh": [self.FakeMesh()], "orig_mesh": [self.FakeMesh()]}
        )
        assert isinstance(result["recon_mesh_0"], wandb.Object3D)
        assert isinstance(result["orig_mesh_0"], wandb.Object3D)
        assert "mesh" not in result and "orig_mesh" not in result

        assert _process_meshes_for_wandb([None], "recon_mesh", 10_000, True) == {}
        subsampled = _process_meshes_for_wandb([self.FakeMesh(10)], "recon_mesh", 4, True)
        assert subsampled["recon_mesh_0_n_points"] == 10
