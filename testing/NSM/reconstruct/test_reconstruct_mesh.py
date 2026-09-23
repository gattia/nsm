"""
``reconstruct_mesh`` and the batch driver ``get_mean_errors``: what they accept, the stages
they run, what they report, and what they hand their collaborators.

The end-to-end runs use the single-object sampled branch: one sphere path with
``get_rand_pts=True``.
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
    """
    ``**kwargs`` used to swallow every key except one, so a misspelling ran at the default
    with no signal at all. With 58 near-synonymous parameters, that is the likeliest error.
    """

    def test_a_misspelled_or_deleted_keyword_raises_by_name(self, sphere_path):
        """
        ``verbose`` was deleted at v0.4.0. ``log_wandb_step`` is named by
        ``reconstruct_latent`` and was never forwarded by ``reconstruct_mesh``.
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
        Every keyword ``kneepipeline/steps/run_nsm.py`` passes. The two that are not named
        parameters are the deprecated one, which works, and ``verbose``, which the consumer
        had to drop before pulling v0.4.0.
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
    The mean-mesh build tested ``is True`` and its user tested truthiness, so ``1`` skipped
    the build and then asked the sampler to register to it.
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
    Only registration reads the mean mesh. ``scale_jointly`` used to build one too: 876,269
    extra decoder evaluations at the default grid, for a mesh nothing read, and it aborted
    the run when that mesh had no surface.
    """

    def test_scale_jointly_alone_builds_no_mean_mesh(self, sphere_path):
        counts = []
        for scale_jointly in (False, True):
            decoder = SphereDecoder()
            run(sphere_path, decoder, scale_jointly=scale_jointly, register_similarity=False)
            counts.append(decoder.n_points_evaluated)
        assert counts[0] == counts[1]

    def test_building_a_mesh_consumes_no_randomness(self):
        """Why dropping the unused build moved no result: it drew from neither generator."""
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
    Two fixed-seed runs agree on every value. Every stage the body times is returned: the
    stage names are read from the source, so a new stage is checked too.
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
    """Ten records answered to the ``verbose`` flag after logging became the mechanism."""
    with caplog.at_level(logging.DEBUG, logger="NSM"):
        run(sphere_path)
    assert "Loaded mesh in" in caplog.text and "Created mesh in" in caplog.text


class TestASubjectMissingASurface:
    """
    ``SCOPE`` §2.5b: fitting from a subset of surfaces is supported. ``compute_recon_loss``
    guarded the reconstructed mesh against ``None`` and read the original unguarded, so the
    metrics the shipped config asks for crashed on it.
    """

    def test_the_fit_decodes_every_surface_and_scores_the_missing_one_nan(self, sphere_path):
        kwargs = dict(decoder=SphereDecoder(objects=2), objects_per_decoder=2)
        result = run([sphere_path, None], return_latent=True, **kwargs)
        assert result["orig_mesh"][1] is None and result["latent"] is not None
        assert len(result["mesh"]) == 2 and all(mesh is not None for mesh in result["mesh"])

        for flag, key in (("calc_symmetric_chamfer", "chamfer"), ("calc_assd", "assd")):
            result = run([sphere_path, None], **{flag: True}, **kwargs)
            assert not np.isnan(result[f"{key}_0"]) and np.isnan(result[f"{key}_1"])


class TestTheKnobsThatDifferByLayer:
    """
    #56: ``chamfer_norm`` and ``sigma_rand_pts`` had a different default at each layer.
    ``chamfer_norm`` is a power, so 1 and 2 reported chamfer in different units. The
    resolved values are the ones every shipped run already used: no config carries
    ``chamfer_norm`` and the trainer passes it commented out (``test_default_config_sync``).
    """

    def test_each_knob_has_one_default_and_it_is_the_shipped_value(self):
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
    """
    #16: ``n_pts_random`` reached the readers under a name neither has, and their
    ``**kwargs`` swallowed it, so 200,000 points were drawn whatever was asked.
    ``get_mean_errors`` forwarded the deprecated ``batch_size_latent_recon`` the same way.
    """

    def test_the_request_arrives_as_n_pts(self, monkeypatch):
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
    #29: ``reconstruct_mesh`` raises ``NoZeroLevelSetError`` when the mean shape has no
    surface, and ``get_mean_errors`` scores NaN so a training run survives its early
    validation epochs. It used to record a zero latent as if fitted (History §10).
    """

    def test_nan_scores_and_no_crash(self, monkeypatch):
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
    #48: ``get_mean_errors`` handed ``Regress.add_latent`` the whole result dict, so a run
    with ``predict_val_variables`` died in ``calc_r2`` after all its reconstructions. The
    stub returns a ``(1, L)`` grad-tracking latent encoding the age, so r2 == 1 proves the
    vector arrived intact.
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
        result = prepare_results_for_wandb(
            {"mesh": [self.FakeMesh()], "orig_mesh": [self.FakeMesh()]}
        )
        assert isinstance(result["recon_mesh_0"], wandb.Object3D)
        assert isinstance(result["orig_mesh_0"], wandb.Object3D)
        assert "mesh" not in result and "orig_mesh" not in result

        assert _process_meshes_for_wandb([None], "recon_mesh", 10_000, True) == {}
        subsampled = _process_meshes_for_wandb([self.FakeMesh(10)], "recon_mesh", 4, True)
        assert subsampled["recon_mesh_0_n_points"] == 10
