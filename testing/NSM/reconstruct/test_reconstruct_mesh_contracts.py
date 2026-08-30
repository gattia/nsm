"""
What ``reconstruct_mesh`` promises about the 58 parameters it takes, the stages it runs,
and what it tells a host about them.

Plan §8.0.J. Five contracts, one file, because they are one shape seen from five sides:
a 409-line function with a 58-parameter signature decides things in more than one place
and reports on itself in fewer places than it measures.

1. **Unknown keywords** are swallowed by ``**kwargs``. With 58 near-synonymous parameter
   names, a misspelling is the likeliest way to call this function wrongly, and it is the
   one way that produces no signal at all.
2. **``register_similarity`` is read twice**, once as ``is True`` and once truthily, so a
   truthy non-``True`` value takes half of each path.
3. **The reference mesh is built on a condition wider than the one that uses it**, so
   ``scale_jointly`` alone pays for a mean shape nothing consults -- and aborts the run if
   that mesh has no surface.
4. **Six stage timings are measured and five are returned.**
5. **Ten of fifteen log records are gated behind the deprecated ``verbose`` flag**, so a
   host that configured logging is not the audience for them.

6. **A subject missing a surface** fits and decodes, but every metric flag on it raises
   -- `SCOPE` §2.5b's supported half, which had no end-to-end test until plan §8.0.N′.

Plus the end-to-end pin the commit-8 extraction is measured against: the full result dict
of a fixed-seed run, which must not move when the body is split into stage helpers.

Strict xfails mark what NSM does not honour yet. Each is retired by the commit that fixes
it.
"""

import ast
import inspect
import json
import logging
import re
from pathlib import Path

import numpy as np
import pytest
import torch

import NSM
import NSM.reconstruct.main as recon_main
from NSM.mesh import create_mesh_adaptive
from NSM.reconstruct.recon_evaluation import compute_recon_loss, get_mean_errors

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


class SphereDecoder(torch.nn.Module):
    """
    An analytic sphere SDF (radius 0.5), one output column, differentiable in the latent.

    The twin of ``test_reconstruct_mesh_options.SphereDecoder``, kept separate because
    this one counts. ``+ 0.0 * latent.sum()`` keeps the output on the latent's graph so
    ``reconstruct_latent``'s backward has a leaf to reach.
    """

    def __init__(self, objects=1):
        super().__init__()
        self.objects = objects
        self.n_points_evaluated = 0

    def forward(self, x=None, latent=None, xyz=None, epoch=None, verbose=False):
        pts = xyz if xyz is not None else x[:, -3:]
        self.n_points_evaluated += pts.shape[0]
        sdf = torch.norm(pts, dim=1, keepdim=True) - 0.5
        if latent is not None:
            sdf = sdf + 0.0 * latent.sum()
        return sdf.repeat(1, self.objects)


class NoZeroLevelSetDecoder(torch.nn.Module):
    """SDF is +1 everywhere, so the mean shape has no surface. See the regression twin."""

    def forward(self, x=None, latent=None, xyz=None, epoch=None, verbose=False):
        n_points = xyz.shape[0] if xyz is not None else x.shape[0]
        return torch.ones(n_points, 1)


@pytest.fixture(scope="module")
def sphere_path(tmp_path_factory):
    pv = pytest.importorskip("pyvista")
    sphere = pv.Sphere(radius=0.5, theta_resolution=18, phi_resolution=18).triangulate()
    path = str(tmp_path_factory.mktemp("recon") / "sphere.vtk")
    sphere.save(path)
    return path


def sampled_run_kwargs(**overrides):
    """The cheap single-object sampled branch, the only end-to-end path in this file."""
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
    return kwargs


# ---------------------------------------------------------------------------
# 1. Unknown keywords
# ---------------------------------------------------------------------------

#: Misspellings of five real parameters, of the kinds that actually happen: a pluralised
#: axis (``n_pts_per_axis``), a dropped plural (``num_iterations``), a trailing typo
#: (``calc_assd``), a transposition (``latent_reg_weight``) and a synonym (``clamp_dist``).
MISSPELLINGS = [
    "n_pts_per_axes",
    "num_iteration",
    "calc_assd_",
    "latent_reg_wieght",
    "clamp_distance",
]


class TestUnknownKeywordsAreRefused:
    """
    ``**kwargs`` used to be inspected for exactly one key, ``batch_size_latent_recon``.
    Every other key reached the end of the function unread, so the caller got the default
    for the parameter they meant to set and no indication that they had not set it.
    """

    @pytest.mark.parametrize("wrong", MISSPELLINGS)
    def test_a_misspelled_parameter_raises(self, sphere_path, wrong):
        """
        Were five strict xfails. Measured before the fix: all five complete the run with no exception, no warning
        and no log record naming the key -- the run used the intended parameter's default
        while the caller believed they had set it. That measurement is why the fix is a
        refusal and not a warning: there was nowhere for a warning to be noticed.
        """
        with pytest.raises(TypeError, match=wrong):
            recon_main.reconstruct_mesh(
                path=sphere_path,
                decoders=SphereDecoder(),
                **sampled_run_kwargs(**{wrong: 999}),
            )

    def test_the_deprecated_key_is_still_accepted(self, sphere_path, caplog):
        """
        ``batch_size_latent_recon`` is the one key ``**kwargs`` is *for*: kneepipeline's
        ``steps/run_nsm.py`` passes it on every fit. Refusing unknown keys must not refuse
        this one -- it warns and runs, as ``test_reconstruct_mesh_options`` also pins.
        """
        with caplog.at_level(logging.WARNING, logger="NSM"):
            recon_main.reconstruct_mesh(
                path=sphere_path,
                decoders=SphereDecoder(),
                **sampled_run_kwargs(batch_size_latent_recon=1),
            )
        assert any(
            "batch_size_latent_recon is deprecated" in r.getMessage() for r in caplog.records
        )

    def test_the_consumers_own_keyword_set_is_accepted(self, sphere_path):
        """
        Every keyword ``kneepipeline/steps/run_nsm.py:185`` passes, together, against a
        signature that is about to start refusing unknown ones. This is the list that must
        never become a ``TypeError``; it is asserted as a set against the signature rather
        than by running a fit, because several of its values need a real model.
        """
        import inspect

        consumer_keywords = {
            "path",
            "decoders",
            "latent_size",
            "num_iterations",
            "l2reg",
            "latent_reg_weight",
            "loss_type",
            "lr",
            "lr_update_factor",
            "n_lr_updates",
            "return_latent",
            "register_similarity",
            "scale_jointly",
            "scale_all_meshes",
            "objects_per_decoder",
            "batch_size_latent_recon",
            "get_rand_pts",
            "n_pts_random",
            "sigma_rand_pts",
            "n_samples_latent_recon",
            "calc_assd",
            "convergence",
            "convergence_patience",
            "clamp_dist",
            "fix_mesh",
            "verbose",
            "return_registration_params",
        }
        named = set(inspect.signature(recon_main.reconstruct_mesh).parameters)
        unknown = consumer_keywords - named
        assert unknown == {"batch_size_latent_recon"}, (
            "the consumer's only unnamed keyword is the deprecated one; anything else "
            f"here would break on a refusal: {sorted(unknown)}"
        )


# ---------------------------------------------------------------------------
# 2. register_similarity, read twice
# ---------------------------------------------------------------------------


class TestRegisterSimilarityIsDecidedOnce:
    """
    The gate that built the mean mesh tested ``register_similarity is True``; the flag
    that *uses* it (``register_to_mean_first``) was truthy. A truthy non-``True`` value
    therefore skipped the build and then asked the sampler to register to it.
    """

    @pytest.mark.parametrize("flag", [True, 1, "similarity"], ids=["True", "one", "similarity"])
    def test_a_truthy_value_takes_one_path(self, sphere_path, flag):
        """
        Were two strict xfails. Measured before the fix: ``1`` and ``"similarity"`` raise a bare
        ``Exception("Must provide mean mesh to register to")`` from
        ``datasets/mesh_sampling.py:149`` -- a file the caller never named, about a
        parameter ``reconstruct_mesh`` does not have.
        """
        result = recon_main.reconstruct_mesh(
            path=sphere_path,
            decoders=SphereDecoder(),
            register_similarity=flag,
            n_pts_per_axis_mean_mesh=24,
            return_registration_params=True,
            **sampled_run_kwargs(),
        )
        assert result["icp_transform"] is not None, "a registered run records its transform"


# ---------------------------------------------------------------------------
# 3. The reference mesh, built wider than it is used
# ---------------------------------------------------------------------------


class TestTheReferenceMeshIsBuiltWhenItIsUsed:
    """
    ``mean_mesh`` has exactly one consumer: the samplers' ``register_to_mean_first``,
    which is set from ``register_similarity`` alone. Building it under ``scale_jointly``
    bought nothing and cost a whole marching-cubes reconstruction.
    """

    def test_scale_jointly_alone_does_not_build_one(self, sphere_path):
        """
        Was a strict xfail. Measured before the fix at the ``n_pts_per_axis_mean_mesh=128`` default:
        524,968 decoder point-evaluations without ``scale_jointly`` and 1,401,237 with it,
        an extra 876,269 for a mesh that is discarded three lines later.

        Removing the build is numerically inert, checked across the commit rather than
        argued: a fixed-seed fit under ``register_similarity=True``, under
        ``scale_jointly=True`` and under neither produced byte-identical latents, vertex
        arrays, chamfer and ASSD before and after. The premise that makes that possible is
        ``test_building_a_reference_mesh_consumes_no_randomness`` below -- had
        ``create_mesh_adaptive`` drawn from either global generator, dropping a call to it
        would have shifted every subsequent sample.
        """
        counts = {}
        for scale_jointly in (False, True):
            decoder = SphereDecoder()
            recon_main.reconstruct_mesh(
                path=sphere_path,
                decoders=decoder,
                scale_jointly=scale_jointly,
                register_similarity=False,
                **sampled_run_kwargs(),
            )
            counts[scale_jointly] = decoder.n_points_evaluated
        assert counts[True] == counts[False], (
            f"scale_jointly cost {counts[True] - counts[False]} extra point-evaluations "
            "for a mean mesh nothing reads"
        )

    def test_scale_jointly_alone_does_not_abort_on_a_surfaceless_mean(self, sphere_path):
        """
        Was a strict xfail, and the sharper half. An under-trained model plus ``scale_jointly=True`` aborts the
        whole reconstruction over a mean mesh that ``register_to_mean_first=False`` was
        never going to consult.
        """
        with pytest.raises(RuntimeError) as excinfo:
            recon_main.reconstruct_mesh(
                path=sphere_path,
                decoders=NoZeroLevelSetDecoder(),
                scale_jointly=True,
                register_similarity=False,
                n_pts_per_axis_mean_mesh=16,
                **sampled_run_kwargs(),
            )
        assert not isinstance(excinfo.value, recon_main.NoZeroLevelSetError)

    def test_a_registered_run_still_refuses_a_surfaceless_mean(self, sphere_path):
        """The path that *does* use the mean mesh keeps #29's refusal."""
        with pytest.raises(recon_main.NoZeroLevelSetError, match="no zero level set"):
            recon_main.reconstruct_mesh(
                path=sphere_path,
                decoders=NoZeroLevelSetDecoder(),
                register_similarity=True,
                n_pts_per_axis_mean_mesh=16,
                **sampled_run_kwargs(),
            )

    def test_building_a_reference_mesh_consumes_no_randomness(self):
        """
        The premise that makes removing the discarded build numerically inert: if
        ``create_mesh_adaptive`` drew from either global generator, dropping a call to it
        would shift every subsequent sample and move the fitted latent. It does not.
        """
        torch.manual_seed(0)
        np.random.seed(0)
        torch_before = torch.get_rng_state().clone()
        numpy_before = np.random.get_state()
        create_mesh_adaptive(
            decoder=SphereDecoder(),
            latent_vector=torch.zeros(1, 8),
            n_pts_per_axis=32,
            search_bounds=(-1.0, 1.0),
            objects=1,
            batch_size=32**3,
            verbose=False,
            device="cpu",
        )
        numpy_after = np.random.get_state()
        assert torch.equal(torch_before, torch.get_rng_state())
        assert numpy_before[1].tolist() == numpy_after[1].tolist()
        assert numpy_before[2] == numpy_after[2]


# ---------------------------------------------------------------------------
# 4. return_timing, against the timings the body measures
# ---------------------------------------------------------------------------


def _timing_names_measured_in_the_source():
    """
    Every stage the body times, read from the source rather than listed here, so a stage
    added later without a ``return_timing`` key turns this red instead of going unnoticed
    the way ``time_calc_recon_loss`` did.

    Two spellings, because the fix changed the one that is current and this test has to
    keep meaning something on either side of it: ``time_x = toc - tic`` was how a stage
    recorded itself before ``_StageTimings``, and ``timings.stage("x", ...)`` is how it
    does now. A scan that matched only the retired spelling would pass by finding nothing.
    """
    source = open(recon_main.__file__, encoding="utf-8").read()
    assigned = set(re.findall(r"\b(time_\w+)\s*=\s*toc", source))
    staged = {f"time_{name}" for name in re.findall(r'timings\.stage\(\s*"(\w+)"', source)}
    measured = assigned | staged
    assert measured, "neither spelling matched -- this scan has stopped measuring anything"
    return measured


class TestReturnTimingCoversEveryStage:
    def test_every_measured_stage_is_returned(self, sphere_path):
        """Was a strict xfail: time_calc_recon_loss was measured and dropped."""
        result = recon_main.reconstruct_mesh(
            path=sphere_path,
            decoders=SphereDecoder(),
            return_timing=True,
            calc_symmetric_chamfer=True,
            **sampled_run_kwargs(),
        )
        returned = {key for key in result if key.startswith("time_")}
        assert _timing_names_measured_in_the_source() - returned == set()

    def test_the_returned_timings_are_plausible_seconds(self, sphere_path):
        """A guard on the recorder that replaces the interleaved ``tic``/``toc`` pairs."""
        result = recon_main.reconstruct_mesh(
            path=sphere_path,
            decoders=SphereDecoder(),
            return_timing=True,
            **sampled_run_kwargs(),
        )
        timings = {key: value for key, value in result.items() if key.startswith("time_")}
        assert timings, "return_timing produced no timings"
        assert all(0.0 <= value < 600.0 for value in timings.values()), timings


# ---------------------------------------------------------------------------
# 5. Log records and the deprecated flag
# ---------------------------------------------------------------------------


def _verbose_gated_log_calls():
    """``logger.*`` calls under an ``if verbose ...:`` inside ``reconstruct_mesh``."""
    source = open(recon_main.__file__, encoding="utf-8").read()
    function = next(
        node
        for node in ast.walk(ast.parse(source))
        if isinstance(node, ast.FunctionDef) and node.name == "reconstruct_mesh"
    )
    gated = []
    for node in ast.walk(function):
        if not isinstance(node, ast.If) or "verbose" not in ast.dump(node.test):
            continue
        for inner in ast.walk(node):
            if (
                isinstance(inner, ast.Call)
                and isinstance(inner.func, ast.Attribute)
                and isinstance(inner.func.value, ast.Name)
                and inner.func.value.id == "logger"
            ):
                gated.append(inner.func.attr)
    return gated


class TestLogRecordsReachAConfiguredHost:
    """
    §8.0.G made logging the mechanism; ten records went on answering to the parameter it
    deprecated. A host that ran ``logging.getLogger("NSM").setLevel(DEBUG)`` -- the exact
    replacement the deprecation warning names -- saw none of them.
    """

    def test_no_log_record_is_gated_on_the_deprecated_flag(self):
        """Was a strict xfail: ten of fifteen, one of them the skipped-surface warning."""
        assert _verbose_gated_log_calls() == []

    def test_a_host_at_debug_sees_the_stage_records(self, sphere_path, caplog):
        """Was a strict xfail: empty, for a host that did exactly what the notice said."""
        with caplog.at_level(logging.DEBUG, logger="NSM"):
            recon_main.reconstruct_mesh(
                path=sphere_path,
                decoders=SphereDecoder(),
                **sampled_run_kwargs(),
            )
        messages = " ".join(record.getMessage() for record in caplog.records)
        assert "Loaded mesh in" in messages
        assert "Created mesh in" in messages

    def test_verbose_true_shows_them_today_and_must_keep_doing_so(self, sphere_path, caplog):
        """
        The bridge attaches at ``DEBUG`` (``_verbose_deprecation.py:82``), so ungating
        cannot take anything away from a ``verbose=True`` caller. ``caplog`` stands in for
        the bridge's handler -- it is a handler on the root, so the bridge declines to add
        its own and the records land here either way.
        """
        with caplog.at_level(logging.DEBUG, logger="NSM"):
            with pytest.warns(DeprecationWarning):
                recon_main.reconstruct_mesh(
                    path=sphere_path,
                    decoders=SphereDecoder(),
                    verbose=True,
                    **sampled_run_kwargs(),
                )
        messages = " ".join(record.getMessage() for record in caplog.records)
        assert "Loaded mesh in" in messages
        assert "Created mesh in" in messages


# ---------------------------------------------------------------------------
# The end-to-end pin the extraction is measured against
# ---------------------------------------------------------------------------


class TestTheWholeResultIsUnchangedByTheSplit:
    """
    Commit 8 splits the body into five stage helpers. This is what makes that provably a
    refactor: the entire result dict of a fixed-seed run -- keys, latent, mesh vertices,
    metrics and registration parameters -- recorded before it and asserted after.

    Not golden numbers in the file: the run is executed twice in the same process and the
    two are compared. That pins *determinism* here and, across the split, the values
    themselves, because the commit that splits is the only thing between two runs of it.
    """

    @staticmethod
    def _run(sphere_path):
        torch.manual_seed(11)
        np.random.seed(11)
        return recon_main.reconstruct_mesh(
            path=sphere_path,
            decoders=SphereDecoder(),
            register_similarity=True,
            n_pts_per_axis_mean_mesh=24,
            calc_symmetric_chamfer=True,
            calc_assd=True,
            return_latent=True,
            return_registration_params=True,
            return_timing=True,
            **sampled_run_kwargs(),
        )

    def test_the_run_is_deterministic_and_complete(self, sphere_path):
        first = self._run(sphere_path)
        second = self._run(sphere_path)

        assert set(first) == set(second)
        assert {"mesh", "orig_mesh", "latent", "icp_transform", "center", "scale"} <= set(first)
        assert any(key.startswith("chamfer") for key in first)
        assert any(key.startswith("assd") for key in first)

        assert torch.equal(first["latent"], second["latent"])
        np.testing.assert_array_equal(
            np.asarray(first["mesh"][0].point_coords),
            np.asarray(second["mesh"][0].point_coords),
        )
        for key, value in first.items():
            if isinstance(value, float) and not key.startswith("time_"):
                assert value == pytest.approx(second[key], rel=0, abs=0), key


# ---------------------------------------------------------------------------
# 6. A subject that is missing a surface (SCOPE §2.5b, issue #67)
# ---------------------------------------------------------------------------


class TestASubjectMissingASurface:
    """
    ``SCOPE`` §2.5b, ruled 2026-08-29: *fitting a latent from a subset of surfaces* is
    supported, *building a dataset from subjects with holes* is not (#67). The supported
    half had no end-to-end test — it was ruled from ``latent_fit.py``, one frame below
    the frame these tests exercise — and plan §8.0.N′ owes it one.

    The fit is indeed fine. What the missing test was hiding is one line up:
    ``compute_recon_loss`` guards the *reconstructed* mesh against ``None`` and reads the
    *original* unguarded, so the capability is unreachable through ``get_mean_errors``,
    which passes ``calc_symmetric_chamfer=config["chamfer"]`` and
    ``calc_assd=config["assd"]`` -- both ``true`` in the shipped config.
    """

    @staticmethod
    def _run(sphere_path, **overrides):
        return recon_main.reconstruct_mesh(
            path=[sphere_path, None],
            decoders=SphereDecoder(objects=2),
            objects_per_decoder=2,
            **sampled_run_kwargs(**overrides),
        )

    def test_the_latent_fits_and_both_surfaces_decode(self, sphere_path):
        """The supported half, asserted rather than inferred: the surface that has no
        original still comes back decoded, because the decoder produces all of them."""
        result = self._run(sphere_path, return_latent=True)
        assert result["orig_mesh"][1] is None
        assert result["latent"] is not None
        assert len(result["mesh"]) == 2
        assert all(mesh is not None for mesh in result["mesh"])

    @pytest.mark.parametrize("flag", ["calc_symmetric_chamfer", "calc_assd"])
    def test_the_metric_the_shipped_config_asks_for_still_runs(self, sphere_path, flag):
        """Before §8.0.N′: ``AttributeError: 'NoneType' object has no attribute
        'point_coords'``, which is what made this capability unreachable in production."""
        result = self._run(sphere_path, **{flag: True})
        suffix = "chamfer" if flag == "calc_symmetric_chamfer" else "assd"
        assert not np.isnan(result[f"{suffix}_0"])
        assert np.isnan(result[f"{suffix}_1"])


class TestTheKnobsThatDifferByLayer:
    """
    #56, parts 1 and 2 — one knob name, a different default at each layer it passes.

    Read off the signatures rather than restated, so the assertion tracks the code:
    ``chamfer_norm`` is a **power**, so 1 and 2 report chamfer in different units, and
    ``sigma_rand_pts`` differs 10×.

    **Resolving them changes no result, and that is measurable rather than argued.**
    The two facts that make it true are pinned next to the shipped config, in
    ``testing/NSM/configs/test_default_config_sync.py``: no config carries a
    ``chamfer_norm`` key, and the trainer passes the argument commented out. So every NSM
    path already runs on ``get_mean_errors``' 2, and ``compute_recon_loss``'s 1 has never
    produced a number. The disposition picks the value already in use at both sites.
    """

    def _default(self, func, name):
        return inspect.signature(func).parameters[name].default

    @pytest.mark.xfail(strict=True, reason="#56 part 1 — resolved in this slice's commit 6")
    def test_chamfer_norm_has_one_default(self):
        defaults = {
            "reconstruct_mesh": self._default(recon_main.reconstruct_mesh, "chamfer_norm"),
            "get_mean_errors": self._default(get_mean_errors, "chamfer_norm"),
            "compute_recon_loss": self._default(compute_recon_loss, "chamfer_norm"),
        }
        assert set(defaults.values()) == {2}, defaults

    @pytest.mark.xfail(strict=True, reason="#56 part 2 — resolved in this slice's commit 6")
    def test_sigma_rand_pts_has_one_default(self):
        defaults = {
            "reconstruct_mesh": self._default(recon_main.reconstruct_mesh, "sigma_rand_pts"),
            "get_mean_errors": self._default(get_mean_errors, "sigma_rand_pts"),
        }
        assert set(defaults.values()) == {0.01}, defaults

    def test_the_value_chosen_is_the_one_the_shipped_config_already_uses(self):
        """
        Both dispositions pick the ShapeMedKnee value, so no shipped run moves. For
        ``sigma_rand_pts`` the config states it outright; ``chamfer_norm`` is the
        ``get_mean_errors`` default the trainer falls through to.
        """
        config = json.loads(
            (Path(NSM.__file__).parent / "configs" / "default_config.json").read_text(
                encoding="utf-8"
            )
        )
        assert config["sigma_rand_pts_recon"] == 0.01
        assert self._default(get_mean_errors, "chamfer_norm") == 2
