"""
End-to-end reconstruction regression through ``reconstruct_mesh``.

``reconstruct_mesh`` is what the downstream consumer actually calls, and before this
module it had one executed line in the whole suite: its ``def``. Everything here goes
through it the way ``kneepipeline/steps/run_nsm.py:170`` does -- a *list* of mesh paths,
every argument by name -- and asserts the eight result keys that consumer reads:
``mesh[0]``, ``mesh[1]``, ``latent``, ``icp_transform``, ``center``, ``scale``,
``assd_0``, ``assd_1``.

The load-bearing, undocumented part of that contract is the ORDER of the ``mesh`` list.
``TestSurfaceOrderContract`` asserts it by geometry rather than by index, so it fails if
the surfaces are ever transposed.
"""

import numpy as np
import pytest
import torch
import vtk
from _harness import (
    COUNT_RTOL,
    FITTED_LATENT_ATOL,
    GEOMETRY_ATOL,
    METRIC_RTOL,
    MIN_HEADROOM,
    RECON_DECODER_ASSET,
    headroom,
    mesh_summary,
    provenance,
    regenerating,
    run_reconstruction,
    train_reconstruction_decoder,
)

from NSM.reconstruct import NoZeroLevelSetError

#: Which synthetic surface each result index is supposed to be. The bone sphere is
#: centred on the origin; the cartilage ellipsoid sits above it. See _harness.SUBJECTS.
BONE, CART = 0, 1

#: The deliberate break: displace the ``PERTURBED_VERTICES`` vertices nearest vertex 0 by
#: ``PERTURBATION`` mesh units. The bone sphere has radius 1.0 and 530 vertices, so ONE
#: vertex moved a quarter of the radius is the smallest geometry change this fixture can
#: express -- and the harness catches it with 34.8x ``FITTED_LATENT_ATOL`` to spare.
#:
#: Keep it at 1. A wider dent is easier to detect and therefore proves less: the measured
#: headroom rises to 69x at 5 vertices and 119x at 10, so raising this number can only
#: make a failing break pass, which is the same mistake as loosening a tolerance wearing a
#: different hat. If this ever drops under ``MIN_HEADROOM``, the fixture or the tolerance
#: is what changed, and that is what wants investigating.
PERTURBATION = 0.25
PERTURBED_VERTICES = 1


def summaries(result):
    return {"bone": mesh_summary(result["mesh"][BONE]), "cart": mesh_summary(result["mesh"][CART])}


class TestConsumerContract:
    """Every key ``steps/run_nsm.py`` reads off the result must be present and usable."""

    def test_result_is_a_dict_with_the_expected_keys(self, reconstruction):
        assert set(reconstruction) >= {
            "mesh",
            "latent",
            "icp_transform",
            "center",
            "scale",
            "assd_0",
            "assd_1",
        }

    def test_mesh_is_a_list_of_one_surface_per_object(self, reconstruction):
        assert isinstance(reconstruction["mesh"], list)
        assert len(reconstruction["mesh"]) == 2
        assert all(m is not None for m in reconstruction["mesh"])

    def test_latent_has_the_configured_shape(self, reconstruction):
        from _harness import LATENT_SIZE

        assert reconstruction["latent"].shape == (1, LATENT_SIZE)

    def test_registration_params_are_the_types_the_consumer_converts(self, reconstruction):
        """
        ``_convert_icp_transform`` accepts vtkTransform, vtkMatrix4x4, ndarray or None. Which
        one arrives here is part of the contract, and nothing in the signature says so.
        """
        assert isinstance(
            reconstruction["icp_transform"],
            (vtk.vtkIterativeClosestPointTransform, vtk.vtkTransform, vtk.vtkMatrix4x4),
        )
        assert np.asarray(reconstruction["center"]).shape == (3,)
        assert np.isscalar(reconstruction["scale"]) or np.asarray(reconstruction["scale"]).ndim == 0

    def test_meshes_expose_point_coords(self, reconstruction):
        for mesh in reconstruction["mesh"]:
            points = np.asarray(mesh.point_coords)
            assert points.ndim == 2 and points.shape[1] == 3 and points.shape[0] > 0


class TestSurfaceOrderContract:
    """
    ``result["mesh"]`` is ordered and the order IS the surface identity. Index 0 = bone,
    index 1 = cartilage is hardcoded at ``steps/run_nsm.py:216,220,232,235`` and stated
    nowhere in the signature, the docstring, or the returned dict -- the same
    undocumented-positional shape as the learning-rate bug.

    The synthetic subjects are built so the two surfaces are separated along z, so these
    assertions are geometric and would fail if the list were ever transposed.
    """

    @staticmethod
    def _centroid_distances(reconstruction, synthetic_meshes):
        """``[i][j]`` = distance from reconstructed surface *i* to input surface *j*."""
        import pyvista as pv

        inputs = [pv.read(path).points.mean(axis=0) for path in synthetic_meshes[0]]
        outputs = [np.asarray(m.point_coords).mean(axis=0) for m in reconstruction["mesh"]]
        return np.array([[float(np.linalg.norm(out - inp)) for inp in inputs] for out in outputs])

    def test_each_result_index_matches_its_input_index(self, reconstruction, synthetic_meshes):
        """
        Scale-free form of the contract: surface *i* of the result must be nearer to
        surface *i* of ``path`` than to the other one. Transposing the list fails this.
        """
        distances = self._centroid_distances(reconstruction, synthetic_meshes)
        assert distances[BONE][BONE] < distances[BONE][CART], distances
        assert distances[CART][CART] < distances[CART][BONE], distances

    def test_the_two_surfaces_are_not_interchangeable(self, reconstruction, synthetic_meshes):
        """The guard: if the inputs were indistinguishable the test above would be empty."""
        distances = self._centroid_distances(reconstruction, synthetic_meshes)
        assert distances[BONE][CART] > 3 * distances[BONE][BONE], distances
        assert distances[CART][BONE] > 3 * distances[CART][CART], distances

    def test_assd_indices_follow_the_same_order(self, reconstruction, synthetic_meshes):
        """
        ``assd_0``/``assd_1`` carry the same positional convention -- the consumer labels
        them ``assd_bone_mm`` / ``assd_cartilage_mm`` on that basis. Each reported value
        must be far below what it would be if measured against the other surface.
        """
        crossed = [
            surface_distance(reconstruction["mesh"][BONE], synthetic_meshes[0][CART]),
            surface_distance(reconstruction["mesh"][CART], synthetic_meshes[0][BONE]),
        ]
        assert reconstruction["assd_0"] * 3 < crossed[0], (reconstruction["assd_0"], crossed[0])
        assert reconstruction["assd_1"] * 3 < crossed[1], (reconstruction["assd_1"], crossed[1])


def surface_distance(reconstructed, original_path):
    """Symmetric mean vertex-to-vertex distance. A proxy, only used for orderings."""
    import pyvista as pv
    from scipy.spatial import cKDTree

    original = np.asarray(pv.read(original_path).points, dtype=float)
    points = np.asarray(reconstructed.point_coords, dtype=float)
    return 0.5 * (
        float(cKDTree(original).query(points)[0].mean())
        + float(cKDTree(points).query(original)[0].mean())
    )


class TestNumericalBaselines:
    def test_fitted_latent_matches_baseline(self, reconstruction, reconstruction_baseline):
        latent = reconstruction["latent"].detach().cpu().numpy().ravel()
        reconstruction_baseline.check("fitted_latent", latent, atol=FITTED_LATENT_ATOL)

    def test_mesh_geometry_matches_baseline(self, reconstruction, reconstruction_baseline):
        reconstruction_baseline.check(
            "mesh_geometry", summaries(reconstruction), atol=GEOMETRY_ATOL
        )

    def test_mesh_point_counts_match_baseline(self, reconstruction, reconstruction_baseline):
        counts = [len(np.asarray(m.point_coords)) for m in reconstruction["mesh"]]
        reconstruction_baseline.check("mesh_point_counts", counts, rtol=COUNT_RTOL)

    def test_surface_metrics_match_baseline(self, reconstruction, reconstruction_baseline):
        reconstruction_baseline.check(
            "assd", [reconstruction["assd_0"], reconstruction["assd_1"]], rtol=METRIC_RTOL
        )

    def test_registration_params_match_baseline(self, reconstruction, reconstruction_baseline):
        reconstruction_baseline.check("scale", float(reconstruction["scale"]), rtol=METRIC_RTOL)
        reconstruction_baseline.check(
            "center", np.asarray(reconstruction["center"], dtype=float), atol=GEOMETRY_ATOL
        )

    def test_reconstruction_is_reproducible_within_a_process(
        self, synthetic_meshes, reconstruction_model, reconstruction
    ):
        """
        The baselines above only mean something if the same inputs give the same answer.
        Re-runs the whole reconstruction under the same seed and compares.
        """
        again = run_reconstruction(synthetic_meshes[0], reconstruction_model)
        assert np.allclose(
            again["latent"].detach().cpu().numpy(),
            reconstruction["latent"].detach().cpu().numpy(),
            atol=0,
            rtol=0,
        )
        assert again["assd_0"] == reconstruction["assd_0"]


class TestDeliberateBreak:
    """
    The second half of "a harness nobody has seen fail is not evidence of anything":
    dent the input bone mesh and confirm the baselines reject the result.

    Each rejection is paired with a ``MIN_HEADROOM`` assertion, so "the break is comfortably
    outside the tolerance" is measured on every run rather than transcribed once. The
    transcribed version was wrong by 4x when it was replaced, in the direction that made the
    break look weaker than it was -- which is the argument for computing it.

    If one of these fails, the break is not the thing to change. Making a deliberate break
    bigger until it is detected proves only that a bigger break is detectable; see
    ``PERTURBED_VERTICES``.
    """

    @pytest.fixture(scope="class")
    def perturbed_reconstruction(self, synthetic_meshes, reconstruction_model, tmp_path_factory):
        import pyvista as pv

        directory = tmp_path_factory.mktemp("perturbed")
        bone = pv.read(synthetic_meshes[0][BONE])
        points = bone.points.copy()
        patch = np.argsort(np.linalg.norm(points - points[0], axis=1))[:PERTURBED_VERTICES]
        points[patch] += np.array([PERTURBATION, 0.0, 0.0], dtype=points.dtype)
        bone.points = points
        bone_path = str(directory / "perturbed_bone.vtk")
        bone.save(bone_path)
        return run_reconstruction([bone_path, synthetic_meshes[0][CART]], reconstruction_model)

    def test_denting_the_bone_changes_the_fitted_latent(
        self, reconstruction, perturbed_reconstruction
    ):
        original = reconstruction["latent"].detach().cpu().numpy().ravel()
        perturbed = perturbed_reconstruction["latent"].detach().cpu().numpy().ravel()
        assert not np.allclose(original, perturbed, atol=FITTED_LATENT_ATOL), (
            f"moving {PERTURBED_VERTICES} bone vertices by {PERTURBATION} left the fitted "
            f"latent inside the harness's tolerance -- the latent baseline would not catch "
            f"a geometry change"
        )

    def test_denting_the_bone_fails_the_latent_baseline(
        self, perturbed_reconstruction, reconstruction_baseline
    ):
        if regenerating():
            pytest.skip("baselines are being rewritten")
        latent = perturbed_reconstruction["latent"].detach().cpu().numpy().ravel()
        with pytest.raises(AssertionError, match="differs from baseline"):
            reconstruction_baseline.check("fitted_latent", latent, atol=FITTED_LATENT_ATOL)

        measured = headroom(
            reconstruction_baseline, "fitted_latent", latent, atol=FITTED_LATENT_ATOL
        )
        assert measured >= MIN_HEADROOM, (
            f"the dent moves the fitted latent only {measured:.1f}x FITTED_LATENT_ATOL "
            f"({FITTED_LATENT_ATOL}), under the MIN_HEADROOM of {MIN_HEADROOM}x. Widen the "
            f"break -- more vertices, not a deeper dent -- never the tolerance."
        )

    def test_denting_the_bone_fails_the_geometry_baseline(
        self, perturbed_reconstruction, reconstruction_baseline
    ):
        if regenerating():
            pytest.skip("baselines are being rewritten")
        summary = summaries(perturbed_reconstruction)
        with pytest.raises(AssertionError, match="differs from baseline"):
            reconstruction_baseline.check("mesh_geometry", summary, atol=GEOMETRY_ATOL)

        measured = headroom(reconstruction_baseline, "mesh_geometry", summary, atol=GEOMETRY_ATOL)
        assert measured >= MIN_HEADROOM, (
            f"the dent moves the mesh geometry only {measured:.1f}x GEOMETRY_ATOL "
            f"({GEOMETRY_ATOL}), under the MIN_HEADROOM of {MIN_HEADROOM}x. Widen the "
            f"break -- more vertices, not a deeper dent -- never the tolerance."
        )


#: Turns ``reconstruct_mesh``'s point draw on. ``RECON_KWARGS`` keeps ``get_rand_pts=False``
#: -- every baselined number above is fitted to the mesh VERTICES -- and with it False the
#: samplers return early and ``reconstruct_mesh``'s ``seed`` argument reaches nothing at
#: all. These tests are the only ones in the suite where that argument does any work.
#:
#: ``n_pts_random`` is honoured since the #16 fix: 200 points per surface, plus the
#: surface vertices ``include_surf_in_pts`` appends. Before it the value was swallowed
#: by the readers' ``**kwargs`` and the 200,000-point default ran — the whole reason
#: each of these reconstructions cost ~4s and there were only five of them.
SAMPLED = dict(get_rand_pts=True, n_pts_random=200)

SAMPLE_SEED = 7


class TestSampledReconstructionIsSeeded:
    """
    ``reconstruct_mesh(seed=...)`` on the multi-object branch, which is the one the
    downstream consumer takes.

    Two things about this path are worth knowing before reading the assertions.

    **The seed has to be handed over under a different name.** ``run_reconstruction``
    declares its own ``seed`` -- the global torch/numpy one -- so it swallows the keyword
    and ``reconstruct_mesh`` keeps its default of ``None``. The harness spells the
    sampling seed ``sample_seed`` for that reason; passing ``seed=`` here would reseed
    torch, leave the draw unseeded, and these tests would still pass three times out of
    four while asserting nothing.

    **``n_pts_random`` reaches the sampler since the #16 fix** — 200 points per surface
    here, plus each surface's own vertices from ``include_surf_in_pts`` (its
    wrong-surface append was #17, fixed in the same era). Before the fix it landed in
    the readers' ``**kwargs`` and the 200,000-per-surface default ran — measured then
    as 400,688 points from a request for 200 — which is why this class was documented
    as expensive. The assertions below never depended on the draw size; only their
    cost did.

    The single-object branch is not covered here: it needs a one-output decoder, which
    this fixture's model is not. (Until #15 unified the sampler keys it was unreachable
    outright -- the sampler returned the draw under ``xyz`` while ``reconstruct_mesh``
    read ``result_["pts"]``.)
    """

    @pytest.fixture(scope="class")
    def seeded_pair(self, synthetic_meshes, reconstruction_model):
        return [
            run_reconstruction(
                synthetic_meshes[0], reconstruction_model, sample_seed=SAMPLE_SEED, **SAMPLED
            )
            for _ in range(2)
        ]

    @pytest.fixture(scope="class")
    def other_seed(self, synthetic_meshes, reconstruction_model):
        return run_reconstruction(
            synthetic_meshes[0], reconstruction_model, sample_seed=SAMPLE_SEED + 1, **SAMPLED
        )

    @pytest.fixture(scope="class")
    def unseeded_pair(self, synthetic_meshes, reconstruction_model):
        return [
            run_reconstruction(
                synthetic_meshes[0], reconstruction_model, sample_seed=None, **SAMPLED
            )
            for _ in range(2)
        ]

    @staticmethod
    def _latent(result):
        return result["latent"].detach().cpu().numpy().ravel()

    def test_the_draw_actually_happened(self, seeded_pair, reconstruction):
        """
        The premise. ``reconstruction`` is the same subject and the same model with
        ``get_rand_pts=False``, so if turning the draw on left the fit unchanged, every
        assertion below would be about a code path that never ran.
        """
        assert not np.allclose(
            self._latent(seeded_pair[0]), self._latent(reconstruction), atol=FITTED_LATENT_ATOL
        )

    def test_the_same_seed_fits_the_same_latent(self, seeded_pair):
        first, second = (self._latent(result) for result in seeded_pair)
        assert np.array_equal(first, second), f"max difference {np.abs(first - second).max():.3e}"

    def test_the_same_seed_reconstructs_the_same_geometry(self, seeded_pair):
        """
        Exact vertex equality, not the decile summary the baselines use: this is one
        process reconstructing one subject twice, so anything short of identical means the
        draw moved.
        """
        first, second = seeded_pair
        for index in (BONE, CART):
            assert np.array_equal(
                np.asarray(first["mesh"][index].point_coords),
                np.asarray(second["mesh"][index].point_coords),
            ), f"surface {index} differs between two runs at the same seed"

    def test_a_different_seed_fits_a_different_latent(self, seeded_pair, other_seed):
        """The guard: without it, "reproducible" would also be satisfied by a dead argument."""
        assert not np.allclose(
            self._latent(seeded_pair[0]), self._latent(other_seed), atol=FITTED_LATENT_ATOL
        )

    def test_an_unseeded_draw_is_not_reproducible(self, unseeded_pair):
        """
        ``sample_seed=None`` is the default and must stay unseeded. Both runs get the same
        global torch and numpy seed from ``run_reconstruction`` and still diverge, which is
        what shows the sampling seed -- not the global state -- is what makes the seeded
        pair above agree.
        """
        first, second = (self._latent(result) for result in unseeded_pair)
        assert not np.allclose(first, second, atol=FITTED_LATENT_ATOL)


class TestTheCommittedDecoder:
    """
    Every assertion above runs on one frozen decoder, loaded from
    ``assets/reconstruction_decoder.pt``. This is the part of that arrangement a reader
    years from now needs: which stack produced the weights the baselines are fitted to.
    """

    def test_it_records_the_stack_it_was_generated_on(self, reconstruction_model):
        """
        ``reconstruction_model`` is requested so the asset is known to exist -- that fixture
        is what loads it, or what writes it on a regeneration run.
        """
        recorded = torch.load(RECON_DECODER_ASSET, weights_only=True)["generated_on"]
        assert set(recorded) == set(provenance()), recorded
        assert recorded["platform"] == "Linux-x86_64", recorded


class TestAFreshlyTrainedDecoder:
    """
    The one thing freezing the decoder took away: with the fixture loading a checkpoint,
    nothing else here checks that a model straight out of ``train_deep_sdf`` can be
    reconstructed from at all.

    Structural only, and that is the point. The numbers a fresh run produces are the chaotic
    ones -- 60 epochs of gradient descent is what moved the reconstruction baselines 763x
    their tolerance under a torch bump -- so this asserts that a surface comes back and the
    latent has the right shape, and pins no value. ``baselines/training.json`` is what pins
    training output, directly and at 8 epochs where it has not yet diverged.

    Costs about 2.5 s on a warm process: ~1.3 s to train the ``RECON_TRAINING_EPOCHS``
    epochs the asset was generated from, ~1.2 s to reconstruct. Training through
    ``_harness.train_reconstruction_decoder`` rather than inline is deliberate -- it keeps
    the asset's regeneration path executed on every run, instead of only when someone sets
    ``NSM_REGENERATE_RECON_DECODER``.
    """

    @pytest.fixture(scope="class")
    def fresh_reconstruction(self, synthetic_meshes, training_dataset, tmp_path_factory):
        model = train_reconstruction_decoder(
            training_dataset, tmp_path_factory.mktemp("fresh_recon_train")
        )
        return run_reconstruction(synthetic_meshes[0], model)

    def test_a_surface_comes_back_for_every_object(self, fresh_reconstruction):
        """``[None, None]`` is what a decoder with no zero level set returns -- see below."""
        meshes = fresh_reconstruction["mesh"]
        assert meshes != [None, None]
        assert all(mesh is not None for mesh in meshes), meshes

    def test_the_fitted_latent_has_the_configured_shape(self, fresh_reconstruction):
        from _harness import LATENT_SIZE

        assert fresh_reconstruction["latent"].shape == (1, LATENT_SIZE)


class NoZeroLevelSetDecoder(torch.nn.Module):
    """
    A decoder whose SDF is +1 everywhere, so its mean shape has no surface.

    Deliberately a test double rather than a real under-trained model: the precondition
    for the early return below is "the zero-latent SDF never changes sign", and whether a
    real model reaches that state depends on its config. This states the precondition
    directly. It matches the calling convention ``mesh/main.decode_sdf`` inspects for --
    keyword ``latent``/``xyz``, with the legacy concatenated form as a fallback.
    """

    def forward(self, x=None, latent=None, xyz=None, epoch=None, verbose=False):
        n_points = xyz.shape[0] if xyz is not None else x.shape[0]
        return torch.ones(n_points, 2)


class TestDecoderWithNoZeroLevelSet:
    """
    What ``reconstruct_mesh`` does when the mean shape has no surface: raises by name
    (#29). The state every model is in before it has learnt a sign change, and the
    first thing anyone wiring up a new architecture will hit.

    Until Aug 2026 it returned a plausible-looking result dict instead -- ``mesh`` of
    Nones, NaN metrics, the untouched zero ``mean_latent`` under ``"latent"`` -- whose
    shape also dropped every key the caller asked for (this class pinned that dict, with
    a strict xfail on the dropped keys, until the fix landed; History §10).
    ``get_mean_errors`` catches the error and scores NaN so a validation epoch survives;
    that seam is pinned in ``test_reconstruct_mesh_options``.
    """

    def test_it_raises_by_name(self, synthetic_meshes):
        with pytest.raises(NoZeroLevelSetError, match="no zero level set"):
            run_reconstruction(synthetic_meshes[0], NoZeroLevelSetDecoder())
