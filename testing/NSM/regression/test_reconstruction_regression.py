"""
End-to-end reconstruction regression through ``reconstruct_mesh``, called the way
``kneepipeline/steps/run_nsm.py`` calls it: a list of mesh paths, every argument by name.
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
#: Keep it at 1. A wider dent proves less: the headroom measured 69x at 5 vertices and 119x
#: at 10, so raising this can only make a failing break pass. If the headroom drops under
#: ``MIN_HEADROOM``, investigate the fixture or the tolerance.
PERTURBATION = 0.25
PERTURBED_VERTICES = 1


def summaries(result):
    return {"bone": mesh_summary(result["mesh"][BONE]), "cart": mesh_summary(result["mesh"][CART])}


class TestConsumerContract:
    """
    Every key ``steps/run_nsm.py`` reads, and the ORDER of ``mesh``: index 0 is bone and 1 is
    cartilage, hardcoded by the consumer and declared nowhere in NSM. The synthetic surfaces
    are separated along z, so the order is asserted by geometry and fails if transposed.
    """

    def test_the_result_has_the_keys_and_types_the_consumer_reads(self, reconstruction):
        """
        Fails if ``reconstruct_mesh`` under kneepipeline's flags drops a key ``run_nsm.py``
        reads, returns other than two non-empty (n, 3) meshes, or changes the type of
        ``icp_transform`` or the shape of ``latent``, ``center`` or ``scale``.
        """
        from _harness import LATENT_SIZE

        assert set(reconstruction) >= {
            "mesh",
            "latent",
            "icp_transform",
            "center",
            "scale",
            "assd_0",
            "assd_1",
        }
        assert isinstance(reconstruction["mesh"], list) and len(reconstruction["mesh"]) == 2
        for mesh in reconstruction["mesh"]:
            points = np.asarray(mesh.point_coords)
            assert points.ndim == 2 and points.shape[1] == 3 and points.shape[0] > 0
        assert reconstruction["latent"].shape == (1, LATENT_SIZE)
        # _convert_icp_transform takes vtkTransform, vtkMatrix4x4, ndarray or None.
        assert isinstance(
            reconstruction["icp_transform"],
            (vtk.vtkIterativeClosestPointTransform, vtk.vtkTransform, vtk.vtkMatrix4x4),
        )
        assert np.asarray(reconstruction["center"]).shape == (3,)
        assert np.asarray(reconstruction["scale"]).ndim == 0

    def test_each_index_is_its_own_surface(self, reconstruction, synthetic_meshes):
        """
        Fails if ``reconstruct_mesh`` returns ``mesh`` or ``assd_i`` in a different order from
        the input paths.

        Each output must be 3x nearer its own input than the other, and ``assd_i`` 3x below
        its distance to the other input.
        """
        import pyvista as pv

        inputs = [pv.read(path).points.mean(axis=0) for path in synthetic_meshes[0]]
        outputs = [np.asarray(m.point_coords).mean(axis=0) for m in reconstruction["mesh"]]
        distance = np.array([[np.linalg.norm(o - i) for i in inputs] for o in outputs])
        assert distance[BONE][CART] > 3 * distance[BONE][BONE], distance
        assert distance[CART][BONE] > 3 * distance[CART][CART], distance

        crossed = [
            surface_distance(reconstruction["mesh"][BONE], synthetic_meshes[0][CART]),
            surface_distance(reconstruction["mesh"][CART], synthetic_meshes[0][BONE]),
        ]
        assert reconstruction["assd_0"] * 3 < crossed[0]
        assert reconstruction["assd_1"] * 3 < crossed[1]


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
    def test_the_reconstruction_matches_baseline(self, reconstruction, reconstruction_baseline):
        """
        Fails if ``reconstruct_mesh`` on the committed decoder moves the fitted latent, mesh
        geometry, point counts, ASSD, ``scale`` or ``center`` past their tolerances from the
        committed baseline.
        """
        latent = reconstruction["latent"].detach().cpu().numpy().ravel()
        reconstruction_baseline.check("fitted_latent", latent, atol=FITTED_LATENT_ATOL)
        reconstruction_baseline.check(
            "mesh_geometry", summaries(reconstruction), atol=GEOMETRY_ATOL
        )
        counts = [len(np.asarray(m.point_coords)) for m in reconstruction["mesh"]]
        reconstruction_baseline.check("mesh_point_counts", counts, rtol=COUNT_RTOL)
        assd = [reconstruction["assd_0"], reconstruction["assd_1"]]
        reconstruction_baseline.check("assd", assd, rtol=METRIC_RTOL)
        reconstruction_baseline.check("scale", float(reconstruction["scale"]), rtol=METRIC_RTOL)
        center = np.asarray(reconstruction["center"], dtype=float)
        reconstruction_baseline.check("center", center, atol=GEOMETRY_ATOL)

    def test_the_same_inputs_give_the_same_answer_exactly(
        self, synthetic_meshes, reconstruction_model, reconstruction
    ):
        """
        Fails if two identically seeded ``reconstruct_mesh`` calls on the vertex path
        (``get_rand_pts=False``) return a different latent or ``assd_0``.
        """
        again = run_reconstruction(synthetic_meshes[0], reconstruction_model)
        assert torch.equal(again["latent"], reconstruction["latent"])
        assert again["assd_0"] == reconstruction["assd_0"]


class TestDeliberateBreak:
    """
    Dent the input bone and confirm the baselines reject the result by at least
    ``MIN_HEADROOM`` times their tolerance. If this fails, do not enlarge the dent: a bigger
    break is easier to detect and proves less (see ``PERTURBED_VERTICES``).
    """

    @pytest.fixture(scope="class")
    def perturbed_reconstruction(self, synthetic_meshes, reconstruction_model, tmp_path_factory):
        import pyvista as pv

        bone = pv.read(synthetic_meshes[0][BONE])
        points = bone.points.copy()
        patch = np.argsort(np.linalg.norm(points - points[0], axis=1))[:PERTURBED_VERTICES]
        points[patch] += np.array([PERTURBATION, 0.0, 0.0], dtype=points.dtype)
        bone.points = points
        bone_path = str(tmp_path_factory.mktemp("perturbed") / "perturbed_bone.vtk")
        bone.save(bone_path)
        return run_reconstruction([bone_path, synthetic_meshes[0][CART]], reconstruction_model)

    def test_denting_the_bone_fails_the_latent_and_geometry_baselines(
        self, perturbed_reconstruction, reconstruction_baseline
    ):
        """
        Fails if ``FITTED_LATENT_ATOL`` or ``GEOMETRY_ATOL`` is loosened, or the baseline
        check weakened, so far that a one-vertex dent in the input bone no longer clears
        ``MIN_HEADROOM`` times the tolerance.
        """
        if regenerating():
            pytest.skip("baselines are being rewritten")
        latent = perturbed_reconstruction["latent"].detach().cpu().numpy().ravel()
        for key, observed, atol in (
            ("fitted_latent", latent, FITTED_LATENT_ATOL),
            ("mesh_geometry", summaries(perturbed_reconstruction), GEOMETRY_ATOL),
        ):
            with pytest.raises(AssertionError, match="differs from baseline"):
                reconstruction_baseline.check(key, observed, atol=atol)
            measured = headroom(reconstruction_baseline, key, observed, atol=atol)
            assert measured >= MIN_HEADROOM, (
                f"the dent moves {key} only {measured:.1f}x its tolerance, under the "
                f"MIN_HEADROOM of {MIN_HEADROOM}x. Widen the break (more vertices), never the "
                f"tolerance."
            )


#: Turns ``reconstruct_mesh``'s point draw on. The baselines above are fitted to the mesh
#: vertices, and with ``get_rand_pts=False`` its ``seed`` argument reaches nothing at all.
SAMPLED = dict(get_rand_pts=True, n_pts_random=200)
SAMPLE_SEED = 7


def test_the_sampled_reconstruction_is_seeded(
    synthetic_meshes, reconstruction_model, reconstruction
):
    """
    Fails if ``reconstruct_mesh``'s ``seed`` stops seeding the multi-surface point draw under
    ``get_rand_pts=True``: the same seed differs, another seed agrees, or ``seed=None``
    repeats under a fixed global seed.

    kneepipeline's model configs set ``get_rand_pts_recon`` false, so production fits draw no
    points and ``seed`` reaches nothing there.

    The harness passes the seed as ``sample_seed``: its own ``seed`` would swallow the keyword,
    reseed torch, leave the draw unseeded, and still pass three times in four. The first
    assert checks that the draw happens at all: the fit must move from the vertex-only one.
    """

    def fit(sample_seed):
        return run_reconstruction(
            synthetic_meshes[0], reconstruction_model, sample_seed=sample_seed, **SAMPLED
        )

    def latent(result):
        return result["latent"].detach().cpu().numpy().ravel()

    first, again = fit(SAMPLE_SEED), fit(SAMPLE_SEED)
    assert not np.allclose(latent(first), latent(reconstruction), atol=FITTED_LATENT_ATOL)
    assert np.array_equal(latent(first), latent(again))
    for index in (BONE, CART):
        np.testing.assert_array_equal(
            first["mesh"][index].point_coords, again["mesh"][index].point_coords
        )
    assert not np.allclose(latent(first), latent(fit(SAMPLE_SEED + 1)), atol=FITTED_LATENT_ATOL)
    assert not np.allclose(latent(fit(None)), latent(fit(None)), atol=FITTED_LATENT_ATOL)


class TestTheCommittedDecoder:
    """Every test above runs on ``assets/reconstruction_decoder.pt``, loaded, not retrained."""

    def test_it_records_the_stack_it_was_generated_on(self, reconstruction_model):
        """
        Fails if the committed decoder records fields other than ``provenance()`` returns,
        or was generated on a platform other than Linux-x86_64.
        """
        recorded = torch.load(RECON_DECODER_ASSET, weights_only=True)["generated_on"]
        assert set(recorded) == set(provenance()), recorded
        assert recorded["platform"] == "Linux-x86_64", recorded


class TestAFreshlyTrainedDecoder:
    """
    With the decoder frozen, nothing else checks that a model straight out of
    ``train_deep_sdf`` can be reconstructed from. No number here is pinned: 60 epochs of
    gradient descent is what moved the baselines 763x their tolerance under a torch bump.
    Instead the trained decoder's ASSD must beat an untrained control of the same
    architecture and seed, which both sides of a torch bump move together (#34). Training
    goes through ``_harness.train_reconstruction_decoder``, so the asset's regeneration path
    runs every time.
    """

    #: Measured 2026-08-26: trained 0.224 / 0.172 against untrained 2.197 / 3.014, ratios of
    #: 9.8x and 17.5x. A floor on "training did something", not a pin on how much.
    MIN_IMPROVEMENT = 3.0

    def test_training_is_what_makes_the_surfaces_fit(
        self, synthetic_meshes, training_dataset, tmp_path_factory
    ):
        """
        Fails if ``reconstruct_mesh`` cannot run on a decoder fresh from ``train_deep_sdf``,
        or that decoder's ASSD is not ``MIN_IMPROVEMENT`` times better than an untrained one's.
        """
        from _harness import LATENT_SIZE, build_model, training_config

        trained = train_reconstruction_decoder(
            training_dataset, tmp_path_factory.mktemp("fresh_recon_train")
        )
        fresh = run_reconstruction(synthetic_meshes[0], trained)
        assert all(mesh is not None for mesh in fresh["mesh"])
        assert fresh["latent"].shape == (1, LATENT_SIZE)

        untrained = build_model(training_config(tmp_path_factory.mktemp("untrained_recon")))
        control = run_reconstruction(synthetic_meshes[0], untrained.eval())
        for key in ("assd_0", "assd_1"):
            ratio = control[key] / fresh[key]
            assert ratio > self.MIN_IMPROVEMENT, f"{key}: only {ratio:.2f}x better than untrained"


class NoZeroLevelSetDecoder(torch.nn.Module):
    """SDF +1 everywhere, so the mean shape has no surface: the state before training."""

    def forward(self, x=None, latent=None, xyz=None, epoch=None):
        return torch.ones((xyz if xyz is not None else x).shape[0], 2)


class TestDecoderWithNoZeroLevelSet:
    """
    ``get_mean_errors`` catches the error and scores NaN; that half is pinned in
    ``test_reconstruct_mesh``.
    """

    def test_it_raises_by_name(self, synthetic_meshes):
        """
        Fails if ``reconstruct_mesh(register_similarity=True)`` returns a result instead of
        raising ``NoZeroLevelSetError`` when the mean shape has no surface
        (KNOWN_ISSUES History 10).
        """
        with pytest.raises(NoZeroLevelSetError, match="no zero level set"):
            run_reconstruction(synthetic_meshes[0], NoZeroLevelSetDecoder())
