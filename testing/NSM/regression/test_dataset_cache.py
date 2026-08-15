"""
The SDF dataset cache: round-trip, cache-key coverage, and the seeding story.

``datasets/sdf_dataset.py`` is the largest module in NSM and the least covered, and it is
the layer an in-memory harness never touches. Everything here is behaviour as it stands
today, bugs included -- several of these tests assert that something *wrong* happens,
because pinning it is what makes a later fix visible.

The cache is keyed by ``md5("_".join(str(p) for p in get_hash_params() + mesh paths))``.
Three parameters that change what gets written are missing from that list, so two runs
differing only in one of them share a key and the second silently reuses the first's data
(``TestUnhashedParametersCollide``). A fourth, ``reference_mesh``, is hashed by object
identity when it is a ``Mesh``, so the cache never hits at all
(``TestReferenceMeshHashing``).
"""

import inspect
import os

import numpy as np
import pytest
from _harness import build_dataset, write_synthetic_meshes

#: One subject, few points: these tests are about keys and content identity, not numbers.
SMALL = dict(n_pts=[600, 600], subsample=64)


@pytest.fixture(scope="module")
def meshes(tmp_path_factory):
    return write_synthetic_meshes(tmp_path_factory.mktemp("cache_meshes"))[:1]


@pytest.fixture(scope="module")
def dataset(meshes, tmp_path_factory):
    return build_dataset(meshes, tmp_path_factory.mktemp("hash_probe"), **SMALL)


def cached_arrays(dataset, index=0):
    """The ``.npz`` this dataset wrote, as a plain dict."""
    return dict(np.load(dataset.data[index]))


def rehash(dataset, mesh_paths, **attributes):
    """
    Recompute the cache key with some constructor parameters changed.

    ``create_hash`` reads ``self.list_hash_params``, which ``__init__`` fills from
    ``get_hash_params()``. Setting the attributes and refilling it exercises exactly the
    key-derivation path without paying for a rebuild.
    """
    original = {name: getattr(dataset, name) for name in attributes}
    original_params = dataset.list_hash_params
    try:
        for name, value in attributes.items():
            setattr(dataset, name, value)
        dataset.list_hash_params = dataset.get_hash_params()
        return dataset.create_hash(mesh_paths)
    finally:
        for name, value in original.items():
            setattr(dataset, name, value)
        dataset.list_hash_params = original_params


class TestCacheRoundTrip:
    def test_a_cache_file_is_written_per_subject(self, dataset, meshes):
        assert len(dataset.data) == len(meshes)
        for path in dataset.data:
            assert os.path.exists(path) and path.endswith(".npz")

    def test_reload_returns_identical_samples(self, meshes, tmp_path_factory):
        """
        Build once, then build again against the same cache with ``load_cache=True`` and a
        different sampling seed. The second build must reuse the first's file byte for
        byte -- if it re-sampled, every number here would move.
        """
        cache = tmp_path_factory.mktemp("roundtrip")
        first = build_dataset(meshes, cache, seed=0, **SMALL)
        reloaded = build_dataset(meshes, cache, seed=999, load_cache=True, **SMALL)

        assert reloaded.data[0] == first.data[0], "cache was not hit"
        original, again = cached_arrays(first), cached_arrays(reloaded)
        assert set(original) == set(again)
        for key in original:
            assert np.array_equal(original[key], again[key]), key

    def test_reloaded_items_match_the_freshly_built_ones(self, meshes, tmp_path_factory):
        """The round trip has to survive ``__getitem__``, not just the file."""
        import torch

        cache = tmp_path_factory.mktemp("roundtrip_items")
        first = build_dataset(meshes, cache, **SMALL)
        reloaded = build_dataset(meshes, cache, load_cache=True, **SMALL)

        torch.manual_seed(0)
        a, _ = first[0]
        torch.manual_seed(0)
        b, _ = reloaded[0]
        assert torch.equal(a["xyz"], b["xyz"])
        assert torch.equal(a["gt_sdf"], b["gt_sdf"])

    def test_the_cache_is_searched_recursively(self, dataset):
        """
        ``find_hash`` walks all of ``loc_save``, not just today's folder -- which is what
        lets a cache written on another day still hit, and also what makes two datasets
        sharing a ``loc_save`` root able to collide across subdirectories.
        """
        cache_file = os.path.basename(dataset.data[0])
        found = dataset.find_hash(filename=cache_file)
        assert found and os.path.basename(found[0]) == cache_file


class TestHashedParametersChangeTheKey:
    """The parameters that are correctly part of the cache key."""

    @pytest.mark.parametrize(
        "attribute,value",
        [
            ("center_pts", False),
            ("norm_pts", False),
            ("fix_mesh", True),
            ("scale_jointly", True),
            ("scale_all_meshes", False),
            ("center_all_meshes", True),
            ("reference_object", 1),
            ("reference_mesh", "some/other/mesh.vtk"),
            ("n_pts", [500, 600]),
            ("p_near_surface", [0.3, 0.4]),
            ("p_further_from_surface", [0.3, 0.4]),
            ("sigma_near", [0.01, None]),
            ("sigma_far", [0.2, None]),
            ("rand_function", "laplace"),
        ],
    )
    def test_changing_it_changes_the_key(self, dataset, meshes, attribute, value):
        baseline = dataset.create_hash(meshes[0])
        assert rehash(dataset, meshes[0], **{attribute: value}) != baseline

    def test_the_mesh_paths_are_part_of_the_key(self, dataset, meshes):
        assert dataset.create_hash(meshes[0]) != dataset.create_hash([meshes[0][1], meshes[0][0]])

    def test_random_seed_is_part_of_the_key(self, dataset, meshes):
        """
        Included for completeness and because it is misleading: the seed reaches the cache
        key and nothing else. See ``TestSeeding``.
        """
        assert rehash(dataset, meshes[0], random_seed=7) != dataset.create_hash(meshes[0])


class TestUnhashedParametersCollide:
    """
    ``mesh_to_scale``, ``uniform_pts_buffer`` and ``subsample`` all change what is written
    into the cache and none of them are in ``get_hash_params``
    (``sdf_dataset.py:1973-1999``). Two runs differing only in one of them therefore share
    a key, and with ``load_cache=True`` -- the production setting -- the second silently
    trains on the first's data.

    **These assert the behaviour NSM should have, and are expected to fail.** Each shows
    that the cached content genuinely differs first, so the ``xfail`` lands on the cache
    key rather than on a vacuous premise. Fixing worklist #1 turns them green, which under
    ``strict=True`` fails the suite until they are un-marked -- see the module docstring.
    """

    @staticmethod
    def _content_differs(meshes, tmp_path_factory, label, **override):
        a = build_dataset(meshes, tmp_path_factory.mktemp(f"{label}_a"), **SMALL)
        b = build_dataset(meshes, tmp_path_factory.mktemp(f"{label}_b"), **dict(SMALL, **override))
        first, second = cached_arrays(a), cached_arrays(b)
        differing = {
            key
            for key in set(first) & set(second)
            if first[key].shape != second[key].shape or not np.array_equal(first[key], second[key])
        }
        return a.create_hash(meshes[0]), b.create_hash(meshes[0]), differing

    @pytest.mark.xfail(strict=True, reason="worklist #1: get_hash_params omits mesh_to_scale")
    def test_mesh_to_scale_must_change_the_cache_key(self, meshes, tmp_path_factory):
        """
        The worst of the three: ``mesh_to_scale`` decides which surface drives centering
        and normalization, so the two runs' cached points and SDFs are in different
        coordinate frames entirely.
        """
        key_a, key_b, differing = self._content_differs(
            meshes, tmp_path_factory, "mts", mesh_to_scale=1
        )
        assert {
            "pts",
            "sdfs",
        } <= differing, f"premise gone: content no longer differs ({differing})"
        assert key_a != key_b, "the cached content differs but the cache key does not"

    @pytest.mark.xfail(strict=True, reason="worklist #1: get_hash_params omits uniform_pts_buffer")
    def test_uniform_pts_buffer_must_change_the_cache_key(self, meshes, tmp_path_factory):
        """It sets the bounds the uniform points are drawn from, so the samples move."""
        key_a, key_b, differing = self._content_differs(
            meshes, tmp_path_factory, "upb", uniform_pts_buffer=0.5
        )
        assert {
            "pts",
            "sdfs",
        } <= differing, f"premise gone: content no longer differs ({differing})"
        assert key_a != key_b, "the cached content differs but the cache key does not"

    @pytest.mark.xfail(strict=True, reason="worklist #1: get_hash_params omits subsample")
    def test_subsample_must_change_the_cache_key(self, meshes, tmp_path_factory):
        """
        Milder but still real: ``subsample`` sets ``samples_per_sign_``, which decides how
        many times ``sdf_pos_neg_idx`` repeats the index arrays -- and those arrays are
        cached. The points themselves are unaffected.
        """
        key_a, key_b, differing = self._content_differs(
            meshes, tmp_path_factory, "sub", subsample=512
        )
        assert any(
            key.startswith(("pos_idx", "neg_idx")) for key in differing
        ), f"premise gone: index arrays no longer differ ({differing})"
        assert key_a != key_b, "the cached index arrays differ but the cache key does not"

    @pytest.mark.xfail(strict=True, reason="worklist #1: colliding runs share a cache file")
    def test_a_changed_parameter_must_not_reuse_the_previous_runs_cache(
        self, meshes, tmp_path_factory
    ):
        """End to end: same cache directory, ``load_cache=True``, only ``mesh_to_scale`` changed."""
        cache = tmp_path_factory.mktemp("collision")
        first = build_dataset(meshes, cache, **SMALL)
        second = build_dataset(meshes, cache, load_cache=True, mesh_to_scale=1, **SMALL)

        assert second.data[0] != first.data[0], "the second run was handed the first run's file"

    @pytest.mark.xfail(
        strict=True, reason="worklist #1: a subsample collision silently unbalances the batch"
    )
    def test_equal_pos_neg_must_hold_after_a_subsample_change(self, meshes, tmp_path_factory):
        """
        What the ``subsample`` collision costs, measured.

        ``sdf_pos_neg_idx`` repeats the negative-index array just far enough for the
        ``subsample`` in force when the cache was written. Reload with a larger one and
        there are not enough entries: ``__getitem__`` takes what there is, then tops the
        batch up with uniformly random points (``sdf_dataset.py:2122-2127``). The
        ``equal_pos_neg=True`` guarantee quietly stops holding, and the surface with the
        fewest interior samples -- the small one, i.e. cartilage in a real dataset -- loses
        the most. Measured at ~4.4x under-representation.

        The reload check that would have caught this compares ``len(data["pos_idx"])``
        against the number of *meshes* (``sdf_dataset.py:1764-1771``), never against the
        subsample the arrays were built for.
        """
        import torch

        base = {k: v for k, v in SMALL.items() if k != "subsample"}
        cache = tmp_path_factory.mktemp("subsample_collision")
        build_dataset(meshes, cache, subsample=64, **base)
        reused = build_dataset(meshes, cache, load_cache=True, subsample=512, **base)
        fresh = build_dataset(
            meshes, tmp_path_factory.mktemp("subsample_fresh"), subsample=512, **base
        )

        def interior_fraction(dataset, surface):
            torch.manual_seed(0)
            item, _ = dataset[0]
            return (item["gt_sdf"][:, surface] < 0).float().mean().item()

        # Surface 1 is the small ellipsoid: fewest interior points, so it is hit hardest.
        assert interior_fraction(reused, 1) == pytest.approx(interior_fraction(fresh, 1), rel=0.25)


class TestReferenceMeshHashing:
    """
    A ``reference_mesh`` passed as a ``Mesh`` object is stringified into the key, and
    ``Mesh.__str__`` includes its memory address -- so the key is per-object. The cache
    can never hit across processes, and inside one process it changes on every
    construction.
    """

    @pytest.mark.xfail(
        strict=True, reason="worklist #2: a Mesh reference_mesh is hashed by memory address"
    )
    def test_two_equal_mesh_objects_must_hash_the_same(self, dataset, meshes):
        """Same geometry, same file, two objects -- the cache key must not care."""
        from pymskt.mesh import Mesh

        one, two = Mesh(meshes[0][0]), Mesh(meshes[0][0])
        assert rehash(dataset, meshes[0], reference_mesh=one) == rehash(
            dataset, meshes[0], reference_mesh=two
        )

    def test_the_address_is_what_leaks_in(self, meshes):
        from pymskt.mesh import Mesh

        assert "0x" in str(Mesh(meshes[0][0])).split("\n")[0]

    def test_a_path_string_hashes_stably(self, dataset, meshes):
        """The same reference given as a path is stable, which is the workaround."""
        assert rehash(dataset, meshes[0], reference_mesh=meshes[0][0]) == rehash(
            dataset, meshes[0], reference_mesh=meshes[0][0]
        )


class TestSeeding:
    """
    ``SDFSamples(random_seed=...)`` is documented as "Random seed". It is never used to
    seed anything: ``grep -n random_seed NSM/datasets/sdf_dataset.py`` finds it stored on
    the instance and appended to the cache key, and nowhere else. NSM calls no seeding
    function at all.

    That leaves two sampling paths with very different reproducibility, and the difference
    is invisible from the constructor. This is the reason the whole harness runs on the
    uniform path.
    """

    def test_the_uniform_path_is_reproducible_under_a_numpy_seed(self, meshes, tmp_path_factory):
        """``sigma=None`` routes through ``get_rand_uniform_pts``, i.e. ``np.random``."""
        a = build_dataset(meshes, tmp_path_factory.mktemp("seed_u_a"), seed=7, **SMALL)
        b = build_dataset(meshes, tmp_path_factory.mktemp("seed_u_b"), seed=7, **SMALL)
        assert np.array_equal(cached_arrays(a)["pts"], cached_arrays(b)["pts"])

    @pytest.mark.xfail(
        strict=True,
        reason="worklist #3 / gattia/pymskt#54: the near-surface sampler cannot be seeded",
    )
    def test_the_near_surface_path_must_be_reproducible(self, meshes, tmp_path_factory):
        """
        With ``sigma`` set, sampling goes through ``pymskt.Mesh.rand_pts_around_surface``,
        which has *two* independent draws a caller cannot reach:

        * the base surface points, via ``pcu.sample_mesh_random(v, f, n, random_seed=0)`` --
          and ``random_seed=0`` means "seed from the current time", not "seed 0";
        * the perturbation offsets, via ``np.random.default_rng()`` with no argument, which
          seeds itself from OS entropy and is independent of ``np.random.seed()``.

        Neither is reachable from NSM, so identical inputs and an identical numpy seed still
        produce different training data. Reported upstream as gattia/pymskt#54.
        """
        near = dict(SMALL, sigma_near=[0.05, 0.05], sigma_far=[0.2, 0.2])
        a = build_dataset(meshes, tmp_path_factory.mktemp("seed_n_a"), seed=7, **near)
        b = build_dataset(meshes, tmp_path_factory.mktemp("seed_n_b"), seed=7, **near)
        assert np.array_equal(cached_arrays(a)["pts"], cached_arrays(b)["pts"])

    def test_a_warm_cache_makes_random_seed_look_like_it_works(self, meshes, tmp_path_factory):
        """
        Not a defect assertion -- this is the mechanism that hides the defect, and it is
        genuinely correct behaviour on its own terms. Two runs with the same ``random_seed``
        get the same cache key, so the second reuses the first's ``.npz`` and is identical
        to it. That is a cache hit working, and it is why nobody noticed.
        """
        near = dict(SMALL, sigma_near=[0.05, 0.05], sigma_far=[0.2, 0.2], random_seed=1234)
        warm_cache = tmp_path_factory.mktemp("seed_shared")
        first = build_dataset(meshes, warm_cache, **near)
        looks_reproducible = build_dataset(meshes, warm_cache, load_cache=True, **near)

        assert np.array_equal(cached_arrays(first)["pts"], cached_arrays(looks_reproducible)["pts"])

    @pytest.mark.xfail(
        strict=True, reason="worklist #3: random_seed feeds the cache key and seeds nothing"
    )
    def test_random_seed_must_make_a_cold_run_reproducible(self, meshes, tmp_path_factory):
        """The same seed against two *cold* caches must give the same samples. It does not."""
        near = dict(SMALL, sigma_near=[0.05, 0.05], sigma_far=[0.2, 0.2], random_seed=1234)
        a = build_dataset(meshes, tmp_path_factory.mktemp("seed_cold_a"), **near)
        b = build_dataset(meshes, tmp_path_factory.mktemp("seed_cold_b"), **near)

        assert np.array_equal(cached_arrays(a)["pts"], cached_arrays(b)["pts"])


class TestConfigurationsThatDoNotRun:
    """
    Constructible-but-uncallable settings: advertised constructor arguments that build fine
    and raise on first use. Each asserts that the option *works*, and is expected to fail.
    """

    @pytest.mark.xfail(
        strict=True, reason="worklist #5: a zero-count sampling combo is passed to pcu anyway"
    )
    def test_zero_sampling_probability_must_sample_nothing(self, meshes, tmp_path_factory):
        """
        ``get_pt_sample_combos`` emits a ``[0, sigma]`` combo and ``get_sample_data_dict``
        calls the sampler with it regardless (``sdf_dataset.py:1820``), so asking for no
        near-surface points is a crash rather than a configuration.
        """
        dataset = build_dataset(
            meshes,
            tmp_path_factory.mktemp("p_zero"),
            p_near_surface=[0.0, 0.0],
            p_further_from_surface=[0.5, 0.5],
            sigma_near=[0.05, 0.05],
            **SMALL,
        )
        assert len(dataset) == len(meshes)

    @pytest.mark.xfail(
        strict=True, reason="worklist #4: store_data_in_memory=True raises UnboundLocalError"
    )
    def test_store_data_in_memory_must_yield_an_item(self, meshes, tmp_path_factory):
        """
        ``MultiSurfaceSDFSamples.__getitem__:2158`` reads ``time_`` and ``size``, which are
        only bound in the ``store_data_in_memory is False`` branch. The single-surface
        ``SDFSamples.__getitem__:1563`` guards the same block correctly, so the two classes
        disagree about the same option.
        """
        dataset = build_dataset(
            meshes, tmp_path_factory.mktemp("in_memory"), store_data_in_memory=True, **SMALL
        )
        item, _ = dataset[0]
        assert {"xyz", "gt_sdf"} <= set(item)

    def test_store_data_in_memory_works_with_load_timing_off(self, meshes, tmp_path_factory):
        """The workaround, recorded so the pairing is documented somewhere."""
        dataset = build_dataset(
            meshes,
            tmp_path_factory.mktemp("in_memory_ok"),
            store_data_in_memory=True,
            test_load_times=False,
            **SMALL,
        )
        item, index = dataset[0]
        assert set(item) == {"xyz", "gt_sdf"} and index == 0

    def test_train_epoch_needs_the_timing_keys(self):
        """
        Why the pairing above is not a real workaround: ``train_epoch`` reads all four
        timing keys unconditionally (``train_deep_sdf.py:578-581``), so the combination
        that avoids the crash produces batches the trainer cannot consume.
        """
        import NSM.train.train_deep_sdf as trainer

        source = inspect.getsource(trainer.train_epoch)
        for key in ("size", "time", "mb_per_sec", "whole_load_time"):
            assert f'sdf_data["{key}"]' in source


class TestCacheLocationDefault:
    """
    ``loc_save``'s default is ``os.environ.get("LOC_SDF_CACHE", ...)`` evaluated as a
    default argument, so it is bound once when ``sdf_dataset`` is imported. Setting the
    env var afterwards has no effect, and a caller who believes it does writes into
    ``~/.cache/nsm_sdf_cache``. The harness passes ``loc_save`` explicitly for this reason.
    """

    @pytest.mark.xfail(strict=True, reason="worklist: LOC_SDF_CACHE is read once, at import time")
    def test_setting_the_env_var_must_change_where_the_cache_goes(self, monkeypatch):
        from NSM.datasets.sdf_dataset import MultiSurfaceSDFSamples

        before = inspect.signature(MultiSurfaceSDFSamples.__init__).parameters["loc_save"].default
        monkeypatch.setenv("LOC_SDF_CACHE", "/nowhere/that/exists")
        after = inspect.signature(MultiSurfaceSDFSamples.__init__).parameters["loc_save"].default
        assert after != before


class TestPointCenteringAndScaling:
    """
    ``get_pts_center_and_scale`` is the normalization every cached sample goes through.
    Two of its documented behaviours are not its behaviours.
    """

    @pytest.mark.xfail(
        strict=True, reason="worklist #6: center= and scale= are rebound before they are read"
    )
    def test_center_and_scale_arguments_must_be_honoured(self):
        """
        Both are rebound before they are read (``sdf_dataset.py:88`` and ``:94``), so
        centering and scaling happen unconditionally and ``center=False, scale=False``
        changes nothing.
        """
        from NSM.datasets.sdf_dataset import get_pts_center_and_scale

        points = np.array([[1.0, 1.0, 1.0], [3.0, 3.0, 3.0]])
        center, _ = get_pts_center_and_scale(points.copy(), center=False, scale=False)
        assert np.allclose(center, [0.0, 0.0, 0.0]), "centering happened despite center=False"

    @pytest.mark.xfail(strict=True, reason="worklist #6: pts is modified in place")
    def test_the_callers_array_must_not_be_mutated(self):
        """
        ``pts -= center`` at ``:91`` writes through to the caller's array. All three in-repo
        call sites pass ``np.copy(...)``, so the convention exists only as a habit at the
        call sites -- a fourth caller written without it gets silently corrupted input.
        """
        from NSM.datasets.sdf_dataset import get_pts_center_and_scale

        points = np.array([[1.0, 1.0, 1.0], [3.0, 3.0, 3.0]])
        get_pts_center_and_scale(points)
        assert np.allclose(points, [[1.0, 1.0, 1.0], [3.0, 3.0, 3.0]])
