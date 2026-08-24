"""
The SDF dataset cache: round-trip, cache-key coverage, and the seeding story.

``datasets/sdf_dataset.py`` is the largest module in NSM and the least covered, and it is
the layer an in-memory harness never touches. Everything here is behaviour as it stands
today, bugs included -- several of these tests assert that something *wrong* happens,
because pinning it is what makes a later fix visible.

The cache is keyed by ``md5(json.dumps({name: value}, sort_keys=True))`` -- a named,
canonical mapping in which every mesh path contributes a content-stable
``(path, size, mtime)`` identity and a loaded ``Mesh`` contributes a geometry digest
(#19, fixed Aug 2026). ``TestFormerlyCollidingParameters`` pins the parameters that used
to change cached content while absent from the key; ``TestReferenceMeshHashing`` pins the
``Mesh``-valued reference that used to hash by memory address.
"""

import inspect
import os
import subprocess
import sys

import numpy as np
import pytest
from _harness import (
    build_dataset,
    build_model,
    build_single_surface_dataset,
    quiet,
    run_training,
    training_config,
    write_synthetic_meshes,
)

#: One subject, few points: these tests are about keys and content identity, not numbers.
SMALL = dict(n_pts=[600, 600], subsample=64)

#: The same, for the single-surface parent class, whose ``n_pts`` is a scalar.
SMALL_SINGLE = dict(n_pts=600, subsample=64)


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

    ``create_hash`` reads ``self.hash_params``, which ``__init__`` fills from
    ``get_hash_params()``. Setting the attributes and refilling it exercises exactly the
    key-derivation path without paying for a rebuild.
    """
    original = {name: getattr(dataset, name) for name in attributes}
    original_params = dataset.hash_params
    try:
        for name, value in attributes.items():
            setattr(dataset, name, value)
        dataset.hash_params = dataset.get_hash_params()
        return dataset.create_hash(mesh_paths)
    finally:
        for name, value in original.items():
            setattr(dataset, name, value)
        dataset.hash_params = original_params


class TestCacheRoundTrip:
    def test_a_cache_file_is_written_per_subject(self, dataset, meshes):
        assert len(dataset.data) == len(meshes)
        for path in dataset.data:
            assert os.path.exists(path) and path.endswith(".npz")

    def test_reload_returns_identical_samples(self, meshes, tmp_path_factory):
        """
        Build once, then build again against the same cache with ``load_cache=True``. The
        second build must reuse the first's file byte for byte.

        Both runs pass ``random_seed=None`` on purpose. That keeps the cache key identical
        -- the seed is part of it -- while leaving sampling unseeded, so a re-sample would
        move every number here rather than reproducing them.
        """
        cache = tmp_path_factory.mktemp("roundtrip")
        first = build_dataset(meshes, cache, seed=0, random_seed=None, **SMALL)
        reloaded = build_dataset(
            meshes, cache, seed=999, random_seed=None, load_cache=True, **SMALL
        )

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


class TestCacheHitMachinery:
    """
    What the cache-hit path does beyond loading: repair, upgrade, and refuse.

    ``get_sample_data_dict`` in both classes wraps the same shell around a hit --
    delete unreadable files, upgrade old layouts in place, coerce to the storage mode
    in force -- and none of it was pinned before the section 8.0.F class-side work
    restructures exactly this code. Each test here is one branch of that shell.
    """

    def test_a_corrupt_cache_file_is_deleted_and_rebuilt(self, meshes, tmp_path_factory):
        """
        ``is_zipfile`` guards the hit before ``np.load`` touches it: a truncated or
        garbage ``.npz`` (a crash mid-write) is deleted and the subject rebuilt, not
        crashed on and not dropped.
        """
        cache = tmp_path_factory.mktemp("badzip")
        first = build_dataset(meshes, cache, **SMALL)
        path = first.data[0]
        with open(path, "wb") as f:
            f.write(b"not a zipfile")

        second = build_dataset(meshes, cache, load_cache=True, **SMALL)

        assert len(second) == len(meshes), "the subject was dropped instead of rebuilt"
        assert second.data[0] == path, "the rebuild landed at a different path"
        rebuilt = cached_arrays(second)
        assert rebuilt["pts"].shape == (sum(SMALL["n_pts"]), 3)

    def test_the_single_surface_class_also_deletes_corrupt_files(
        self, bone_meshes, tmp_path_factory
    ):
        """``SDFSamples.get_sample_data_dict`` is a separate copy of the same shell."""
        cache = tmp_path_factory.mktemp("badzip_single")
        first = build_single_surface_dataset(bone_meshes[:1], cache, **SMALL_SINGLE)
        path = first.data[0]
        with open(path, "wb") as f:
            f.write(b"junk")

        second = build_single_surface_dataset(
            bone_meshes[:1], cache, load_cache=True, **SMALL_SINGLE
        )
        assert len(second) == 1 and second.data[0] == path
        assert cached_arrays(second)["pts"].shape == (SMALL_SINGLE["n_pts"], 3)

    def test_a_pre_overlap_pass_cache_is_upgraded_and_resaved(self, meshes, tmp_path_factory):
        """
        ``remove_overlapping_points`` runs on every hit, so a cache written before the
        overlap pass existed is shrunk and resaved in place. The index lists are NOT
        recomputed on this path: the resave condition that would recompute them compares
        ``len(data["pos_idx"])`` against the number of surfaces, which overlap removal
        does not change. The poisoned row here is the last one, pruned from every index
        list first, precisely so the in-range guard below stays out of the way and the
        resave branch is what this test exercises.
        """
        cache = tmp_path_factory.mktemp("overlap_upgrade")
        first = build_dataset(meshes, cache, **SMALL)
        path = first.data[0]
        arrays = dict(np.load(path))
        n = arrays["sdfs"].shape[0]
        arrays["sdfs"] = arrays["sdfs"].copy()
        arrays["sdfs"][-1, :] = -0.05  # inside BOTH surfaces: anatomically impossible
        for key in [k for k in arrays if k.startswith(("pos_idx", "neg_idx", "surf_idx"))]:
            arrays[key] = arrays[key][arrays[key] != n - 1]
        np.savez(path, **arrays)

        second = build_dataset(meshes, cache, load_cache=True, **SMALL)

        assert second.data[0] == path, "the upgrade should hit, not rebuild"
        upgraded = dict(np.load(path))
        assert upgraded["sdfs"].shape[0] == n - 1, "the overlapping row was not removed on disk"
        assert np.array_equal(upgraded["pos_idx_0"], arrays["pos_idx_0"]), (
            "the index lists were recomputed -- the length-based resave condition "
            "must have changed"
        )

    def test_out_of_range_cached_indices_delete_and_rebuild(self, meshes, tmp_path_factory):
        """
        ``test_if_idx_in_range`` guards against index lists that outlived their point
        set (an overlap pass shrank ``xyz`` after they were computed). Such a file is
        deleted and the subject rebuilt from the meshes -- served as-is, those indices
        would read the wrong rows or step off the end of the array.
        """
        cache = tmp_path_factory.mktemp("out_of_range")
        first = build_dataset(meshes, cache, **SMALL)
        path = first.data[0]
        arrays = dict(np.load(path))
        poisoned = arrays["pos_idx_0"].copy()
        poisoned[0] = arrays["pts"].shape[0] + 100
        arrays["pos_idx_0"] = poisoned
        np.savez(path, **arrays)

        second = build_dataset(meshes, cache, load_cache=True, **SMALL)

        assert len(second) == len(meshes)
        rebuilt = cached_arrays(second)
        assert rebuilt["pos_idx_0"].max() < rebuilt["pts"].shape[0], "the poison survived"

    def test_a_pre_index_layout_cache_is_upgraded_on_the_single_surface_class(
        self, bone_meshes, tmp_path_factory
    ):
        """
        The upgrade ``SDFSamples.get_sample_data_dict`` documents -- "caches from before
        the ``pos_idx`` layout are upgraded in place" -- never fired until Aug 2026: its
        condition was ``"pos_idx" not in data``, and ``unpack_numpy_data`` puts the key
        there unconditionally, as an EMPTY list when the group is absent from the file
        (``unpack_pts``). A pre-index-layout cache was served untouched and
        ``__getitem__`` died on it with ``IndexError: list index out of range`` (verified
        by execution, 2026-08-24) -- always a crash, never wrong results, so no History
        entry. The condition now checks the unpacked length, the same idea as the multi
        class's ``n_meshes`` length check, which never had the defect.
        """
        cache = tmp_path_factory.mktemp("backfill_single")
        first = build_single_surface_dataset(bone_meshes[:1], cache, **SMALL_SINGLE)
        path = first.data[0]
        arrays = dict(np.load(path))
        stripped = {
            k: v for k, v in arrays.items() if not k.startswith(("pos_idx", "neg_idx", "surf_idx"))
        }
        np.savez(path, **stripped)

        second = build_single_surface_dataset(
            bone_meshes[:1], cache, load_cache=True, **SMALL_SINGLE
        )

        upgraded = dict(np.load(path))
        assert np.array_equal(upgraded["pts"], arrays["pts"]), "the subject was resampled"
        assert np.array_equal(
            upgraded["pos_idx_0"], arrays["pos_idx_0"]
        ), "the backfilled indices differ from the originally computed ones"
        item, _ = second[0]
        assert {"xyz", "gt_sdf"} <= set(item)

    def test_the_multi_class_backfills_missing_index_lists(self, meshes, tmp_path_factory):
        """The working counterpart of the xfail above, pinned so it stays working."""
        cache = tmp_path_factory.mktemp("backfill_multi")
        first = build_dataset(meshes, cache, **SMALL)
        path = first.data[0]
        arrays = dict(np.load(path))
        stripped = {
            k: v for k, v in arrays.items() if not k.startswith(("pos_idx", "neg_idx", "surf_idx"))
        }
        np.savez(path, **stripped)

        second = build_dataset(meshes, cache, load_cache=True, **SMALL)

        upgraded = dict(np.load(path))
        assert np.array_equal(upgraded["pts"], arrays["pts"]), "the subject was resampled"
        for key in ("pos_idx_0", "pos_idx_1", "neg_idx_0", "neg_idx_1"):
            assert np.array_equal(upgraded[key], arrays[key]), key
        item, _ = second[0]
        assert {"xyz", "gt_sdf"} <= set(item)

    def test_a_disk_built_cache_reloads_into_either_storage_mode(self, meshes, tmp_path_factory):
        """
        ``store_data_in_memory`` is a serving choice, not a property of the cache: the
        same ``.npz`` serves a disk-backed dataset (``data`` holds the path) and an
        in-memory one (``data`` holds the unpacked dict), and the batches drawn from
        the two are identical.
        """
        import torch

        cache = tmp_path_factory.mktemp("store_modes")
        disk = build_dataset(meshes, cache, **SMALL)
        memory = build_dataset(meshes, cache, load_cache=True, store_data_in_memory=True, **SMALL)

        assert isinstance(disk.data[0], str)
        assert isinstance(memory.data[0], dict)

        torch.manual_seed(0)
        from_disk, _ = disk[0]
        torch.manual_seed(0)
        from_memory, _ = memory[0]
        assert torch.equal(from_disk["xyz"], from_memory["xyz"])
        assert torch.equal(from_disk["gt_sdf"], from_memory["gt_sdf"])

    def test_every_subject_started_is_logged(self, tmp_path_factory):
        """
        ``MultiSurfaceSDFSamples.get_sample_data_dict`` appends each subject to
        ``list_meshes_started_loading.log`` in ``loc_save`` before doing anything else,
        so a crash mid-build names its subject. The log appends across builds, cache
        hits included.
        """
        subjects = write_synthetic_meshes(tmp_path_factory.mktemp("logged_meshes"))[:2]
        cache = tmp_path_factory.mktemp("logged")
        build_dataset(subjects, cache, **SMALL)

        log = os.path.join(str(cache), "list_meshes_started_loading.log")
        assert os.path.exists(log)
        with open(log, encoding="utf-8") as f:
            lines = f.read().splitlines()
        assert lines == [str(subject) for subject in subjects]

        build_dataset(subjects, cache, load_cache=True, **SMALL)
        with open(log, encoding="utf-8") as f:
            assert len(f.read().splitlines()) == 2 * len(subjects)


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
        The seed changes the samples, so it has to change the key -- otherwise two seeds
        would share one cached file. See ``TestSeeding``.
        """
        assert rehash(dataset, meshes[0], random_seed=7) != dataset.create_hash(meshes[0])


class TestFormerlyCollidingParameters:
    """
    ``mesh_to_scale`` and ``uniform_pts_buffer`` change what is written into the cache
    and were absent from ``get_hash_params`` until Aug 2026 (#19 (a)): two runs
    differing only in one of them shared a key, and with ``load_cache=True`` -- the
    production setting -- the second silently trained on the first's data. Each test
    still shows the cached content genuinely differs before asserting the keys do, so
    a parameter that stops mattering shows up as a dead premise rather than a vacuous
    pass. ``subsample`` is the deliberate exception: it stays OUT of the key, because
    its only cached-content effect is the index padding (see the xfails below, resolved
    by decoupling rather than keying).
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

    def test_mesh_to_scale_must_change_the_cache_key(self, meshes, tmp_path_factory):
        """
        The worst of them: ``mesh_to_scale`` decides which surface drives centering
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

    @pytest.mark.xfail(strict=True, reason="#19: get_hash_params omits subsample")
    def test_subsample_must_change_the_cache_key(self, meshes, tmp_path_factory):
        """
        Milder but still real: ``subsample`` sets ``samples_per_sign_``, which decides how
        many times ``sdf_pos_neg_idx`` repeats the index arrays -- and those arrays are
        cached. The points themselves are unaffected.

        The repeat count is ``samples_per_sign // available + 1``, so the two subsamples
        have to straddle a multiple of the number of samples of that sign for the arrays to
        differ at all. Near-surface sampling leaves ~240 negatives per surface here, so 64
        vs 512 both round to a repeat of 1 and the premise assertion below would go off.
        """
        key_a, key_b, differing = self._content_differs(
            meshes, tmp_path_factory, "sub", subsample=2048
        )
        assert any(
            key.startswith(("pos_idx", "neg_idx")) for key in differing
        ), f"premise gone: index arrays no longer differ ({differing})"
        assert key_a != key_b, "the cached index arrays differ but the cache key does not"

    def test_a_changed_parameter_must_not_reuse_the_previous_runs_cache(
        self, meshes, tmp_path_factory
    ):
        """End to end: same cache directory, ``load_cache=True``, only ``mesh_to_scale`` changed."""
        cache = tmp_path_factory.mktemp("collision")
        first = build_dataset(meshes, cache, **SMALL)
        second = build_dataset(meshes, cache, load_cache=True, mesh_to_scale=1, **SMALL)

        assert second.data[0] != first.data[0], "the second run was handed the first run's file"

    @pytest.mark.xfail(
        strict=True, reason="#19: a subsample collision silently unbalances the batch"
    )
    def test_equal_pos_neg_must_hold_after_a_subsample_change(self, meshes, tmp_path_factory):
        """
        What the ``subsample`` collision costs, measured.

        ``sdf_pos_neg_idx`` repeats the negative-index array just far enough for the
        ``subsample`` in force when the cache was written. Reload with a larger one and
        there are not enough entries: ``MultiSurfaceSDFSamples.__getitem__`` takes what
        there is, then tops the batch up with uniformly random points. The
        ``equal_pos_neg=True`` guarantee quietly stops holding, and the surface with the
        fewest interior samples -- the small one, i.e. cartilage in a real dataset -- loses
        the most.

        Measured at 1.6x under-representation (interior fraction 0.20 against a fresh
        0.32), and the gap only opens once the reloaded ``subsample`` exceeds the cached
        point count. On the uniform sampling path this harness used to run on it was 4.4x
        at a far smaller subsample, because uniform points rarely land inside a small
        ellipsoid; near-surface sampling puts ~20% of them inside, which is both more
        realistic and a much softer landing for this bug.

        The reload check that would have caught this
        (``MultiSurfaceSDFSamples.get_sample_data_dict``) compares ``len(data["pos_idx"])``
        against the number of *meshes*, never against the subsample the arrays were
        built for.
        """
        import torch

        base = {k: v for k, v in SMALL.items() if k != "subsample"}
        cache = tmp_path_factory.mktemp("subsample_collision")
        build_dataset(meshes, cache, subsample=64, **base)
        reused = build_dataset(meshes, cache, load_cache=True, subsample=4096, **base)
        fresh = build_dataset(
            meshes, tmp_path_factory.mktemp("subsample_fresh"), subsample=4096, **base
        )

        def interior_fraction(dataset, surface):
            torch.manual_seed(0)
            item, _ = dataset[0]
            return (item["gt_sdf"][:, surface] < 0).float().mean().item()

        # Surface 1 is the small ellipsoid: fewest interior points, so it is hit hardest.
        assert interior_fraction(reused, 1) == pytest.approx(interior_fraction(fresh, 1), rel=0.25)


class TestMeshContentInTheKey:
    """
    The cache key notices when a mesh file's *content* changes, not only its path
    (#19 (b), fixed Aug 2026): each path contributes ``(path, size, mtime)``, so an
    in-place edit moves the key without any file being read.
    """

    def test_an_in_place_mesh_edit_must_change_the_key(self, tmp_path_factory):
        """
        Overwrite a subject's mesh at the same path with different geometry: the stale
        cached samples must not be served, so the key has to move. Until Aug 2026 the
        key hashed the path string alone and stood still through any edit.
        """
        import pyvista as pv

        subject = write_synthetic_meshes(tmp_path_factory.mktemp("editable"))[:1]
        dataset = build_dataset(subject, tmp_path_factory.mktemp("edit_cache"), **SMALL)
        key_before = dataset.create_hash(subject[0])

        edited = pv.Sphere(radius=0.5, theta_resolution=30, phi_resolution=30).triangulate()
        edited.save(subject[0][0])

        assert dataset.create_hash(subject[0]) != key_before


class TestReferenceMeshHashing:
    """
    A ``reference_mesh`` passed as a ``Mesh`` object contributes a digest of its
    geometry to the key (#19 (c), fixed Aug 2026). Until then it was stringified, and
    ``Mesh.__str__`` includes the memory address -- the key was per-object, so a
    dataset with a ``Mesh`` reference could never hit its own cache.
    """

    def test_two_equal_mesh_objects_must_hash_the_same(self, dataset, meshes):
        """Same geometry, same file, two objects -- the cache key must not care."""
        from pymskt.mesh import Mesh

        one, two = Mesh(meshes[0][0]), Mesh(meshes[0][0])
        assert rehash(dataset, meshes[0], reference_mesh=one) == rehash(
            dataset, meshes[0], reference_mesh=two
        )

    def test_a_path_string_hashes_stably(self, dataset, meshes):
        """The same reference given as a path is stable -- formerly the only workaround."""
        assert rehash(dataset, meshes[0], reference_mesh=meshes[0][0]) == rehash(
            dataset, meshes[0], reference_mesh=meshes[0][0]
        )


class TestSeeding:
    """
    What ``SDFSamples(random_seed=...)`` reproduces, and what it deliberately does not.

    A seed makes both sampling paths reproducible from cold, which is what the rest of this
    harness is built on -- every baselined number comes from a seeded near-surface dataset.
    ``random_seed=None`` is the other half of the contract: it leaves sampling on the legacy
    global numpy stream, so old callers keep getting old numbers.
    """

    def test_the_uniform_path_is_reproducible_under_a_numpy_seed(self, meshes, tmp_path_factory):
        """
        The compatibility contract, not a leftover: with ``random_seed=None`` the uniform
        path still draws through ``np.random.uniform``, i.e. the legacy global stream, so a
        caller who seeds numpy and passes no ``random_seed`` gets exactly the numbers they
        always did. Routing the unseeded path through ``default_rng`` instead would be a
        different stream and would silently change every such caller's data.
        """
        uniform = dict(SMALL, sigma_near=[None, None], sigma_far=[None, None], random_seed=None)
        a = build_dataset(meshes, tmp_path_factory.mktemp("seed_u_a"), seed=7, **uniform)
        b = build_dataset(meshes, tmp_path_factory.mktemp("seed_u_b"), seed=7, **uniform)
        assert np.array_equal(cached_arrays(a)["pts"], cached_arrays(b)["pts"])

    def test_the_near_surface_path_must_be_reproducible(self, meshes, tmp_path_factory):
        """
        The path production uses. It goes through ``pymskt.Mesh.rand_pts_around_surface``,
        which has two independent draws -- the base surface points from
        ``pcu.sample_mesh_random`` and the perturbation offsets from a ``default_rng`` --
        and both take the seed NSM derives for that surface. Neither was reachable from
        NSM before pymskt 0.1.21, which is why this used to be an ``xfail``.
        """
        near = dict(SMALL, sigma_near=[0.05, 0.05], sigma_far=[0.2, 0.2])
        a = build_dataset(meshes, tmp_path_factory.mktemp("seed_n_a"), seed=7, **near)
        b = build_dataset(meshes, tmp_path_factory.mktemp("seed_n_b"), seed=7, **near)
        assert np.array_equal(cached_arrays(a)["pts"], cached_arrays(b)["pts"])

    def test_random_seed_must_make_a_cold_run_reproducible(self, meshes, tmp_path_factory):
        """The same seed against two *cold* caches gives the same samples."""
        near = dict(SMALL, sigma_near=[0.05, 0.05], sigma_far=[0.2, 0.2], random_seed=1234)
        a = build_dataset(meshes, tmp_path_factory.mktemp("seed_cold_a"), **near)
        b = build_dataset(meshes, tmp_path_factory.mktemp("seed_cold_b"), **near)

        assert np.array_equal(cached_arrays(a)["pts"], cached_arrays(b)["pts"])


#: Builds one dataset in a fresh interpreter: ``sys.argv[1]`` is the cache directory,
#: ``sys.argv[2]`` is "1" for ``multiprocessing=True``, ``sys.argv[3]`` is a mesh
#: directory the CALLER has already populated. Both invocations reuse those files
#: unmodified -- rewriting them per invocation would move their ``(path, size, mtime)``
#: identity and with it the cache key -- so the two builds produce the same cache
#: *filenames*, which is what lets the caller pair them up.
_BUILD_IN_SUBPROCESS = f"""
import glob
import os
import sys
sys.path.insert(0, {os.path.dirname(os.path.abspath(__file__))!r})
from _harness import build_dataset

bones = sorted(glob.glob(os.path.join(sys.argv[3], "*_bone.vtk")))
build_dataset(
    [[bone, bone.replace("_bone.vtk", "_cart.vtk")] for bone in bones],
    sys.argv[1],
    random_seed=1234,
    multiprocessing=sys.argv[2] == "1",
    n_processes=2,
    **{SMALL!r},
)
"""


def _cached_by_name(cache_dir):
    """``{basename: path}`` for every ``.npz`` under a cache directory."""
    found = {}
    for root, _, names in os.walk(cache_dir):
        for name in names:
            if name.endswith(".npz"):
                found[name] = os.path.join(root, name)
    return found


class TestSeedDerivation:
    """
    The per-draw seed derivation, pinned property by property.

    ``derive_seed`` hands every (subject, sampling-combo, surface) its own seed, derived
    from the run seed and the *bytes of the subject's meshes*. All five of the properties
    below are silent when they break -- the data still looks like data -- so each says what
    a reader loses if it stops holding.
    """

    def test_different_seeds_give_different_data(self, meshes, tmp_path_factory):
        """
        If this fails the seed is not reaching the sampler at all, and every "reproducible"
        claim here is really just a cache hit.
        """
        a = build_dataset(meshes, tmp_path_factory.mktemp("derive_1234"), random_seed=1234, **SMALL)
        b = build_dataset(meshes, tmp_path_factory.mktemp("derive_5678"), random_seed=5678, **SMALL)
        assert not np.array_equal(cached_arrays(a)["pts"], cached_arrays(b)["pts"])

    def test_the_two_sampling_combos_draw_different_points(self, meshes, tmp_path_factory):
        """
        Ask for the near and far passes with identical parameters -- same sigma, same
        count -- and they must still produce different points.

        ``rand_pts_around_surface`` picks base points on the surface and then perturbs
        them, so one seed shared across the two combos means both passes perturb the *same*
        base points. The dataset would then carry half as many distinct surface locations
        as it appears to, at every sigma, and nothing downstream would notice.
        """
        identical = dict(
            SMALL,
            sigma_near=[0.02, 0.02],
            sigma_far=[0.02, 0.02],
            p_near_surface=[0.4, 0.4],
            p_further_from_surface=[0.4, 0.4],
            random_seed=99,
        )
        dataset = build_dataset(meshes, tmp_path_factory.mktemp("combos"), **identical)

        # pt_sample_combos is [near, far, uniform]; each contributes sum(n_pts) points to
        # the front of `pts`, in order.
        near_count, far_count = (sum(combo[0]) for combo in dataset.pt_sample_combos[:2])
        assert near_count == far_count, "the two combos must be the same size to compare"
        points = cached_arrays(dataset)["pts"]
        near, far = points[:near_count], points[near_count : near_count + far_count]

        assert not np.array_equal(near, far)

    def test_the_mesh_list_order_does_not_change_a_subjects_data(self, tmp_path_factory):
        """
        Reverse ``list_mesh_paths`` and every subject must keep its own samples.

        This is why the derivation is keyed on the mesh contents and not on ``enumerate``'s
        index, and it is the property most likely to be "simplified" back out: an index is
        right there in the loop. Keyed positionally, adding one subject to the front of a
        training list would resample every other subject -- while their cache keys, and so
        their cached files, stayed valid.
        """
        two = write_synthetic_meshes(tmp_path_factory.mktemp("order_meshes"))[:2]
        forward = build_dataset(two, tmp_path_factory.mktemp("order_fwd"), random_seed=321, **SMALL)
        reverse = build_dataset(
            list(reversed(two)), tmp_path_factory.mktemp("order_rev"), random_seed=321, **SMALL
        )

        for index in range(2):
            mine = cached_arrays(forward, index)["pts"]
            counterpart = cached_arrays(reverse, 1 - index)["pts"]
            positional = cached_arrays(reverse, index)["pts"]
            assert np.array_equal(mine, counterpart), f"subject {index} was resampled"
            assert not np.array_equal(mine, positional), (
                f"subject {index} matches whatever is at its position, so this test cannot "
                f"tell the two derivations apart"
            )

    def test_moving_the_meshes_does_not_change_the_data(self, tmp_path_factory):
        """
        The same mesh bytes at two different absolute paths, same ``random_seed``, must
        sample identically.

        The two cache *keys* differ -- the path is still hashed into them -- so the second
        build is a genuine cold resample that happens to land on the same answer, not a
        cache hit. That is the whole point of keying the seed on contents: the seed decides
        which points get drawn, so relocating a dataset must not silently redraw it.
        """
        original = write_synthetic_meshes(tmp_path_factory.mktemp("here"))[:1]
        moved = write_synthetic_meshes(tmp_path_factory.mktemp("there"))[:1]
        assert original[0] != moved[0], "the two copies must be at different paths"

        near = dict(SMALL, random_seed=4242)
        a = build_dataset(original, tmp_path_factory.mktemp("moved_a"), **near)
        b = build_dataset(moved, tmp_path_factory.mktemp("moved_b"), **near)

        assert a.create_hash(original[0]) != b.create_hash(moved[0]), "cache keys must differ"
        assert np.array_equal(cached_arrays(a)["pts"], cached_arrays(b)["pts"])

    def test_multiprocessing_does_not_change_the_data(self, tmp_path_factory):
        """
        ``multiprocessing=True`` must produce the same cache as ``multiprocessing=False``.

        ``Pool`` forks, so every worker inherits one copy of the parent's global numpy
        state. Before the seed was threaded through, that state was the only thing driving
        the sampler and the pooled build reproduced none of the serial one -- all three
        subjects differed. That path is still live: rerun this with ``random_seed=None``
        and the same three comparisons come back ``[False, False, False]``.

        Both datasets are built in *separate* processes. Building one in-process and
        forking for the other hangs -- a pre-existing fork-after-VTK hazard, unrelated to
        seeding.
        """
        mesh_dir = str(tmp_path_factory.mktemp("mp_meshes"))
        write_synthetic_meshes(mesh_dir)
        caches = [str(tmp_path_factory.mktemp("mp_off")), str(tmp_path_factory.mktemp("mp_on"))]
        for cache, flag in zip(caches, ("0", "1")):
            finished = subprocess.run(
                [sys.executable, "-c", _BUILD_IN_SUBPROCESS, cache, flag, mesh_dir],
                capture_output=True,
                text=True,
                timeout=600,
            )
            assert finished.returncode == 0, finished.stderr[-2000:]

        serial, parallel = (_cached_by_name(cache) for cache in caches)
        assert sorted(serial) == sorted(parallel) and len(serial) == 3, (serial, parallel)
        for name in sorted(serial):
            assert np.array_equal(
                np.load(serial[name])["pts"], np.load(parallel[name])["pts"]
            ), f"{name} differs between the serial and pooled builds"


@pytest.fixture(scope="module")
def bone_meshes(tmp_path_factory):
    """``[bone, bone]`` -- single paths, not pairs, which is what ``SDFSamples`` takes."""
    pairs = write_synthetic_meshes(tmp_path_factory.mktemp("single_meshes"))[:2]
    return [pair[0] for pair in pairs]


class TestSingleSurfaceSDFSamples:
    """
    The same seeding contract, on the single-surface PARENT class.

    ``SDFSamples`` is not ``MultiSurfaceSDFSamples`` with one surface: it has its own
    ``get_sample_data_dict``, ``get_pt_sample_combos`` and ``__getitem__``, and its own
    call to ``read_mesh_get_sampled_pts`` -- the *other* sampler, not the one the subclass
    uses. Nothing ``TestSeeding`` and ``TestSeedDerivation`` establish above carries over
    to any of it, and until this class existed nothing in ``testing/`` constructed an
    ``SDFSamples`` at all.

    The four properties below are the subclass's, restated. They are the ones that would
    let a seeded run silently stop being reproducible.
    """

    @pytest.fixture(scope="class")
    def dataset(self, bone_meshes, tmp_path_factory):
        return build_single_surface_dataset(
            bone_meshes, tmp_path_factory.mktemp("single_build"), **SMALL_SINGLE
        )

    @pytest.fixture(scope="class")
    def seeded_pair(self, bone_meshes, tmp_path_factory):
        """The same ``random_seed`` against two *cold*, separate caches."""
        return [
            build_single_surface_dataset(
                bone_meshes, tmp_path_factory.mktemp(f"single_seeded_{label}"), **SMALL_SINGLE
            )
            for label in ("a", "b")
        ]

    def test_it_builds_and_caches_one_file_per_subject(self, dataset, bone_meshes):
        assert len(dataset.data) == len(bone_meshes)
        for path in dataset.data:
            assert os.path.exists(path) and path.endswith(".npz")

    def test_the_scalar_n_pts_is_the_point_count_that_lands_in_the_cache(self, dataset):
        """
        The parent's ``get_sample_data_dict`` preallocates ``data["xyz"]`` with the
        scalar ``self.n_pts`` where the subclass uses ``sum(self.n_pts)`` over its
        per-surface list. Both are right for their own class; this pins that the scalar
        one is, so a later attempt to unify the two cannot quietly truncate this path.
        """
        assert cached_arrays(dataset)["pts"].shape == (SMALL_SINGLE["n_pts"], 3)

    def test_the_same_seed_reproduces_a_cold_run(self, seeded_pair):
        """
        Two separate cache directories, so neither run can be reading the other's file --
        the warm-cache illusion that hid the unseeded sampler for as long as it did.
        """
        first, second = seeded_pair
        assert first.data[0] != second.data[0], "the two runs shared a cache file"
        for index in range(len(first.data)):
            assert np.array_equal(
                cached_arrays(first, index)["pts"], cached_arrays(second, index)["pts"]
            ), f"subject {index} did not reproduce"

    def test_a_different_seed_gives_different_points(
        self, seeded_pair, bone_meshes, tmp_path_factory
    ):
        """The guard on the test above: without this, "reproducible" could just mean inert."""
        other = build_single_surface_dataset(
            bone_meshes, tmp_path_factory.mktemp("single_other_seed"), seed=5678, **SMALL_SINGLE
        )
        assert not np.array_equal(cached_arrays(seeded_pair[0])["pts"], cached_arrays(other)["pts"])

    def test_an_unseeded_run_is_not_reproducible(self, bone_meshes, tmp_path_factory):
        """
        ``random_seed=None`` must stay unseeded. Both runs get the same ``np.random.seed``
        and still differ, which is the point: on the near-surface path the draw happens
        inside ``pymskt.Mesh.rand_pts_around_surface``, off the global stream, so
        ``random_seed`` is the only thing that can make it reproducible.
        """
        unseeded = [
            build_single_surface_dataset(
                bone_meshes,
                tmp_path_factory.mktemp(f"single_unseeded_{label}"),
                random_seed=None,
                **SMALL_SINGLE,
            )
            for label in ("a", "b")
        ]
        assert not np.array_equal(
            cached_arrays(unseeded[0])["pts"], cached_arrays(unseeded[1])["pts"]
        )


class TestFormerlyUncallableConfigurations:
    """
    Advertised constructor arguments that used to build fine and crash on first use.
    Each test asserts the option now works; the crashes they replace were #22 and #23,
    fixed Aug 2026.
    """

    def test_zero_sampling_probability_samples_nothing(self, meshes, tmp_path_factory):
        """
        ``get_pt_sample_combos`` emits a ``[0, sigma]`` combo when a probability is 0,
        and ``get_sample_data_dict`` now skips it (#23) instead of handing
        ``point_cloud_utils`` an empty point cloud to crash on. The remaining combos
        still fill the whole preallocated buffer -- the random share absorbs what the
        probabilities leave over, so nothing is silently left at zero.
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
        arrays = cached_arrays(dataset)
        # A skipped combo must not leave a hole of never-written rows in the buffer.
        assert not np.any(np.all(arrays["pts"] == 0, axis=1))
        item, _ = dataset[0]
        assert {"xyz", "gt_sdf"} <= set(item)

    def test_zero_probability_on_the_single_surface_class(self, bone_meshes, tmp_path_factory):
        """``SDFSamples.get_sample_data_dict`` is separate code from the subclass's."""
        dataset = build_single_surface_dataset(
            bone_meshes[:1],
            tmp_path_factory.mktemp("p_zero_single"),
            p_near_surface=0.0,
            p_further_from_surface=0.5,
            **SMALL_SINGLE,
        )
        item, _ = dataset[0]
        assert {"xyz", "gt_sdf"} <= set(item)

    def test_store_data_in_memory_yields_an_item(self, meshes, tmp_path_factory):
        """
        ``MultiSurfaceSDFSamples.__getitem__`` read ``time_`` and ``size``, which are only
        bound when a disk load happened, so ``store_data_in_memory=True`` raised
        ``UnboundLocalError`` (#22). It now carries the same guard as the single-surface
        ``SDFSamples.__getitem__`` always did: timing keys are emitted only when a load
        was actually timed.
        """
        dataset = build_dataset(
            meshes, tmp_path_factory.mktemp("in_memory"), store_data_in_memory=True, **SMALL
        )
        item, _ = dataset[0]
        assert set(item) == {"xyz", "gt_sdf"}

    @pytest.fixture(scope="class")
    def timing_free_dataset(self, meshes, tmp_path_factory):
        """In memory with load timing off -- formerly the half-workaround for #22."""
        return build_dataset(
            meshes,
            tmp_path_factory.mktemp("in_memory_ok"),
            store_data_in_memory=True,
            test_load_times=False,
            **SMALL,
        )

    def test_store_data_in_memory_works_with_load_timing_off(self, timing_free_dataset):
        item, index = timing_free_dataset[0]
        assert set(item) == {"xyz", "gt_sdf"} and index == 0

    def test_the_trainer_consumes_batches_without_timing_keys(
        self, timing_free_dataset, tmp_path_factory
    ):
        """
        ``train_epoch`` used to read all four load-timing keys unconditionally, which is
        what made #22 unfixable by the dataset guard alone: the combination that avoided
        the crash produced batches the trainer could not consume, so no combination of
        the two flags both constructed and trained. The keys are now optional
        diagnostics on both sides -- emitted only when a disk load was timed, accumulated
        and logged only when present.

        Asserted by running the trainer rather than by grepping its source: a grep for
        ``sdf_data["size"]`` lies in both directions -- red on a harmless rename, green
        on an unguarded read that crashes.

        ``samples_per_object_per_batch`` has to follow ``SMALL``'s ``subsample``: mismatch
        them and the run dies at the batch concatenation, several steps before the reads
        this is about.
        """
        config = training_config(tmp_path_factory.mktemp("in_memory_train"))
        config.update(
            {
                "n_epochs": 1,
                "checkpoint_epochs": 1,
                "save_frequency": 1,
                "samples_per_object_per_batch": SMALL["subsample"],
            }
        )
        records, _ = run_training(config, build_model(config), timing_free_dataset)
        assert len(records) == 1
        assert "loss" in records[0]


class TestScaleJointlyInMemory:
    """
    ``scale_jointly=True`` with ``store_data_in_memory=True`` never constructed before
    the #69 fix: the in-memory branch of ``norm_and_scale_all_meshes`` read the
    flattened ``new_pts_0``-style keys that exist only in the ``.npz`` cache layout,
    and it also omitted ``joint_scale_buffer``. Until the fix this was a strict-xfail
    pin whose ``raises=KeyError`` made a KeyError-only half-fix a plain failure; the
    body asserts the buffered domain, so it now guards both halves of the fix: both
    storage modes compute the shared frame and ``__getitem__`` applies it per batch.
    """

    def test_an_in_memory_dataset_lands_inside_the_buffered_domain(self, meshes, tmp_path_factory):
        """
        ``joint_scale_buffer=9`` makes the shared scale 10x the observed max radius, so
        every batch coordinate lands within ~0.1-0.2 of the origin; an unbuffered
        scaling leaves the near-surface points at radius ~0.5-1.1. The 0.25 threshold
        sits severalfold clear of both, on any draw.
        """
        dataset = build_dataset(
            meshes,
            tmp_path_factory.mktemp("joint_mem"),
            center_pts=False,
            norm_pts=False,
            scale_jointly=True,
            joint_scale_buffer=9.0,
            store_data_in_memory=True,
            **SMALL,
        )
        item, _ = dataset[0]
        assert item["xyz"].norm(dim=1).max() < 0.25


class TestCacheLocationDefault:
    """
    ``loc_save=None`` resolves ``LOC_SDF_CACHE`` when the dataset is CONSTRUCTED. Until
    Aug 2026 the environment read was a default argument, evaluated once at import (#24),
    so setting the variable afterwards had no effect and the cache silently went to
    ``~/.cache/nsm_sdf_cache``. The harness still passes ``loc_save`` explicitly
    everywhere else so its tests can never depend on the developer's environment.
    """

    def test_setting_the_env_var_changes_where_the_cache_goes(
        self, meshes, monkeypatch, tmp_path_factory
    ):
        cache_root = tmp_path_factory.mktemp("env_cache")
        monkeypatch.setenv("LOC_SDF_CACHE", str(cache_root))
        dataset = build_dataset(meshes, "ignored-by-override", loc_save=None, **SMALL)
        assert dataset.loc_save == str(cache_root)
        assert dataset.data[0].startswith(str(cache_root))

    def test_a_blank_env_var_counts_as_unset(self, meshes, monkeypatch, tmp_path_factory):
        """
        The downstream consumer blanks the variable rather than unsetting it
        (``kneepipeline/steps/run_nsm.py``), and ``""`` must mean the home default: a
        literally-empty ``loc_save`` would root the cache -- and ``find_hash``'s
        recursive walk -- at the current working directory.
        """
        fake_home = tmp_path_factory.mktemp("fake_home")
        monkeypatch.setenv("HOME", str(fake_home))
        monkeypatch.setenv("LOC_SDF_CACHE", "")
        dataset = build_dataset(meshes, "ignored-by-override", loc_save=None, **SMALL)
        assert dataset.loc_save == os.path.join(str(fake_home), ".cache", "nsm_sdf_cache")


class TestPointCenteringAndScaling:
    """
    ``get_pts_center_and_scale`` is the normalization every cached sample goes through.
    Two of its documented behaviours are not its behaviours.
    """

    def test_center_and_scale_are_not_accepted_as_arguments(self):
        """
        They were removed rather than honoured, so this asserts they are gone.

        Both were shadowed by the values computed from them before they were read, so
        neither had any effect at any value. Honouring them instead would have been the
        harmful fix: every caller passes ``scale=norm_pts``, which defaults to ``False``
        at all four definition sites and is unset in the shipped configs, so an
        authoritative argument would stop scaling on a default run -- measured, a point
        cloud of max radius 24.95 stays at 24.95 instead of normalizing to 1.0. That
        changes the coordinate frame of every dataset, checkpoint and reconstruction
        NSM has ever produced. See #20.
        """
        from NSM.datasets.sdf_dataset import get_pts_center_and_scale

        taken = inspect.signature(get_pts_center_and_scale).parameters
        assert "center" not in taken
        assert "scale" not in taken

    def test_centering_and_scaling_still_happen_unconditionally(self):
        """
        The behaviour the removal must preserve: both operations always run.

        This is the half that goes red if someone reinstates the arguments and wires
        them up, because the defaults would then switch scaling off.
        """
        from NSM.datasets.sdf_dataset import get_pts_center_and_scale

        points = np.array([[1.0, 1.0, 1.0], [3.0, 3.0, 3.0]])
        center, scale, normalized = get_pts_center_and_scale(points, return_pts=True)

        assert np.allclose(center, [2.0, 2.0, 2.0]), "centering did not happen"
        assert np.isclose(
            np.max(np.linalg.norm(normalized, axis=-1)), 1.0
        ), "scaling did not happen"

    def test_the_callers_array_must_not_be_mutated(self):
        """
        ``pts -= center`` used to write through to the caller's array. All three in-repo
        call sites carried a defensive ``np.copy(...)``; the copy now lives inside the
        function, where a fourth caller gets it for free. See #21.
        """
        from NSM.datasets.sdf_dataset import get_pts_center_and_scale

        points = np.array([[1.0, 1.0, 1.0], [3.0, 3.0, 3.0]])
        get_pts_center_and_scale(points)
        assert np.allclose(points, [[1.0, 1.0, 1.0], [3.0, 3.0, 3.0]])


class TestUniformSamplingCube:
    """
    The uniform-sampling cube the two samplers draw from when ``sigma`` is None.

    The single- and multi-mesh samplers carried private copies of the cube arithmetic and
    they had diverged (#40, fixed Aug 2026): in both, ``mins`` was rebound before ``maxs``
    read it, so a nonzero ``uniform_pts_buffer`` grew the cube more above than below; and
    only the single-mesh copy clipped its draws, to +/-(1 + buffer/2), piling the
    truncated samples onto the clip faces. Both now share
    ``get_buffered_cube_mins_maxs`` and neither clips.
    """

    def test_the_buffer_expands_the_cube_symmetrically(self):
        from NSM.datasets.sdf_dataset import get_buffered_cube_mins_maxs, get_cube_mins_maxs

        rng = np.random.default_rng(0)
        pts = rng.normal(size=(500, 3)) + [5.0, -2.0, 0.5]
        mins0, maxs0 = get_cube_mins_maxs(pts)
        mins, maxs = get_buffered_cube_mins_maxs(pts, 0.5)

        assert np.allclose((mins + maxs) / 2, (mins0 + maxs0) / 2), "the centre moved"
        assert np.allclose(maxs - mins, 1.5 * (maxs0 - mins0)), "span must grow by 1+buffer"

    def test_the_two_samplers_draw_from_the_same_cube(self, meshes):
        """
        Same mesh, same buffer, uniform path: a normalized mesh spans a +/-1 cube, and
        ``uniform_pts_buffer=0.5`` widens it to +/-1.5 in both samplers. Before the fix
        the single-mesh draw was clipped to +/-1.25 and the multi-mesh one spanned
        -1.50/+1.56 -- so each bound assertion below fails against one of the two old
        behaviours.
        """
        from NSM.datasets.sdf_dataset import (
            read_mesh_get_sampled_pts,
            read_meshes_get_sampled_pts,
        )

        path = meshes[0][0]
        kwargs = dict(center_pts=True, norm_pts=True, fix_mesh=False, get_random=True)
        with quiet():
            single = read_mesh_get_sampled_pts(
                path, sigma=None, n_pts=4000, uniform_pts_buffer=0.5, seed=0, **kwargs
            )
            multi = read_meshes_get_sampled_pts(
                [path], sigma=[None], n_pts=[4000], uniform_pts_buffer=0.5, seed=0, **kwargs
            )

        for label, pts in (("single", single["pts"]), ("multi", multi["pts"])):
            assert pts.min() >= -1.5 and pts.max() <= 1.5, f"{label}: cube too large"
            assert pts.max() > 1.4 and pts.min() < -1.4, f"{label}: cube did not reach its bounds"
            assert abs(pts.max() + pts.min()) < 0.1, f"{label}: cube is asymmetric"

    def test_pts_surface_return_types_match(self, meshes):
        """
        ``pts_surface`` was a Python list from the single-mesh sampler and an int64 array
        from the multi-mesh one -- the last of #40's three divergences.
        """
        from NSM.datasets.sdf_dataset import (
            read_mesh_get_sampled_pts,
            read_meshes_get_sampled_pts,
        )

        path = meshes[0][0]
        kwargs = dict(center_pts=True, norm_pts=True, fix_mesh=False, get_random=True)
        with quiet():
            single = read_mesh_get_sampled_pts(path, sigma=0.05, n_pts=200, seed=0, **kwargs)
            multi = read_meshes_get_sampled_pts([path], sigma=[0.05], n_pts=[200], seed=0, **kwargs)

        for label, result in (("single", single), ("multi", multi)):
            surface = result["pts_surface"]
            assert isinstance(surface, np.ndarray), label
            assert surface.dtype == np.int64, label
            assert surface.shape == (200,), label


class TestEmptySignedSamples:
    """
    ``sdf_pos_neg_idx`` divided by zero whenever a surface had no positive or no negative
    samples (#41, fixed Aug 2026). Now: a surface nothing draws from -- missing (None), or
    allotted no subsample share -- yields empty index lists and is handled; a drawn-from
    surface missing a sign raises a ``ValueError`` naming the surface.
    """

    def test_a_nested_surface_raises_a_named_error(self, tmp_path_factory):
        """
        One surface inside another loses every interior point to
        ``remove_overlapping_points``, leaving it with no negative samples. The harness's
        synthetic subjects are built disjoint precisely to stay clear of this
        (``_harness.SUBJECTS``); here the nesting is deliberate.
        """
        import pyvista as pv

        directory = tmp_path_factory.mktemp("nested_meshes")
        outer = pv.Sphere(radius=1.0, theta_resolution=24, phi_resolution=24).triangulate()
        inner = pv.Sphere(radius=0.4, theta_resolution=24, phi_resolution=24).triangulate()
        outer_path = os.path.join(str(directory), "outer.vtk")
        inner_path = os.path.join(str(directory), "inner.vtk")
        outer.save(outer_path)
        inner.save(inner_path)

        with pytest.raises(ValueError, match="Surface 1 has no negative"):
            build_dataset(
                [[outer_path, inner_path]], tmp_path_factory.mktemp("nested_cache"), **SMALL
            )

    def test_a_missing_surface_is_handled_as_empty(self, dataset):
        """
        An all-NaN SDF column is a missing (None) surface -- ``read_meshes`` fills the
        column with NaN for a ``None`` path. Empty index lists are the contract:
        ``__getitem__``'s ``randperm(0)`` draws nothing from them.

        A direct method call, because the end-to-end None-surface path currently dies
        earlier, at ``get_sample_data_dict``'s preallocated buffer write -- a separate
        defect from this one (#67).
        """
        import torch

        gt_sdf = torch.stack(
            [torch.linspace(-1.0, 1.0, 10), torch.full((10,), float("nan"))], dim=1
        )
        pos, neg, surf = dataset.sdf_pos_neg_idx({"gt_sdf": gt_sdf, "xyz": torch.zeros(10, 3)})

        assert pos[0].numel() > 0 and neg[0].numel() > 0
        assert pos[1].numel() == 0 and neg[1].numel() == 0 and surf[1].numel() == 0

    @pytest.mark.xfail(
        strict=True, reason="#67: a None surface dies at the preallocated buffer write"
    )
    def test_a_none_surface_subject_must_build(self, meshes, tmp_path_factory):
        """
        The fdfe902 feature: a subject may be missing a structure. The build currently
        dies in ``get_sample_data_dict`` -- ``data["xyz"]`` expects ``sum(n_pts_)`` rows
        per combo while the sampler returns only the non-None surfaces' points -- which
        is why the NaN-column handling above is reachable only by direct call.
        """
        dataset = build_dataset(
            [[meshes[0][0], None]],
            tmp_path_factory.mktemp("none_surface"),
            store_data_in_memory=True,
            save_cache=False,
            **SMALL,
        )
        item, _ = dataset[0]
        assert {"xyz", "gt_sdf"} <= set(item)

    def test_the_single_surface_class_also_raises_by_name(self):
        """``SDFSamples.sdf_pos_neg_idx`` is separate code with the same division."""
        from types import SimpleNamespace

        import torch

        from NSM.datasets.sdf_dataset import SDFSamples

        all_positive = {"gt_sdf": torch.linspace(0.1, 1.0, 10)}
        with pytest.raises(ValueError, match="no negative SDF samples"):
            SDFSamples.sdf_pos_neg_idx(SimpleNamespace(subsample=64), all_positive)


class TestConstructorContract:
    """The declared constructor surface of ``MultiSurfaceSDFSamples`` (#43, fixed Aug 2026)."""

    def test_subsample_none_is_refused_at_construction(self, meshes, tmp_path_factory):
        """
        ``None`` -- the documented default until Aug 2026 -- used to construct and then
        crash in ``get_samples_per_sign`` on a cold cache, or skip joint normalization
        and return unnormalized points on a warm one. There is no working default, so
        construction refuses by name.
        """
        with pytest.raises(ValueError, match="subsample must be a positive int"):
            build_dataset(
                meshes, tmp_path_factory.mktemp("none_sub"), **dict(SMALL, subsample=None)
            )

    def test_joint_scale_buffer_is_accepted_and_reaches_normalization(
        self, meshes, tmp_path_factory
    ):
        """
        ``joint_scale_buffer`` sets the margin on the joint max radius -- 0.1 in every
        shipped multi-surface dataset -- and the constructor refused it with a
        ``TypeError`` until Aug 2026. The parent's default happens to equal the
        production value, which is why nothing noticed. Whether it belongs in the cache
        key is #19's business (it does not change cached bytes), deliberately not
        asserted here.
        """
        joint = dict(SMALL, scale_jointly=True, center_pts=False, norm_pts=False)
        cache = tmp_path_factory.mktemp("joint_buffer")
        narrow = build_dataset(meshes, cache, joint_scale_buffer=0.1, **joint)
        wide = build_dataset(meshes, cache, load_cache=True, joint_scale_buffer=0.25, **joint)

        assert wide.max_radius / narrow.max_radius == pytest.approx(1.25 / 1.1, rel=1e-6)


class TestMeshSubjects:
    """
    A subject passed as an in-memory ``Mesh`` -- which the ``isinstance(..., (str, Mesh))``
    branches in ``preprocess_inputs`` and ``load_reference_mesh`` advertise -- has never
    built end to end in either class (determined by execution, 2026-08-24):

    * Both readers gate on ``os.path.exists(path)``, which returns ``False`` for a
      ``Mesh`` object, so the subject is "skipped" as a missing path: the reader returns
      None and ``__init__`` silently drops it. The dataset comes back shorter than the
      subject list -- possibly empty -- with no error.
    * Seeded (``random_seed`` set), the single class dies even earlier:
      ``mesh_content_key`` iterates what it is given when it is not a path, and
      iterating a ``Mesh`` raises ``KeyError: 'Index (0) not understood...'``.
    * The multi class stringifies each ``Mesh`` into the cache key, i.e. by memory
      address -- moot while the subject never builds, but it means fixing the build
      alone would resurrect the ``TestReferenceMeshHashing`` defect one level down.

    Both pins assert the behaviour the branches advertise. Issue text is drafted in the
    section 8.0.F slice PR for the maintainer to file; #19's identity routing covers
    what runs today, which is paths.
    """

    @pytest.mark.xfail(strict=True, reason="a Mesh subject has never built (draft issue, slice PR)")
    def test_a_mesh_subject_must_build_on_the_single_surface_class(
        self, bone_meshes, tmp_path_factory
    ):
        from pymskt.mesh import Mesh

        dataset = build_single_surface_dataset(
            [Mesh(bone_meshes[0])],
            tmp_path_factory.mktemp("mesh_subject_single"),
            store_data_in_memory=True,
            save_cache=False,
            **SMALL_SINGLE,
        )
        assert len(dataset) == 1, "the Mesh subject was silently dropped"
        item, _ = dataset[0]
        assert {"xyz", "gt_sdf"} <= set(item)

    @pytest.mark.xfail(strict=True, reason="a Mesh subject has never built (draft issue, slice PR)")
    def test_a_mesh_subject_must_build_on_the_multi_surface_class(self, meshes, tmp_path_factory):
        from pymskt.mesh import Mesh

        dataset = build_dataset(
            [[Mesh(path) for path in meshes[0]]],
            tmp_path_factory.mktemp("mesh_subject_multi"),
            store_data_in_memory=True,
            save_cache=False,
            **SMALL,
        )
        assert len(dataset) == 1, "the Mesh subject was silently dropped"
        item, _ = dataset[0]
        assert {"xyz", "gt_sdf"} <= set(item)


class TestReferenceMeshFromSubjectIndex:
    """
    ``reference_mesh=<int>`` names a subject to register everyone else to (#61, fixed
    Aug 2026).
    """

    def test_an_integer_reference_with_combined_surfaces_builds(self, tmp_path_factory):
        """
        With ``mesh_to_scale=[0, 1]``, subject 0's two surfaces are combined into the
        registration target. This path raised ``UnboundLocalError`` one statement before
        the combine result -- a pyvista ``PolyData`` with no ``save_mesh`` -- would have
        broken anyway; ``combine_meshes`` now keeps its declared ``Mesh`` return type.
        """
        from pymskt.mesh import Mesh

        subjects = write_synthetic_meshes(tmp_path_factory.mktemp("ref_meshes"))[:2]
        dataset = build_dataset(
            subjects,
            tmp_path_factory.mktemp("ref_cache"),
            mesh_to_scale=[0, 1],
            reference_mesh=0,
            **SMALL,
        )

        assert isinstance(dataset.reference_mesh, Mesh)
        assert len(dataset) == 2
        item, _ = dataset[0]
        assert {"xyz", "gt_sdf"} <= set(item)
