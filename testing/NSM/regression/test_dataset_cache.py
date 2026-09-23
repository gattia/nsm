"""
The SDF dataset cache: round trip, what reaches the cache key, and what the seed does.

The key is ``md5(json.dumps({name: value}, sort_keys=True))``. Each mesh path contributes
``(path, size, mtime)`` and a loaded ``Mesh`` a digest of its geometry (#19).
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


@pytest.fixture(scope="module")
def bone_meshes(tmp_path_factory):
    """``[bone, bone]`` -- single paths, not pairs, which is what ``SDFSamples`` takes."""
    pairs = write_synthetic_meshes(tmp_path_factory.mktemp("single_meshes"))[:2]
    return [pair[0] for pair in pairs]


#: Builds the same subjects pooled into ``sys.argv[1]``, then serially into ``sys.argv[2]``,
#: from meshes the caller wrote to ``sys.argv[3]``. Both builds read the same files, so they
#: produce the same cache filenames. A fresh process, pooled first: forking after the parent
#: has built with VTK hangs.
_BUILD_IN_SUBPROCESS = f"""
import glob
import os
import sys
sys.path.insert(0, {os.path.dirname(os.path.abspath(__file__))!r})
from _harness import build_dataset

bones = sorted(glob.glob(os.path.join(sys.argv[3], "*_bone.vtk")))
subjects = [[bone, bone.replace("_bone.vtk", "_cart.vtk")] for bone in bones]
for cache, pooled in ((sys.argv[1], True), (sys.argv[2], False)):
    build_dataset(
        subjects, cache, random_seed=1234, multiprocessing=pooled, n_processes=2, **{SMALL!r}
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


class TestCacheRoundTrip:
    def test_a_reload_serves_the_first_builds_file_byte_for_byte(self, meshes, tmp_path_factory):
        """
        ``random_seed=None`` on both builds keeps the key the same while leaving sampling
        unseeded, so a re-sample would change every number. The first build's arrays are
        read before the second build runs: a re-sample writes to the same path. ``find_hash``
        walks all of ``loc_save``, which is what lets a cache written on another day still hit.
        """
        cache = tmp_path_factory.mktemp("roundtrip")
        first = build_dataset(meshes, cache, seed=0, random_seed=None, **SMALL)
        original = cached_arrays(first)
        reloaded = build_dataset(
            meshes, cache, seed=999, random_seed=None, load_cache=True, **SMALL
        )
        assert len(first.data) == len(meshes) and first.data[0].endswith(".npz")
        assert reloaded.data[0] == first.data[0], "the cache was not hit"
        again = cached_arrays(reloaded)
        assert original.keys() == again.keys()
        assert all(np.array_equal(original[key], again[key]) for key in original)

        name = os.path.basename(first.data[0])
        assert os.path.basename(first.find_hash(filename=name)[0]) == name

    def test_a_disk_cache_serves_either_storage_mode(self, meshes, tmp_path_factory):
        import torch

        cache = tmp_path_factory.mktemp("store_modes")
        disk = build_dataset(meshes, cache, **SMALL)
        memory = build_dataset(meshes, cache, load_cache=True, store_data_in_memory=True, **SMALL)
        assert isinstance(disk.data[0], str) and isinstance(memory.data[0], dict)
        torch.manual_seed(0)
        from_disk, _ = disk[0]
        torch.manual_seed(0)
        from_memory, _ = memory[0]
        assert torch.equal(from_disk["xyz"], from_memory["xyz"])

    def test_every_subject_started_is_logged(self, tmp_path_factory):
        """So a crash mid-build names its subject. The log appends across builds."""
        subjects = write_synthetic_meshes(tmp_path_factory.mktemp("logged_meshes"))[:2]
        cache = tmp_path_factory.mktemp("logged")
        log = os.path.join(str(cache), "list_meshes_started_loading.log")
        build_dataset(subjects, cache, **SMALL)
        with open(log, encoding="utf-8") as f:
            assert f.read().splitlines() == [str(subject) for subject in subjects]
        build_dataset(subjects, cache, load_cache=True, **SMALL)
        with open(log, encoding="utf-8") as f:
            assert len(f.read().splitlines()) == 4


class TestCacheHitRepair:
    """What a cache hit does besides loading: delete, rebuild, or upgrade in place."""

    def test_an_unreadable_file_is_rebuilt_in_both_classes(
        self, meshes, bone_meshes, tmp_path_factory
    ):
        """
        A truncated ``.npz`` (a crash mid-write) is deleted and the subject rebuilt at the
        same path. So is one whose index lists point past its points: served as-is they
        would read the wrong rows.
        """
        for label, build, subjects, n_pts in (
            ("multi", build_dataset, meshes, sum(SMALL["n_pts"])),
            ("single", build_single_surface_dataset, bone_meshes[:1], SMALL_SINGLE["n_pts"]),
        ):
            small = SMALL if label == "multi" else SMALL_SINGLE
            cache = tmp_path_factory.mktemp(f"badzip_{label}")
            path = build(subjects, cache, **small).data[0]
            with open(path, "wb") as f:
                f.write(b"not a zipfile")
            again = build(subjects, cache, load_cache=True, **small)
            assert len(again) == len(subjects) and again.data[0] == path
            assert cached_arrays(again)["pts"].shape == (n_pts, 3)

        cache = tmp_path_factory.mktemp("out_of_range")
        path = build_dataset(meshes, cache, **SMALL).data[0]
        arrays = dict(np.load(path))
        arrays["pos_idx_0"] = arrays["pos_idx_0"].copy()
        arrays["pos_idx_0"][0] = arrays["pts"].shape[0] + 100
        np.savez(path, **arrays)
        rebuilt = cached_arrays(build_dataset(meshes, cache, load_cache=True, **SMALL))
        assert rebuilt["pos_idx_0"].max() < rebuilt["pts"].shape[0]

    def test_a_pre_overlap_pass_cache_is_shrunk_and_resaved(self, meshes, tmp_path_factory):
        """
        ``remove_overlapping_points`` runs on every hit. The index lists are not recomputed
        on this path, so the poisoned row is pruned from them first, keeping the in-range
        guard out of the way.
        """
        cache = tmp_path_factory.mktemp("overlap_upgrade")
        path = build_dataset(meshes, cache, **SMALL).data[0]
        arrays = dict(np.load(path))
        n = arrays["sdfs"].shape[0]
        arrays["sdfs"] = arrays["sdfs"].copy()
        arrays["sdfs"][-1, :] = -0.05  # inside both surfaces
        for key in [k for k in arrays if k.startswith(("pos_idx", "neg_idx", "surf_idx"))]:
            arrays[key] = arrays[key][arrays[key] != n - 1]
        np.savez(path, **arrays)

        assert build_dataset(meshes, cache, load_cache=True, **SMALL).data[0] == path
        upgraded = dict(np.load(path))
        assert upgraded["sdfs"].shape[0] == n - 1
        assert np.array_equal(upgraded["pos_idx_0"], arrays["pos_idx_0"])

    def test_missing_index_lists_are_backfilled_in_both_classes(
        self, meshes, bone_meshes, tmp_path_factory
    ):
        """
        The single-surface upgrade never fired until Aug 2026: ``unpack_numpy_data`` always
        sets ``pos_idx``, as an empty list when absent, so the ``not in`` test was always
        false and ``__getitem__`` raised ``IndexError``. It now checks the length.
        """
        for label, build, subjects, small in (
            ("multi", build_dataset, meshes, SMALL),
            ("single", build_single_surface_dataset, bone_meshes[:1], SMALL_SINGLE),
        ):
            cache = tmp_path_factory.mktemp(f"backfill_{label}")
            path = build(subjects, cache, **small).data[0]
            arrays = dict(np.load(path))
            prefixes = ("pos_idx", "neg_idx", "surf_idx")
            np.savez(path, **{k: v for k, v in arrays.items() if not k.startswith(prefixes)})

            again = build(subjects, cache, load_cache=True, **small)
            upgraded = dict(np.load(path))
            assert np.array_equal(upgraded["pts"], arrays["pts"]), "the subject was resampled"
            for key in [k for k in arrays if k.startswith(("pos_idx", "neg_idx"))]:
                assert np.array_equal(upgraded[key], arrays[key]), (label, key)
            assert {"xyz", "gt_sdf"} <= set(again[0][0])


class TestHashedParametersChangeTheKey:
    def test_every_hashed_parameter_changes_the_key(self, dataset, meshes):
        """The seed changes the samples, so it is in the key, as are the mesh paths."""
        baseline = dataset.create_hash(meshes[0])
        for attribute, value in (
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
            ("random_seed", 7),
        ):
            assert rehash(dataset, meshes[0], **{attribute: value}) != baseline, attribute
        assert dataset.create_hash([meshes[0][1], meshes[0][0]]) != baseline


class TestFormerlyCollidingParameters:
    """
    ``mesh_to_scale`` and ``uniform_pts_buffer`` change the cached content and were missing
    from the key (#19a), so with ``load_cache=True`` a second run silently trained on the
    first's data. The premise, that the content differs, is asserted first.

    ``subsample`` is deliberately not in the key. Its one effect on cached content, the
    index padding, moved to draw time, sized by the ``subsample`` then in force.
    """

    @staticmethod
    def _keys_and_differing_content(meshes, tmp_path_factory, label, **override):
        a = build_dataset(meshes, tmp_path_factory.mktemp(f"{label}_a"), **SMALL)
        b = build_dataset(meshes, tmp_path_factory.mktemp(f"{label}_b"), **dict(SMALL, **override))
        first, second = cached_arrays(a), cached_arrays(b)
        differing = {
            key
            for key in set(first) & set(second)
            if first[key].shape != second[key].shape or not np.array_equal(first[key], second[key])
        }
        return a.create_hash(meshes[0]), b.create_hash(meshes[0]), differing

    def test_what_changes_the_content_changes_the_key_and_subsample_does_neither(
        self, meshes, tmp_path_factory
    ):
        for label, override in (
            ("mts", {"mesh_to_scale": 1}),
            ("upb", {"uniform_pts_buffer": 0.5}),
        ):
            key_a, key_b, differing = self._keys_and_differing_content(
                meshes, tmp_path_factory, label, **override
            )
            assert {"pts", "sdfs"} <= differing, f"premise gone: {label} no longer changes content"
            assert key_a != key_b, label

        key_a, key_b, differing = self._keys_and_differing_content(
            meshes, tmp_path_factory, "sub", subsample=2048
        )
        assert key_a == key_b and differing == set()

        cache = tmp_path_factory.mktemp("collision")
        first = build_dataset(meshes, cache, **SMALL)
        second = build_dataset(meshes, cache, load_cache=True, mesh_to_scale=1, **SMALL)
        assert second.data[0] != first.data[0], "the second run was handed the first run's file"

    def test_equal_pos_neg_holds_after_a_subsample_change(self, meshes, tmp_path_factory):
        """
        The padding used to be cached for the build's ``subsample``, so a larger one topped
        batches up with uniform points and ``equal_pos_neg`` stopped holding: 1.6x
        under-representation of the small surface's interior (0.20 against 0.32).
        """
        import torch

        base = {k: v for k, v in SMALL.items() if k != "subsample"}
        cache = tmp_path_factory.mktemp("subsample_collision")
        build_dataset(meshes, cache, subsample=64, **base)
        reused = build_dataset(meshes, cache, load_cache=True, subsample=4096, **base)
        fresh = build_dataset(meshes, tmp_path_factory.mktemp("sub_fresh"), subsample=4096, **base)

        def interior_fraction(dataset):
            torch.manual_seed(0)
            return (dataset[0][0]["gt_sdf"][:, 1] < 0).float().mean().item()

        assert interior_fraction(reused) == pytest.approx(interior_fraction(fresh), rel=0.25)


class TestMeshContentInTheKey:
    def test_an_in_place_mesh_edit_changes_the_key(self, tmp_path_factory):
        """#19b: the key hashed the path alone, so stale samples outlived an edit."""
        import pyvista as pv

        subject = write_synthetic_meshes(tmp_path_factory.mktemp("editable"))[:1]
        dataset = build_dataset(subject, tmp_path_factory.mktemp("edit_cache"), **SMALL)
        before = dataset.create_hash(subject[0])
        pv.Sphere(radius=0.5, theta_resolution=30, phi_resolution=30).triangulate().save(
            subject[0][0]
        )
        assert dataset.create_hash(subject[0]) != before


class TestReferenceMeshHashing:
    def test_equal_meshes_hash_the_same(self, dataset, meshes):
        """
        #19c: a ``Mesh`` reference was stringified, and its ``__str__`` includes a memory
        address, so such a dataset never hit its own cache.
        """
        from pymskt.mesh import Mesh

        one, two = Mesh(meshes[0][0]), Mesh(meshes[0][0])
        assert rehash(dataset, meshes[0], reference_mesh=one) == rehash(
            dataset, meshes[0], reference_mesh=two
        )


class TestSeeding:
    """
    A seed makes both sampling paths reproducible from a cold cache. ``random_seed=None``
    leaves the uniform path on numpy's global stream, so an old caller that seeds numpy
    still gets the numbers it always did.
    """

    def test_seeded_builds_reproduce_on_both_paths(self, meshes, tmp_path_factory):
        for label, options in (
            ("uniform", dict(sigma_near=[None, None], sigma_far=[None, None], random_seed=None)),
            ("near", dict(sigma_near=[0.05, 0.05], sigma_far=[0.2, 0.2])),
            ("cold", dict(sigma_near=[0.05, 0.05], sigma_far=[0.2, 0.2], random_seed=1234)),
        ):
            a, b = (
                build_dataset(
                    meshes, tmp_path_factory.mktemp(f"seed_{label}"), seed=7, **{**SMALL, **options}
                )
                for _ in range(2)
            )
            assert np.array_equal(cached_arrays(a)["pts"], cached_arrays(b)["pts"]), label


class TestSeedDerivation:
    """
    ``derive_seed`` gives each (subject, sampling combo, surface) its own seed, from the run
    seed and the bytes of the subject's meshes. Each property below is silent when it breaks.
    """

    def test_the_seed_reaches_each_draw_separately(self, meshes, tmp_path_factory):
        """
        Different seeds give different data, or "reproducible" would just mean a cache hit.
        The near and far passes, asked for identical parameters, still draw different base
        points, or the dataset carries half the surface locations it appears to.
        """
        a, b = (
            build_dataset(meshes, tmp_path_factory.mktemp(f"derive_{s}"), random_seed=s, **SMALL)
            for s in (1234, 5678)
        )
        assert not np.array_equal(cached_arrays(a)["pts"], cached_arrays(b)["pts"])

        identical = dict(
            SMALL,
            sigma_near=[0.02, 0.02],
            sigma_far=[0.02, 0.02],
            p_near_surface=[0.4, 0.4],
            p_further_from_surface=[0.4, 0.4],
            random_seed=99,
        )
        dataset = build_dataset(meshes, tmp_path_factory.mktemp("combos"), **identical)
        near_count, far_count = (sum(combo[0]) for combo in dataset.pt_sample_combos[:2])
        points = cached_arrays(dataset)["pts"]
        assert near_count == far_count
        assert not np.array_equal(points[:near_count], points[near_count : 2 * near_count])

    def test_order_and_location_do_not_change_a_subjects_data(self, tmp_path_factory):
        """
        Keyed on mesh contents, not position: adding a subject to the front of a list would
        otherwise resample every other subject while their cached files stayed valid. And
        the same bytes at another path, a genuine cold resample, land on the same points.
        """
        two = write_synthetic_meshes(tmp_path_factory.mktemp("order_meshes"))[:2]
        forward = build_dataset(two, tmp_path_factory.mktemp("fwd"), random_seed=321, **SMALL)
        reverse = build_dataset(two[::-1], tmp_path_factory.mktemp("rev"), random_seed=321, **SMALL)
        for index in range(2):
            mine = cached_arrays(forward, index)["pts"]
            assert np.array_equal(mine, cached_arrays(reverse, 1 - index)["pts"])
            assert not np.array_equal(mine, cached_arrays(reverse, index)["pts"])

        moved = write_synthetic_meshes(tmp_path_factory.mktemp("there"))[:1]
        a = build_dataset(two[:1], tmp_path_factory.mktemp("here_a"), random_seed=4242, **SMALL)
        b = build_dataset(moved, tmp_path_factory.mktemp("there_b"), random_seed=4242, **SMALL)
        assert a.create_hash(two[0]) != b.create_hash(moved[0])
        assert np.array_equal(cached_arrays(a)["pts"], cached_arrays(b)["pts"])

    def test_multiprocessing_does_not_change_the_data(self, tmp_path_factory):
        """
        ``Pool`` forks, and before the seed was threaded through, the forked global numpy
        state drove the sampler: all three subjects differed. With ``random_seed=None`` they
        still do (checked 2026-09-23 with this script).
        """
        mesh_dir = str(tmp_path_factory.mktemp("mp_meshes"))
        write_synthetic_meshes(mesh_dir)
        pooled, serial = str(tmp_path_factory.mktemp("mp_on")), str(
            tmp_path_factory.mktemp("mp_off")
        )
        finished = subprocess.run(
            [sys.executable, "-c", _BUILD_IN_SUBPROCESS, pooled, serial, mesh_dir],
            capture_output=True,
            text=True,
            timeout=600,
        )
        assert finished.returncode == 0, finished.stderr[-2000:]

        serial, parallel = _cached_by_name(serial), _cached_by_name(pooled)
        assert sorted(serial) == sorted(parallel) and len(serial) == 3
        for name in serial:
            assert np.array_equal(np.load(serial[name])["pts"], np.load(parallel[name])["pts"])


class TestSingleSurfaceSDFSamples:
    """
    ``SDFSamples`` is not ``MultiSurfaceSDFSamples`` with one surface: it has its own
    ``get_sample_data_dict``, ``__getitem__`` and sampler, so the seeding contract is
    asserted on it separately.
    """

    def test_it_caches_its_scalar_n_pts_and_its_seed_reproduces(
        self, bone_meshes, tmp_path_factory
    ):
        first, second = (
            build_single_surface_dataset(
                bone_meshes, tmp_path_factory.mktemp(f"single_{label}"), **SMALL_SINGLE
            )
            for label in ("a", "b")
        )
        assert len(first.data) == len(bone_meshes)
        assert cached_arrays(first)["pts"].shape == (SMALL_SINGLE["n_pts"], 3)
        assert first.data[0] != second.data[0], "the two runs shared a cache file"
        for index in range(len(first.data)):
            assert np.array_equal(
                cached_arrays(first, index)["pts"], cached_arrays(second, index)["pts"]
            )

        other = build_single_surface_dataset(
            bone_meshes, tmp_path_factory.mktemp("single_other"), seed=5678, **SMALL_SINGLE
        )
        assert not np.array_equal(cached_arrays(first)["pts"], cached_arrays(other)["pts"])

    def test_an_unseeded_run_is_not_reproducible(self, bone_meshes, tmp_path_factory):
        """
        The near-surface draw happens inside pymskt, off numpy's global stream, so
        ``random_seed`` is the only thing that can make it reproducible.
        """
        a, b = (
            build_single_surface_dataset(
                bone_meshes,
                tmp_path_factory.mktemp(f"unseeded_{label}"),
                random_seed=None,
                **SMALL_SINGLE,
            )
            for label in ("a", "b")
        )
        assert not np.array_equal(cached_arrays(a)["pts"], cached_arrays(b)["pts"])


class TestFormerlyUncallableConfigurations:
    """Constructor options that built fine and crashed on first use (#22, #23, #69)."""

    def test_a_zero_sampling_probability_samples_nothing(
        self, meshes, bone_meshes, tmp_path_factory
    ):
        """
        #23: the empty combo reached ``point_cloud_utils`` and crashed. The other combos
        still fill the whole buffer, so no row is left at zero.
        """
        dataset = build_dataset(
            meshes,
            tmp_path_factory.mktemp("p_zero"),
            p_near_surface=[0.0, 0.0],
            p_further_from_surface=[0.5, 0.5],
            sigma_near=[0.05, 0.05],
            **SMALL,
        )
        assert not np.any(np.all(cached_arrays(dataset)["pts"] == 0, axis=1))
        assert {"xyz", "gt_sdf"} <= set(dataset[0][0])
        single = build_single_surface_dataset(
            bone_meshes[:1],
            tmp_path_factory.mktemp("p_zero_single"),
            p_near_surface=0.0,
            p_further_from_surface=0.5,
            **SMALL_SINGLE,
        )
        assert {"xyz", "gt_sdf"} <= set(single[0][0])

    def test_an_in_memory_dataset_trains_with_or_without_load_timing(
        self, meshes, tmp_path_factory
    ):
        """
        #22: ``store_data_in_memory=True`` raised ``UnboundLocalError`` on the timing keys,
        and ``train_epoch`` read all four timing keys unconditionally, so no combination of
        the two flags both built and trained. Asserted by running the trainer.
        """
        for timing in (True, False):
            dataset = build_dataset(
                meshes,
                tmp_path_factory.mktemp(f"in_memory_{timing}"),
                store_data_in_memory=True,
                test_load_times=timing,
                **SMALL,
            )
            item, index = dataset[0]
            assert set(item) == {"xyz", "gt_sdf"} and index == 0

        config = training_config(tmp_path_factory.mktemp("in_memory_train"))
        config.update(
            {
                "n_epochs": 1,
                "checkpoint_epochs": 1,
                "save_frequency": 1,
                "samples_per_object_per_batch": SMALL["subsample"],
            }
        )
        records, _ = run_training(config, build_model(config), dataset)
        assert len(records) == 1 and "loss" in records[0]

    def test_scale_jointly_works_in_either_storage_mode(self, meshes, tmp_path_factory):
        """
        #69: the in-memory branch read ``.npz``-only keys and omitted ``joint_scale_buffer``.
        ``joint_scale_buffer=9`` puts every batch within ~0.1-0.2 of the origin; an
        unbuffered scale leaves points at ~0.5-1.1, so 0.25 separates them on any draw.
        """
        joint = dict(
            SMALL, scale_jointly=True, center_pts=False, norm_pts=False, joint_scale_buffer=9.0
        )
        disk, memory = (
            build_dataset(
                meshes, tmp_path_factory.mktemp(f"joint_{mode}"), store_data_in_memory=mode, **joint
            )
            for mode in (False, True)
        )
        np.testing.assert_allclose(disk.center, memory.center, rtol=1e-6)
        np.testing.assert_allclose(disk.max_radius, memory.max_radius, rtol=1e-6)
        assert disk[0][0]["xyz"].norm(dim=1).max() < 0.25
        assert memory[0][0]["xyz"].norm(dim=1).max() < 0.25


def test_the_cache_location_is_read_when_the_dataset_is_built(
    meshes, monkeypatch, tmp_path_factory
):
    """
    #24: ``LOC_SDF_CACHE`` was a default argument, read once at import. A blank value means
    the home default, since kneepipeline blanks it rather than unsetting it, and an empty
    ``loc_save`` would root the cache in the working directory.
    """
    cache_root = tmp_path_factory.mktemp("env_cache")
    monkeypatch.setenv("LOC_SDF_CACHE", str(cache_root))
    dataset = build_dataset(meshes, "ignored-by-override", loc_save=None, **SMALL)
    assert dataset.loc_save == str(cache_root) and dataset.data[0].startswith(str(cache_root))

    fake_home = tmp_path_factory.mktemp("fake_home")
    monkeypatch.setenv("HOME", str(fake_home))
    monkeypatch.setenv("LOC_SDF_CACHE", "")
    dataset = build_dataset(meshes, "ignored-by-override", loc_save=None, **SMALL)
    assert dataset.loc_save == os.path.join(str(fake_home), ".cache", "nsm_sdf_cache")


class TestPointCenteringAndScaling:
    def test_centering_and_scaling_still_happen_unconditionally(self):
        """
        The ``center`` and ``scale`` arguments were overwritten before being read, and were
        removed (#20). Honouring them would have stopped scaling on every default run
        (``scale=norm_pts``, default False): a cloud of radius 24.95 would stay 24.95. The
        caller's array is not modified (#21).
        """
        from NSM.datasets.sdf_dataset import get_pts_center_and_scale

        parameters = inspect.signature(get_pts_center_and_scale).parameters
        assert "center" not in parameters and "scale" not in parameters

        points = np.array([[1.0, 1.0, 1.0], [3.0, 3.0, 3.0]])
        center, _, normalized = get_pts_center_and_scale(points, return_pts=True)
        assert np.allclose(center, [2.0, 2.0, 2.0])
        assert np.isclose(np.max(np.linalg.norm(normalized, axis=-1)), 1.0)
        assert np.allclose(points, [[1.0, 1.0, 1.0], [3.0, 3.0, 3.0]])


class TestUniformSamplingCube:
    """
    The two samplers had private copies of the cube arithmetic, and they had diverged (#40):
    the buffer grew the cube more above than below, only one clipped, and ``pts_surface``
    was a list from one and an array from the other.
    """

    def test_both_samplers_draw_from_one_symmetric_cube(self, meshes):
        from NSM.datasets.sdf_dataset import (
            get_buffered_cube_mins_maxs,
            get_cube_mins_maxs,
            read_mesh_get_sampled_pts,
            read_meshes_get_sampled_pts,
        )

        pts = np.random.default_rng(0).normal(size=(500, 3)) + [5.0, -2.0, 0.5]
        mins0, maxs0 = get_cube_mins_maxs(pts)
        mins, maxs = get_buffered_cube_mins_maxs(pts, 0.5)
        assert np.allclose((mins + maxs) / 2, (mins0 + maxs0) / 2)
        assert np.allclose(maxs - mins, 1.5 * (maxs0 - mins0))

        path = meshes[0][0]
        kwargs = dict(center_pts=True, norm_pts=True, fix_mesh=False, get_random=True, seed=0)
        with quiet():
            single = read_mesh_get_sampled_pts(
                path, sigma=None, n_pts=4000, uniform_pts_buffer=0.5, **kwargs
            )
            multi = read_meshes_get_sampled_pts(
                [path], sigma=[None], n_pts=[4000], uniform_pts_buffer=0.5, **kwargs
            )
        for label, result in (("single", single), ("multi", multi)):
            points = result["pts"]
            assert -1.5 <= points.min() < -1.4 and 1.4 < points.max() <= 1.5, label
            assert abs(points.max() + points.min()) < 0.1, label
            surface = result["pts_surface"]
            assert isinstance(surface, np.ndarray) and surface.dtype == np.int64, label


class TestEmptySignedSamples:
    """
    ``sdf_pos_neg_idx`` divided by zero when a surface had no samples of one sign (#41).
    A missing surface now gives empty index lists; a drawn-from surface missing a sign
    raises, naming the surface.
    """

    def test_a_surface_missing_a_sign_is_named_and_a_missing_surface_is_empty(
        self, dataset, tmp_path_factory
    ):
        """
        One surface inside another loses its interior to ``remove_overlapping_points``. The
        harness subjects are disjoint to avoid exactly this. A missing surface is an all-NaN
        column, and is checked by direct call: the end-to-end path dies earlier (#67).
        """
        import pyvista as pv
        import torch

        from NSM.datasets.sdf_dataset import SDFSamples

        directory = tmp_path_factory.mktemp("nested_meshes")
        paths = []
        for name, radius in (("outer", 1.0), ("inner", 0.4)):
            path = os.path.join(str(directory), f"{name}.vtk")
            pv.Sphere(radius=radius, theta_resolution=24, phi_resolution=24).triangulate().save(
                path
            )
            paths.append(path)
        with pytest.raises(ValueError, match="Surface 1 has no negative"):
            build_dataset([paths], tmp_path_factory.mktemp("nested_cache"), **SMALL)

        gt_sdf = torch.stack([torch.linspace(-1.0, 1.0, 10), torch.full((10,), float("nan"))], 1)
        pos, neg, surf = dataset.sdf_pos_neg_idx({"gt_sdf": gt_sdf, "xyz": torch.zeros(10, 3)})
        assert pos[0].numel() > 0 and neg[0].numel() > 0
        assert pos[1].numel() == neg[1].numel() == surf[1].numel() == 0

        from types import SimpleNamespace

        with pytest.raises(ValueError, match="no negative SDF samples"):
            SDFSamples.sdf_pos_neg_idx(
                SimpleNamespace(subsample=64), {"gt_sdf": torch.linspace(0.1, 1.0, 10)}
            )

    @pytest.mark.xfail(
        strict=True, reason="#67: a None surface dies at the preallocated buffer write"
    )
    def test_a_none_surface_subject_must_build(self, meshes, tmp_path_factory):
        dataset = build_dataset(
            [[meshes[0][0], None]],
            tmp_path_factory.mktemp("none_surface"),
            store_data_in_memory=True,
            save_cache=False,
            **SMALL,
        )
        assert {"xyz", "gt_sdf"} <= set(dataset[0][0])


def test_the_constructor_refuses_no_subsample_and_honours_joint_scale_buffer(
    meshes, tmp_path_factory
):
    """
    #43: ``subsample=None`` was the documented default, and it crashed on a cold cache or
    skipped normalization on a warm one. ``joint_scale_buffer`` was refused with a
    ``TypeError``, unnoticed because the parent's default equals the production 0.1.
    """
    with pytest.raises(ValueError, match="subsample must be a positive int"):
        build_dataset(meshes, tmp_path_factory.mktemp("none_sub"), **dict(SMALL, subsample=None))

    joint = dict(SMALL, scale_jointly=True, center_pts=False, norm_pts=False)
    cache = tmp_path_factory.mktemp("joint_buffer")
    narrow = build_dataset(meshes, cache, joint_scale_buffer=0.1, **joint)
    wide = build_dataset(meshes, cache, load_cache=True, joint_scale_buffer=0.25, **joint)
    assert wide.max_radius / narrow.max_radius == pytest.approx(1.25 / 1.1, rel=1e-6)


@pytest.mark.xfail(strict=True, reason="a Mesh subject has never built (draft issue, slice PR)")
@pytest.mark.parametrize("single", [True, False], ids=["single", "multi"])
def test_a_mesh_subject_must_build(single, meshes, bone_meshes, tmp_path_factory):
    """
    Both classes advertise ``Mesh`` subjects, and neither builds one. The readers test
    ``os.path.exists`` and drop a ``Mesh`` as a missing path, silently. Seeded, the single
    class fails earlier: ``mesh_content_key`` iterates the ``Mesh``.
    """
    from pymskt.mesh import Mesh

    options = dict(store_data_in_memory=True, save_cache=False)
    if single:
        subjects, build, small = [Mesh(bone_meshes[0])], build_single_surface_dataset, SMALL_SINGLE
    else:
        subjects, build, small = [[Mesh(p) for p in meshes[0]]], build_dataset, SMALL
    dataset = build(subjects, tmp_path_factory.mktemp("mesh_subject"), **options, **small)
    assert len(dataset) == 1, "the Mesh subject was silently dropped"
    assert {"xyz", "gt_sdf"} <= set(dataset[0][0])


def test_an_integer_reference_with_combined_surfaces_builds(tmp_path_factory):
    """
    #61: ``reference_mesh=0`` with ``mesh_to_scale=[0, 1]`` combines subject 0's surfaces
    into the registration target. It raised ``UnboundLocalError``.
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
    assert isinstance(dataset.reference_mesh, Mesh) and len(dataset) == 2
    assert {"xyz", "gt_sdf"} <= set(dataset[0][0])
