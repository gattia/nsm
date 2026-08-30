"""Turn meshes into the SDF samples a decoder trains on, and cache the result.

Two dataset classes, one for a single surface per subject and one for several:
:class:`SDFSamples` and :class:`MultiSurfaceSDFSamples`. Both do the same four
things per subject -- read the meshes, put them in a common frame, draw points and
compute signed distances, cache -- and the leaf helpers for each live in
``NSM/datasets/utils.py`` and ``NSM/datasets/mesh_sampling.py``.

Three things here are traps rather than details:

* **NSM never builds one of these from a config.** ``train_deep_sdf`` takes an
  already-built dataset, so ``default_config.json``'s dataset keys are a
  specification the caller translates into constructor arguments.
* **The cache key does not cover every parameter that changes the samples**
  (#19), so a run pointed at a stale cache silently reuses the old points.
* **``multiprocessing=True`` after an in-process build deadlocks** (#25) -- see
  that parameter's entry in :class:`SDFSamples`.

Coordinate conventions -- what ``center_pts``, ``norm_pts`` and ``scale_jointly``
each do, and which of them ``sigma_near``/``sigma_far`` are expressed in -- are in
``docs/ARCHITECTURE.md`` and ``docs/KNOWN_ISSUES.md`` §3; they are not
interchangeable and #3 is open against the sigma half.
"""

import gc
import hashlib
import json
import logging
import multiprocessing
import os
import time
import warnings
import zipfile
from datetime import datetime
from multiprocessing import Pool

import numpy as np
import point_cloud_utils as pcu
import pymskt as mskt
import torch
import vtk
from pymskt.mesh import Mesh
from vtk.util.numpy_support import numpy_to_vtk, vtk_to_numpy

logger = logging.getLogger(__name__)

try:
    from pympler import muppy, tracker  # asizeof, summary, muppy, tracker
except ModuleNotFoundError:
    logger.debug(
        "Pympler not installed, cannot use asizeof - if trying to debug memory usage, install pympler"
    )


today_date = datetime.now().strftime("%b_%d_%Y")


from .._verbose_deprecation import honour_verbose

# Moved to .utils / .mesh_sampling in the §8.0 decomposition (slice A) and re-imported
# here permanently: NSM.datasets and NSM.datasets.sdf_dataset are both live import
# paths for these names (reconstruct/main.py uses both), so they are public API of
# this module, not an implementation detail of the move.
from .mesh_sampling import (  # noqa: F401
    read_mesh_get_sampled_pts,
    read_meshes_get_sampled_pts,
)
from .utils import (  # noqa: F401
    check_probabilities,
    check_probabilities_sum,
    combine_meshes,
    derive_seed,
    get_buffered_cube_mins_maxs,
    get_cube_mins_maxs,
    get_pts_center_and_scale,
    get_rand_uniform_pts,
    is_zipfile,
    mesh_content_key,
    meshfix,
    unpack_numpy_data,
    unpack_pts,
)

# --- Cache-key machinery ---------------------------------------------------------
#
# A subject's cache key answers one question: "would this exact configuration write
# these exact bytes?" It is built in three steps:
#
#   1. get_hash_params() -> a {name: value} dict of every constructor parameter that
#      changes cached content. Computed once in __init__ and stored as
#      self.hash_params (MultiSurfaceSDFSamples extends the same dict with its four
#      extra parameters).
#   2. create_hash(loc_mesh) copies that dict and adds "mesh_paths": the subject's
#      own mesh(es), each reduced to an identity token by _identity().
#   3. The dict is serialized with json.dumps(sort_keys=True) and md5'd. Sorting by
#      name is what makes the key positionless -- no entry's meaning depends on
#      where it sits, so adding or reordering entries can never silently swap two
#      parameters' roles (the LR-bug shape).
#
# An identity token is whichever of these fits what was passed:
#
#   a path        -> ("/data/subject0_bone.vtk", 51284, 1756071234.56)  # size, mtime
#   a loaded Mesh -> ("geometry", "303a4609...")                # md5 of points+faces
#   missing/None  -> ("/gone.vtk", None, None)
#
# so editing a mesh file in place moves the key (its stat changes) without anything
# re-reading mesh bytes on a cache hit.

#: Version stamp inside every cache key. Bump it when a change alters what gets written
#: into the cached ``.npz`` while every keyed parameter stands still -- one integer
#: instead of a fresh #19. Format 2 (Aug 2026): named canonical key, content-stable
#: identities, unpadded index lists.
CACHE_FORMAT = 2


def _file_identity(path):
    """
    A path's identity in the cache key: ``(path, st_size, st_mtime)``.

    The stat is what makes an in-place mesh edit move the key without reading the file
    -- hashing full contents would read every mesh on every construction, cache hits
    included, while a stat is free (#19 (b)). A path that cannot be stat'ed (missing, or
    None for a subject's missing surface, #67) contributes ``(path, None, None)`` rather
    than raising: the samplers skip that subject a moment later.
    """
    try:
        stat_result = os.stat(path)
    except (OSError, TypeError, ValueError):
        return (str(path), None, None)
    return (str(path), stat_result.st_size, stat_result.st_mtime)


def _mesh_geometry_digest(mesh):
    """
    A loaded ``Mesh``'s identity in the cache key: md5 over its point and face bytes.

    Object identity -- ``str(mesh)`` embeds the memory address -- made any key holding a
    ``Mesh`` per-object, so it could never hit across constructions (#19 (c)). The
    geometry is what the cached samples actually depend on.
    """
    digest = hashlib.md5()
    digest.update(np.ascontiguousarray(np.asarray(mesh.points)).tobytes())
    digest.update(np.ascontiguousarray(np.asarray(mesh.faces)).tobytes())
    return digest.hexdigest()


def _identity(value):
    """One cache-key identity token: geometry digest for a ``Mesh``, stat identity else."""
    if issubclass(type(value), Mesh):
        return ("geometry", _mesh_geometry_digest(value))
    return _file_identity(value)


def _draw_sign_share(indices, samples_per_sign):
    """
    Draw one sign's share of a batch from its RAW index set.

    The set is tiled (``repeat``) until it holds at least ``samples_per_sign`` entries
    -- sized by the subsample in force NOW, not the one at build time, which is what
    decouples cached bytes from ``subsample`` (#19) -- then permuted and truncated.
    For an unchanged subsample the tiled array is byte-identical to what the cache
    used to store, so batches are bit-identical to the padded-cache era. An empty set
    (a surface nothing draws from: missing, or all one sign under a zero share) draws
    nothing, exactly as before -- ``randperm(0)`` runs either way, so the RNG stream
    does not shift.
    """
    if indices.numel() > 0:
        indices = indices.repeat(samples_per_sign // indices.numel() + 1)
    perm = torch.randperm(indices.size(0))[:samples_per_sign]
    return indices[perm]


class SDFSamples(torch.utils.data.Dataset):
    """
    Dataset class for sampling SDFs from meshes.

    Args:
        list_mesh_paths (list): List of paths to meshes
        subsample (int): Number of points each __getitem__ returns. Required, and must be
            a positive int -- there is no working default (#43).
        n_pts (int, optional): Number of points to sample. Defaults to 500000.
        p_near_surface (float, optional): Proportion of points to sample near the surface. Defaults to 0.4.
        p_further_from_surface (float, optional): Proportion of points to sample further from the surface. Defaults to 0.4.
        sigma_near (float, optional): Standard deviation/scale of the distribution for points near the surface. Defaults to 0.01.
        sigma_far (float, optional): Standard deviation/scale of the distribution for points further from the surface. Defaults to 0.1.
        rand_function (str, optional): Distribution to sample from. Defaults to 'normal'. Also supports 'laplace'.
        center_pts (bool, optional): Defaults to True.
        norm_pts (bool, optional): Defaults to False. Together they decide *whether*
            per-subject normalization runs, not which half of it: if either is True
            each subject is both centered and scaled (see ``docs/KNOWN_ISSUES.md``
            § Open on ``center_pts``/``norm_pts``). Only False/False leaves coordinates
            alone, which is what ``scale_jointly`` requires.
        scale_method (str, optional): Method to scale the points. Defaults to 'max_rad'.
        scale_jointly (bool, optional): Whether to center and scale all subjects together
            after loading (norm_and_scale_all_meshes) instead of per subject; requires
            center_pts=False and norm_pts=False. Works whether subjects are held in
            memory (``store_data_in_memory=True``) or reloaded from the ``.npz`` cache
            per batch (False): either way the stored data stays in the unscaled frame
            and the shared frame is applied per batch in ``__getitem__``.
            Defaults to False.
        joint_scale_buffer (float, optional): Margin added to the joint max radius when
            scale_jointly is True, so unseen subjects slightly larger than the training
            set still fit inside the model's domain. Defaults to 0.1.
        loc_save (str, optional): Directory for the cached files. Defaults to the
            LOC_SDF_CACHE environment variable, read when the dataset is constructed
            (an empty value counts as unset), else ~/.cache/nsm_sdf_cache.
        save_cache (bool, optional): Whether to save the cached files. Defaults to True.
        load_cache (bool, optional): Whether to load the cached files. Defaults to True.
        random_seed (int, optional): Seeds the sampling, and is part of the cache key.
            Every subject/surface/sigma draw gets its own seed derived from it, keyed on
            the subject's mesh *contents* rather than on list position or path, so neither
            reordering list_mesh_paths nor moving the meshes changes any subject's data.
            Defaults to None, which leaves sampling unseeded -- the historical behaviour.
            Reproducible sampling requires mskt>=0.1.21.
        reference_mesh (Mesh, str, int or list, optional): What every subject is
            similarity-registered to before sampling; None skips registration. Accepts a
            loaded Mesh, a path, an index into list_mesh_paths, or a list of paths --
            see load_reference_mesh for how each resolves. Similarity = rigid + uniform
            scale, so each subject comes out at the reference's size: between-subject
            size does not survive registration, under scale_jointly or otherwise.
            Defaults to None.
        verbose (bool, optional): Whether to print verbose output. Defaults to False.
        equal_pos_neg (bool, optional): Draw half of every batch from positive-SDF
            samples and half from negative, instead of uniformly. Defaults to True.
        fix_mesh (bool, optional): Whether to fix the meshes (using meshfix). Defaults to True.
        print_filename (bool, optional): Whether to print the filename when loading. Defaults to False.
        multiprocessing (bool, optional): Build/load subjects in a Pool(n_processes).
            Also makes a reference_mesh spill to disk so workers can share it
            (see load_reference_mesh). Defaults to True.

            **A True build deadlocks if an earlier build in the same process ran with
            False** (#25). Fork-after-VTK: the Pool inherits a VTK/OpenMP state the
            in-process build left behind, and the workers sit idle forever -- no message,
            no traceback, no timeout. The trigger is narrower than "a second dataset", and
            the narrowness is the useful part: True -> True is fine (measured 2.6 s then
            0.4 s), False -> True never returns (5.7 s then nothing). Since True is this
            default, the shape that hangs is an ordinary one -- build the train split
            in-process, then the val split. Build both the same way, or build each in its
            own process. Not fixed: a `spawn` context would change behaviour nobody has
            asked for, and this constraint is cheaper than that. Worked around in
            ``test_dataset_cache.TestSeedDerivation::test_multiprocessing_does_not_change_the_data``,
            which builds its two datasets in separate subprocesses for this reason.
        n_processes (int, optional): Pool size when multiprocessing. Defaults to 2.
        store_data_in_memory (bool, optional): Keep every subject's sample dict in
            memory (True), or keep only its cache path and reload the .npz on every
            __getitem__ (False). False requires save_cache=True. Defaults to False.
        debug_memory (bool, optional): Print a pympler memory-summary diff every 100th
            subject load (requires pympler installed). Defaults to False.
        test_load_times (bool, optional): Include time/size/mb_per_sec/whole_load_time
            in each disk-backed __getitem__ batch. Optional diagnostics, not batch
            contract: in-memory items never carry them (#22). Defaults to True.
        uniform_pts_buffer (float, optional): Expansion of the uniform sampling cube;
            see get_buffered_cube_mins_maxs. Part of the cache key. Defaults to 0.0.

    Notes:
        ``__getitem__`` returns ``(batch, idx)``: ``batch["xyz"]`` is (subsample, 3)
        and ``batch["gt_sdf"]`` (subsample,), float32, plus the load-time diagnostics
        when enabled.

        Caches are one ``.npz`` per subject under ``loc_save/<Mon_DD_YYYY>/``, the date
        fixed at import time; lookups search all of ``loc_save`` recursively, so hits
        cross dates. The cache key is a named, content-stable mapping -- see
        get_hash_params and create_hash; keys from before ``cache_format`` 2 (Aug 2026)
        never hit again, so an old cache directory is reclaimable disk.
    """

    @honour_verbose
    def __init__(
        self,
        list_mesh_paths,
        subsample,
        n_pts=500000,
        p_near_surface=0.4,
        p_further_from_surface=0.4,
        sigma_near=0.01,
        sigma_far=0.1,
        rand_function="normal",
        center_pts=True,
        norm_pts=False,
        scale_method="max_rad",
        scale_jointly=False,
        joint_scale_buffer=0.1,
        loc_save=None,
        save_cache=True,
        load_cache=True,
        random_seed=None,
        reference_mesh=None,
        verbose=False,
        equal_pos_neg=True,
        fix_mesh=True,
        print_filename=False,
        multiprocessing=True,
        n_processes=2,
        store_data_in_memory=False,
        debug_memory=False,
        test_load_times=True,
        uniform_pts_buffer=0.0,
    ):

        # subsample has no working default: every build path divides by it or
        # multiplies with it, so None used to crash downstream in
        # get_samples_per_sign / sdf_pos_neg_idx instead of here (#43).
        if not isinstance(subsample, (int, np.integer)) or subsample <= 0:
            raise ValueError(
                f"subsample must be a positive int -- the number of points each "
                f"__getitem__ returns -- got {subsample!r}."
            )

        # Resolved at call time so setting LOC_SDF_CACHE before construction works; it
        # was frozen into the signature at import time until Aug 2026 (#24). An empty
        # value counts as unset, so a caller blanking the variable gets the home-cache
        # default rather than a cache rooted at the working directory.
        if loc_save is None:
            loc_save = os.environ.get("LOC_SDF_CACHE") or os.path.join(
                os.path.expanduser("~"), ".cache", "nsm_sdf_cache"
            )

        # p_near_surface & p_further_from_surface must be >=0, <=1
        # sum of p_near_surface & p_further_from_surface must be <=1
        if isinstance(p_near_surface, (list, tuple)) & isinstance(
            p_further_from_surface, (list, tuple)
        ):
            for p_near, p_far in zip(p_near_surface, p_further_from_surface):
                check_probabilities(p_near)
                check_probabilities(p_far)
                check_probabilities_sum(p_near, p_far)
        elif isinstance(p_near_surface, float) & isinstance(p_further_from_surface, float):
            check_probabilities(p_near_surface)
            check_probabilities(p_further_from_surface)
            check_probabilities_sum(p_near_surface, p_further_from_surface)
        else:
            raise ValueError(
                "p_near_surface & p_further_from_surface must be floats or lists/tuples of floats"
            )

        self.list_mesh_paths = list_mesh_paths
        self.subsample = subsample
        self.n_pts = n_pts
        self.p_near_surface = p_near_surface
        self.p_further_from_surface = p_further_from_surface
        self.sigma_near = sigma_near
        self.sigma_far = sigma_far
        self.rand_function = rand_function
        self.center_pts = center_pts
        self.norm_pts = norm_pts
        self.scale_method = scale_method
        self.scale_jointly = scale_jointly
        self.joint_scale_buffer = joint_scale_buffer
        self.loc_save = loc_save
        self.random_seed = random_seed
        self.reference_mesh = reference_mesh
        self.verbose = verbose
        self.equal_pos_neg = equal_pos_neg
        self.fix_mesh = fix_mesh
        self.load_cache = load_cache
        self.save_cache = save_cache
        self.print_filename = print_filename
        self.multiprocessing = multiprocessing
        self.n_processes = n_processes
        self.store_data_in_memory = store_data_in_memory
        self.debug_memory = debug_memory
        self._memory_tracker = None
        self._memory_counter = 0
        self.test_load_times = test_load_times
        self.uniform_pts_buffer = uniform_pts_buffer

        # if store_data_in_memory is False & save_cache is False, then raise error
        if (self.store_data_in_memory is False) and (self.save_cache is False):
            raise ValueError(
                "If store_data_in_memory is False, then save_cache must be True."
                "when data not stored in memory, it is loaded from disk - but data is"
                "not saved to disk when save_cache is False."
            )

        # set defaults so can use same 'norm_and_scale_all_meshes' function
        # for single and multiple meshes. The hasattr guards are an initialization-order
        # contract: a subclass that wants its own values (MultiSurfaceSDFSamples does)
        # must set these attributes BEFORE calling super().__init__.
        if not hasattr(self, "reference_object"):
            self.reference_object = 0
        if not hasattr(self, "n_meshes"):
            self.n_meshes = 1

        self.max_radius = None
        self.center = None

        # preprocess inputs before proceeding
        self.preprocess_inputs()

        # Computed BEFORE load_reference_mesh, which mutates reference_mesh (an int or
        # path becomes a loaded Mesh; under multiprocessing, None plus a spill path) --
        # _reference_identity resolves the constructor's form.
        self.hash_params = self.get_hash_params()

        if save_cache is True:
            self.cache_folder = os.path.join(self.loc_save, today_date)
            os.makedirs(self.cache_folder, exist_ok=True)

        # get the combinations of points and sigmas to sample
        self.pt_sample_combos = self.get_pt_sample_combos()

        # preallocate reference mesh path to None
        self.reference_mesh_path = None

        if self.reference_mesh is not None:
            self.load_reference_mesh()

        # function to allow calling additional internal functions from subclasses.
        self.run_before_loading_data()

        self.data = []
        # Wrap this loading loop in a multiprocessing pool
        # This gate survives the §8.0.N ungating deliberately: log arguments evaluate
        # eagerly, and sched_getaffinity is a probe run solely to be logged.
        if self.verbose is True:
            try:
                logger.debug("CPU affinity:%s", os.sched_getaffinity(0))
            except AttributeError:
                # sched_getaffinity is not available on all platforms (eg., mac/windows)
                logger.debug("CPU affinity not available on this platform")
        if self.multiprocessing is True:
            with Pool(processes=self.n_processes) as pool:
                self.data = pool.map(self.load_mesh_step, self.list_mesh_paths)
        else:
            self.data = [self.load_mesh_step(loc_mesh) for loc_mesh in self.list_mesh_paths]

        # remove mesh paths that failed to load
        self.list_mesh_paths = [
            x for idx, x in enumerate(self.list_mesh_paths) if self.data[idx] is not None
        ]
        # remove data that failed to load
        self.data = [x for x in self.data if x is not None]

        if self.scale_jointly is True:
            self.norm_and_scale_all_meshes()

    def print_memory_summary(self):
        """Print a pympler summary diff every 100th call (``debug_memory=True`` only)."""
        if self._memory_tracker is None:
            self._memory_tracker = tracker.SummaryTracker()

        # every 100th iteration, print the memory summary
        if self._memory_counter % 100 == 0:
            self._memory_tracker.print_diff()

            # all_objects = muppy.get_objects()
            # numpy_arrays = [obj for obj in all_objects if isinstance(obj, np.ndarray)]
            # refs = gc.get_referrers(numpy_arrays[0])
            # print('REFERENCES TO NUMPY ARRAY')
            # print(refs)
        # size_info = asizeof.asized(self, detail=1)
        # print(size_info)
        # all_objects = muppy.get_objects()
        # memory_summary = summary.summarize(all_objects)
        # if self._memory_summary is not None:
        # self._memory_summary = memory_summary

        self._memory_counter += 1

    def run_before_loading_data(self):
        """Subclass hook, called after setup but before any subject loads."""
        pass

    def load_mesh_step(self, loc_mesh):
        """
        Per-subject worker: build or load one subject via ``get_sample_data_dict``.

        Returns its result unchanged -- a sample dict, a cache path, or None for a
        failed subject, which ``__init__`` then drops from ``list_mesh_paths`` and
        ``data``.
        """
        logger.debug("Loading mesh: %s", loc_mesh)

        if self.debug_memory is True:
            self.print_memory_summary()

        if self.multiprocessing is True:
            try:
                os.sched_setaffinity(0, range(multiprocessing.cpu_count()))
            except AttributeError:
                # sched_setaffinity is not available on all platforms (eg., mac/windows).
                # Forking a Pool worker resets CPU affinity on Linux; elsewhere there is
                # nothing to reset, so skipping it is the correct no-op rather than a
                # degraded path. Matches the guard on sched_getaffinity above.
                pass
        data = self.get_sample_data_dict(loc_mesh)

        if data is None:
            logger.warning("Skipping mesh: %s", loc_mesh)
            logger.warning("Error in loading")

        logger.debug("Data type: %s", type(data))
        logger.debug("Finished loading mesh: %s", loc_mesh)

        gc.collect()

        return data

    def _subject_new_pts(self, data):
        """
        One subject's per-surface ``new_pts`` arrays, from either storage mode.

        A disk-backed entry is the subject's ``.npz`` path, which flattens the list to
        ``new_pts_{i}`` keys; an in-memory entry holds ``new_pts`` as a list (of
        tensors in ``SDFSamples``, arrays in ``MultiSurfaceSDFSamples``, with None for
        a missing surface -- #67).
        """
        if self.store_data_in_memory is True:
            return [None if pts is None else np.asarray(pts) for pts in data["new_pts"]]
        data_ = np.load(data)
        return [data_[f"new_pts_{mesh_idx}"] for mesh_idx in range(self.n_meshes)]

    def norm_and_scale_all_meshes(self):
        """
        Compute the shared frame for every subject (``scale_jointly=True``).

        The shared center is the across-subject mean of each subject's
        ``reference_object`` surface centroid -- the other surfaces follow the reference,
        they do not pull on it. The shared scale is the largest radius any surface of any
        subject reaches from that center, grown by ``joint_scale_buffer`` so unseen
        subjects slightly larger than the training set still land inside the model's
        domain. One frame for everyone removes per-subject position/size as a source of
        variation.

        Nothing is rescaled here: the result is stored as ``self.center`` /
        ``self.max_radius`` and applied per batch in ``__getitem__``, so cached ``.npz``
        files and in-memory sample dicts alike stay in the unscaled frame.
        """
        logger.debug("Computing centering and scaling...")
        tic = time.time()
        centers = []
        for data in self.data:
            new_pts = self._subject_new_pts(data)
            centers.append(np.mean(new_pts[self.reference_object], axis=0))
        # new center:
        center = np.mean(centers, axis=0)

        logger.debug("Done computing centers")

        max_radii = []
        # for each data, comput the max radius (from the new/global center)
        for data in self.data:
            max_radius = 0
            for xyz in self._subject_new_pts(data):
                if xyz is None:
                    continue
                centered_xyz = xyz - center
                radii = np.linalg.norm(centered_xyz, axis=-1)
                max_radius_ = np.max(radii)
                if max_radius_ > max_radius:
                    max_radius = max_radius_
            max_radii.append(max_radius)
        max_radius = np.max(max_radii)
        # make the biggest radius a bit bigger than observed to enable model to
        # generalize to unseen data that is slightly larger than the observed data.
        max_radius = max_radius * (1 + self.joint_scale_buffer)
        logger.debug("Done computing max radii")

        self.max_radius = max_radius.astype(np.float32)
        self.center = center.astype(np.float32)
        toc = time.time()
        logger.info("Finished computing centering and scaling in %.3fs", toc - tic)
        logger.info("\tMax radius: %s", self.max_radius)
        logger.info("\tCenter: %s", self.center)

    def preprocess_inputs(self):
        """
        Validate/normalize constructor inputs before any data loads. Subclasses extend.

        Raises:
            ValueError: If ``scale_jointly`` is combined with ``center_pts`` or
                ``norm_pts`` -- joint scaling requires untouched per-subject coordinates.
        """

        if self.scale_jointly is True:
            if self.center_pts is True:
                raise ValueError(
                    "Scale jointly assumes centering at end... so center should be False"
                )
            if self.norm_pts is True:
                raise ValueError(
                    "Scale jointly assumes normalizing at end... so norm should be False"
                )

    def get_dict_pts(self, data, pts_name):
        """Flatten ``data[pts_name]`` to ``{pts_name}_{i}`` keys for ``np.savez``."""
        dict_pts = {}
        if isinstance(data[pts_name], list):
            for idx_, orig_pts_ in enumerate(data[pts_name]):
                dict_pts[f"{pts_name}_{idx_}"] = orig_pts_
        else:
            dict_pts[f"{pts_name}_0"] = data[pts_name]
        return dict_pts

    def save_data_to_cache(self, data, file_hash, filepath=None):
        """
        Write one subject's sample dict to a ``.npz`` cache file.

        The on-disk spelling differs from the in-memory one: ``xyz`` is stored as
        ``pts``, ``gt_sdf`` as ``sdfs``, and list-valued entries are flattened to
        indexed keys (``new_pts_0``, ...). ``unpack_numpy_data`` reverses all of it.

        Args:
            data (dict): Dictionary of data to save
            file_hash (str): Cache key; names the file ``{file_hash}.npz``
            filepath (str, optional): Write here instead (used to upgrade an existing
                cache file in place). Defaults to None.
        """
        # if want to cache, and new... then save.
        if filepath is None:
            filepath = os.path.join(self.cache_folder, f"{file_hash}.npz")
        dict_pts = {}
        dict_pts.update(self.get_dict_pts(data, "orig_pts"))
        dict_pts.update(self.get_dict_pts(data, "new_pts"))

        additional_keys = [
            "pos_idx",
            "neg_idx",
            "surf_idx",
            "center",
            "max_radius",
            "max_radius_xyz",
        ]
        for key in additional_keys:
            if key in data:
                dict_pts.update(self.get_dict_pts(data, key))
                # dict_pts[key] = data[key]

        # add pos/negative point indices

        np.savez(filepath, pts=data["xyz"], sdfs=data["gt_sdf"], **dict_pts)

    def get_sample_data_dict(self, loc_mesh):
        """
        Build or load one subject's samples; return them, or the path they are cached at.

        This shell runs once, for both classes; the class-specific halves are two
        private hooks. On a cache hit (``load_cache=True``): unreadable ``.npz`` files
        are deleted and the next candidate tried, then ``_upgrade_cached_layout``
        brings old cache layouts up to date -- resaving the file in place, or deleting
        it to force a rebuild. On a miss: ``_build_subject`` samples the subject from
        its mesh(es). The result is coerced to the storage mode in force: the sample
        dict itself (``store_data_in_memory=True``) or its cache path (False, the
        default).

        Args:
            loc_mesh (str or list): The subject's mesh path(s)

        Returns:
            dict, str or None: Sample dict, cache path, or None when the mesh failed
            to load -- ``__init__`` then drops the subject.
        """

        # The subject's cache key names its file: <key>.npz, searched recursively
        # across every date folder under loc_save (find_hash stops at the first match).
        file_hash = self.create_hash(loc_mesh)
        cached_file = self.find_hash(filename=f"{file_hash}.npz")

        file_loaded = False

        # --- Hit path: try to serve the subject from its cached .npz. -------------
        if (len(cached_file) > 0) and (self.load_cache is True):
            logger.info("Loading cached file")
            for cache_path in cached_file:
                # A corrupt file (a crash mid-write) is deleted rather than crashed
                # on; with no candidate left, the subject rebuilds below. Two guards
                # because both fail modes exist: not-a-zip-at-all, and a zip whose
                # central directory is broken (np.load raises BadZipFile).
                if not is_zipfile(cache_path):
                    logger.warning("DELETING BAD ZIP FILE: %s", cache_path)
                    os.remove(cache_path)
                    continue

                try:
                    data = unpack_numpy_data(np.load(cache_path))
                except zipfile.BadZipFile:
                    logger.warning("DELETING BAD ZIP FILE: %s", cache_path)
                    os.remove(cache_path)
                    continue

                # The class-specific upgrade decides one of three outcomes:
                #   rebuild -> the file is beyond repair: delete it and fall through
                #              to the miss path
                #   resave  -> data was upgraded in memory: rewrite the file, use it
                #   neither -> the file is current: use it as-is
                data, resave_data, rebuild = self._upgrade_cached_layout(data, cache_path)

                if rebuild:
                    logger.warning("\tDeleting file...")
                    os.remove(cache_path)
                    break

                if resave_data:
                    # resave data to cache - overwriting original.
                    self.save_data_to_cache(data, file_hash, filepath=cache_path)

                file_loaded = True
                break

        # --- Miss path: no usable cache, so sample the subject from its mesh(es). --
        if file_loaded is False:
            data = self._build_subject(loc_mesh)
            if data is None:
                return None

            if self.save_cache is True:
                self.save_data_to_cache(data, file_hash)
                cache_path = os.path.join(self.cache_folder, f"{file_hash}.npz")

        # --- Storage-mode coercion: hand back the sample dict itself (in-memory
        # datasets keep it) or the file's path (disk-backed datasets reload the
        # .npz on every __getitem__). ----------------------------------------------
        if self.store_data_in_memory is False:
            logger.debug("updating data to be cache path")
            # change the data to be the path to the saved cache file
            data = cache_path

        return data

    def _upgrade_cached_layout(self, data, cache_path):
        """
        Bring one cached subject up to the current layout, on a hit.

        Returns ``(data, resave, rebuild)``: ``resave`` asks the shell to overwrite
        the cache file with the upgraded ``data``; ``rebuild`` asks it to delete the
        file and rebuild from the meshes. Here: a cache from before the index-list
        layout gets its sign indices computed and resaved. ``unpack_numpy_data``
        always sets the index keys -- an absent group comes back as an EMPTY list --
        so the check is on length, not presence.
        """
        # Only one legacy layout exists for this class: caches written before sign
        # indices were stored at all. An empty unpacked list means the file has no
        # index group -- compute the indices now and ask the shell to resave, so the
        # upgrade is paid once per file, not on every load.
        if len(data["pos_idx"]) == 0 or len(data["neg_idx"]) == 0 or len(data["surf_idx"]) == 0:
            pos_idx, neg_idx, surf_idx = self.sdf_pos_neg_idx(data)
            data["pos_idx"] = pos_idx
            data["neg_idx"] = neg_idx
            data["surf_idx"] = surf_idx
            return data, True, False
        return data, False, False

    def _build_subject(self, loc_mesh):
        """
        Sample one subject cold -- the per-combo sampling loop, the class-specific
        half of ``get_sample_data_dict``. Each ``pt_sample_combos`` entry is drawn
        via ``read_mesh_get_sampled_pts``, with a per-combo seed derived from
        ``random_seed`` and keyed on the mesh contents.

        Returns:
            dict or None: The sample dict, indices included; None when the mesh
            failed to load.
        """
        logger.debug("Creating SDF Samples")
        if self.print_filename is True:
            logger.debug("%s", loc_mesh)
        data = {
            "xyz": torch.zeros((self.n_pts, 3)),
            "gt_sdf": torch.zeros((self.n_pts)),
        }
        pts_idx = 0

        if self.multiprocessing is True:
            if self.reference_mesh_path is not None:
                reference_mesh = Mesh(self.reference_mesh_path)
            else:
                reference_mesh = None
        else:
            reference_mesh = self.reference_mesh

        logger.debug("type of reference mesh: %s", type(reference_mesh))
        logger.debug("ref mesh path: %s", self.reference_mesh_path)

        # Keyed on the mesh contents, not on the subject's index and not on the cache
        # hash: an index would resample every subject when the list is reordered, and
        # the cache hash contains the mesh path, so it would resample everyone when
        # the data is moved. Read once here rather than per combo.
        content_key = mesh_content_key(loc_mesh) if self.random_seed is not None else None

        for idx_, (n_pts_, sigma_) in enumerate(self.pt_sample_combos):
            # A zero-count combo (p_near_surface=0, p_further_from_surface=0, or the
            # two summing to 1) samples nothing; passing it through would crash in
            # point_cloud_utils on an empty point cloud (#23). The seed key stays
            # idx_, so skipping one combo does not re-seed the others.
            if n_pts_ == 0:
                continue
            result_ = read_mesh_get_sampled_pts(
                loc_mesh,
                sigma=sigma_,
                n_pts=n_pts_,
                rand_function=self.rand_function,
                center_pts=self.center_pts,
                norm_pts=self.norm_pts,
                scale_method=self.scale_method,
                get_random=True,
                fix_mesh=self.fix_mesh,
                register_to_mean_first=False if reference_mesh is None else True,
                mean_mesh=reference_mesh,
                uniform_pts_buffer=self.uniform_pts_buffer,
                seed=derive_seed(self.random_seed, content_key, idx_),
            )

            if result_ is None:
                return None

            xyz_ = result_["pts"]
            sdfs_ = result_["sdf"]

            data["xyz"][pts_idx : pts_idx + n_pts_, :] = torch.from_numpy(xyz_).float()
            data["gt_sdf"][pts_idx : pts_idx + n_pts_] = torch.from_numpy(sdfs_).float()
            pts_idx += n_pts_

            if "orig_pts" not in data:
                # First combo that actually ran -- not necessarily combo 0, which a
                # zero count skips. Convert list of arrays to tensors.
                data["orig_pts"] = [torch.from_numpy(pts).float() for pts in result_["orig_pts"]]
                data["new_pts"] = [torch.from_numpy(pts).float() for pts in result_["new_pts"]]

        pos_idx, neg_idx, surf_idx = self.sdf_pos_neg_idx(data)
        data["pos_idx"] = pos_idx
        data["neg_idx"] = neg_idx
        data["surf_idx"] = surf_idx

        return data

    def get_pt_sample_combos(self):
        """
        The three sampling passes: near-surface, far-surface, and uniform.

        Counts follow ``p_near_surface`` / ``p_further_from_surface``; whatever the two
        (truncated) shares leave of ``n_pts`` is drawn uniformly from the buffered cube,
        marked by sigma None.

        Returns:
            list: List of [n_pts, sigma] pairs, one per pass
        """

        n_p_near_surface = int(self.n_pts * self.p_near_surface)
        n_p_further_from_surface = int(self.n_pts * self.p_further_from_surface)
        n_p_random = self.n_pts - n_p_near_surface - n_p_further_from_surface

        pt_sample_combos = [
            [n_p_near_surface, self.sigma_near],
            [n_p_further_from_surface, self.sigma_far],
            [n_p_random, None],
        ]

        return pt_sample_combos

    def sdf_pos_neg_idx(self, data):
        """
        Index the samples by SDF sign.

        The RAW index sets, not padded ones: the equal-share tiling happens at draw
        time (``_draw_sign_share``), sized by the subsample then in force, so cached
        bytes do not depend on ``subsample`` (#19).

        Args:
            data (dict): Dictionary of sampled points and SDFs

        Returns:
            tuple: (pos_idx, neg_idx, surf_idx) index tensors into ``data["xyz"]``

        Raises:
            ValueError: If every sample has the same sign -- equal batches cannot be
                drawn, and a mesh with no interior or no exterior samples is degenerate
                or unclosed (#41). Raising here keeps the failure at build time rather
                than a silent one-sign draw later.
        """

        pos_idx = (data["gt_sdf"] > 0).nonzero(as_tuple=True)[0]
        neg_idx = (data["gt_sdf"] < 0).nonzero(as_tuple=True)[0]
        surf_idx = (data["gt_sdf"] == 0).nonzero(as_tuple=True)[0]

        for sign, idx_ in (("positive", pos_idx), ("negative", neg_idx)):
            if idx_.numel() == 0:
                raise ValueError(
                    f"The mesh yielded no {sign} SDF samples, so equal positive/negative "
                    f"batches cannot be drawn from it. Is the mesh degenerate or unclosed?"
                )

        return pos_idx, neg_idx, surf_idx

    def find_hash(self, filename="hashed_filename.npz"):
        """
        Search the cache tree for ``filename``, stopping at the first match.

        Walks all of ``loc_save`` -- every date folder, not just today's -- so a cache
        written on an earlier day still hits.

        Args:
            filename (str, optional): Hashed filename. Defaults to 'hashed_filename.npz'.

        Returns:
            list: Zero or one path(s); the first match wins.
        """

        files = []
        for p, d, f in os.walk(self.loc_save):
            for filename_ in f:
                if filename_ == filename:
                    files.append(os.path.join(p, filename_))
                    logger.info("File found in cache: %s", files[-1])
                    return files

        return files

    def load_reference_mesh(self):
        """
        Resolve ``reference_mesh`` into a loaded ``Mesh`` -- or a path workers reload.

        Accepted forms: a ``Mesh``, used as-is; a path string; an int, indexing
        ``list_mesh_paths`` -- a multi-surface subject resolves to its registration
        surface(s), ``mesh_to_scale``, combined into one mesh when that is a list
        (#61); or a list of paths, indexed by ``reference_object``.

        With ``multiprocessing=True`` the mesh is then written to a timestamped ``.vtk``
        in the cache folder and ``self.reference_mesh`` set back to None: pool workers
        reload it from ``self.reference_mesh_path`` rather than receiving the object
        itself, and the timestamp keeps concurrent runs' reference meshes apart.

        Raises:
            TypeError: If reference mesh is not a string, int, list of strings, or
                mesh.Mesh object
        """

        logger.debug("Loading reference mesh:  %s", self.reference_mesh)

        if issubclass(type(self.reference_mesh), Mesh):
            pass
        elif isinstance(self.reference_mesh, int):
            if isinstance(self.list_mesh_paths[0], (str, Mesh)):
                self.reference_mesh = Mesh(self.list_mesh_paths[self.reference_mesh])
            elif isinstance(self.list_mesh_paths[0], (list, tuple)):
                # Multi-surface subject: the reference is the surface(s) that drive
                # registration -- combined into one mesh when mesh_to_scale is a list.
                subject = self.list_mesh_paths[self.reference_mesh]
                if isinstance(self.mesh_to_scale, (list, tuple)):
                    meshes = [Mesh(subject[idx]) for idx in self.mesh_to_scale]
                    self.reference_mesh = combine_meshes(meshes, list(range(len(meshes))))
                else:
                    self.reference_mesh = Mesh(subject[self.mesh_to_scale])
            else:
                raise TypeError("provided list_meshes wrong type")
        elif isinstance(self.reference_mesh, str):
            self.reference_mesh = Mesh(self.reference_mesh)
        elif isinstance(self.reference_mesh, list):
            # below will throw error in SDFSamples, but will work in MultiSurfaceSDFSamples
            # where self.mesh_to_scale is defined & a list/tuple type likely
            # (reference_object vs mesh_to_scale: see the MultiSurfaceSDFSamples docstring)
            self.reference_mesh = Mesh(self.reference_mesh[self.reference_object])
        else:
            raise TypeError(
                "Reference mesh must be a string, list of strings, or mesh.Mesh object, not",
                type(self.reference_mesh),
            )

        logger.debug("type of reference mesh: %s", type(self.reference_mesh))

        if self.multiprocessing is True:
            # update reference mesh path to be a has on the current time - so as to not end up with
            # multiple training runs of different tissues using the same reference mesh.
            # this happens because the random seed is set - so all models get the same random number.
            hashed_time = hashlib.md5(str(int(time.time())).encode()).hexdigest()
            self.reference_mesh_path = os.path.join(
                self.cache_folder, f"REFERENCE_MESH_{hashed_time}.vtk"
            )
            self.reference_mesh.save_mesh(self.reference_mesh_path)
            self.reference_mesh = None

    def _reference_identity(self):
        """
        ``reference_mesh``'s contribution to the cache key, content-stable.

        Resolution mirrors ``load_reference_mesh``, from the constructor's form: an int
        indexes ``list_mesh_paths`` (a multi-surface subject resolves to its
        ``mesh_to_scale`` surface(s)), a list is indexed by ``reference_object``, and
        the resulting path(s) -- or a directly-passed ``Mesh`` -- go through
        ``_identity``. A raw int used to be hashed as itself, so reordering
        ``list_mesh_paths`` re-aimed the reference while the key stood still -- the
        positional-coupling defect one level up.
        """
        reference = self.reference_mesh
        if reference is None:
            # No registration happens, so there is nothing to key.
            return None
        if issubclass(type(reference), Mesh):
            # A loaded mesh: key its geometry, not its object id.
            return _identity(reference)
        if isinstance(reference, int):
            # An index into list_mesh_paths: resolve to that subject's actual file(s),
            # so the key follows the file rather than the list position.
            subject = self.list_mesh_paths[reference]
            if isinstance(subject, (list, tuple)):
                # Multi-surface subject: the reference is the surface(s) that drive
                # registration, i.e. mesh_to_scale (an int, or a list to combine).
                to_scale = self.mesh_to_scale
                indices = to_scale if isinstance(to_scale, (list, tuple)) else [to_scale]
                return [_identity(subject[idx]) for idx in indices]
            # Single-surface subject: one path.
            return [_identity(subject)]
        if isinstance(reference, str):
            # A path, given directly.
            return [_identity(reference)]
        if isinstance(reference, (list, tuple)):
            # A list of per-surface paths: the reference is the reference_object'th.
            return [_identity(reference[self.reference_object])]
        # Any other type is refused by load_reference_mesh a moment after this runs.
        return [str(reference)]

    def get_hash_params(self):
        """
        The named entries of the cache key: everything, besides the subject's own
        meshes, that changes what ``get_sample_data_dict`` writes into the cache.

        ``cache_format`` versions the key itself (see ``CACHE_FORMAT``). Deliberately
        absent, settled in the section 8.0.F statement: ``joint_scale_buffer`` (the
        shared frame is stored on the dataset and applied per batch -- it never touches
        a cached byte) and ``mesh_names`` (names the surfaces, does not change samples).

        Returns:
            dict: ``{name: value}``, canonically serialized by ``create_hash``
        """

        return {
            "cache_format": CACHE_FORMAT,
            "n_pts": self.n_pts,
            "p_near_surface": self.p_near_surface,
            "p_further_from_surface": self.p_further_from_surface,
            "sigma_near": self.sigma_near,
            "sigma_far": self.sigma_far,
            "center_pts": self.center_pts,
            "norm_pts": self.norm_pts,
            "scale_method": self.scale_method,
            "rand_function": self.rand_function,
            "reference_mesh": self._reference_identity(),
            "fix_mesh": self.fix_mesh,
            "scale_jointly": self.scale_jointly,
            "uniform_pts_buffer": self.uniform_pts_buffer,
            "random_seed": self.random_seed,
        }

    def create_hash(self, loc_mesh):
        """
        The cache key for one subject: md5 over a canonical, named serialization.

        ``hash_params`` (see ``get_hash_params``) is extended with the subject's
        ``mesh_paths`` -- one content-stable identity per mesh, in order, so an in-place
        edit moves the key (``_file_identity``) -- and serialized with
        ``json.dumps(sort_keys=True)``. No entry's meaning depends on its position.
        ``subsample`` is deliberately not in the key: it no longer changes cached
        bytes -- index sets are stored raw and padded at draw (``_draw_sign_share``)
        -- and batch size is a serving parameter, not a property of the data.

        Args:
            loc_mesh (str, Mesh or list): The subject's mesh(es)

        Returns:
            str: Hashed string
        """

        # The parameter half of the key, computed once in __init__ (a copy, so the
        # stored dict is never mutated here)...
        params = dict(self.hash_params)
        # ...plus the subject half: one identity token per mesh, in surface order --
        # order is meaningful here ([bone, cart] is not [cart, bone]), which a plain
        # JSON list preserves.
        if isinstance(loc_mesh, (list, tuple)):
            params["mesh_paths"] = [_identity(mesh) for mesh in loc_mesh]
        else:
            params["mesh_paths"] = [_identity(loc_mesh)]
        # sort_keys makes the serialization canonical; default=str catches any stray
        # non-JSON value (e.g. a numpy scalar) rather than crashing the build.
        serialized = json.dumps(params, sort_keys=True, default=str)
        return hashlib.md5(serialized.encode()).hexdigest()

    def __len__(self):
        """Number of subjects that loaded successfully (failures are dropped)."""

        return len(self.data)

    def __getitem__(self, idx):
        """
        One training batch for subject ``idx``.

        Disk-backed datasets (``store_data_in_memory=False``) reload the subject's
        ``.npz`` on every call. With ``equal_pos_neg``, ``subsample / 2`` rows are drawn
        from each sign's index list, topped up with unconstrained draws when the halves
        round short. Under joint scaling the shared center/scale is applied here, to the
        batch, not to the cache.

        Args:
            idx (int): Subject index

        Returns:
            tuple: ``(batch, idx)``. ``batch["xyz"]`` is (subsample, 3) and
            ``batch["gt_sdf"]`` (subsample,), float32. The timing keys (``time``,
            ``size``, ``mb_per_sec``, ``whole_load_time``) appear only when
            ``test_load_times=True`` and the item came from disk (#22).
        """

        tic_whole_load = time.time()

        if self.store_data_in_memory is False:
            # if not storing in memory, then load from cache
            tic = time.time()
            data_ = np.load(self.data[idx])
            toc = time.time()
            time_ = toc - tic

            # get size of the numpy file in mb
            size = os.path.getsize(self.data[idx]) / 1e6

            if self.equal_pos_neg is True:
                list_keys_unpack = ["pos_idx", "neg_idx"]
            else:
                list_keys_unpack = []
            data_ = unpack_numpy_data(data_, list_additional_keys=list_keys_unpack)
        elif self.store_data_in_memory is True:
            # if storing in memory, then just get the data
            data_ = self.data[idx]
        else:
            raise ValueError("store_data_in_memory must be True or False")

        if self.subsample is not None:
            if self.equal_pos_neg is True:
                tic_rand_sample = time.time()
                samples_per_sign = int(self.subsample / 2)

                if isinstance(data_["pos_idx"], list):
                    idx_pos = _draw_sign_share(data_["pos_idx"][0], samples_per_sign)
                elif isinstance(data_["pos_idx"], torch.Tensor):
                    idx_pos = _draw_sign_share(data_["pos_idx"], samples_per_sign)
                else:
                    raise ValueError("pos_idx must be a list or tensor")

                if isinstance(data_["neg_idx"], list):
                    idx_neg = _draw_sign_share(data_["neg_idx"][0], samples_per_sign)
                elif isinstance(data_["neg_idx"], torch.Tensor):
                    idx_neg = _draw_sign_share(data_["neg_idx"], samples_per_sign)
                else:
                    raise ValueError("neg_idx must be a list or tensor")
                toc_rand_sample = time.time()
                logger.debug("rand sample time: %ss", toc_rand_sample - tic_rand_sample)

                tic_cat = time.time()
                idx_ = torch.cat((idx_pos, idx_neg), dim=0)
                toc_cat = time.time()
                logger.debug("concat time: %ss", toc_cat - tic_cat)

                if len(idx_) < self.subsample:
                    # if we don't have enough points, then just take random points
                    tic_rand = time.time()
                    perm = torch.randperm(data_["xyz"].size(0))
                    _idx_ = perm[: self.subsample - len(idx_)]
                    idx_ = torch.cat([idx_, _idx_], dim=0)
                    toc_rand = time.time()
                    logger.debug("rand additional sub sample time: %ss", toc_rand - tic_rand)

            else:
                perm = torch.randperm(data_["xyz"].size(0))
                idx_ = perm[: self.subsample]

            logger.debug(
                "idx_ size: %s idx_ min: %s idx_ max: %s", idx_.size(), idx_.min(), idx_.max()
            )
            logger.debug("equal neg pos %s", self.equal_pos_neg)

            # unpack the data
            xyz = data_["xyz"][idx_, :]
            sdf = data_["gt_sdf"][idx_]

            if (self.max_radius is not None) and (self.center is not None):
                # if normalizing at the group level, then normalize here.
                tic_norm = time.time()
                xyz = (xyz - self.center) / self.max_radius
                sdf = sdf / self.max_radius
                toc_norm = time.time()
                logger.debug("norm time: %ss", toc_norm - tic_norm)

            data_ = {
                "xyz": xyz,
                "gt_sdf": sdf,
            }

            toc_whole_load = time.time()
            time_whole_load = toc_whole_load - tic_whole_load

            if (self.test_load_times is True) and (self.store_data_in_memory is False):
                data_["time"] = time_
                data_["size"] = size
                data_["mb_per_sec"] = size / time_
                data_["whole_load_time"] = time_whole_load

        return data_, idx


class MultiSurfaceSDFSamples(SDFSamples):
    """
    Dataset class for sampling SDFs from multiple mesh surfaces with support for
    multi-surface rigid registration.

    Extends SDFSamples to handle multiple anatomical surfaces simultaneously,
    such as bone + cartilage or medial + lateral menisci.

    Args:
        list_mesh_paths (list): One entry per subject, each a list of per-surface mesh
            paths in a fixed surface order, e.g. ``[[bone, cart], ...]``. A None entry
            marks a subject's missing surface -- accepted here, but the build path for
            it has never worked end to end (#67).

        mesh_names (list of str, optional): Human-readable names for the surfaces, in
            the same order as each subject's mesh-path list -- the one place that
            ordering is defined, which is why the names belong here rather than in a
            free-floating config key (#52). ``train_deep_sdf`` adopts them into
            ``config["mesh_names"]`` (and refuses a disagreeing config), so they end up
            in ``model_params_config.json``. Deliberately NOT in the cache key: names
            do not change sampled data.

        mesh_to_scale (int or list): Index(es) of mesh(es) to use for registration and scaling.
            - If int: Uses single mesh for registration (original behavior)
            - If list: Combines multiple meshes for joint registration
            Example: mesh_to_scale=[0, 1] for medial + lateral menisci registration

        reference_object (int): Index of the surface whose sampled points anchor
            centering, and which element of a list-valued reference_mesh is used.
            A separate knob from mesh_to_scale (which surface(s) drive registration
            and scaling); the two are not kept in sync, and why they are separate is
            an open question inherited from the original implementation.

        scale_all_meshes (bool): Scale using every surface's points (True, default) or
            only ``mesh_to_scale``'s (False). See read_meshes_get_sampled_pts.
        center_all_meshes (bool): Center on every surface's points (True) or only
            ``mesh_to_scale``'s (False, default).

        n_pts (int or list): Per-surface sample counts; a scalar or one-element list is
            broadcast to every surface. The per-surface floats (p_near_surface,
            p_further_from_surface, sigma_near, sigma_far) broadcast the same way.

        Other args: Same as SDFSamples parent class

    Notes:
        - When mesh_to_scale is a list, meshes are combined with the pymskt Mesh `+`
          operator (see combine_meshes)
        - Joint registration improves alignment when multiple related surfaces should
          be considered together rather than individually
        - ``__getitem__`` returns ``batch["gt_sdf"]`` with shape (subsample,
          n_surfaces): every surface's signed distance to every sampled point, not just
          the surface the point was drawn around.
    """

    @honour_verbose
    def __init__(
        self,
        list_mesh_paths,
        subsample,
        n_pts=500000,
        p_near_surface=0.4,
        p_further_from_surface=0.4,
        sigma_near=0.01,
        sigma_far=0.1,
        rand_function="normal",
        center_pts=True,
        norm_pts=False,
        scale_method="max_rad",
        scale_jointly=False,
        joint_scale_buffer=0.1,
        loc_save=None,
        save_cache=True,
        load_cache=True,
        random_seed=None,
        reference_mesh=None,
        verbose=False,
        equal_pos_neg=True,
        fix_mesh=True,
        print_filename=False,
        test_load_times=True,
        uniform_pts_buffer=0.0,
        # Multi surface specific
        scale_all_meshes=True,
        center_all_meshes=False,
        mesh_to_scale=0,
        reference_object=0,
        store_data_in_memory=False,
        multiprocessing=True,
        debug_memory=False,
        n_processes=2,
        mesh_names=None,
    ):
        # Validate before any file I/O so a bad declaration fails at construction.
        if mesh_names is not None and len(mesh_names) != len(list_mesh_paths[0]):
            raise ValueError(
                f"mesh_names has {len(mesh_names)} entries but each subject has "
                f"{len(list_mesh_paths[0])} surfaces. The names must match each "
                f"subject's mesh-path list, in order."
            )
        self.mesh_names = mesh_names

        # if n_pts is not a list, then make it a list that is
        # the same length as the number of meshes.
        if not isinstance(n_pts, (list, tuple)):
            n_pts = [n_pts] * len(list_mesh_paths[0])
        if len(n_pts) == 1 and len(list_mesh_paths[0]) > 1:
            n_pts = n_pts * len(list_mesh_paths[0])

        self.times = []
        self.data_size = []
        self.mb_per_sec = []
        self.test_load_times = test_load_times
        # Multi surface specific
        self.mesh_to_scale = mesh_to_scale
        self.total_n_pts = sum(n_pts)
        self.scale_all_meshes = scale_all_meshes
        self.center_all_meshes = center_all_meshes
        self.n_meshes = len(list_mesh_paths[0])
        self.reference_object = reference_object

        super().__init__(
            list_mesh_paths=list_mesh_paths,
            subsample=subsample,
            n_pts=n_pts,
            p_near_surface=p_near_surface,
            p_further_from_surface=p_further_from_surface,
            sigma_near=sigma_near,
            sigma_far=sigma_far,
            rand_function=rand_function,
            center_pts=center_pts,
            norm_pts=norm_pts,
            scale_method=scale_method,
            scale_jointly=scale_jointly,
            joint_scale_buffer=joint_scale_buffer,
            loc_save=loc_save,
            save_cache=save_cache,
            load_cache=load_cache,
            random_seed=random_seed,
            reference_mesh=reference_mesh,
            verbose=verbose,
            equal_pos_neg=equal_pos_neg,
            fix_mesh=fix_mesh,
            print_filename=print_filename,
            store_data_in_memory=store_data_in_memory,
            multiprocessing=multiprocessing,
            n_processes=n_processes,
            debug_memory=debug_memory,
            test_load_times=test_load_times,
            uniform_pts_buffer=uniform_pts_buffer,
        )

    def preprocess_inputs(self):
        """As the parent's, plus: count the surfaces and broadcast scalar per-surface
        parameters (p_near_surface, sigma_near, ...) into per-surface lists."""
        super().preprocess_inputs()

        if isinstance(self.list_mesh_paths[0], (list, tuple)):
            self.n_meshes = len(self.list_mesh_paths[0])
        elif isinstance(self.list_mesh_paths[0], (str, Mesh)):
            self.n_meshes = len(self.list_mesh_paths)

        if not isinstance(self.p_near_surface, (list, int)):
            self.p_near_surface = [self.p_near_surface] * self.n_meshes
        if not isinstance(self.p_further_from_surface, (list, int)):
            self.p_further_from_surface = [self.p_further_from_surface] * self.n_meshes
        if not isinstance(self.sigma_near, (list, int)):
            self.sigma_near = [self.sigma_near] * self.n_meshes
        if not isinstance(self.sigma_far, (list, int)):
            self.sigma_far = [self.sigma_far] * self.n_meshes
        if not isinstance(self.n_pts, (list, int)):
            self.n_pts = [self.n_pts] * self.n_meshes

    def run_before_loading_data(self):
        """Precompute each surface's per-sign batch share before any subject loads."""
        self.get_samples_per_sign()

    def test_if_idx_in_range(self, data):
        """
        Whether every pos/neg index actually points into ``data["xyz"]``.

        Guards against stale caches: ``remove_overlapping_points`` shrinks the point
        set, so index lists computed before an overlap pass can exceed it. A False
        return makes ``get_sample_data_dict`` delete the cache file and rebuild.
        """
        n_pts = data["xyz"].shape[0]

        for name in ["pos_idx", "neg_idx"]:
            indices = data[name]
            max_idx = 0
            for tensor in indices:
                if tensor.numel() == 0:
                    # A missing (None) surface has empty index lists; torch.max
                    # raises on an empty tensor, and empty is trivially in range.
                    continue
                max_idx = torch.max(tensor)
                if max_idx >= n_pts:
                    return False

        return True

    def get_sample_data_dict(self, loc_meshes):
        """
        The parent's shell, after three multi-surface pre-steps: normalize
        ``loc_meshes`` to a list, append the subject to
        ``list_meshes_started_loading.log`` in ``loc_save`` (a crash mid-build names
        its subject), and refresh the per-sign batch shares. The class-specific
        halves are ``_build_subject`` (per-surface sampling) and
        ``_upgrade_cached_layout``.

        Args:
            loc_meshes (list or str): The subject's per-surface mesh path(s).

        Returns:
            dict, str or None: As the parent -- sample dict, cache path, or None for a
            failed subject.
        """
        if type(loc_meshes) not in (tuple, list):
            loc_meshes = [loc_meshes]

        with open(os.path.join(self.loc_save, "list_meshes_started_loading.log"), "a") as f:
            f.write(str(loc_meshes) + "\n")

        # get the number of points to sample per mesh / sign(in/out or pos/neg)
        self.get_samples_per_sign()

        return super().get_sample_data_dict(loc_meshes)

    def _upgrade_cached_layout(self, data, cache_path):
        """
        As the parent's, for the multi-surface cache layout: the overlap pass runs on
        every hit, so a pre-overlap-pass cache shrinks and resaves (without touching
        the index lists -- overlap removal does not change how many surfaces there
        are, which is what the recompute condition checks); index lists missing or of
        the wrong surface count are recomputed; and indices that outlived their point
        set -- an overlap pass shrank ``xyz`` after they were computed -- ask for
        delete-and-rebuild, because served as-is they would read the wrong rows or
        step off the end of the array.
        """
        # Three checks, and the order matters: `resave_data` accumulates across the
        # first two, and the in-range guard runs LAST so a file that fails it is
        # deleted, never resaved.
        resave_data = False

        # (1) Overlap pass. Caches written before remove_overlapping_points existed
        # still hold points labeled inside two surfaces; dropping them shrinks
        # xyz/gt_sdf. The pass is idempotent, so on a current cache it removes
        # nothing (in_in == 0) and costs one scan.
        data, in_in = self.remove_overlapping_points(data)

        if in_in > 0:
            resave_data = True

        # (2) Index lists absent, or built for a different number of surfaces:
        # recompute them. The comparison is per-surface list length against
        # n_meshes -- note this deliberately does NOT fire when (1) shrank the
        # point set, because overlap removal changes row counts, not surface counts.
        if (
            ("pos_idx" not in data)
            or (len(data["pos_idx"]) != self.n_meshes)
            or ("neg_idx" not in data)
            or (len(data["neg_idx"]) != self.n_meshes)
            or ("surf_idx" not in data)
            or (len(data["surf_idx"]) != self.n_meshes)
        ):
            logger.debug("getting pos/neg")
            pos_idx, neg_idx, surf_idx = self.sdf_pos_neg_idx(data)
            data["pos_idx"] = pos_idx
            data["neg_idx"] = neg_idx
            data["surf_idx"] = surf_idx

            resave_data = True

        # (3) Every index must still point inside xyz. Exactly because (2) does not
        # recompute after an overlap shrink, a pre-overlap cache upgraded by (1) can
        # carry indices that now step past the smaller point set; served as-is they
        # would read the wrong rows, so the whole file is rebuilt from the meshes.
        if self.test_if_idx_in_range(data) is False:
            logger.warning("Indices out of range! %s", cache_path)
            return data, False, True

        return data, resave_data, False

    def _build_subject(self, loc_meshes):
        """
        As the parent's, per surface: ``gt_sdf`` is built (sum(n_pts), n_surfaces),
        with a missing (None) surface's column all-NaN; the ICP transform from the
        first combo carries into the rest; and ``remove_overlapping_points`` drops
        points labeled inside two or more surfaces before the indices are computed.
        """
        logger.debug("Creating SDF Samples")
        if self.print_filename is True:
            logger.debug("%s", loc_meshes)

        data = {
            "xyz": torch.zeros((sum(self.n_pts), 3)),
            "gt_sdf": torch.zeros((sum(self.n_pts), len(loc_meshes))),
        }
        pts_idx = 0
        icp_transform = None

        if self.multiprocessing is True:
            if self.reference_mesh_path is not None:
                reference_mesh = Mesh(self.reference_mesh_path)
            else:
                reference_mesh = None
        else:
            reference_mesh = self.reference_mesh

        logger.debug("type of reference mesh: %s", type(reference_mesh))
        logger.debug("ref mesh path: %s", self.reference_mesh_path)

        content_key = mesh_content_key(loc_meshes) if self.random_seed is not None else None

        for idx_, (n_pts_, sigma_) in enumerate(self.pt_sample_combos):
            # A combo asked to sample nothing anywhere would crash in
            # point_cloud_utils on an empty point cloud (#23). The seed key stays
            # idx_, so skipping one combo does not re-seed the others.
            if sum(n_pts_) == 0:
                continue
            tic = time.time()
            result_ = read_meshes_get_sampled_pts(
                loc_meshes,
                sigma=sigma_,
                n_pts=n_pts_,
                rand_function=self.rand_function,
                center_pts=self.center_pts,
                norm_pts=self.norm_pts,
                scale_method=self.scale_method,
                get_random=True,
                fix_mesh=self.fix_mesh,
                register_to_mean_first=False if reference_mesh is None else True,  #
                mean_mesh=reference_mesh,  #
                uniform_pts_buffer=self.uniform_pts_buffer,
                # Multi surface specific
                mesh_to_scale=self.mesh_to_scale,
                scale_all_meshes=self.scale_all_meshes,
                center_all_meshes=self.center_all_meshes,
                icp_transform=icp_transform,
                seed=derive_seed(self.random_seed, content_key, idx_),
            )

            if result_ is None:
                return None

            icp_transform = result_["icp_transform"]

            toc = time.time()
            logger.debug("%s - %s: %ss", idx_, sigma_, toc - tic)

            if "orig_pts" not in data:
                # First combo that actually ran -- not necessarily combo 0,
                # which a zero count skips.
                data["orig_pts"] = result_["orig_pts"]
                data["new_pts"] = result_["new_pts"]

            xyz_ = result_["pts"]
            sdfs_ = result_["sdf"]

            data["xyz"][pts_idx : pts_idx + sum(n_pts_), :] = torch.from_numpy(xyz_).float()

            for mesh_idx, _sdfs_ in enumerate(sdfs_):
                if _sdfs_ is None:
                    # If mesh was None, fill with NaN to indicate missing data
                    # don't need this now.. but can handle training on incomplete data in the future.
                    data["gt_sdf"][pts_idx : pts_idx + sum(n_pts_), mesh_idx] = float("nan")
                else:
                    data["gt_sdf"][pts_idx : pts_idx + sum(n_pts_), mesh_idx] = torch.from_numpy(
                        _sdfs_
                    ).float()
            pts_idx += sum(n_pts_)

        # Drop points that have are labeled as being inside
        # 2 objects - clearly this is an error.
        data, in_in = self.remove_overlapping_points(data)

        logger.debug("getting pos/neg")
        pos_idx, neg_idx, surf_idx = self.sdf_pos_neg_idx(data)
        data["pos_idx"] = pos_idx
        data["neg_idx"] = neg_idx
        data["surf_idx"] = surf_idx

        return data

    def get_samples_per_sign(self):
        """
        Each surface's per-sign share of a batch: ``subsample`` split across surfaces
        in proportion to their ``n_pts``, halved per sign, truncated. Truncation means
        the shares can sum below ``subsample``; ``__getitem__`` tops the batch up with
        unconstrained draws. Stored as ``self.samples_per_sign_``.
        """
        samples_per_mesh = [
            int((n_pts_ / self.total_n_pts) * self.subsample) for n_pts_ in self.n_pts
        ]

        # setup samples per sign
        self.samples_per_sign_ = []
        for subsample_ in samples_per_mesh:
            samples_per_sign = int(subsample_ / 2)
            logger.debug("%s", samples_per_sign)
            self.samples_per_sign_.append(samples_per_sign)

    def remove_overlapping_points(self, data):
        """
        Drop points labeled inside two or more surfaces -- anatomically impossible,
        so such a point is a segmentation/meshing error and would teach the model a
        false interior.

        All-NaN columns (missing surfaces) are excluded from the count. Removal
        shrinks ``xyz``/``gt_sdf``, which is why index lists must be recomputed after
        this runs (see ``test_if_idx_in_range``).

        Args:
            data (dict): Sample dict with ``gt_sdf`` of shape (n, n_surfaces)

        Returns:
            tuple: ``(data, n_removed)``; a nonzero count on a cache hit triggers a
            resave.
        """
        sdf_ = data["gt_sdf"].clone()

        # Check if we have None values (represented as NaN)
        non_none_mask = ~torch.isnan(sdf_).all(dim=0)

        if non_none_mask.sum() < 2:
            return data, 0  # Can't have overlaps with fewer than 2 surfaces

        # Only process non-None columns for overlap detection
        sdf_filtered = sdf_[:, non_none_mask]

        # "Overlapping" means inside two or more surfaces, where inside is a strictly
        # negative SDF. Count per point — a sign-sum test is equivalent to this count
        # only at exactly two surfaces.
        inside_count = torch.sum(sdf_filtered < 0, dim=1)

        out_all = torch.sum(inside_count == 0)
        in_one = torch.sum(inside_count == 1)
        in_in = torch.sum(inside_count >= 2)

        # Create mask for points to keep (not overlapping)
        keep_mask = inside_count < 2

        # Apply the mask to remove overlapping points from the full dataset
        # This preserves the None columns while removing problematic points
        data["gt_sdf"] = data["gt_sdf"][keep_mask, :]
        data["xyz"] = data["xyz"][keep_mask, :]

        logger.debug("inside_count shape %s", inside_count.shape)
        logger.debug("inside_count %s", inside_count)
        logger.debug("outside all surfaces %s", out_all)
        logger.debug("inside exactly one %s", in_one)
        logger.debug("inside two or more %s", in_in)
        logger.info("Removed %s overlapping points", in_in)

        return data, in_in

    def get_pt_sample_combos(self):
        """
        As the parent's, with per-surface counts: each pass pairs a count list and a
        sigma list, and the uniform pass carries one None per surface.

        Returns:
            list: List of [n_pts_list, sigma_list] pairs, one per pass
        """
        n_p_near_surface = [
            int(n_pts_ * p_near) for n_pts_, p_near in zip(self.n_pts, self.p_near_surface)
        ]
        n_p_further_from_surface = [
            int(n_pts_ * p_far) for n_pts_, p_far in zip(self.n_pts, self.p_further_from_surface)
        ]
        n_p_random = [
            n_pts_ - n_p_near - n_p_far
            for n_pts_, n_p_near, n_p_far in zip(
                self.n_pts, n_p_near_surface, n_p_further_from_surface
            )
        ]

        pt_sample_combos = [
            [n_p_near_surface, self.sigma_near],
            [n_p_further_from_surface, self.sigma_far],
            [
                n_p_random,
                [
                    None,
                ]
                * self.n_meshes,
            ],
        ]

        return pt_sample_combos

    def get_hash_params(self):
        """As the parent's (the per-surface lists ride in the shared entries), plus the
        multi-surface parameters -- ``mesh_to_scale`` above all: it decides which surface
        drives centering and normalization, so two runs differing in it are in different
        coordinate frames entirely (#19 (a), unkeyed until Aug 2026)."""
        params = super().get_hash_params()
        params.update(
            {
                "mesh_to_scale": self.mesh_to_scale,
                "reference_object": self.reference_object,
                "scale_all_meshes": self.scale_all_meshes,
                "center_all_meshes": self.center_all_meshes,
            }
        )
        return params

    def sdf_pos_neg_idx(self, data):
        """
        Per-surface sign indices.

        As the parent's, per surface, and RAW like the parent's: the equal-share
        tiling happens at draw time (``_draw_sign_share``), so cached bytes do not
        depend on ``subsample`` (#19). A surface nothing is drawn from -- a zero
        share, or the all-NaN column of a missing surface -- keeps empty index lists
        instead of raising.

        Args:
            data (dict): Sample dict with ``gt_sdf`` of shape (n, n_surfaces)

        Returns:
            tuple: (pos_idx, neg_idx, surf_idx) -- each a list with one index tensor
            per surface, indexing into ``data["xyz"]``

        Raises:
            ValueError: If a surface that *is* drawn from has every sample on one side
                (#41) -- e.g. a surface nested inside another loses every interior
                point to remove_overlapping_points.
        """

        pos_idx = []
        neg_idx = []
        surf_idx = []
        logger.debug("data %s %s", data["xyz"].shape, data["gt_sdf"].shape)

        for mesh_idx in range(self.n_meshes):

            samples_per_sign = self.samples_per_sign_[mesh_idx]

            mesh_sdfs = data["gt_sdf"][:, mesh_idx].clone()
            pos_idx_ = (mesh_sdfs > 0).nonzero(as_tuple=True)[0]
            neg_idx_ = (mesh_sdfs < 0).nonzero(as_tuple=True)[0]
            surf_idx_ = (mesh_sdfs == 0).nonzero(as_tuple=True)[0]

            # A surface nothing is drawn from may be empty: an all-NaN column is a
            # missing (None) surface, and a zero subsample share means __getitem__ never
            # samples it. Its empty index lists are handled -- randperm(0) draws nothing.
            surface_is_drawn_from = samples_per_sign > 0 and not torch.isnan(mesh_sdfs).all()

            if surface_is_drawn_from:
                for sign, idx_ in (("positive", pos_idx_), ("negative", neg_idx_)):
                    if idx_.numel() == 0:
                        # A surface with no interior samples trains to garbage (#41);
                        # raising here keeps the failure at build time rather than a
                        # silent one-sign draw later.
                        raise ValueError(
                            f"Surface {mesh_idx} has no {sign} SDF samples, so its "
                            f"equal positive/negative batch share cannot be drawn. A "
                            f"surface nested inside another loses every interior point "
                            f"to remove_overlapping_points."
                        )

            pos_idx.append(pos_idx_)
            neg_idx.append(neg_idx_)
            surf_idx.append(surf_idx_)

        return pos_idx, neg_idx, surf_idx

    def __getitem__(self, idx):
        """
        One training batch for subject ``idx``.

        As the parent's, with the equal-sign draw done per surface: each surface
        contributes its ``samples_per_sign_`` share from each of its own index lists
        (a zero-share surface contributes nothing), and the batch is topped up with
        unconstrained draws when the truncated shares sum below ``subsample``.

        Args:
            idx (int): Subject index

        Returns:
            tuple: ``(batch, idx)``. ``batch["xyz"]`` is (subsample, 3) and
            ``batch["gt_sdf"]`` (subsample, n_surfaces), float32. Timing keys as in
            the parent (#22).
        """
        tic_whole_load = time.time()
        if self.store_data_in_memory is False:
            # if not storing in memory, then load from cache

            # if self.test_load_times is True:
            tic = time.time()
            data_ = np.load(self.data[idx])
            toc = time.time()
            time_ = toc - tic
            # self.times.append(time_)

            # get size of the numpy file in mb
            size = os.path.getsize(self.data[idx]) / 1e6
            # self.sizes.append(size)

            # self.mb_per_sec.append(size / time_)

            logger.debug("size: %smb, time: %ss, mb/s: %smb/s", size, time_, size / time_)

            if self.equal_pos_neg is True:
                list_keys_unpack = ["pos_idx", "neg_idx"]
            else:
                list_keys_unpack = []
            tic_unpack = time.time()
            data_ = unpack_numpy_data(data_, list_additional_keys=list_keys_unpack)
            toc_unpack = time.time()
            logger.debug("unpack time: %ss", toc_unpack - tic_unpack)

        elif self.store_data_in_memory is True:
            # if storing in memory, then just get the data
            data_ = self.data[idx]
        else:
            raise ValueError("store_data_in_memory must be True or False")

        if self.subsample is not None:
            if self.equal_pos_neg is True:
                # get number of points for each mesh
                # this is weighted by the number of points in the mesh
                # relative to the total number of points in the dataset
                # samples_per_mesh = [int((n_pts_/self.total_n_pts) * self.subsample) for n_pts_ in self.n_pts]
                idx_ = []
                for mesh_idx, samples_per_sign in enumerate(self.samples_per_sign_):
                    tic_mesh = time.time()
                    # get number of positive and negative points for this mesh
                    # samples_per_sign = int(subsample_/2)
                    logger.debug("samples_per_sign %s", samples_per_sign)
                    logger.debug("mesh idx %s", mesh_idx)
                    logger.debug("data_ pos %s", data_["pos_idx"])

                    if samples_per_sign == 0:
                        continue

                    # get random indices for positive and negative points
                    idx_pos = _draw_sign_share(data_["pos_idx"][mesh_idx], samples_per_sign)
                    idx_neg = _draw_sign_share(data_["neg_idx"][mesh_idx], samples_per_sign)

                    # combine positive and negative indices
                    idx_ += [idx_pos, idx_neg]
                    toc_mesh = time.time()
                    logger.debug("mesh %s time: %ss", mesh_idx, toc_mesh - tic_mesh)

                tic_cat = time.time()
                # combine indices for all meshes
                idx_ = torch.cat(idx_, dim=0)
                toc_cat = time.time()
                logger.debug("cat time: %ss", toc_cat - tic_cat)

                if len(idx_) < self.subsample:
                    # if we don't have enough points, then just take random points
                    tic_rand = time.time()
                    perm = torch.randperm(data_["xyz"].size(0))
                    _idx_ = perm[: self.subsample - len(idx_)]
                    idx_ = torch.cat([idx_, _idx_], dim=0)
                    toc_rand = time.time()
                    logger.debug("rand additional sub sample time: %ss", toc_rand - tic_rand)

            else:
                perm = torch.randperm(data_["xyz"].size(0))
                idx_ = perm[: self.subsample]

            logger.debug(
                "idx_ size: %s idx_ min: %s idx_ max: %s", idx_.size(), idx_.min(), idx_.max()
            )
            logger.debug("equal neg pos %s", self.equal_pos_neg)

            xyz = data_["xyz"][idx_, :]
            sdf = data_["gt_sdf"][idx_, :]

            if (self.max_radius is not None) and (self.center is not None):
                tic_scaling = time.time()
                xyz = (xyz - self.center) / self.max_radius
                sdf = sdf / self.max_radius
                toc_scaling = time.time()
                logger.debug("scaling time: %ss", toc_scaling - tic_scaling)

            data_ = {
                "xyz": xyz,
                "gt_sdf": sdf,
            }

            toc_whole_load = time.time()

            # Same guard as SDFSamples.__getitem__: in-memory items have no disk load to
            # time, so the timing keys are only emitted when one was measured (#22).
            if (self.test_load_times is True) and (self.store_data_in_memory is False):
                data_["time"] = time_
                data_["size"] = size
                data_["mb_per_sec"] = size / time_
                data_["whole_load_time"] = toc_whole_load - tic_whole_load

        return data_, idx
