from . import correspondence_metrics, interpolate, refine_mesh, triangle_metrics
from .main import *  # noqa: F401,F403  # re-export; see docs/ARCHITECTURE.md star-import trap

# What this package exports; see NSM/models/__init__.py for the rule and what it does not do.
#
# The four submodules on the import line above are here because they were not reachable
# from `NSM.mesh` at all -- `from .main import *` binds `main` and nothing else, so 2,104
# lines, 16% of the library, could only be reached by naming the submodule (ARCHITECTURE
# section 2.1). Binding them is purely additive: it removes no name, and every dependency
# they have is one `main` already imported, so `import NSM.mesh` costs 0.002 s more and
# loads no new top-level module.
__all__ = [  # noqa: F405 - every name below comes through the star re-export above
    "coarse_bounds_from_sign_change",
    "correspondence_metrics",
    "create_grid_samples",
    "create_grid_samples_in_bounds",
    "create_mesh",
    "create_mesh_adaptive",
    "crop_sdf_to_narrow_band",
    "decode_sdf",
    "get_sdfs",
    "interpolate",
    "main",
    "refine_mesh",
    "scale_mesh",
    "scale_mesh_",
    "sdf_grid_to_mesh",
    "sdf_grid_to_mesh_vtk",
    "triangle_metrics",
]
