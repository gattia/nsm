import importlib.metadata as _metadata
import logging
import os

from . import utils

# The stdlib library idiom: NSM emits nothing until the host configures logging, and no
# "No handlers could be found" noise is possible. A library never configures the root
# logger -- reconstruct/main.py did, at module scope, until Aug 2026 (plan §8.0.G, #58).
logging.getLogger("NSM").addHandler(logging.NullHandler())

# os.environ["MKL_NUM_THREADS"] = "1"
# os.environ["OMP_NUM_THREADS"] = "1"
# os.environ["OPENBLAS_NUM_THREADS"] = "1"
# os.environ["NUMEXPR_NUM_THREADS"] = "1"


# The version of the installed distribution, which setuptools-scm derives from the git
# tag at build time. Not a literal: one sat here saying 0.2.0 while the tree was 269
# commits and 34 breaking changes past that tag, and one sat at 0.0.1 for years before it.
#
# The fallback is not decoration. `importlib.metadata` answers about *installed*
# distributions, and NSM is reachable without being one -- the downstream consumer inserts
# its checkout on `sys.path` at runtime. Raising here would break `import NSM` itself.
try:
    __version__ = _metadata.version("NSM")
except _metadata.PackageNotFoundError:  # pragma: no cover - needs an uninstalled checkout
    __version__ = "0.0.0+unknown"
