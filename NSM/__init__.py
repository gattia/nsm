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


__version__ = "0.2.0"
