"""
One-release bridge for the deprecated ``verbose=`` parameter (#58, plan §8.0.G).

Nothing here is permanent API. NSM's output goes through ``logging`` from Aug 2026 on;
``verbose=`` survives one release so that callers who pass it do not lose their output
without notice -- all of it, at ``DEBUG``, since that is where most of what it used to
print now sits.

DELETE THIS FILE at v0.4.0, together with every ``verbose`` parameter it decorates and
the ``@honour_verbose`` line above each of them. That removal is Breaking and belongs
with the release that carries it.

Why the flag is *honoured* rather than warn-and-no-op, which is what deprecating it
would normally mean: a ``DeprecationWarning`` is invisible under Python's default filter
outside ``__main__`` (measured 2026-08-26), and the consumer we ship to calls NSM from
inside a module. Warn-and-no-op would therefore be indistinguishable, for them, from
deleting their output silently. The warning is the announcement; this release of overlap
is what makes it an announcement rather than a fait accompli.

Two rules the implementation encodes:

* **A host that has configured logging is never overridden.** The handler goes on only
  when no handler anywhere in the ``NSM.*`` ancestry would already receive the records
  -- which is what ``logging`` itself checks when it dispatches.
* **One notice per user call, not per internal hop.** ``reconstruct_mesh`` forwards
  ``verbose`` down through several bridged functions; the depth counter means only the
  outermost installs a handler and only the outermost warns.

Background: ``docs/KNOWN_ISSUES.md`` § History, and the CHANGELOG's Deprecated entry.
"""

import functools
import inspect
import logging
import sys
import threading
import warnings
from contextlib import contextmanager

#: The root of NSM's logger hierarchy. ``NSM/__init__.py`` puts the NullHandler here.
LOGGER_NAME = "NSM"

_REPLACEMENT = (
    'configure logging instead -- logging.getLogger("NSM").setLevel(logging.DEBUG) '
    "with a handler on it, or logging.basicConfig(level=logging.INFO)"
)

_depth = threading.local()


def _would_be_handled(logger):
    """Does any handler already stand to receive this logger's records?

    Mirrors ``logging.Logger.callHandlers``: walk the ancestry while ``propagate``
    holds. The ``NullHandler`` NSM installs on itself does not count -- it exists to
    silence the "no handlers" warning, not to deliver anything.
    """
    while logger is not None:
        if any(not isinstance(h, logging.NullHandler) for h in logger.handlers):
            return True
        logger = logger.parent if logger.propagate else None
    return False


@contextmanager
def _bridged_output():
    """Show every ``NSM.*`` record on stderr for the duration, if nobody else would.

    DEBUG, not INFO. Most of what ``verbose=True`` used to print is per-step chatter,
    which the conversion put at ``debug`` -- so an INFO bridge would honour the flag in
    name and drop three lines in four, which is the silent loss this module exists to
    prevent.
    """
    logger = logging.getLogger(LOGGER_NAME)
    if _would_be_handled(logger):
        yield
        return
    handler = logging.StreamHandler(sys.stderr)
    handler.setFormatter(logging.Formatter("%(name)s - %(levelname)s - %(message)s"))
    level = logger.level
    logger.addHandler(handler)
    logger.setLevel(min(level, logging.DEBUG) if level else logging.DEBUG)
    try:
        yield
    finally:
        logger.removeHandler(handler)
        logger.setLevel(level)
        handler.close()


def honour_verbose(func):
    """Deprecate ``func``'s ``verbose`` parameter while still honouring it.

    Warns when the caller supplied the flag, and routes NSM's log records to stderr
    while it is truthy -- including when it is truthy by default, which is the only way
    ``mesh/interpolate.update_positions`` keeps the output it has always produced.

    Not applied to the two functions whose ``verbose`` is a *required* parameter
    (``SDFSamples.load_mesh_step``, ``_process_meshes_for_wandb``): a required parameter
    means every call site is inside NSM, under a public entry point that is bridged
    already.
    """
    parameters = inspect.signature(func).parameters
    if "verbose" not in parameters:
        raise TypeError(f"{func.__qualname__} has no 'verbose' parameter to deprecate")
    index = list(parameters).index("verbose")
    default = parameters["verbose"].default

    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        supplied = "verbose" in kwargs or len(args) > index
        requested = (
            kwargs["verbose"] if "verbose" in kwargs else args[index] if supplied else default
        )
        if not requested:
            return func(*args, **kwargs)
        outermost = getattr(_depth, "value", 0) == 0
        if outermost and supplied:
            warnings.warn(
                f"{func.__qualname__}(verbose=...) is deprecated and will be removed in "
                f"v0.4.0; {_REPLACEMENT}. Until then the flag still shows NSM's output, "
                "on stderr rather than stdout.",
                DeprecationWarning,
                stacklevel=2,
            )
        if not outermost:
            return func(*args, **kwargs)
        _depth.value = 1
        try:
            with _bridged_output():
                return func(*args, **kwargs)
        finally:
            _depth.value = 0

    return wrapper
