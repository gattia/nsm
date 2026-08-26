"""
The one-release ``verbose=`` bridge (``NSM/_verbose_deprecation.py``, plan §8.0.G).

Two things have to hold at once, and they are why the bridge is not a one-liner: a
caller who passes ``verbose=True`` and has configured nothing still sees NSM's output,
while a host that *has* configured logging is never overridden or duplicated.

The mechanism is exercised through synthetic decorated functions, so these tests do not
move when the ``print`` conversion reaches a subpackage. The real entry points appear
only in the deprecation-notice test, which is the contract external callers see.

``unconfigured_host`` is not optional scaffolding, and it is a context manager rather
than a plain fixture for a reason: pytest's own logging plugin attaches a handler to the
root logger, so under the suite NSM's ancestry always looks configured and the bridge
correctly declines to install anything. That handler goes on *after* fixture setup, so
only something entered inside the test body can strip it. Without this, every assertion
here would pass vacuously.
"""

import logging
import warnings
from contextlib import contextmanager, suppress

import pytest

from NSM._verbose_deprecation import LOGGER_NAME, honour_verbose


@pytest.fixture
def logging_state():
    """Restore the ``"NSM"`` and root loggers afterwards, whatever a test does to them."""
    loggers = (logging.getLogger(LOGGER_NAME), logging.getLogger())
    saved = [(lg, list(lg.handlers), lg.level, lg.propagate) for lg in loggers]
    yield loggers[0]
    for lg, handlers, level, propagate in saved:
        lg.handlers[:] = handlers
        lg.setLevel(level)
        lg.propagate = propagate


@pytest.fixture
def unconfigured_host(logging_state):
    """``with unconfigured_host() as nsm_logger``: no handler anywhere in the ancestry."""

    @contextmanager
    def stripped():
        root = logging.getLogger()
        saved = list(root.handlers)
        root.handlers[:] = []
        logging_state.handlers[:] = [
            h for h in logging_state.handlers if isinstance(h, logging.NullHandler)
        ]
        try:
            yield logging_state
        finally:
            root.handlers[:] = saved

    return stripped


class _Collector(logging.Handler):
    """Stands in for a host's handler; records what it is given."""

    def __init__(self, sink):
        super().__init__(level=logging.DEBUG)
        self._sink = sink

    def emit(self, record):
        self._sink.append(record)


@honour_verbose
def speak(verbose=False, message="hello from NSM"):
    logging.getLogger("NSM.testing.speak").info(message)
    logging.getLogger("NSM.testing.speak").debug("%s, quietly", message)
    return "returned"


class TestTheFlagStillShowsTheCallerSomething:
    def test_an_unconfigured_host_gets_the_records_on_stderr(self, unconfigured_host, capsys):
        with unconfigured_host():
            assert speak(verbose=True) == "returned"
        captured = capsys.readouterr()
        assert "hello from NSM" in captured.err
        assert captured.out == ""

    def test_debug_records_reach_stderr_too(self, unconfigured_host, capsys):
        """
        The bridge is at DEBUG, not INFO. Most of what ``verbose=True`` used to print
        is per-step chatter, which the conversion put at ``debug``; an INFO bridge
        would honour the flag in name and silently drop most of its output.
        """
        with unconfigured_host():
            speak(verbose=True)
        assert "hello from NSM, quietly" in capsys.readouterr().err

    def test_without_the_flag_nothing_is_emitted(self, unconfigured_host, capsys):
        with unconfigured_host():
            speak()
        assert capsys.readouterr() == ("", "")

    def test_the_handler_and_level_do_not_outlive_the_call(self, unconfigured_host, capsys):
        with unconfigured_host() as nsm_logger:
            before, level = list(nsm_logger.handlers), nsm_logger.level
            speak(verbose=True)
            capsys.readouterr()
            assert nsm_logger.handlers == before
            assert nsm_logger.level == level
            logging.getLogger("NSM.testing.speak").info("after the call")
        assert "after the call" not in capsys.readouterr().err

    def test_the_flag_still_reaches_the_function(self):
        @honour_verbose
        def record(a, verbose=False):
            return (a, verbose)

        assert record(1, True) == (1, True)  # positional
        assert record(1, verbose=True) == (1, True)  # keyword
        assert record(1) == (1, False)  # defaulted


class TestAConfiguredHostIsNotOverridden:
    def test_a_handler_on_the_NSM_logger_wins(self, unconfigured_host, capsys):
        records = []
        with unconfigured_host() as nsm_logger:
            nsm_logger.addHandler(_Collector(records))
            nsm_logger.setLevel(logging.INFO)
            installed = len(nsm_logger.handlers)

            speak(verbose=True)

            assert len(nsm_logger.handlers) == installed
        # The host set INFO; its level stands, so the debug record is filtered out.
        assert [r.getMessage() for r in records] == ["hello from NSM"]
        assert capsys.readouterr().err == ""

    def test_a_handler_further_up_the_ancestry_wins_too(self, unconfigured_host, capsys):
        """
        ``logging.basicConfig()`` configures the *root*, not ``"NSM"``. Checking only
        ``NSM.handlers`` would add a second handler and show every record twice.
        """
        records = []
        with unconfigured_host():
            root = logging.getLogger()
            root.addHandler(_Collector(records))
            root.setLevel(logging.INFO)

            speak(verbose=True)

        # The host set INFO; its level stands, so the debug record is filtered out.
        assert [r.getMessage() for r in records] == ["hello from NSM"]
        assert capsys.readouterr().err == ""

    def test_propagate_false_makes_the_ancestry_irrelevant(self, unconfigured_host, capsys):
        """A host that isolated ``"NSM"`` gets the bridge, not the root's handler."""
        with unconfigured_host() as nsm_logger:
            logging.getLogger().addHandler(_Collector([]))
            nsm_logger.propagate = False

            speak(verbose=True)

        assert "hello from NSM" in capsys.readouterr().err


class TestOnePerUserCallNotPerInternalHop:
    """``reconstruct_mesh`` forwards ``verbose`` down through bridged callees."""

    def test_only_the_outermost_installs_a_handler(self, unconfigured_host, capsys):
        @honour_verbose
        def outer(verbose=False):
            return inner(verbose=verbose)

        @honour_verbose
        def inner(verbose=False):
            return len(logging.getLogger(LOGGER_NAME).handlers)

        with unconfigured_host() as nsm_logger:
            expected = len(nsm_logger.handlers) + 1
            assert outer(verbose=True) == expected
            assert len(nsm_logger.handlers) == expected - 1
        capsys.readouterr()

    def test_only_the_outermost_warns(self, unconfigured_host):
        @honour_verbose
        def outer(verbose=False):
            return inner(verbose=verbose)

        @honour_verbose
        def inner(verbose=False):
            return None

        with unconfigured_host(), pytest.warns(DeprecationWarning) as raised:
            outer(verbose=True)
        assert len(raised) == 1
        assert "outer" in str(raised[0].message)


class TestTheDeprecationNotice:
    #: One real entry point per subpackage. Each warns in the wrapper, before the
    #: function body runs, so deliberately invalid arguments cost nothing and reach no
    #: mesh, model or file.
    REPRESENTATIVE_CALLS = {
        "datasets": ("NSM.datasets.mesh_sampling", "read_meshes_get_sampled_pts"),
        "mesh": ("NSM.mesh.main", "create_mesh"),
        "models": ("NSM.models.triplanar", "TriplanarDecoder.forward"),
        "reconstruct": ("NSM.reconstruct.main", "reconstruct_mesh"),
        "train": ("NSM.train.train_deep_sdf_multi_head", "train_epoch"),
        "utils": ("NSM.utils", "adjust_learning_rate"),
    }

    @pytest.mark.parametrize("subpackage", sorted(REPRESENTATIVE_CALLS))
    def test_every_subpackage_announces_it(self, subpackage, unconfigured_host):
        import importlib

        module_name, attribute = self.REPRESENTATIVE_CALLS[subpackage]
        target = importlib.import_module(module_name)
        for part in attribute.split("."):
            target = getattr(target, part)

        with pytest.warns(DeprecationWarning) as raised:
            with suppress(Exception):
                target(verbose=True)
        message = str(raised[0].message)
        assert attribute.split(".")[-1] in message
        assert "v0.4.0" in message
        assert 'logging.getLogger("NSM")' in message

    def test_a_defaulted_flag_is_honoured_without_a_notice(self, unconfigured_host, capsys):
        """
        ``mesh/interpolate.update_positions`` defaults ``verbose=True``. Honouring the
        default is what keeps its output; warning about a parameter the caller never
        wrote would be a notice about someone else's choice.
        """

        @honour_verbose
        def loud(verbose=True):
            logging.getLogger("NSM.testing.loud").info("default-on output")

        with unconfigured_host(), warnings.catch_warnings(record=True) as raised:
            warnings.simplefilter("always")
            loud()
        assert [w for w in raised if issubclass(w.category, DeprecationWarning)] == []
        assert "default-on output" in capsys.readouterr().err

    def test_decorating_a_function_without_verbose_is_a_TypeError(self):
        with pytest.raises(TypeError, match="no 'verbose' parameter"):

            @honour_verbose
            def unbridgeable(a, b):
                return a + b
