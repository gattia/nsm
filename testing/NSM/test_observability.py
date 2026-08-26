"""
Characterization tests for what NSM writes and where, written immediately before the
§8.0.G conversion from ``print`` to ``logging`` (#58).

Two things are pinned, and they pull in opposite directions:

* **stdout is a contract surface.** ``kneepipeline/steps/run_nsm.py`` runs each NSM fit
  in a subprocess with ``capture_output=True`` and then parses ``json.loads`` of the
  *last line* of stdout. Anything NSM prints after that line breaks the consumer. The
  parse must keep working across this slice.
* **stdout is not NSM's to write to.** A library's diagnostics belong on the host's
  logging handlers, which default to stderr. That one is a ``strict`` xfail here: today
  NSM prints an import-time notice and its deprecated-kwarg notices to stdout.

The third pin is the ``logging.basicConfig`` at ``NSM/reconstruct/main.py`` module
scope, which reconfigures the *host process's root logger* on any
``import NSM.reconstruct`` (``reconstruct/__init__.py`` star-imports ``.main``). Also a
strict xfail: it fires today, invisibly, in the consumer's process.

Every check runs in a subprocess, because all three are properties of a fresh
interpreter: once ``NSM`` is imported, the import-time effects cannot be observed again.
"""

import ast
import json
import pathlib
import subprocess
import sys
import textwrap

import pytest

#: Emits an NSM diagnostic and nothing else, then raises before any real work: the
#: ``batch_size_latent_recon`` deprecation notice fires at the top of ``reconstruct_mesh``
#: and the invalid ``path`` aborts on the next check. No model, no meshes, no GPU.
_EMIT_A_DIAGNOSTIC = """
from NSM.reconstruct.main import reconstruct_mesh
try:
    reconstruct_mesh(path=42, decoders=None, latent_size=8, batch_size_latent_recon=1)
except ValueError:
    pass
"""


def _run(*bodies):
    """Run ``bodies`` in a fresh interpreter; return its CompletedProcess.

    Each fragment is dedented on its own, so a module-level constant and an
    indented literal can be concatenated without the second losing its indentation.
    """
    script = "\n".join(textwrap.dedent(body) for body in bodies)
    completed = subprocess.run(
        [sys.executable, "-c", script],
        capture_output=True,
        text=True,
        timeout=300,
    )
    assert completed.returncode == 0, completed.stderr[-2000:]
    return completed


class TestTheConsumerStdoutContract:
    """
    The shape ``_fit_nsm_subprocess`` uses: NSM does its talking, the caller's own
    ``print(json.dumps(...))`` goes last, and the parent reads that last line back.
    """

    def test_the_last_stdout_line_still_parses_as_json(self):
        completed = _run(
            _EMIT_A_DIAGNOSTIC,
            """
            import json
            print(json.dumps({"loss": 0.5, "latent": [0.0, 1.0]}, default=str))
            """,
        )
        last_line = completed.stdout.strip().split("\n")[-1]
        assert json.loads(last_line) == {"loss": 0.5, "latent": [0.0, 1.0]}


class TestNSMOwnsNoStream:
    """
    The target state of the slice. Both are ``strict`` xfails: they fail today, and the
    conversion commits unmark them.
    """

    def test_importing_and_calling_writes_nothing_to_stdout(self):
        completed = _run("import NSM.reconstruct", _EMIT_A_DIAGNOSTIC)
        assert completed.stdout == ""

    def test_an_unconfigured_host_sees_no_log_records(self):
        """
        What the ``NullHandler`` on the ``"NSM"`` logger buys: records from ``NSM.*``
        find a handler, so ``logging.lastResort`` never fires and even a ``warning``
        stays silent until the host asks for it. That is the stdlib idiom's known
        consequence, not an oversight -- ``verbose=`` is the bridge for callers who
        want the output without configuring logging.
        """
        completed = _run(
            """
            import logging
            import NSM.reconstruct.recon_evaluation as recon_evaluation
            recon_evaluation.logger.info("an info record")
            recon_evaluation.logger.warning("a warning record")
            """
        )
        assert "an info record" not in completed.stderr
        assert "a warning record" not in completed.stderr

    def test_importing_does_not_reconfigure_the_host_root_logger(self):
        """
        Recorded before and after the import in one process, so the comparison is
        against *that* interpreter's defaults rather than a hard-coded level.
        """
        completed = _run(
            """
            import json, logging, sys
            root = logging.getLogger()
            def snapshot():
                return {"level": root.level, "handlers": [type(h).__name__ for h in root.handlers]}
            before = snapshot()
            import NSM.reconstruct
            print(json.dumps({"before": before, "after": snapshot()}), file=sys.stderr)
            """
        )
        recorded = json.loads(completed.stderr.strip().split("\n")[-1])
        assert recorded["after"] == recorded["before"]


#: The Logger methods that emit a record. ``addHandler`` and friends are configuration,
#: which ``_verbose_deprecation`` does to a *local* named ``logger``.
EMIT_METHODS = {"debug", "info", "warning", "error", "exception", "critical", "log"}


class TestTheConversionHolds:
    """
    Structural pins over ``NSM/`` itself: what the §8.0.G conversion established, so a
    later slice reintroducing a ``print`` or an f-string log line goes red rather than
    unnoticed. ``train/deprecated/`` is out of scope until §8.0.P.
    """

    #: A script's own output on its own stdout is not the library speaking.
    ALLOWED_PRINTS = {"NSM/configs/generate_sdf_default_config.py"}

    def test_no_print_survives_outside_the_generator_script(self):
        offenders = [
            f"{path}:{node.lineno}"
            for path, tree in _library_modules()
            for node in ast.walk(tree)
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id == "print"
            and path not in self.ALLOWED_PRINTS
        ]
        assert offenders == []

    def test_every_log_call_defers_its_formatting(self):
        """
        ``%``-style, not an f-string or a pre-built string: a suppressed record must
        cost no formatting. The hot ones sit in per-batch and per-step loops, and this
        is the only thing standing between them and a silent per-iteration cost.

        Note what this does *not* buy: the *arguments* are still evaluated eagerly. A
        log line whose argument is expensive to compute belongs behind a guard, which
        is where the remaining comprehension-valued ones already sit.
        """
        offenders = []
        for path, tree in _library_modules():
            for node in ast.walk(tree):
                if not (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)):
                    continue
                if not (isinstance(node.func.value, ast.Name) and node.func.value.id == "logger"):
                    continue
                if node.func.attr not in EMIT_METHODS:
                    continue
                first = node.args[0] if node.args else None
                built = isinstance(first, ast.JoinedStr) or (
                    isinstance(first, ast.BinOp) and isinstance(first.op, (ast.Mod, ast.Add))
                )
                built = built or (
                    isinstance(first, ast.Call)
                    and isinstance(first.func, ast.Attribute)
                    and first.func.attr == "format"
                )
                if built:
                    offenders.append(f"{path}:{node.lineno}")
        assert offenders == []

    def test_every_module_that_speaks_has_its_own_logger(self):
        """
        One ``getLogger(__name__)`` per speaking module, so the ``NSM.*`` hierarchy is
        real: a host can silence ``NSM.datasets`` without silencing reconstruction.
        """
        missing = []
        for path, tree in _library_modules():
            speaks = any(
                isinstance(node, ast.Call)
                and isinstance(node.func, ast.Attribute)
                and isinstance(node.func.value, ast.Name)
                and node.func.value.id == "logger"
                and node.func.attr in EMIT_METHODS
                for node in ast.walk(tree)
            )
            defines = any(
                isinstance(node, ast.Assign)
                and any(getattr(t, "id", None) == "logger" for t in node.targets)
                for node in tree.body
            )
            if speaks and not defines:
                missing.append(path)
        assert missing == []


def _library_modules():
    """(repo-relative path, parsed module) for every ``NSM/`` file outside ``deprecated/``."""
    root = pathlib.Path(__file__).resolve().parents[2] / "NSM"
    for path in sorted(root.rglob("*.py")):
        if "deprecated" in path.parts:
            continue
        relative = path.relative_to(root.parent).as_posix()
        yield relative, ast.parse(path.read_text(encoding="utf-8"))
