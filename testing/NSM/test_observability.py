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

import json
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

    @pytest.mark.xfail(
        strict=True,
        reason="#58: NSM prints its diagnostics to stdout (plan §8.0.G, unmarks in the conversion)",
    )
    def test_importing_and_calling_writes_nothing_to_stdout(self):
        completed = _run("import NSM.reconstruct", _EMIT_A_DIAGNOSTIC)
        assert completed.stdout == ""

    @pytest.mark.xfail(
        strict=True,
        reason="#58: reconstruct/main.py calls logging.basicConfig at module scope",
    )
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
