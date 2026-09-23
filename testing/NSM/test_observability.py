"""
What NSM writes and where (#58).

* **stdout belongs to the caller.** ``kneepipeline/steps/run_nsm.py`` runs each fit in a
  subprocess and parses the *last line* of stdout as JSON, so NSM must print nothing.
* **Logging is the host's to configure.** NSM logs through ``logging.getLogger(__name__)``
  with a ``NullHandler`` on ``"NSM"``, and never reconfigures the root logger.
"""

import ast
import json
import pathlib
import subprocess
import sys

#: One interpreter start costs about 6 s of imports, so every runtime property is
#: checked in the same one. ``reconstruct_mesh`` logs the ``batch_size_latent_recon``
#: deprecation first, then refuses the invalid ``path``: a diagnostic with no real work.
_PROBE = """
import json, logging, sys
root = logging.getLogger()
before = {"level": root.level, "handlers": [type(h).__name__ for h in root.handlers]}

import NSM.reconstruct
from NSM.reconstruct import recon_evaluation
from NSM.reconstruct.main import reconstruct_mesh

after = {"level": root.level, "handlers": [type(h).__name__ for h in root.handlers]}
try:
    reconstruct_mesh(path=42, decoders=None, latent_size=8, batch_size_latent_recon=1)
except ValueError:
    pass
recon_evaluation.logger.info("an info record")
recon_evaluation.logger.warning("a warning record")
print(json.dumps({"before": before, "after": after}), file=sys.stderr)
print(json.dumps({"loss": 0.5}))
"""


def test_nsm_writes_nothing_to_stdout_and_leaves_host_logging_alone():
    completed = subprocess.run(
        [sys.executable, "-c", _PROBE], capture_output=True, text=True, timeout=300
    )
    assert completed.returncode == 0, completed.stderr[-2000:]

    # The consumer's JSON is the only thing on stdout.
    assert completed.stdout.strip().split("\n") == ['{"loss": 0.5}']
    # An unconfigured host sees no NSM record, not even a warning.
    assert "an info record" not in completed.stderr
    assert "a warning record" not in completed.stderr
    # Importing NSM did not reconfigure the root logger.
    roots = json.loads(completed.stderr.strip().split("\n")[-1])
    assert roots["after"] == roots["before"]


#: ``Logger`` methods that emit a record.
EMIT_METHODS = {"debug", "info", "warning", "error", "exception", "critical", "log"}

#: A script's own output on its own stdout is not the library speaking.
ALLOWED_PRINTS = {"NSM/configs/generate_sdf_default_config.py"}

#: Outside the documented surface (``SCOPE`` §2.4), so not held to the gate rule.
UNDOCUMENTED_SURFACE = {"NSM/reconstruct/reconstruct_latent_S3.py"}


def _library_modules():
    root = pathlib.Path(__file__).resolve().parents[2] / "NSM"
    for path in sorted(root.rglob("*.py")):
        yield path.relative_to(root.parent).as_posix(), ast.parse(path.read_text(encoding="utf-8"))


def _log_calls(tree):
    for node in ast.walk(tree):
        if (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and isinstance(node.func.value, ast.Name)
            and node.func.value.id == "logger"
            and node.func.attr in EMIT_METHODS
        ):
            yield node


def test_the_library_speaks_only_through_lazily_formatted_log_calls():
    """
    No ``print``. And no f-string, ``%`` or ``.format`` message: a suppressed record must
    cost no formatting, and several sit in per-batch loops.
    """
    offenders = []
    for path, tree in _library_modules():
        for node in ast.walk(tree):
            if (
                isinstance(node, ast.Call)
                and isinstance(node.func, ast.Name)
                and node.func.id == "print"
                and path not in ALLOWED_PRINTS
            ):
                offenders.append(f"print at {path}:{node.lineno}")
        for node in _log_calls(tree):
            first = node.args[0] if node.args else None
            built = isinstance(first, ast.JoinedStr)
            built |= isinstance(first, ast.BinOp) and isinstance(first.op, (ast.Mod, ast.Add))
            built |= (
                isinstance(first, ast.Call)
                and isinstance(first.func, ast.Attribute)
                and first.func.attr == "format"
            )
            if built:
                offenders.append(f"eager format at {path}:{node.lineno}")
    assert offenders == []


def test_every_record_reaches_a_host_that_asks_for_it():
    """
    Each module that logs has its own ``logger``, so a host can silence ``NSM.datasets``
    alone. No log call sits inside ``if flag:`` where ``flag`` is a parameter of the same
    function: that was the ``verbose=`` pattern, removed at v0.4.0. It is checked under any
    parameter name, and was confirmed by adding such a gate and seeing it fail.
    ``logger.isEnabledFor`` guards are allowed: they read the host's level.
    """

    def gating_name(test):
        if isinstance(test, ast.Name):
            return test.id
        if (
            isinstance(test, ast.Compare)
            and isinstance(test.left, ast.Name)
            and len(test.ops) == 1
            and isinstance(test.ops[0], ast.Is)
            and isinstance(test.comparators[0], ast.Constant)
        ):
            return test.left.id
        return None

    offenders = []
    for path, tree in _library_modules():
        defines_logger = any(
            isinstance(node, ast.Assign)
            and any(getattr(t, "id", None) == "logger" for t in node.targets)
            for node in tree.body
        )
        if any(_log_calls(tree)) and not defines_logger:
            offenders.append(f"no module logger in {path}")
        if path in UNDOCUMENTED_SURFACE:
            continue
        for function in ast.walk(tree):
            if not isinstance(function, (ast.FunctionDef, ast.AsyncFunctionDef)):
                continue
            args = function.args
            parameters = {a.arg for a in args.args + args.posonlyargs + args.kwonlyargs}
            for node in ast.walk(function):
                if not isinstance(node, ast.If) or node.orelse or not node.body:
                    continue
                if gating_name(node.test) in parameters and all(
                    isinstance(s, ast.Expr)
                    and isinstance(s.value, ast.Call)
                    and ast.unparse(s.value.func).startswith("logger.")
                    for s in node.body
                ):
                    offenders.append(f"gated log call at {path}:{node.lineno}")
    assert offenders == []
