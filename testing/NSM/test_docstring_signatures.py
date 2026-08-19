"""
A docstring may not document a parameter the function does not have.

This is the mechanically checkable slice of Phase 2. Most of the docstring problems in
this repo are semantic -- the text describes behaviour the body does not have -- and no
test can see those. What a test *can* see is a name in an ``Args:`` block that is not in
the signature, which is what happens when a parameter is renamed, removed, or commented
out and the prose is left behind.

Both instances it found when written were live defects rather than typos:

- ``read_mesh_get_sampled_pts`` documented ``return_orig_mesh``, ``return_new_mesh`` and
  ``return_orig_pts`` as parameters "Defaults to False", while the body listed all three as
  deprecated and printed "always True" -- the opposite. NSM's own call site still passed two
  of them, so every mesh load emitted a deprecation line.
- ``compute_recon_loss`` documented ``orig_pts``, which is commented out of its signature,
  and omitted ``orig_meshes`` which replaced it.

The inverse check -- a parameter that exists but is undocumented -- is deliberately not
asserted. Plenty of functions here document only the interesting arguments, and failing on
that would be noise rather than signal.
"""

import ast
import re
from pathlib import Path

import pytest

NSM = Path(__file__).resolve().parents[2] / "NSM"

# encoding="utf-8" on every read below is load-bearing, not decoration: something else in
# the suite resets the locale to ASCII, so a bare read_text() passes in isolation and dies
# with UnicodeDecodeError on the first non-ASCII character once the full suite runs.

# Section headings that may appear inside or after an Args block. Without these, `Note:`
# and friends parse as parameter names.
SECTIONS = re.compile(
    r"^(Args|Arguments|Parameters|Returns?|Raises?|Yields?|Notes?|Examples?|References|"
    r"Attributes|See Also|Warns?|Warnings?|Todo)\s*:?\s*$"
)
ARGS_START = re.compile(r"^(Args|Arguments|Parameters)\s*:?\s*$")
PARAM_LINE = re.compile(r"^(\*{0,2}\w+)\s*(\([^)]*\))?\s*:")


def documented_parameters(docstring):
    """Names in the ``Args:`` block of a Google-style docstring."""
    found, in_args = set(), False
    for raw in (docstring or "").split("\n"):
        line = raw.strip()
        if ARGS_START.match(line):
            in_args = True
            continue
        if in_args and SECTIONS.match(line):
            in_args = False
            continue
        if in_args:
            match = PARAM_LINE.match(line)
            if match:
                found.add(match.group(1).lstrip("*"))
    return found


def real_parameters(node):
    args = node.args
    names = {a.arg for a in [*args.posonlyargs, *args.args, *args.kwonlyargs]}
    if args.vararg:
        names.add(args.vararg.arg)
    if args.kwarg:
        names.add(args.kwarg.arg)
    return names - {"self", "cls"}


def _functions():
    for path in sorted(NSM.rglob("*.py")):
        try:
            tree = ast.parse(path.read_text(encoding="utf-8"))
        except SyntaxError:  # pragma: no cover - not this test's problem
            continue
        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                if ast.get_docstring(node):
                    yield path.relative_to(NSM.parent), node


DOCUMENTED = [
    (str(rel), node.lineno, node.name)
    for rel, node in _functions()
    if documented_parameters(ast.get_docstring(node))
]


def test_enough_docstrings_are_parsed_for_this_to_mean_anything():
    """Guards against the parser silently matching nothing and the suite reading green."""
    assert len(DOCUMENTED) >= 40, f"only parsed {len(DOCUMENTED)} Args blocks"


@pytest.mark.parametrize("path,lineno,name", DOCUMENTED, ids=[f"{p}:{n}" for p, _, n in DOCUMENTED])
def test_a_documented_parameter_exists(path, lineno, name):
    tree = ast.parse((NSM.parent / path).read_text(encoding="utf-8"))
    node = next(
        n
        for n in ast.walk(tree)
        if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))
        and n.name == name
        and n.lineno == lineno
    )
    phantom = documented_parameters(ast.get_docstring(node)) - real_parameters(node)
    assert not phantom, (
        f"{path}:{lineno} {name}() documents {sorted(phantom)}, " f"which is not in its signature"
    )
