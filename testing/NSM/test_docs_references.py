"""
The docs cite code by symbol, not by line number, and this asserts the symbols exist.

Line numbers were tried and did not survive: a seven-line portability fix in
``sdf_dataset.py`` moved every citation below it, and a ``black`` pass over
``triplanar.py`` and ``reconstruct/main.py`` moves them again. A checker that verified
line numbers would have gone red on every reformat and produced recurring transcription
work; the numbers were removed instead, and this checks what replaced them.

Scope, deliberately narrow: only **dotted** references are checked -- ``Class.method`` or
``module.function`` -- and only when the leading component names something in ``NSM/``.
Those are the cross-file citations that rot when code is renamed or moved. A bare
single-word reference in backticks is indistinguishable from a parameter name in prose
(``padding``, ``subsample``) and is not checked; keep cross-file citations dotted so this
test can see them.
"""

import ast
import re
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
NSM = REPO / "NSM"
DOCS = [REPO / "docs" / n for n in ("KNOWN_ISSUES.md", "SCOPE.md", "ARCHITECTURE.md")]

# Backticked token that is a dotted identifier and nothing else: no call parens, no
# subscripts, no path separators, no file extension.
TOKEN = re.compile(r"`([A-Za-z_][A-Za-z0-9_]*(?:\.[A-Za-z_][A-Za-z0-9_]*)+)`")

# `sdf_dataset.py` is a filename, not a symbol, and parses as a dotted identifier.
FILE_SUFFIXES = {"py", "toml", "yml", "yaml", "json", "md", "cfg", "txt", "in"}

# A CamelCase head is a class reference and must resolve -- otherwise renaming a class
# makes its citations vanish from this check instead of failing it, which is the hole a
# line-number checker would also have had. These are the CamelCase names the docs may
# legitimately mention that are not NSM classes; adding to this set is a deliberate act.
NOT_NSM_CLASSES = {
    "NSM",  # the package itself: `NSM.datasets`, `NSM.__version__` are paths, not symbols
    "Mesh",  # pymskt
}
CAMEL = re.compile(r"^[A-Z][A-Za-z0-9]*$")


def _from_nsm(node):
    """Is this ``from ... import ...`` pulling from inside NSM?"""
    return node.level > 0 or (node.module or "").split(".")[0] == "NSM"


def _qualnames(path):
    """
    Every def/class in a file as a dotted qualname, plus instance attributes and
    names the module re-exports.

    ``self.padding = padding`` inside a class body registers ``Class.padding``: the docs
    cite attributes as well as methods, and an attribute that is renamed is exactly the
    kind of drift worth catching.

    ``from .triangle_metrics import get_faces`` registers ``get_faces`` too, because
    ``NSM.mesh.refine_mesh.get_faces`` **is** a working reference -- Python binds the
    name in the importing module. Without this the check under-approximates a module's
    surface and rejects citations that resolve: ``refine_mesh.get_faces`` (§8.0.I, where
    the function moved but the import path was kept deliberately) and ``deep_sdf.Sine``
    (§8.0.H, same shape) were both false negatives until this was added. A ``def`` that
    is *deleted* rather than moved still fails, which is the drift being caught.

    **NSM-internal imports only**, and the narrowing is load-bearing: registering every
    ``from x import y`` puts third-party names into the module index, and since
    ``TOP_LEVEL`` is what decides whether a citation is even ours, ``from torch import
    nn`` made the docs' ``nn.Sequential`` / ``nn.Embedding`` / ``nn.ModuleList`` look
    like NSM symbols and fail. Measured: three failures.
    """
    out = set()

    def attrs(classnode, prefix):
        for node in ast.walk(classnode):
            targets = []
            if isinstance(node, ast.Assign):
                targets = node.targets
            elif isinstance(node, ast.AnnAssign):
                targets = [node.target]
            for t in targets:
                if (
                    isinstance(t, ast.Attribute)
                    and isinstance(t.value, ast.Name)
                    and t.value.id == "self"
                ):
                    out.add(prefix + t.attr)

    def walk(node, prefix):
        for child in ast.iter_child_nodes(node):
            if isinstance(child, ast.ImportFrom) and not prefix and _from_nsm(child):
                out.update(a.asname or a.name for a in child.names if a.name != "*")
            if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                qual = prefix + child.name
                out.add(qual)
                if isinstance(child, ast.ClassDef):
                    attrs(child, qual + ".")
                walk(child, qual + ".")

    walk(ast.parse(path.read_text(encoding="utf-8")), "")
    return out


def _index():
    """{module_stem: {qualnames}} plus the set of class names defined anywhere."""
    by_module, classes = {}, set()
    for py in NSM.rglob("*.py"):
        try:
            quals = _qualnames(py)
        except SyntaxError:  # pragma: no cover - a broken file is not this test's problem
            continue
        by_module[py.stem] = by_module.get(py.stem, set()) | quals
        classes |= {q for q in quals if "." not in q}
    return by_module, classes


INDEX, TOP_LEVEL = _index()
ALL_QUALS = {q for quals in INDEX.values() for q in quals}


def _citations():
    for doc in DOCS:
        if not doc.exists():
            continue
        for token in TOKEN.findall(doc.read_text(encoding="utf-8")):
            if token.rsplit(".", 1)[-1] in FILE_SUFFIXES:
                continue
            head = token.split(".")[0]
            if head in NOT_NSM_CLASSES:
                continue
            # Ours if the head names an NSM module or top-level symbol -- or if it simply
            # looks like a class, so that a renamed class fails rather than disappearing.
            if head in INDEX or head in TOP_LEVEL or CAMEL.match(head):
                yield doc.name, token


CITATIONS = sorted(set(_citations()))


def test_the_docs_cite_at_least_a_handful_of_symbols():
    """Guards against the regex silently matching nothing and the suite reading green."""
    assert len(CITATIONS) >= 15, f"only found {len(CITATIONS)} dotted citations: {CITATIONS}"


@pytest.mark.parametrize("doc,token", CITATIONS, ids=[f"{d}:{t}" for d, t in CITATIONS])
def test_a_cited_symbol_exists(doc, token):
    head, rest = token.split(".", 1)
    if head in INDEX:  # module.symbol
        assert rest in INDEX[head], f"{doc} cites `{token}`, but {head}.py has no {rest}"
    else:  # Class.method, module not named
        assert token in ALL_QUALS, f"{doc} cites `{token}`, which is defined nowhere in NSM/"
