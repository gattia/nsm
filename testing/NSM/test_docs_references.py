"""
Names that prose cites must exist in the code: NSM symbols in ``docs/``, test names in
``docs/`` and ``NSM/``, and parameters in a docstring's ``Args:`` block.

Citations are by name, not line number, because line numbers move on every reformat.
"""

import ast
import re
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
NSM = REPO / "NSM"
TESTING = REPO / "testing"
DOCS = [REPO / "docs" / n for n in ("KNOWN_ISSUES.md", "SCOPE.md", "ARCHITECTURE.md")]

# encoding="utf-8" on every read is required: something in the suite resets the locale to
# ASCII, so a bare read_text() fails on the first non-ASCII character under the full suite.


def _read(path):
    return path.read_text(encoding="utf-8")


# ---------------------------------------------------------------------------
# NSM symbols cited in docs/
# ---------------------------------------------------------------------------

#: A backticked dotted identifier: no call parens, subscripts or path separators.
TOKEN = re.compile(r"`([A-Za-z_][A-Za-z0-9_]*(?:\.[A-Za-z_][A-Za-z0-9_]*)+)`")

#: ``sdf_dataset.py`` is a filename, not a symbol.
FILE_SUFFIXES = {"py", "toml", "yml", "yaml", "json", "md", "cfg", "txt", "in"}

#: CamelCase heads that are not NSM classes. A CamelCase head must otherwise resolve, so
#: renaming a class fails its citations instead of silently dropping them from the check.
NOT_NSM_CLASSES = {"NSM", "Mesh"}
CAMEL = re.compile(r"^[A-Z][A-Za-z0-9]*$")


def _from_nsm(node):
    return node.level > 0 or (node.module or "").split(".")[0] == "NSM"


def _qualnames(path):
    """
    Every def and class in a file as a dotted qualname, plus ``self.x`` attributes and the
    names the module imports from inside NSM.

    Imports count because ``refine_mesh.get_faces`` is a working reference: Python binds
    the name in the importing module. Only NSM-internal imports count. Registering
    ``from torch import nn`` made ``nn.Sequential`` look like an NSM symbol.
    """
    out = set()

    def attrs(classnode, prefix):
        for node in ast.walk(classnode):
            targets = node.targets if isinstance(node, ast.Assign) else []
            if isinstance(node, ast.AnnAssign):
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

    walk(ast.parse(_read(path)), "")
    return out


def _index():
    """``{module_stem: {qualnames}}`` and the set of top-level names."""
    by_module, top_level = {}, set()
    for py in NSM.rglob("*.py"):
        quals = _qualnames(py)
        by_module[py.stem] = by_module.get(py.stem, set()) | quals
        top_level |= {q for q in quals if "." not in q}
    return by_module, top_level


def test_every_nsm_symbol_the_docs_cite_exists():
    """
    Fails if ``KNOWN_ISSUES.md``, ``SCOPE.md`` or ``ARCHITECTURE.md`` cites a backticked
    dotted name, such as ``module.function`` or ``Class.method``, that no NSM file defines.
    """
    index, top_level = _index()
    all_quals = {q for quals in index.values() for q in quals}

    checked, missing = 0, []
    for doc in DOCS:
        for token in sorted(set(TOKEN.findall(_read(doc)))):
            head, rest = token.split(".", 1)
            if token.rsplit(".", 1)[-1] in FILE_SUFFIXES or head in NOT_NSM_CLASSES:
                continue
            if head in index:
                checked += 1
                if rest not in index[head]:
                    missing.append(f"{doc.name}: `{token}`")
            elif head in top_level or CAMEL.match(head):
                checked += 1
                if token not in all_quals:
                    missing.append(f"{doc.name}: `{token}`")

    assert checked >= 15, f"only {checked} citations matched; the regex has stopped working"
    assert missing == []


# ---------------------------------------------------------------------------
# Test names cited in docs/ and NSM/
# ---------------------------------------------------------------------------

#: Where a citation of a test is a claim that the test exists today. CHANGELOG.md is left
#: out: it records what was true at each release.
CITING = [
    *DOCS,
    REPO / "README.md",
    REPO / "DEVELOPMENT.md",
    TESTING / "NSM" / "regression" / "README.md",
    *NSM.rglob("*.py"),
]

INLINE_CODE = re.compile(r"`+([^`\n]+)`+")
TEST_NAME = re.compile(r"\b(Test[A-Z]\w*|test_\w+)\b")


def test_every_test_name_cited_in_prose_exists():
    """
    Fails if inline code in the three ``docs/`` files, ``README.md``, ``DEVELOPMENT.md``, the
    regression README or an ``NSM/`` source file names a ``TestX``, ``test_x`` or
    ``test_x.py`` that ``testing/`` does not define.

    Renaming or merging a test otherwise leaves a "Pinned by" line that points nowhere.
    """
    defined = set()
    for path in TESTING.rglob("*.py"):
        defined.add(path.stem)
        for node in ast.walk(ast.parse(_read(path))):
            if isinstance(node, (ast.ClassDef, ast.FunctionDef)):
                defined.add(node.name)
    # NSM has its own identifiers that start with test_, such as `test_load_times`.
    for path in NSM.rglob("*.py"):
        for node in ast.walk(ast.parse(_read(path))):
            if isinstance(node, (ast.FunctionDef, ast.arg)):
                defined.add(getattr(node, "name", None) or node.arg)

    cited, missing = 0, []
    for path in CITING:
        for span in INLINE_CODE.findall(_read(path)):
            for name in TEST_NAME.findall(span.replace(".py", "")):
                cited += 1
                if name not in defined:
                    missing.append(f"{path.relative_to(REPO)}: {name}")

    assert cited >= 20, f"only {cited} test citations matched; the regex has stopped working"
    assert sorted(set(missing)) == []


# ---------------------------------------------------------------------------
# Docstring Args blocks
# ---------------------------------------------------------------------------

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
        elif in_args and SECTIONS.match(line):
            in_args = False
        elif in_args and PARAM_LINE.match(line):
            found.add(PARAM_LINE.match(line).group(1).lstrip("*"))
    return found


def test_no_docstring_documents_a_parameter_its_function_lacks():
    """
    Fails if an ``NSM/`` function's ``Args:`` or ``Parameters`` block documents a name its
    signature lacks, as a renamed or removed parameter leaves behind.

    An undocumented parameter is not checked: many functions document only some.
    """
    parsed, phantoms = 0, []
    for path in sorted(NSM.rglob("*.py")):
        for node in ast.walk(ast.parse(_read(path))):
            if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                continue
            documented = documented_parameters(ast.get_docstring(node))
            if not documented:
                continue
            parsed += 1
            args = node.args
            real = {a.arg for a in [*args.posonlyargs, *args.args, *args.kwonlyargs]}
            real |= {a.arg for a in (args.vararg, args.kwarg) if a}
            real -= {"self", "cls"}
            extra = documented - real
            if extra:
                phantoms.append(f"{path.relative_to(REPO)}:{node.lineno} {node.name} {extra}")

    assert parsed >= 40, f"only {parsed} Args blocks parsed; the parser has stopped working"
    assert phantoms == []


# ---------------------------------------------------------------------------
# KNOWN_ISSUES.md § Open: the summary table against its entries
# ---------------------------------------------------------------------------


def test_the_open_summary_table_and_its_entries_are_the_same_set():
    """
    Fails if a row of the summary table in ``KNOWN_ISSUES.md`` § Open links to no ``###``
    entry in that section, or an entry has no row linking to it.
    """
    text = _read(REPO / "docs" / "KNOWN_ISSUES.md")
    section = text[text.index("\n# Open\n") : text.index("\n# History\n")]

    def anchor(heading):
        return "#" + re.sub(r"[^\w\- ]", "", heading.lower()).strip().replace(" ", "-")

    rows = [
        link
        for line in section.splitlines()
        if line.startswith("|") and not line.startswith("|---") and "| Severity |" not in line
        for link in re.findall(r"\]\((#[^)]+)\)", line.split("|")[1])
    ]
    entries = [anchor(line[4:]) for line in section.splitlines() if line.startswith("### ")]
    assert sorted(rows) == sorted(entries)
