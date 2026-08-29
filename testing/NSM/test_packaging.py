"""
What a built wheel contains, and where its version comes from.

Nothing else in the suite builds a distribution, so every claim about one has until now
been read off ``pyproject.toml`` rather than measured -- and ``SCOPE.md`` §5's claim was
half wrong for it: it says a wheel contains neither the config generator nor
``default_config.json``, and the generator has been shipping all along
(``[tool.setuptools.packages.find]`` takes ``namespaces = true`` by default, so
``NSM.configs`` is found without an ``__init__.py``). What is missing is package *data*.

**The wheel is built from a copy of the tracked files, never from the checkout in place.**
A developer's tree carries a stale ``build/lib/``, and ``--no-build-isolation`` reuses it:
this repo's still holds ``NSM/dependencies``, deleted in PR #64, and a working-tree build
silently packages it. The copy also has no ``.git``, which is deliberate -- it is the
GitHub source-zip case, and after §8.0.O it is what proves the version is derived rather
than written down.

Cost: about 1.5 s per build, measured. There are two.
"""

import re
import shutil
import subprocess
import sys
import zipfile
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]


def tracked_copy(destination):
    """The tracked files at their working-tree content, without ``.git``."""
    listing = subprocess.run(["git", "ls-files", "-z"], cwd=REPO, capture_output=True, check=True)
    for name in listing.stdout.decode("utf-8").split("\0"):
        if not name:
            continue
        source = REPO / name
        if not source.exists():  # deleted-but-not-staged
            continue
        target = destination / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)
    return destination


def build_wheel(directory, out):
    """
    ``pip wheel`` without build isolation, so no network and no re-resolve.

    ``encoding="utf-8"`` rather than ``text=True``: the latter decodes with the *locale's*
    preferred encoding, which is ``US-ASCII`` on a GitHub macOS runner, and pip prints its
    error banner with a ``\u00d7``. A failed build therefore raised ``UnicodeDecodeError``
    from ``subprocess`` instead of reaching the assertion that prints why it failed -- the
    diagnostic destroyed by the thing it existed to diagnose. Same class as the repo-wide
    "read files with encoding=utf-8 explicitly" rule in ``CLAUDE.md``.
    """
    return subprocess.run(
        [
            sys.executable,
            "-m",
            "pip",
            "wheel",
            "--no-deps",
            "--no-build-isolation",
            "-w",
            str(out),
            str(directory),
        ],
        capture_output=True,
        encoding="utf-8",
        errors="replace",
    )


@pytest.fixture(scope="module")
def wheel(tmp_path_factory):
    """One built wheel: its path and its entry list."""
    if not (REPO / ".git").exists():
        pytest.skip("needs the git checkout to enumerate tracked files")
    root = tmp_path_factory.mktemp("dist")
    result = build_wheel(tracked_copy(root / "src"), root / "out")
    assert result.returncode == 0, result.stdout + result.stderr
    built = sorted((root / "out").glob("*.whl"))
    assert len(built) == 1, built
    return built[0], zipfile.ZipFile(built[0]).namelist()


class TestWhatShips:
    def test_the_config_generator_ships(self, wheel):
        """``SCOPE.md`` §5 said it does not. It does, and always did."""
        _, names = wheel
        assert "NSM/configs/generate_sdf_default_config.py" in names

    def test_the_default_config_ships(self, wheel):
        """
        Was a strict xfail. The file every doc in the repo calls "the shipped
        ``default_config.json``" was in no built distribution: ``find`` packages modules,
        not data. It worked only because every install so far has been editable.

        Byte-compared, not just listed -- a data file that ships stale is the version of
        this defect nobody would notice.
        """
        path, names = wheel
        assert "NSM/configs/default_config.json" in names
        shipped = zipfile.ZipFile(path).read("NSM/configs/default_config.json")
        assert shipped == (REPO / "NSM" / "configs" / "default_config.json").read_bytes()

    def test_the_wheel_is_named_for_the_package(self, wheel):
        """
        The legible failure for "the build backend could not read `pyproject.toml`".
        setuptools below 61 ignores the `[project]` table instead of refusing it, and
        produces `UNKNOWN-0.0.0` with none of `NSM/` inside -- measured against 58.1.0,
        which is what a GitHub macOS runner ships. Every assertion in this class would
        fail on that wheel, and none of them would say why.
        """
        path, _ = wheel
        assert path.name.split("-")[0] == "nsm", path.name

    def test_every_subpackage_ships(self, wheel):
        _, names = wheel
        shipped = {n.split("/")[1] for n in names if n.startswith("NSM/") and n.count("/") > 1}
        assert {"configs", "datasets", "mesh", "models", "reconstruct", "train"} <= shipped

    def test_nothing_deleted_reappears(self, wheel):
        """``NSM/dependencies`` left with PR #64 and is still in this checkout's build/."""
        _, names = wheel
        assert not [n for n in names if n.startswith("NSM/dependencies/")]


class TestWhereTheVersionComesFrom:
    def test_a_tree_with_no_git_metadata_still_builds(self, wheel):
        """
        The GitHub source-zip case. ``setuptools-scm`` fails outright on a tree with no
        ``.git`` unless a fallback is declared, so this is the assertion that turns red if
        the fallback is ever dropped -- measured, not reasoned: without it ``pip wheel``
        stops at "Getting requirements to build wheel".
        """
        path, _ = wheel
        assert path.exists()

    def test_the_version_is_derived_and_not_written_down(self, wheel):
        """
        Was a strict xfail: the wheel came out as ``0.2.0``, the literal.

        With no ``.git`` present a derived version can only be the declared fallback,
        where a version written into a file comes through unchanged instead. That is the
        whole failure mode -- ``NSM.__version__`` said ``0.2.0`` for 269 commits and 34
        breaking changes, and ``0.0.1`` for years before that.
        """
        fallback = re.search(
            r'^fallback_version\s*=\s*"([^"]+)"',
            (REPO / "pyproject.toml").read_text(encoding="utf-8"),
            re.MULTILINE,
        )
        assert fallback, "pyproject.toml declares no fallback_version"
        path, _ = wheel
        assert path.name.split("-")[1] == fallback.group(1)

    def test_no_source_file_hardcodes_the_version(self):
        """Was a strict xfail: ``__version__ = "0.2.0"`` sat at the bottom of the file."""
        source = (REPO / "NSM" / "__init__.py").read_text(encoding="utf-8")
        assert not re.search(r'^__version__\s*=\s*["\']', source, re.MULTILINE)

    def test_only_one_nsm_distribution_is_discoverable(self):
        """
        The one way the derived version can still be wrong, and it is invisible.

        ``importlib.metadata`` scans ``sys.path``, and an editable install puts the source
        tree on it -- so a leftover ``NSM.egg-info/PKG-INFO`` in the repo root is found
        *before* the ``dist-info`` in site-packages and wins. Measured: with a stale one
        present, ``version("NSM")`` reported ``0.3.1.dev2`` while site-packages held
        ``0.3.1.dev3``; deleting it gave ``dev3`` immediately, and not even
        ``pip install -e . --force-reinstall --no-cache-dir`` cleared it, because pip
        rewrites site-packages and never touches the source tree. ``make clean`` does.

        That matters more here than it would elsewhere: the point of deriving the version
        from the tag is that it cannot silently go stale, and this is the remaining way it
        can.

        **Asserted on the set of discoverable distributions, not by comparing
        ``version("NSM")`` against the egg-info** -- the first draft did that and could not
        fail, because ``version()`` reads the shadow too and was being compared with
        itself. Verified by planting a mismatched ``PKG-INFO``: the first version stayed
        green, this one goes red.
        """
        from importlib.metadata import distributions

        found = {}
        for dist in distributions():
            name = (dist.metadata.get("Name") or "").lower().replace("-", "_")
            if name == "nsm":
                found[str(getattr(dist, "_path", dist))] = dist.version
        if not found:
            pytest.skip("NSM is importable but not installed")

        assert len(set(found.values())) == 1, (
            f"more than one NSM distribution is on sys.path and they disagree: {found}. "
            f"importlib.metadata takes the first, so NSM.__version__ reports whichever "
            f"comes earlier -- usually a stale *.egg-info in the repo root. Run "
            f"`make clean`."
        )

    def test_the_reported_version_matches_the_installed_distribution(self):
        """
        True before and after, and the point of the change is that it stays true without
        anyone maintaining it. Skipped when NSM is on ``sys.path`` but not installed --
        which is a real deployment: ``kneepipeline`` inserts a path at runtime.
        """
        from importlib.metadata import PackageNotFoundError, version

        import NSM

        try:
            installed = version("NSM")
        except PackageNotFoundError:
            pytest.skip("NSM is importable but not installed")
        assert NSM.__version__ == installed


class TestTheSupportedPythons:
    """
    ``requires-python`` is a promise, and it was inherited rather than made.

    It said ``>=3.7`` from the original project boilerplate (``fe403ad``) until v0.3.0,
    which was never true of NSM: every runtime dependency requires 3.9, so on 3.7 or 3.8
    the dependency set does not resolve at all.

    Computed rather than transcribed, because the floor moves under us -- one dependency
    raising its own floor to 3.10 silently makes our declaration false again, and nothing
    else in the repo would notice.
    """

    #: The runtime dependencies, from ``requirements.txt``. Read rather than hardcoded so a
    #: dependency added there is covered without anyone remembering this test exists.
    @staticmethod
    def runtime_requirements():
        import re

        text = (REPO / "requirements.txt").read_text(encoding="utf-8")
        names = []
        for line in text.splitlines():
            line = line.split("#")[0].strip()
            if line:
                names.append(re.split(r"[<>=!~\[]", line)[0].strip())
        return names

    def test_we_do_not_promise_a_python_no_dependency_supports(self):
        from importlib.metadata import PackageNotFoundError, distribution

        from packaging.specifiers import SpecifierSet
        from packaging.version import Version

        declared = re.search(
            r'^requires-python\s*=\s*"([^"]+)"',
            (REPO / "pyproject.toml").read_text(encoding="utf-8"),
            re.MULTILINE,
        )
        assert declared, "pyproject.toml declares no requires-python"
        ours = SpecifierSet(declared.group(1))

        candidates = [f"3.{minor}" for minor in range(7, 15)]
        admitted = [v for v in candidates if ours.contains(Version(v))]
        assert admitted, f"requires-python {declared.group(1)!r} admits no known Python"
        lowest = admitted[0]

        checked, refusing = 0, {}
        for name in self.runtime_requirements():
            try:
                # .get(), not [] -- a distribution whose metadata omits the field raises
                # KeyError on indexing, and which distributions are discoverable depends
                # on what earlier tests put on sys.path. Passed alone, failed in suite.
                spec = distribution(name).metadata.get("Requires-Python")
            except PackageNotFoundError:
                continue
            if not spec:
                continue
            checked += 1
            if not SpecifierSet(spec).contains(Version(lowest)):
                refusing[name] = spec

        assert checked >= 5, f"only {checked} dependencies had metadata to check"
        assert not refusing, (
            f"pyproject.toml promises Python {lowest}, which these dependencies refuse: "
            f"{refusing}. Raise requires-python to match them."
        )


SUBPACKAGES = ["NSM.datasets", "NSM.mesh", "NSM.models", "NSM.reconstruct", "NSM.train"]


def belongs_to_nsm(obj):
    """
    Is this NSM's own -- either defined in it, or one of its submodules?

    Both halves are needed. ``NSM.train`` exports three submodules and nothing else, so a
    rule of "no modules in ``__all__``" would be wrong for it; a rule of "has an
    ``NSM.*`` ``__module__``" would be wrong for every module object.
    """
    import types

    if isinstance(obj, types.ModuleType):
        return obj.__name__.startswith("NSM.")
    return str(getattr(obj, "__module__", "")).startswith("NSM")


class TestPublicApiDeclaration:
    """
    ``__all__`` per subpackage -- Phase 0's last open deliverable (``SCOPE.md`` §3.3) and
    §10.1's stated gate for 1.0.0.

    It is deliberately not at the top level: ``NSM/__init__.py`` imports only ``utils``,
    so a top-level ``__all__`` would force every subpackage to import eagerly and pull
    ``wandb``, ``vtk`` and a root-logger reconfiguration into every ``import NSM``.
    """

    @pytest.mark.parametrize("name", SUBPACKAGES)
    def test_each_subpackage_declares_what_is_public(self, name):
        """Was a strict xfail: ``grep -rn __all__ NSM/`` returned nothing at all."""
        import importlib

        module = importlib.import_module(name)
        declared = getattr(module, "__all__", None)
        assert isinstance(declared, list) and declared
        for entry in declared:
            assert hasattr(module, entry), f"{name}.__all__ names {entry}, which is unbound"
            assert belongs_to_nsm(getattr(module, entry)), f"{name}.__all__ claims {entry}"

    @pytest.mark.parametrize("name", SUBPACKAGES)
    def test_a_star_import_binds_exactly_what_is_declared(self, name):
        """
        The half of ``__all__`` that changes behaviour. Executed rather than reasoned
        about: ``exec`` is the only way to run a star-import inside a function.
        """
        import importlib

        namespace = {}
        exec(f"from {name} import *", namespace)  # noqa: S102 - the behaviour under test
        bound = {n for n in namespace if not n.startswith("__")}
        assert bound == set(importlib.import_module(name).__all__)

    @pytest.mark.parametrize("name", SUBPACKAGES)
    def test_what_a_star_import_binds_today(self, name):
        """
        ``__all__`` controls ``from X import *`` and states intent. It does **not** unbind
        ``NSM.datasets.torch``, and this is the assertion that stops the next reader
        concluding otherwise: the foreign names are still there afterwards.

        ``NSM.train`` is the contrast -- it re-exports three of its own submodules and
        imports nothing into the package namespace, so it has always been clean.
        """
        import importlib

        module = importlib.import_module(name)
        foreign = [
            n
            for n in dir(module)
            if not n.startswith("_") and not belongs_to_nsm(getattr(module, n))
        ]
        if name == "NSM.train":
            assert foreign == []
        else:
            assert foreign, f"{name} stopped leaking -- this measurement needs redoing"
