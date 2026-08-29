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
    for name in listing.stdout.decode().split("\0"):
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
    """``pip wheel`` without build isolation, so no network and no re-resolve."""
    result = subprocess.run(
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
        text=True,
    )
    return result


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

    @pytest.mark.xfail(strict=True, reason="§8.0.O: the version is the literal in NSM/__init__.py")
    def test_the_version_is_derived_and_not_written_down(self, wheel):
        """
        With no ``.git`` present, a derived version can only be the declared fallback. A
        version written into a file would come through unchanged instead, which is the
        whole failure mode: ``NSM.__version__`` has said ``0.2.0`` for 269 commits and 34
        breaking changes.
        """
        fallback = re.search(
            r'^fallback_version\s*=\s*"([^"]+)"',
            (REPO / "pyproject.toml").read_text(encoding="utf-8"),
            re.MULTILINE,
        )
        assert fallback, "pyproject.toml declares no fallback_version"
        path, _ = wheel
        assert path.name.split("-")[1] == fallback.group(1)

    @pytest.mark.xfail(strict=True, reason="§8.0.O: NSM/__init__.py hardcodes __version__")
    def test_no_source_file_hardcodes_the_version(self):
        source = (REPO / "NSM" / "__init__.py").read_text(encoding="utf-8")
        assert not re.search(r'^__version__\s*=\s*["\']', source, re.MULTILINE)

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

    @pytest.mark.xfail(strict=True, reason="§8.0.O: no subpackage declares __all__")
    @pytest.mark.parametrize("name", SUBPACKAGES)
    def test_each_subpackage_declares_what_is_public(self, name):
        import importlib

        module = importlib.import_module(name)
        declared = getattr(module, "__all__", None)
        assert isinstance(declared, list) and declared
        for entry in declared:
            assert hasattr(module, entry), f"{name}.__all__ names {entry}, which is unbound"
            assert belongs_to_nsm(getattr(module, entry)), f"{name}.__all__ claims {entry}"

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
