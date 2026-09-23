"""
What a built wheel contains, and where its version comes from.

The wheel is built from a copy of the tracked files, not from the checkout. A developer's
``build/lib/`` can hold modules deleted long ago, such as ``NSM/dependencies``, and
``--no-build-isolation`` reuses it. The copy also has no ``.git``, which is the GitHub
source-zip case and what proves the version is derived.
"""

import importlib
import re
import shutil
import subprocess
import sys
import types
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

    ``encoding="utf-8"``, not ``text=True``: a GitHub macOS runner's locale is US-ASCII, and
    pip's error banner has a non-ASCII character, so a failed build raised
    ``UnicodeDecodeError`` before its output could be printed.
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
    def test_the_wheel_carries_every_subpackage_and_the_default_config(self, wheel):
        """
        ``default_config.json`` is package data and shipped in no wheel until it was
        declared. It is byte-compared, since a stale copy would pass a listing check.

        The name check is the readable failure for a build backend that cannot read
        ``pyproject.toml``: setuptools below 61 ignores ``[project]`` and produces
        ``UNKNOWN-0.0.0`` with none of ``NSM/`` inside (measured against 58.1.0).
        """
        path, names = wheel
        assert path.name.split("-")[0] == "nsm", path.name
        shipped = {n.split("/")[1] for n in names if n.startswith("NSM/") and n.count("/") > 1}
        assert {"configs", "datasets", "mesh", "models", "reconstruct", "train"} <= shipped
        assert "NSM/configs/generate_sdf_default_config.py" in names
        assert (
            zipfile.ZipFile(path).read("NSM/configs/default_config.json")
            == (REPO / "NSM" / "configs" / "default_config.json").read_bytes()
        )
        assert not [n for n in names if n.startswith("NSM/dependencies/")]


class TestWhereTheVersionComesFrom:
    def test_the_version_is_derived_and_not_written_down(self, wheel):
        """
        The wheel is built with no ``.git``, so a derived version can only be the declared
        fallback. A version written into a source file would come through instead:
        ``NSM.__version__`` said ``0.2.0`` for 269 commits.
        """
        fallback = re.search(
            r'^fallback_version\s*=\s*"([^"]+)"',
            (REPO / "pyproject.toml").read_text(encoding="utf-8"),
            re.MULTILINE,
        )
        assert fallback, "pyproject.toml declares no fallback_version"
        path, _ = wheel
        assert path.name.split("-")[1] == fallback.group(1)
        source = (REPO / "NSM" / "__init__.py").read_text(encoding="utf-8")
        assert not re.search(r'^__version__\s*=\s*["\']', source, re.MULTILINE)

    def test_one_nsm_distribution_is_discoverable_and_it_is_the_one_imported(self):
        """
        ``importlib.metadata`` scans ``sys.path``, and an editable install puts the source
        tree on it, so a stale ``NSM.egg-info`` in the repo root wins over site-packages.
        Measured: it reported ``0.3.1.dev2`` while site-packages held ``0.3.1.dev3``, and
        ``pip install -e . --force-reinstall`` did not clear it. ``make clean`` does.

        Asserted on the set of distributions: comparing ``version("NSM")`` with the
        egg-info could not fail, because ``version()`` reads the shadow too.
        """
        from importlib.metadata import distributions, version

        import NSM

        found = {
            str(getattr(dist, "_path", dist)): dist.version
            for dist in distributions()
            if (dist.metadata.get("Name") or "").lower().replace("-", "_") == "nsm"
        }
        if not found:
            pytest.skip("NSM is importable but not installed")
        assert (
            len(set(found.values())) == 1
        ), f"NSM distributions disagree: {found}. Run make clean."
        assert NSM.__version__ == version("NSM")


def test_requires_python_admits_no_version_a_dependency_refuses():
    """
    ``requires-python`` said ``>=3.7`` until v0.3.0, and every runtime dependency needs
    3.9. Computed from the installed metadata, so a dependency raising its floor fails here.
    """
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
    lowest = next(v for v in (f"3.{m}" for m in range(7, 15)) if ours.contains(Version(v)))

    checked, refusing = 0, {}
    for line in (REPO / "requirements.txt").read_text(encoding="utf-8").splitlines():
        name = re.split(r"[<>=!~\[]", line.split("#")[0].strip())[0].strip()
        if not name:
            continue
        try:
            # .get(): a distribution whose metadata omits the field raises on [].
            spec = distribution(name).metadata.get("Requires-Python")
        except PackageNotFoundError:
            continue
        if spec:
            checked += 1
            if not SpecifierSet(spec).contains(Version(lowest)):
                refusing[name] = spec

    assert checked >= 5, f"only {checked} dependencies had metadata to check"
    assert not refusing, f"requires-python admits {lowest}, which these refuse: {refusing}"


class TestPublicApiDeclaration:
    """
    ``__all__`` per subpackage (``SCOPE.md`` §3.3). Not at the top level: ``NSM/__init__.py``
    imports only ``utils``, and a top-level ``__all__`` would import every subpackage.
    """

    def test_each_subpackage_declares_exactly_what_a_star_import_binds(self):
        """
        Every declared name resolves and is NSM's own, and ``from X import *`` binds
        exactly the declaration. ``__all__`` does not unbind ``NSM.datasets.torch``.
        """
        problems = []
        for name in ["NSM.datasets", "NSM.mesh", "NSM.models", "NSM.reconstruct", "NSM.train"]:
            module = importlib.import_module(name)
            declared = getattr(module, "__all__", None)
            if not declared:
                problems.append(f"{name} declares no __all__")
                continue
            for entry in declared:
                obj = getattr(module, entry, None)
                owner = obj.__name__ if isinstance(obj, types.ModuleType) else obj.__module__
                if obj is None or not str(owner).startswith("NSM"):
                    problems.append(f"{name}.__all__ names {entry}")
            namespace = {}
            exec(f"from {name} import *", namespace)  # noqa: S102 - the behaviour under test
            if {n for n in namespace if not n.startswith("__")} != set(declared):
                problems.append(f"from {name} import * binds something else")
        assert problems == []
