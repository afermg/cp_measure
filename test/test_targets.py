"""Tests for the symbol-level target mapper ``cp_measure._bench.targets``.

The mapper's whole reason to exist is to resolve a PR diff to *exactly* the measurement functions
it changes — at symbol granularity, so a shared-helper edit selects only the features whose call
graph reaches that helper, not every feature whose module merely imports the helper's module.

The core tests are hermetic: they build a throwaway git repo with a miniature package (two
features sharing a util module but using different helpers) and assert the mapper distinguishes
them. A final, environment-guarded test checks the real PRs #74/#75 when their refs are present.
"""

import json
import subprocess

import pytest

from cp_measure._bench import targets

# Miniature package mirroring the src/cp_measure/** layout the mapper scans. Two features live in
# the SAME module and import from the SAME util module, but use DIFFERENT helpers — file-closure
# selection cannot tell them apart; symbol-level must.
_BASE_FILES = {
    "src/cp_measure/__init__.py": "",
    "src/cp_measure/utils.py": (
        "def _helper_a(x):\n    return x + 1\n\n\ndef _helper_b(x):\n    return x - 1\n"
    ),
    "src/cp_measure/core/__init__.py": "",
    "src/cp_measure/core/sizeshape.py": (
        "from cp_measure.utils import _helper_a, _helper_b\n\n\n"
        "def get_zernike(masks, pixels):\n    return _helper_a(masks)\n\n\n"
        "def get_feret(masks, pixels):\n    return _helper_b(masks)\n"
    ),
    "src/cp_measure/core/texture.py": (
        "from cp_measure.utils import _helper_b\n\n\n"
        "def get_texture(masks, pixels):\n    return _helper_b(pixels)\n"
    ),
    "src/cp_measure/multimask/__init__.py": "",
    "src/cp_measure/multimask/measureobjectneighbors.py": (
        "from cp_measure.utils import _helper_a\n\n\n"
        "def measureobjectneighbors(masks1, masks2):\n    return _helper_a(masks1)\n"
    ),
    "src/cp_measure/_unrelated.py": "def unused():\n    return 0\n",
}

_MINI_SUPPORTED = {
    "zernike": ("cp_measure.core.sizeshape", "get_zernike", 1, "core"),
    "feret": ("cp_measure.core.sizeshape", "get_feret", 1, "core"),
    "texture": ("cp_measure.core.texture", "get_texture", 1, "core"),
}
_MINI_UNSUPPORTED = {
    "multimask:neighbors": (
        "cp_measure.multimask.measureobjectneighbors",
        "measureobjectneighbors",
    ),
}


def _git(repo, *args):
    subprocess.run(
        ["git", "-C", str(repo), *args], check=True, capture_output=True, text=True
    )


def _write(repo, files):
    for rel, content in files.items():
        p = repo / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(content)


@pytest.fixture
def repo(tmp_path, monkeypatch):
    """A git repo at the base state; returns a helper to apply a change and resolve it."""
    _git(tmp_path, "init", "-q")
    _git(tmp_path, "config", "user.email", "t@t.t")
    _git(tmp_path, "config", "user.name", "t")
    _write(tmp_path, _BASE_FILES)
    _git(tmp_path, "add", "-A")
    _git(tmp_path, "commit", "-qm", "base")
    base = subprocess.run(
        ["git", "-C", str(tmp_path), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()

    monkeypatch.setattr(targets, "SUPPORTED", _MINI_SUPPORTED)
    monkeypatch.setattr(targets, "_UNSUPPORTED_ROOTS", _MINI_UNSUPPORTED)
    monkeypatch.chdir(tmp_path)

    def apply_and_resolve(changes):
        _write(tmp_path, changes)
        _git(tmp_path, "add", "-A")
        _git(tmp_path, "commit", "-qm", "change")
        return targets.resolve(base, "HEAD")

    return apply_and_resolve


def _features(result):
    return sorted(b["feature"] for b in result["benchmarked"])


def test_shared_helper_edit_selects_only_reaching_feature(repo):
    # Edit _helper_a, used ONLY by get_zernike. file-closure would also pick feret (same module)
    # and anything importing utils; symbol-level must pick zernike alone.
    changed = dict(_BASE_FILES)
    changed["src/cp_measure/utils.py"] = changed["src/cp_measure/utils.py"].replace(
        "return x + 1", "return x + 100"
    )
    result = repo({"src/cp_measure/utils.py": changed["src/cp_measure/utils.py"]})
    assert result["state"] == "benchmarked"
    assert _features(result) == ["zernike"]


def test_shared_helper_used_by_two_features(repo):
    # _helper_b is used by get_feret AND get_texture (different modules) — both selected, nothing else.
    src = _BASE_FILES["src/cp_measure/utils.py"].replace("return x - 1", "return x - 2")
    result = repo({"src/cp_measure/utils.py": src})
    assert _features(result) == ["feret", "texture"]


def test_direct_feature_edit(repo):
    src = _BASE_FILES["src/cp_measure/core/sizeshape.py"].replace(
        "return _helper_b(masks)", "return _helper_b(masks) + 1"
    )
    result = repo({"src/cp_measure/core/sizeshape.py": src})
    assert _features(result) == ["feret"]
    assert next(b for b in result["benchmarked"] if b["feature"] == "feret")["direct"]


def test_non_measurement_change_is_empty(repo):
    result = repo({"src/cp_measure/_unrelated.py": "def unused():\n    return 1\n"})
    assert result["state"] == "empty"
    assert result["benchmarked"] == []


def test_multimask_change_is_skipped_unsupported(repo):
    src = _BASE_FILES["src/cp_measure/multimask/measureobjectneighbors.py"].replace(
        "return _helper_a(masks1)", "return _helper_a(masks1) + 1"
    )
    result = repo({"src/cp_measure/multimask/measureobjectneighbors.py": src})
    assert result["state"] == "skipped-unsupported"
    assert result["benchmarked"] == []
    assert "multimask:neighbors" in result["skipped_unsupported"]


def test_supported_table_matches_bulk_registries():
    # The entry-point table must cover exactly the real registries. Compare on FUNCTION identity
    # (module, name) + arity, not on feature labels — the registry key is branch-dependent (main
    # spells it "ferret", later branches "feret") and the label is cosmetic; (module, function) is
    # the stable identity the harness actually imports.
    from cp_measure import bulk

    def identify(fn):
        fn = getattr(
            fn, "func", fn
        )  # unwrap functools.partial (legacy-wrapped registries)
        return (fn.__module__, fn.__name__)

    registry = {}
    for fn in bulk.get_core_measurements().values():
        registry[identify(fn)] = 1
    for fn in bulk.get_correlation_measurements().values():
        registry[identify(fn)] = 2

    # Every registry function must be covered by exactly one table entry (module + a name
    # candidate + matching arity), and there must be no orphan table entries.
    assert len(targets.SUPPORTED) == len(registry)
    for module, func, arity, _reg in targets.SUPPORTED.values():
        candidates = (func,) if isinstance(func, str) else func
        matched = [(module, name) for name in candidates if (module, name) in registry]
        assert len(matched) == 1, (
            f"{module}:{func} must match exactly one registry function"
        )
        assert registry[matched[0]] == arity


def _ref_exists(ref):
    return (
        subprocess.run(
            ["git", "rev-parse", "--verify", "-q", ref], capture_output=True
        ).returncode
        == 0
    )


@pytest.mark.skipif(
    not (
        _ref_exists("origin/main")
        and _ref_exists("origin/perf/zernike-vectorize")
        and _ref_exists("origin/perf/radial-zernike-vectorize")
    ),
    reason="PR refs not present in this checkout",
)
def test_real_zernike_prs_resolve_precisely():
    # The motivating fixture: a _zernike_scores helper change must select only the zernike features,
    # not the 6 features whose modules import utils.
    r74 = targets.resolve("origin/main", "origin/perf/zernike-vectorize")
    assert _features(r74) == ["zernike"]
    r75 = targets.resolve(
        "origin/perf/zernike-vectorize", "origin/perf/radial-zernike-vectorize"
    )
    assert _features(r75) == ["radial_zernikes"]


def test_cli_outputs_json(repo, capsys):
    repo({"src/cp_measure/_unrelated.py": "def unused():\n    return 2\n"})
    targets.main(["--base", "HEAD~1", "--head", "HEAD"])
    out = json.loads(capsys.readouterr().out)
    assert out["state"] == "empty"
