"""Resolve a PR diff to the cp_measure measurement functions it actually changes.

Run as ``python -m cp_measure._bench.targets --base <ref> --head <ref>``; prints a JSON object
describing which measurement functions the benchmark harness should time.

The resolution is **symbol-level**, not file-level. A change to a shared helper (e.g.
``utils._zernike_scores``) selects only the measurement functions whose call graph actually
reaches that symbol — not every function whose module merely imports ``utils``. Conversely the
function-vs-symbol graph is rooted at an explicit entry-point table (the public ``get_*``
features), so lazy imports in ``bulk.py`` cannot cause an entry-point to be missed.

It reads everything from git refs (``git show <ref>:<path>``), so it works without checking the
tree out and matches CI, where ``head`` is the checked-out PR SHA. The diff is taken against the
merge-base (``base...head``), which is correct for stacked PRs as long as ``base`` is the PR's
real base branch.

Output states (never collapsed):
- ``benchmarked``  — at least one supported feature is impacted; ``benchmarked`` lists them.
- ``skipped-unsupported`` — the PR changes measurement code we don't benchmark in v1
  (multimask / numba backends); ``skipped_unsupported`` names it. Distinct from "no change".
- ``empty`` — the PR changes no measurement code at all (docs, CI, an unused helper).
"""

from __future__ import annotations

import argparse
import ast
import json
import re
import subprocess
import warnings
from collections import defaultdict


def _parse(source: str) -> ast.Module:
    # Source at arbitrary refs may contain invalid escape sequences etc.; we only need the
    # structure, so silence those warnings rather than spamming the benchmark log.
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", SyntaxWarning)
        return ast.parse(source)


# Supported features: feature name -> (module, function, arity, registry). Arity is the number of
# intensity-image arguments the harness must supply (core: 1, correlation: 2). The feature names
# mirror bulk.py's _CORE / _CORRELATION registries; test_targets cross-checks that they stay in
# sync, so this explicit table can't silently drift from the real registry.
_CORE_MOD = "cp_measure.core"
SUPPORTED = {
    "radial_distribution": (
        f"{_CORE_MOD}.measureobjectintensitydistribution",
        "get_radial_distribution",
        1,
        "core",
    ),
    "radial_zernikes": (
        f"{_CORE_MOD}.measureobjectintensitydistribution",
        "get_radial_zernikes",
        1,
        "core",
    ),
    "intensity": (f"{_CORE_MOD}.measureobjectintensity", "get_intensity", 1, "core"),
    "sizeshape": (f"{_CORE_MOD}.measureobjectsizeshape", "get_sizeshape", 1, "core"),
    "zernike": (f"{_CORE_MOD}.measureobjectsizeshape", "get_zernike", 1, "core"),
    # function renamed across branches: get_ferret (main, a typo) -> get_feret (later).
    "feret": (
        f"{_CORE_MOD}.measureobjectsizeshape",
        ("get_feret", "get_ferret"),
        1,
        "core",
    ),
    "texture": (f"{_CORE_MOD}.measuretexture", "get_texture", 1, "core"),
    "granularity": (f"{_CORE_MOD}.measuregranularity", "get_granularity", 1, "core"),
    "costes": (
        f"{_CORE_MOD}.measurecolocalization",
        "get_correlation_costes",
        2,
        "correlation",
    ),
    "pearson": (
        f"{_CORE_MOD}.measurecolocalization",
        "get_correlation_pearson",
        2,
        "correlation",
    ),
    "manders_fold": (
        f"{_CORE_MOD}.measurecolocalization",
        "get_correlation_manders_fold",
        2,
        "correlation",
    ),
    "rwc": (
        f"{_CORE_MOD}.measurecolocalization",
        "get_correlation_rwc",
        2,
        "correlation",
    ),
}

# Measurement code we knowingly don't benchmark in v1; a PR touching only these is
# "skipped-unsupported", not "empty". Roots are resolved at HEAD (numba dir may not exist).
_UNSUPPORTED_ROOTS = {
    "multimask:measureobjectneighbors": (
        "cp_measure.multimask.measureobjectneighbors",
        "measureobjectneighbors",
    ),
    "multimask:measureobjectoverlap": (
        "cp_measure.multimask.measureobjectoverlap",
        "measureobjectoverlap",
    ),
}
# Any symbol whose module starts with one of these is itself unsupported (numba backends).
_UNSUPPORTED_MODULE_PREFIXES = ("cp_measure.core.numba",)

_MAX_BENCHMARKED = 12  # runtime-safety cap; excess is reported, never silently dropped


def _git(*args: str) -> str:
    return subprocess.run(
        ["git", *args], check=True, capture_output=True, text=True
    ).stdout


def _list_py(ref: str) -> dict[str, str]:
    """{module name -> repo path} for every ``src/cp_measure/**.py`` at ``ref``."""
    out = _git("ls-tree", "-r", "--name-only", ref, "--", "src/cp_measure")
    modules = {}
    for path in out.splitlines():
        if path.endswith(".py"):
            modules[_module_name(path)] = path
    return modules


def _read(ref: str, path: str) -> str:
    return _git("show", f"{ref}:{path}")


def _module_name(path: str) -> str:
    rel = path.split("src/", 1)[-1][: -len(".py")]
    parts = rel.split("/")
    if parts[-1] == "__init__":
        parts = parts[:-1]
    return ".".join(parts)


def _end(node: ast.AST) -> int:
    return getattr(node, "end_lineno", node.lineno)


def _symbol_names(node: ast.stmt) -> list[str]:
    """Top-level symbol name(s) a statement defines (def/class/module-level assignment)."""
    if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
        return [node.name]
    if isinstance(node, ast.Assign):
        return [t.id for t in node.targets if isinstance(t, ast.Name)]
    if isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
        return [node.target.id]
    return []


def _dotted(node: ast.AST) -> str | None:
    """Reconstruct ``a.b.c`` from a (possibly nested) Attribute/Name node."""
    parts = []
    while isinstance(node, ast.Attribute):
        parts.append(node.attr)
        node = node.value
    if isinstance(node, ast.Name):
        parts.append(node.id)
        return ".".join(reversed(parts))
    return None


def _resolve_import(
    node: ast.ImportFrom | ast.Import, mod: str, module_set: set[str]
) -> dict:
    """Map local names bound by one import statement to ('symbol'|'module', target)."""
    out: dict[str, tuple[str, str]] = {}
    if isinstance(node, ast.Import):
        for alias in node.names:
            if alias.name.startswith("cp_measure"):
                out[alias.asname or alias.name] = ("module", alias.name)
        return out
    # ImportFrom
    if node.level:  # relative: resolve against the current package
        base = mod.rsplit(".", node.level)[0] if "." in mod else ""
        pkg = f"{base}.{node.module}" if node.module else base
    else:
        pkg = node.module or ""
    if not pkg.startswith("cp_measure"):
        return out
    for alias in node.names:
        full = f"{pkg}.{alias.name}"
        if full in module_set:  # importing a submodule
            out[alias.asname or alias.name] = ("module", full)
        else:  # importing a symbol from a module
            out[alias.asname or alias.name] = ("symbol", f"{pkg}:{alias.name}")
    return out


def build_graph(ref: str):
    """Return (defs, edges, modules). ``defs``: sym_id -> module. ``edges``: sym_id -> set(sym_id)."""
    modules = _list_py(ref)
    module_set = set(modules)
    trees = {mod: _parse(_read(ref, path)) for mod, path in modules.items()}

    defs: dict[str, str] = {}
    top_names: dict[str, dict[str, str]] = defaultdict(dict)
    for mod, tree in trees.items():
        for node in tree.body:
            for name in _symbol_names(node):
                sym = f"{mod}:{name}"
                defs[sym] = mod
                top_names[mod][name] = sym

    edges: dict[str, set[str]] = defaultdict(set)
    for mod, tree in trees.items():
        module_imports = {}
        for node in tree.body:
            if isinstance(node, (ast.Import, ast.ImportFrom)):
                module_imports.update(_resolve_import(node, mod, module_set))
        local = top_names[mod]
        for node in tree.body:
            names = _symbol_names(node)
            if not names:
                continue
            refs = _referenced_symbols(
                node, mod, module_imports, local, defs, module_set
            )
            for name in names:
                edges[f"{mod}:{name}"] |= refs
    return defs, edges, modules


def _referenced_symbols(node, mod, module_imports, local, defs, module_set) -> set[str]:
    """Package symbols referenced inside one top-level statement's subtree."""
    # Local (function-body) imports extend the module-level import map for this symbol only.
    imports = dict(module_imports)
    for sub in ast.walk(node):
        if isinstance(sub, (ast.Import, ast.ImportFrom)) and sub is not node:
            imports.update(_resolve_import(sub, mod, module_set))

    refs: set[str] = set()
    for sub in ast.walk(node):
        if isinstance(sub, ast.Attribute):
            dotted = _dotted(sub)
            if dotted:
                hit = _resolve_dotted(dotted, imports, module_set)
                if hit in defs:
                    refs.add(hit)
        elif isinstance(sub, ast.Name) and isinstance(sub.ctx, ast.Load):
            if sub.id in local:
                refs.add(local[sub.id])
            elif sub.id in imports and imports[sub.id][0] == "symbol":
                sym = imports[sub.id][1]
                if sym in defs:
                    refs.add(sym)
    return refs


def _resolve_dotted(dotted: str, imports: dict, module_set: set[str]) -> str | None:
    """Resolve ``a.b.c`` to a ``module:symbol`` id, or None."""
    parts = dotted.split(".")
    # Longest module prefix wins: cp_measure.core.x.get_zernike -> (cp_measure.core.x, get_zernike)
    for i in range(len(parts) - 1, 0, -1):
        prefix = ".".join(parts[:i])
        if prefix in module_set:
            return f"{prefix}:{parts[i]}"
    # Otherwise the head name may be an imported module or symbol.
    head = parts[0]
    if head in imports:
        kind, target = imports[head]
        if kind == "module" and len(parts) >= 2:
            return f"{target}:{parts[1]}"
        if kind == "symbol":
            return target
    return None


_HUNK = re.compile(r"^@@ -\d+(?:,\d+)? \+(\d+)(?:,(\d+))? @@", re.M)


def changed_symbols(base: str, head: str) -> set[str]:
    """Top-level symbols (def/class/module-assignment) overlapping a changed HEAD line range."""
    diff = _git("diff", "-U0", f"{base}...{head}", "--", "src/cp_measure")
    per_file: dict[str, list[tuple[int, int]]] = defaultdict(list)
    current = None
    for line in diff.splitlines():
        if line.startswith("+++ b/"):
            current = line[len("+++ b/") :]
        elif line.startswith("+++ ") and "/dev/null" in line:
            current = None
        elif current and (m := _HUNK.match(line)):
            start = int(m.group(1))
            count = int(m.group(2)) if m.group(2) is not None else 1
            end = start + max(count, 1) - 1
            per_file[current].append((start, end))

    changed: set[str] = set()
    for path, ranges in per_file.items():
        if not path.endswith(".py") or not path.startswith("src/cp_measure"):
            continue
        try:
            tree = _parse(_read(head, path))
        except subprocess.CalledProcessError:
            continue  # deleted at head
        mod = _module_name(path)
        for node in tree.body:
            names = _symbol_names(node)
            if not names:
                continue
            lo, hi = node.lineno, _end(node)
            if any(not (r_end < lo or r_start > hi) for r_start, r_end in ranges):
                for name in names:
                    changed.add(f"{mod}:{name}")
    return changed


def _reachable(root: str, edges: dict[str, set[str]]) -> set[str]:
    seen, stack = set(), [root]
    while stack:
        cur = stack.pop()
        if cur in seen:
            continue
        seen.add(cur)
        stack.extend(edges.get(cur, ()))
    return seen


def resolve(base: str, head: str) -> dict:
    defs, edges, _ = build_graph(head)
    changed = changed_symbols(base, head)

    benchmarked = []
    for feature, (module, func, arity, registry) in SUPPORTED.items():
        # `func` may be a tuple of name candidates (a function renamed across branches, e.g.
        # get_ferret -> get_feret); use whichever exists at this ref.
        candidates = (func,) if isinstance(func, str) else func
        root = next(
            (f"{module}:{fn}" for fn in candidates if f"{module}:{fn}" in defs), None
        )
        if root is None:
            continue  # feature not present at this ref
        reach = _reachable(root, edges)
        if reach & changed:
            benchmarked.append(
                {
                    "feature": feature,
                    "module": module,
                    "function": root.split(":", 1)[1],
                    "arity": arity,
                    "registry": registry,
                    "direct": root
                    in changed,  # the feature fn itself changed (vs a helper)
                }
            )

    # Unsupported: changed measurement code we don't benchmark in v1.
    unsupported = set()
    for label, (module, func) in _UNSUPPORTED_ROOTS.items():
        root = f"{module}:{func}"
        if root in defs and _reachable(root, edges) & changed:
            unsupported.add(label)
    for sym in changed:
        if sym.split(":", 1)[0].startswith(_UNSUPPORTED_MODULE_PREFIXES):
            unsupported.add(sym)

    benchmarked.sort(key=lambda b: (not b["direct"], b["feature"]))  # direct hits first
    dropped = [b["feature"] for b in benchmarked[_MAX_BENCHMARKED:]]
    benchmarked = benchmarked[:_MAX_BENCHMARKED]

    if benchmarked:
        state = "benchmarked"
    elif unsupported:
        state = "skipped-unsupported"
    else:
        state = "empty"

    return {
        "state": state,
        "base": base,
        "head": head,
        "benchmarked": benchmarked,
        "skipped_unsupported": sorted(unsupported),
        "changed_symbol_count": len(changed),
        "dropped": dropped,
    }


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--base", required=True, help="PR base ref (merge-base is taken vs head)"
    )
    parser.add_argument("--head", required=True, help="PR head ref / SHA")
    args = parser.parse_args(argv)
    print(json.dumps(resolve(args.base, args.head), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
