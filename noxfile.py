import nox

PYTHON_VERSIONS = ["3.10", "3.11", "3.12", "3.13", "3.14"]

nox.options.default_venv_backend = "uv"


@nox.session(python=PYTHON_VERSIONS)
def tests(session: nox.Session) -> None:
    """Run the test suite under each supported Python version."""
    try:
        session.install(
            "pytest",
            "pytest-cov",
            "pytest-markdown-docs",
            "six",  # centrosome runtime dep, not declared in its metadata
            ".",
        )
    except Exception:
        session.skip(
            f"Could not install dependencies for Python {session.python} "
            "(likely missing binary wheels for C extensions — run via CI instead)"
        )
    session.run("pytest", "test/", "-q")


@nox.session(python=PYTHON_VERSIONS)
def mypy(session: nox.Session) -> None:
    """Run static type checking against each supported Python version."""
    # Install mypy + numpy only; cp_measure installed without its C-extension
    # dependencies so mypy can resolve the package source without needing
    # mahotas/centrosome wheels (--ignore-missing-imports covers the rest).
    session.install("mypy>=1.0,<1.20", "pathspec>=0.10,<1.0", "numpy")
    session.install("--no-deps", ".")
    session.run(
        "mypy",
        "src/cp_measure/core/",
        f"--python-version={session.python}",
        "--ignore-missing-imports",
    )
