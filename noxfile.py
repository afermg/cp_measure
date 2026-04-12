import nox

PYTHON_VERSIONS = ["3.10", "3.11", "3.12", "3.13", "3.14"]

nox.options.default_venv_backend = "uv"


@nox.session(python=PYTHON_VERSIONS)
def tests(session: nox.Session) -> None:
    """Run the test suite under each supported Python version."""
    session.install(
        "pytest",
        "pytest-cov",
        "pytest-markdown-docs",
        "six",  # centrosome runtime dep, not declared in its metadata
        ".",
    )
    session.run("pytest", "test/", "-q")


@nox.session(python=PYTHON_VERSIONS)
def mypy(session: nox.Session) -> None:
    """Run static type checking against each supported Python version."""
    session.install("mypy>=1.0,<1.20", "pathspec>=0.10,<1.0", "numpy", ".")
    session.run(
        "mypy",
        "src/cp_measure/core/",
        f"--python-version={session.python}",
        "--ignore-missing-imports",
    )
