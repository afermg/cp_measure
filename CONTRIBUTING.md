# Contributing to cp_measure

Thank you for your interest in contributing to cp_measure! We welcome contributions from the community and appreciate your efforts to help improve this project.

## Project Philosophy

cp_measure aims to be **lean in dependencies and complexity to maximize its accessibility for the ML/Bio communities**. We strive to provide programmatic access to CellProfiler measurements while keeping the codebase as transparent and self-contained as possible.

## Getting Started

1. Fork the repository on GitHub
2. Clone your fork locally:
   ```bash
   git clone git@github.com:YOUR_USERNAME/cp_measure.git
   cd cp_measure
   ```
3. Set up the development environment using [uv](https://github.com/astral-sh/uv?tab=readme-ov-file#installation):
   ```bash
   uv sync --all-extras
   ```

## Types of Contributions

We welcome various types of contributions:

- **Documentation improvements**: Help clarify usage, add examples, or improve docstrings
- **Bug fixes**: Report and fix issues you encounter
- **Performance improvements**: Optimize existing measurements while maintaining compatibility
- **New features**: Propose and implement new measurement modules (please discuss first via issues)
- **Tests**: Add or improve test coverage, especially numerical validation against CellProfiler

## Development Process

### Before You Start

1. Check existing issues and pull requests to avoid duplicate work
2. For significant changes, open an issue to discuss your proposal first

### Making Changes

1. Create a new branch for your changes:
   ```bash
   git checkout -b feature/your-feature-name
   ```

2. Follow the existing code style and conventions:
   - Each measurement module should be as self-contained as possible
   - (Recommended) Use float32 values between 0 and 1 for images to match CellProfiler

3. Write or update tests for your changes

4. Run the test suite:
   ```bash
   uv run pytest --cov --color=yes --cov-report=xml
   ```

### Code Quality Standards

All contributions must pass:
- **Linting**: `uv run ruff check src/`
- **Formatting**: `uv run ruff format src/`
- **Tests**: All existing and new tests must pass

### Submitting Your Contribution

1. Commit your changes with clear, descriptive commit messages
2. Push your branch to your fork
3. Create a pull request against the main branch of the original repository
4. In the PR description:
   - Explain what changes you've made and why
   - Reference any related issues
   - Include any relevant test results or benchmarks

## Testing Guidelines

- Unit tests should be added for all new functionality
- Tests go in the `test/` directory following the existing structure
- When adding new measurements, include validation against CellProfiler's output where possible
- CI runs on Python 3.10 and 3.12, on both Ubuntu and macOS

## Current Roadmap & Priorities

The project is actively developing in these areas:

1. **Speed improvements**: JIT compilation and optional GPU support
2. **Numerical validation**: Comprehensive tests comparing cp_measure to CellProfiler outputs
3. **Interface formalization**: Standardizing input/output formats for 2D/3D images and masks
4. **Documentation**: Improving examples and API documentation

## Communication

- **Issues**: Use GitHub issues for bug reports, feature requests, and general discussions
- **Pull Requests**: For code contributions and documentation improvements
- **Discussions**: Feel free to start discussions about potential improvements or use cases

## Notes on Project Relationship to CellProfiler/cellprofiler_library

cp_measure focuses specifically on reimplementing CellProfiler's measurement modules with a unified interface, while the CellProfiler team's `cellprofiler_library` project aims to port many CellProfiler modules, not only measurement modules. While there may be some overlap in the future, cp_measure maintains its focus on:
- Minimal dependencies
- Transparent, self-contained measurement implementations
- Unified interface for ML/computational biology workflows
- Performance optimizations (GPU support, JIT compilation)

## Questions?

If you have any questions about contributing, please open an issue and we'll be happy to help!

Thank you for contributing to cp_measure!
