from pathlib import Path

_PACKAGE_DIR = Path(__file__).resolve().parent


def _find_project_dir() -> Path:
    """Nearest ancestor of the package that looks like the project root.

    In a source checkout `src/scibs/config.py` sits three levels under the
    root, but a non-editable install drops the package into site-packages
    where no such root exists -- so look for a marker rather than counting
    directories, and fall back to the working directory.
    """
    for candidate in _PACKAGE_DIR.parents:
        if (candidate / "pyproject.toml").is_file() or (candidate / ".git").exists():
            return candidate
    return Path.cwd()


# Project paths
PROJECT_DIR = _find_project_dir()
PROJECT_SRC_DIR = PROJECT_DIR / "src"
PROJECT_TESTS_DIR = PROJECT_DIR / "tests"

# Everything the scripts generate (final meshes, plot screenshots, debug
# extraction dumps) lands here, inside the project itself.
PROJECT_OUTPUT_DIR = PROJECT_DIR / "out"

# C code source paths
SOURCE_DIR = PROJECT_DIR.parent / "SCIBS"
SOURCE_DATA_DIR = SOURCE_DIR / "data"
SOURCE_CODE_DIR = SOURCE_DIR / "source"
