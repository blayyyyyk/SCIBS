import os
from pathlib import Path

# Project paths
PROJECT_DIR = Path(__file__).parent.parent
PROJECT_SRC_DIR = PROJECT_DIR / "src"
PROJECT_TESTS_DIR = PROJECT_DIR / "tests"

# C code source paths
SOURCE_DIR = PROJECT_DIR.parent / "SCIBS"
SOURCE_DATA_DIR = SOURCE_DIR / "data"
SOURCE_CODE_DIR = SOURCE_DIR / "source"


