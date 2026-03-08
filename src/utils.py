import numpy as np


def is_model(array: np.ndarray):
    if not isinstance(array, np.ndarray):
        return False

    names = array.dtype.names
    if names is None:
        return False

    return "cell" in names and "node" in names


def load_tri(path: str) -> tuple[np.ndarray, np.ndarray]:
    with open(path, "r") as f:
        vert_count = int(f.readline().split()[0])

    verts = np.loadtxt(
        path, skiprows=1, usecols=(1, 2, 3), dtype=np.float64, max_rows=vert_count
    )
    ids = np.loadtxt(
        path, skiprows=vert_count + 2, usecols=(1, 2, 3), dtype=np.int64
    )  # skip vertex header (+1), skip index header (+1), skip vertex entries (+vertex_count)
    return verts, ids
