import numpy as np

def is_model(array: np.ndarray):
    if not isinstance(array, np.ndarray):
        return False

    names = array.dtype.names
    if names is None:
        return False

    return "cell" in names and "node" in names