from numpy.typing import ArrayLike
from typing import Callable, Optional, Sequence, Iterable

import numpy as np
import scipy.io as sio

from .mat import is_model


def _read_mat(in_path: str, struct_filter: Optional[Callable[[str], bool]] = None) -> dict[str, np.ndarray]:
    mat_contents = sio.loadmat(in_path)
    out = {}
    for i, (struct_name, struct_content) in enumerate(mat_contents.values()):
        # check if struct is what we want
        if struct_filter is not None:
            if not struct_filter(struct_name): continue

        # check if struct is a valid model
        if not is_model(struct_content): continue

        out[struct_name] = mat_contents[struct_name]

    return out

def _write_mat(out_path: str, data: dict[str, np.ndarray]) -> None:
    mat_contents = sio.loadmat(out_path)
    for k, v in data.items():
        if not k in mat_contents: continue
        mat_contents[k] = v

    for key in ['__header__', '__version__', '__globals__']:
        if not k in mat_contents: continue
        mat_contents.pop(key, None)

    sio.savemat(out_path, mat_contents)


def read_tet_mat(in_path: str) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    mat_contents = _read_mat(in_path)
    out = {}
    for k, v in mat_contents.items():
        pts = mat_contents[k]['node'][0, 0]
        ids = mat_contents[k]['cell'][0, 0]
        out[k] = (pts, ids)

    return out

def write_tet(out_path: str, data):
    reshape = {}
    for k, (pts, ids) in data.items():
        dtype = np.dtype([('node', pts.dtype), ('cell', ids.dtype)])
        reshape[k] = np.array([(pts, ids)], dtype=dtype)

    _write_mat(out_path, reshape)

def read_tri(in_path: str) -> tuple[np.ndarray, np.ndarray]:
    """Loads a custom .tri file into vertices and face indices."""
    with open(in_path, "r") as f:
        vert_count = int(f.readline().split()[0])

    verts = np.loadtxt(
        in_path, skiprows=1, usecols=(1, 2, 3), dtype=np.float64, max_rows=vert_count
    )
    ids = np.loadtxt(
        in_path, skiprows=vert_count + 2, usecols=(1, 2, 3), dtype=np.int64
    )
    return verts, ids


def write_tri(out_path: str, verts: np.ndarray, faces: np.ndarray):
    """Saves vertices and faces back to the custom .tri format."""
    with open(out_path, "w") as f:
        f.write(f"{len(verts)}\n")
        for i, v in enumerate(verts):
            f.write(f"{i+1} {v[0]:.6f} {v[1]:.6f} {v[2]:.6f}\n")

        f.write(f"{len(faces)}\n")
        for i, face in enumerate(faces):
            f.write(f"{i+1} {face[0]+1} {face[1]+1} {face[2]+1}\n")


def read_node(in_path: str) -> np.ndarray:
    """Reads TetGen .node and .ele files."""
    pts_data = np.loadtxt(in_path, skiprows=1)
    pnt = pts_data[:, 1:4] # Skip ID, take x, y, z
    return pnt


def read_ele(in_path: str) -> np.ndarray:
    ele_data = np.loadtxt(in_path, skiprows=1, dtype=int)
    ele = ele_data[:, 1:5] - 1 # Skip ID, take 4 nodes, convert to 0-based indexing
    return ele


def read_pts(in_path: str) -> np.ndarray:
    return np.loadtxt(
        in_path, skiprows=1, usecols=(0, 1, 2), dtype=np.float64
    ) - 1 # convert to 0-indexing from 1-indexing

def read_tet(in_path: str) -> np.ndarray:
    return np.loadtxt(
        in_path, skiprows=1, usecols=(0, 1, 2, 3), dtype=np.int64
    ) - 1 # convert to 0-indexing from 1-indexing

def write_el(out_path: str, mapping_indices: np.ndarray):
    out = np.stack([
        np.arange(len(mapping_indices)), mapping_indices
    ], axis=1)
    np.savetxt(out_path, out, fmt='%d', delimiter=' ', header=str(len(mapping_indices)), comments='')

def write_tot(out_path: str, tet_of_tri: np.ndarray, local_nodes: np.ndarray):
    """Writes the .tot format"""
    r_stack = np.stack([
        np.arange(len(tet_of_tri)), tet_of_tri + 1
    ], axis=-1) # save as 1-indexed
    out = np.concat([r_stack, local_nodes + 1], axis=-1)
    np.savetxt(out_path, out, fmt='%d', delimiter=' ', header=str(len(tet_of_tri)), comments='')
