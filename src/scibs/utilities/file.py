from numpy.typing import ArrayLike
from typing import Callable, Optional, Sequence, Iterable
import struct

import numpy as np
import scipy.io as sio

from .mat import is_model


def write_mat(file_prefix: str, pts: np.ndarray, ids: np.ndarray, struct_name: str):
    mat_contents = sio.loadmat(f"{file_prefix}.mat")
    
    if not is_model(mat_contents[struct_name]):
        raise ValueError(f"Struct {struct_name} is not a valid model")
    
    mat_contents[struct_name]['node'][0, 0] = pts
    mat_contents[struct_name]['cell'][0, 0] = ids + 1

    # remove metadata to avoid warnings
    try:
        for key in ['__header__', '__version__', '__globals__']:
            mat_contents.pop(key, None)
    except:
        pass

    sio.savemat(f"{file_prefix}_E.mat", mat_contents)


def read_mat(file_name: str, struct_name: str) -> tuple[np.ndarray, np.ndarray]:
    mat_contents = sio.loadmat(file_name)
    
    if not is_model(mat_contents[struct_name]):
        raise ValueError(f"Struct {struct_name} is not a valid model")
    
    pts = mat_contents[struct_name]['node'][0, 0]
    ids = mat_contents[struct_name]['cell'][0, 0] - 1
    
    return pts, ids


def read_tri(in_path: str) -> tuple[np.ndarray, np.ndarray]:
    """Loads a custom .tri file into vertices and face indices."""
    with open(in_path, "r") as f:
        vert_count = int(f.readline().split()[0])

    verts = np.loadtxt(
        in_path, skiprows=1, usecols=(1, 2, 3), dtype=np.float64, max_rows=vert_count
    )
    ids = np.loadtxt(
        in_path, skiprows=vert_count + 2, usecols=(1, 2, 3), dtype=np.int64
    ) - 1
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

def write_tot(out_path: str, data: np.ndarray):
    """Writes the .tot format"""
    out = np.concat([
        np.arange(len(data))[:, None], data + 1
    ], axis=-1) # save as 1-indexed
    np.savetxt(out_path, out, fmt='%d', delimiter=' ', header=str(len(data)), comments='')

def read_tot(in_path: str):
    return np.loadtxt(in_path, skiprows=1, usecols=[1, 2, 3, 4], dtype=np.int32, comments=None) - 1

def read_el(in_path: str):
    return np.loadtxt(in_path, skiprows=1, usecols=[1], dtype=np.int32, comments=None)

def read_pot(in_path: str, dtype=None):
    with open(in_path, 'rb') as f:
        header = f.read(8)
        assert header.startswith(b';;mbfmat'), "invalid file format"
        f.seek(8, 1) # skip a double
        nr = struct.unpack('<i', f.read(4))[0]
        nc = struct.unpack('<i', f.read(4))[0]
        mat = np.fromfile(f, "<d")
        if nc > 1:
            mat = mat.reshape(nr, nc)
            
        if dtype:
            mat = mat.astype(dtype)
        
        return mat

def write_pot(out_path: str, data: np.ndarray):
    """
    Writes a 1D numpy array to a .pot file matching the format of read_pot.
    """
    # ensure flattness
    data = np.asarray(data).astype(np.float64).flatten()
    
    nr = len(data)
    nc = 1  # strictly saving a 1D array w/ 1 column
    
    with open(out_path, 'wb') as f:
        # write header
        f.write(b';;mbfmat') # 8 bytes
        
        # write empty placeholder padding
        f.write(struct.pack('<d', 0.0)) # 8 bytes
        
        # write number of rows and columns
        f.write(struct.pack('<i', nr)) # little-endian, int, 32-bit
        f.write(struct.pack('<i', nc)) # little-endian, int, 32-bit
        
        # write the matrix data
        data.astype('<d').tofile(f) # little-endian, float, 64-bit
        
        
def write_tetwarp(out_path, history):
    with open(out_path, 'w') as f:
        f.write(f"{len(history)} 5\n")
        for h in history:
            f.write(f"{h[0] + 1:>8} {h[1]:>8} {h[2] + 1:>8} {h[3] + 1:>8} {h[4] if len(h) > 4 else 0:>8}\n")
            
        f.close()


def read_electrode_descr(in_path: str):
    electrode = np.loadtxt(
        in_path,
        skiprows=1,
        usecols=(1, 2, 3, 4),
        ndmin=2,
        comments=["height", "n layers"],
    )
    electrode_pts, electrode_radii = np.split(electrode, [3], axis=1)
    electrode_radii = electrode_radii.flatten()
    return electrode_pts, electrode_radii