import numpy as np
from typing import Union
import os, struct

FileDescriptorOrPath = Union[int, str, bytes, os.PathLike[str], os.PathLike[bytes]]

def read_pot(path: FileDescriptorOrPath, dtype=None):
    with open(path, 'rb') as f:
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

def write_pot(path: FileDescriptorOrPath, data: np.ndarray):
    """
    Writes a 1D numpy array to a .pot file matching the format of read_pot.
    """
    # ensure flattness
    data = np.asarray(data).astype(np.float64).flatten()
    
    nr = len(data)
    nc = 1  # strictly saving a 1D array w/ 1 column
    
    with open(path, 'wb') as f:
        # write header
        f.write(b';;mbfmat') # 8 bytes
        
        # write empty placeholder padding
        f.write(struct.pack('<d', 0.0)) # 8 bytes
        
        # write number of rows and columns
        f.write(struct.pack('<i', nr)) # little-endian, int, 32-bit
        f.write(struct.pack('<i', nc)) # little-endian, int, 32-bit
        
        # write the matrix data
        data.astype('<d').tofile(f) # little-endian, float, 64-bit  

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
