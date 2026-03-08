from pymatreader import read_mat

import struct
import numpy as np

def read_pot(path: str, dtype=None):
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
        
        
if __name__ == "__main__":
    read_pot("data/MNI152_E.pot")