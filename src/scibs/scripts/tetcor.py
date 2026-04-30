from typing import Optional
import os
from argparse import ArgumentParser

import numpy as np
import scipy.io as sio

from scibs.utilities.mat import is_model


def calculate_signed_volumes(pts, ids):
    # check if ids are 1-indexed
    if ids.min() == 1 and ids.max() == pts.shape[0]:
        ids -= 1 # shift to be 0-indexed

    tet_verts = pts[ids]
    e0 = tet_verts[:, 1] - tet_verts[:, 0]
    e1 = tet_verts[:, 2] - tet_verts[:, 0]
    e2 = tet_verts[:, 3] - tet_verts[:, 0]
    cross = np.cross(e0, e1)
    jacobian_det = np.sum(e2 * cross, axis=1)
    return jacobian_det / 6.0 # volume is 1/6 of jacobian

def unpack_tets(mat_contents, struct_name) -> tuple[np.ndarray, np.ndarray]:
    pts = mat_contents[struct_name]['node'][0, 0]
    ids = mat_contents[struct_name]['cell'][0, 0]
    return pts, ids





def save_tets(mat_contents, output_matlab_file_name: str, new_pts: np.ndarray, new_ids: np.ndarray, struct_name: str):
    mat_contents[struct_name]['node'][0, 0] = new_pts
    mat_contents[struct_name]['cell'][0, 0] = new_ids

    # remove metadata to avoid warnings
    try:
        for key in ['__header__', '__version__', '__globals__']:
            mat_contents.pop(key, None)
    except:
        pass

    sio.savemat(output_matlab_file_name, mat_contents)

def tetcor(pts: np.ndarray, ids: np.ndarray) -> tuple[np.ndarray, int]:
    volumes = calculate_signed_volumes(pts, ids)
    negative_ids = np.where(volumes < 0)[0]
    n_corrected = len(negative_ids)
    if n_corrected > 0:
        # swap indices to invert volume of tetrahedron
        temp = ids[negative_ids, 2].copy()
        ids[negative_ids, 2] = ids[negative_ids, 3]
        ids[negative_ids, 3] = temp

    return ids, n_corrected


def main(file_name, output, struct_name: Optional[str] = None):
    mat_contents = sio.loadmat(file_name)
    print(mat_contents)

    # load from matlab file
    contents = {}
    for _struct_name, struct_content in mat_contents.items():
        if not is_model(struct_content): continue
        if struct_name and struct_name != _struct_name: continue
        _name = struct_name if struct_name else _struct_name
        contents[_name] = unpack_tets(mat_contents, _name)

    # run tetcor
    out = {}
    for name, (pts, ids) in contents.items():
        pts_copy = pts.copy()
        new_ids, _n_corrected = tetcor(pts_copy, ids)
        out[name] = (pts_copy, new_ids)

    # write output to matlab file
    for _struct_name, (new_pts, new_ids) in out.items():
        save_tets(mat_contents, output, new_pts, new_ids, _struct_name)

    print(f"Output saved at {output}")