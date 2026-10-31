from typing import Optional
import os
from argparse import ArgumentParser

import numpy as np
import scipy.io as sio

from scibs.utilities.file import output_dir
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

    # Force direct extraction dump to the shared project output folder during function execution
    extraction_path = str(output_dir() / "extraction_tetcor.txt")
    print(f"Dumping extraction data directly to: {extraction_path}")
    with open(extraction_path, "w") as f:
        f.write("=== CORRECTED TET IDS ===\n")
        np.savetxt(f, ids, fmt="%d")

        f.write("\n=== N_CORRECTED ===\n")
        f.write(f"{n_corrected}\n")

        f.write("\n=== CORRECTED TET INDICES ===\n")
        np.savetxt(f, negative_ids.reshape(-1, 1), fmt="%d")

    return ids, n_corrected


def drop_degenerate_tets(pts: np.ndarray, ids: np.ndarray, rel_tol: float = 1e-10):
    """Removes tetrahedra with effectively zero volume.

    `tetcor` fixes element *orientation*, but a flat (coplanar-node) element has
    no orientation to fix: its Jacobian is singular, so the FEM element stiffness
    matrix -- which needs 1/(6V) -- is undefined, and assembling it puts NaN/Inf
    into the global system or makes it singular. Such an element occupies no
    volume and so carries no conductivity either, meaning deleting it leaves both
    the domain and the physics unchanged; it is purely an artifact.

    These come out of `tetwarp`, not the input mesh: replaying a surface edge
    split onto the volume can insert the new node almost exactly coplanar with an
    adjacent interior tetrahedron, and the resulting split is flat. Cleaning the
    surface first (`--clean-slivers`) removes nearly all of them, but it cannot
    see this condition, which depends on the interior tet's shape rather than the
    surface triangle's.

    `rel_tol` is relative to the median element volume, so it is scale-free.
    Raises if removal would orphan a node (one left in no element at all), since
    that is a free degree of freedom and would make the stiffness matrix singular
    -- a louder failure than the one being fixed, so it must never pass silently.
    """
    tet_verts = pts[ids]
    e0 = tet_verts[:, 1] - tet_verts[:, 0]
    e1 = tet_verts[:, 2] - tet_verts[:, 0]
    e2 = tet_verts[:, 3] - tet_verts[:, 0]
    volumes = np.abs(np.sum(e2 * np.cross(e0, e1), axis=1)) / 6.0

    positive = volumes[volumes > 0]
    scale = np.median(positive) if len(positive) else 1.0
    degenerate = volumes <= rel_tol * scale
    n_dropped = int(degenerate.sum())
    if n_dropped == 0:
        return ids, 0

    kept = ids[~degenerate]
    orphaned = np.setdiff1d(np.unique(ids[degenerate]), np.unique(kept))
    if len(orphaned) > 0:
        raise ValueError(
            f"Dropping {n_dropped} degenerate tetrahedra would orphan "
            f"{len(orphaned)} node(s) (indices {orphaned[:10].tolist()}), leaving free "
            f"degrees of freedom that make the stiffness matrix singular. The mesh "
            f"needs repair around these nodes rather than element removal."
        )
    return kept, n_dropped


def compact_nodes(pts: np.ndarray, ids: np.ndarray, potentials: Optional[np.ndarray] = None):
    """Drops nodes no element references, reindexing `ids` (and `potentials`).

    `cleanup_slivers` merges vertex pairs, and `drop_degenerate_tets` removes
    elements; either can leave a node behind that belongs to no element. Such a
    node is a free degree of freedom -- an all-zero row in the assembled
    stiffness matrix -- so a solver that derives its DOFs from the node array
    rather than from element connectivity gets a singular system.

    `potentials` is indexed by node, so it has to be permuted in lockstep.
    Letting the two fall out of alignment would silently relabel which nodes
    belong to which electrode, which is far worse than an unused node, so the
    caller must pass it whenever one exists.
    """
    used = np.unique(ids)
    if len(used) == len(pts):
        return pts, ids, potentials, 0

    remap = np.full(len(pts), -1, dtype=np.int64)
    remap[used] = np.arange(len(used))
    new_pts = pts[used]
    new_ids = remap[ids]
    new_pots = None if potentials is None else np.asarray(potentials)[used]
    return new_pts, new_ids, new_pots, int(len(pts) - len(used))


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