from argparse import ArgumentParser
from functools import partial
from typing import Optional

import numpy as np

from src.scibs.utilities.file import read_pts, read_tet, write_el, write_tot, write_tri


def tet2tri(tet_pts: np.ndarray, tet_ids: np.ndarray, surface_only: bool = True):
    # extract the 4 faces for each tetrahedron.
    faces = np.vstack([
        tet_ids[:, [0, 1, 2]],
        tet_ids[:, [1, 3, 2]],
        tet_ids[:, [2, 3, 0]],
        tet_ids[:, [3, 1, 0]]
    ])

    og_tet_ids = np.arange(len(tet_ids))

    # maintain tracking of original tet index and local node orientation
    tet_of_tri = np.tile(og_tet_ids, 4)
    local_nodes = np.vstack([
        np.tile([0, 1, 2], (len(tet_ids), 1)),
        np.tile([1, 3, 2], (len(tet_ids), 1)),
        np.tile([2, 3, 0], (len(tet_ids), 1)),
        np.tile([3, 1, 0], (len(tet_ids), 1))
    ])

    el = None
    tot = None
    if surface_only:
        sorted_faces = np.sort(faces, axis=1)
        _, inverse_idx, counts = np.unique(sorted_faces, axis=0, return_inverse=True, return_counts=True)
        face_counts = counts[inverse_idx]

        surface_mask = face_counts == 1

        if (face_counts > 2).any():
            print("Warning: These tetrahedrons touch more than one other tetrahedron (mesh anomaly detected).")

        tri_ids = faces[surface_mask]
        final_tet_of_tri = tet_of_tri[surface_mask]
        final_local_nodes = local_nodes[surface_mask]

        # Compact unused points and create map
        used_pnt_mask = np.zeros(len(tet_pts), dtype=bool)
        used_pnt_mask[tri_ids.flatten()] = True

        old_indices = np.where(used_pnt_mask)[0]
        old_to_new = np.full(len(tet_pts), -1, dtype=int)
        old_to_new[old_indices] = np.arange(len(old_indices))

        # Update mappings
        tet_pts = tet_pts[old_indices]
        tri_ids = old_to_new[tri_ids]

        el = old_indices
        tot = np.concat([final_tet_of_tri[:, None], final_local_nodes], axis=-1)
    else:
        tri_ids = faces

    return tri_ids, tet_pts, el, tot
    

def main(pts_name: str, tet_name: Optional[str] = None, tri_name: Optional[str] = None, el_name: Optional[str] = None, tot_name: Optional[str] = None, surface_only: bool = True, verbose: bool = False):
    opt_kwargs = [tet_name, tri_name, el_name, tot_name]
    if any(opt_kwargs[:2]) and not all(opt_kwargs[:2]):
        raise ValueError("If one of tetName or triName is provided, both must be provided.")
    elif not any(opt_kwargs[:2]):
        tet_name = pts_name
        tri_name = pts_name

    if surface_only and not any(opt_kwargs):
        el_name = pts_name
        tot_name = pts_name

    tet_pts = read_pts(f"{pts_name}.pts")
    tet_ids = read_tet(f"{tet_name}.tet")

    tri_ids, tet_pts, el, tot = tet2tri(tet_pts, tet_ids, surface_only=surface_only)
    if surface_only and el_name and el is not None:
        print(f"Writing mapping to {el_name}...")
        write_el(f"{el_name}.el", el)

    if surface_only and tot_name and tot is not None:
        print(f"Writing cell topology to {tot_name}...")
        write_tot(f"{tot_name}.tot", tot)

    print(f"Writing {len(tri_ids)} triangles to {tri_name}.tri ...")
    write_tri(f"{tri_name}.tri", tet_pts, tri_ids)

    print("Done.")