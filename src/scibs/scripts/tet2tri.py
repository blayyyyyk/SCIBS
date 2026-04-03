from argparse import ArgumentParser
from functools import partial
from typing import Optional

import numpy as np

from src.scibs.utilities.file import read_pts, read_tet, write_el, write_tot, write_tri


def tet2tri(pts_name: str, tet_name: Optional[str] = None, tri_name: Optional[str] = None, el_name: Optional[str] = None, tot_name: Optional[str] = None, surface_only: bool = False, verbose: bool = False):
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

        if el_name:
            print(f"Writing mapping to {el_name}...")
            write_el(f"{el_name}.el", old_indices)

        if tot_name:
            print(f"Writing cell topology to {tot_name}...")
            write_tot(f"{tot_name}.tot", final_tet_of_tri, final_local_nodes)


    else:
        tri_ids = faces

    if verbose:
        print(f"Writing {len(tri_ids)} triangles to {tri_name}.tri ...")

    write_tri(f"{tri_name}.tri", tet_pts, tri_ids)

    if verbose:
        print(f"Done.")


# parser.set_defaults(func=main)
# if __name__ == "__main__":
#     # parse arguments
#     import os
#     prog = os.path.basename(__file__)
#     parser = ArgumentParser(prog=prog, parents=[])
#     args = parser.parse_args()
#     if hasattr(args, "func"):
#         args.func(args)
#     else:
#         parser.print_help() # print help if no/invalid mode specified
