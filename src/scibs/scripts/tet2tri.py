from typing import Optional
from src.scibs.utilities.file import read_pts, read_tet, write_el, write_tot, write_tri
from argparse import ArgumentParser
import numpy as np
from functools import partial


# Process Mode Parsing #
parser = ArgumentParser(usage="tet2tri [-options] ptsName tetName triName [elName [totName]]\n\ttet2tri [-options] name", add_help=False)
parser.add_argument('-s', help="output surface triangles only", action="store_true")
parser.add_argument("ptsName", type=str)
parser.add_argument("tetName", nargs="?", type=str)
parser.add_argument("triName", nargs="?", type=str)
parser.add_argument("elName", nargs="?", type=str)
parser.add_argument("totName", nargs="?", type=str)


def tet2tri(pts_file, tet_file, tri_file, el_file: Optional[str] = None, tot_file: Optional[str] = None, surface_only=False, verbose=False):
    tet_pts = read_pts(pts_file)
    tet_ids = read_tet(tet_file)

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

        if el_file:
            print(f"Writing mapping to {el_file}...")
            write_el(el_file, old_indices)

        if tot_file:
            print(f"Writing cell topology to {tot_file}...")
            write_tot(tot_file, final_tet_of_tri, final_local_nodes)


    else:
        tri_ids = faces
        
    if verbose:
        print(f"Writing {len(tri_ids)} triangles to {tri_file}...")
    
    write_tri(tri_file, tet_pts, tri_ids)

    if verbose:
        print(f"Done.")


def main(args):
    pos_args = [getattr(args, arg) for arg in ["tetName", "triName", "elName", "totName"] if getattr(args, arg)]

    if len(pos_args) not in [0, 2, 3, 4]:
        parser.error("missing positional arguments")
        
    if not args.tetName:
        args.tetName = args.ptsName
    
    if not args.triName:
        args.triName = args.ptsName
        
    if args.s and len(pos_args) == 0:
        args.elName = args.ptsName
        args.totName = args.ptsName
    
    pts_file = f"{args.ptsName}.pts"
    tet_file = f"{args.tetName}.tet"
    tri_file = f"{args.triName}.tri"
    el_file = f"{args.elName}.el"
    tot_file = f"{args.totName}.tot"

    tet2tri(pts_file, tet_file, tri_file, el_file, tot_file, surface_only=args.s)


parser.set_defaults(func=main)
if __name__ == "__main__":
    # parse arguments
    import os
    prog = os.path.basename(__file__)
    parser = ArgumentParser(prog=prog, parents=[])
    args = parser.parse_args()
    if hasattr(args, "func"):
        args.func(args)
    else:
        parser.print_help() # print help if no/invalid mode specified
    
    
    
