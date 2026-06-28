from scibs.scripts.elecpatch import elecpatch
from time import time

import scipy.io as sio

from scibs.scripts.elecpotsurf import elecpotsurf
from scibs.scripts.tet2tri import tet2tri
from scibs.scripts.tetcor import tetcor
from scibs.scripts.tetwarp import tetwarp
from scibs.scripts.trielec import trielec
from scibs.utilities.file import (
    read_electrode_descr,
    read_mat,
    read_pts,
    read_tet,
    write_mat,
)
from scibs.utilities.graphics import visualize_mesh, visualize_tet_volume, visualize_potentials


def pipeline(file_prefix: str, struct_name: str):
    start = time()
    
    print("Loading mesh")
    tet_pts, tet_ids = read_mat(f"{file_prefix}.mat", struct_name)

    print("Running tet2tri")
    tri_ids, tri_pts_old, el_map, tot_map = tet2tri(tet_pts, tet_ids)
    assert el_map is not None and tot_map is not None

    print("Running trielec")
    electrode_pts, electrode_radii = read_electrode_descr(f"{file_prefix}_elec_descr.txt")
    tri_pts, tri_ids, face_labels, warp_lines = trielec(
        tri_pts_old, tri_ids, electrode_pts, electrode_radii
    )

    print("Running tetwarp")
    tet_pts, tet_ids, history, new_el_map = tetwarp(tet_pts, el_map, tot_map, tet_ids, warp_lines)

    # print("Running elecpotsurf")
    # electrode_pots = elecpotsurf(tri_ids, face_labels, len(tet_pts), new_el_map)

    print("Running tetcor (Base Mesh)")
    tet_ids, n_corrected = tetcor(tet_pts, tet_ids)

    
    print("Running elecpatch (Extrusion)")
    # Make sure to pass new_el_map here!
    tet_pts, tet_ids = elecpatch(
        tri_pts, tri_ids, face_labels, tet_pts, tet_ids, new_el_map, height=2.0
    )

    print("Running tetcor (Extruded Geometry)")
    tet_ids, n_corrected = tetcor(tet_pts, tet_ids)

    visualize_tet_volume(tet_pts, tet_ids)

    print("Saving mesh")
    write_mat(file_prefix, tet_pts, tet_ids, struct_name)

    print(f"Mesh saved to {file_prefix}_E.mat")
    end = time()

    print(f"Time Elapsed: {end - start:.4f}s")  # Fixed time calculation