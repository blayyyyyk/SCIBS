from scibs.scripts.tetwarp import tetwarp
from scibs.scripts.tet2tri import tet2tri
from scibs.scripts.trielec import trielec
from scibs.scripts.tetcor import tetcor
from scibs.utilities.file import read_electrode_descr, read_pts, read_tet, read_mat, write_mat
import scipy.io as sio

def pipeline(file_prefix: str, struct_name: str):
    # load from matlab file
    print("Loading mesh")
    tet_pts, tet_ids = read_mat(f"{file_prefix}.mat", struct_name)
    
    
    print("Running tet2tri")

    tri_ids, tri_pts, el_map, tot_map = tet2tri(tet_pts, tet_ids)
    assert el_map is not None and tot_map is not None

    print("Running trielec")
    electrode_pts, electrode_radii = read_electrode_descr(f"{file_prefix}_elec_descr.txt")
    tri_pts, tri_ids, face_labels, warp_lines = trielec(
        tri_pts, tri_ids, electrode_pts, electrode_radii
    )

    print("Running tetwarp")
    new_pts, new_tets, history = tetwarp(tet_pts, el_map, tot_map, tet_ids, warp_lines)

    print("Running tetcor")
    corrected_tets, n_corrected = tetcor(new_pts, new_tets)
    print(corrected_tets.shape, tet_ids.shape, n_corrected, new_pts.shape)

    print("Saving mesh")
    write_mat(file_prefix, new_pts, corrected_tets, struct_name)
    
    print(f"Mesh saved to {file_prefix}_E.mat")
    
