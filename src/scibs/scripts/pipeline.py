from pathlib import Path
from time import time
from typing import Optional
# Imports from individual script files
from scibs.scripts.elecpatch import elecpatch
from scibs.scripts.elecpotsurf import check_electrode_conflicts, elecpotsurf
from scibs.scripts.tet2tri import tet2tri
from scibs.scripts.tetcor import compact_nodes, drop_degenerate_tets, tetcor
from scibs.scripts.tetwarp import tetwarp
from scibs.scripts.trielec import trielec

# Utility imports
from scibs.utilities.file import (
    output_dir,
    read_electrode_descr,
    read_mat,
    write_mat,
)
from scibs.utilities.graphics import visualize_potentials, visualize_tet_volume


def pipeline(
    file_prefix: str, struct_name: str, clean_slivers: bool = False,
    min_electrode_gap: Optional[float] = None,
):
    start = time()

    # Route the final mesh/potentials and the visualization screenshots to the
    # project's own `out/` folder instead of alongside the (possibly
    # read-only/shared) source data, so results are easy to find regardless of
    # where the input lives.
    run_name = Path(file_prefix).name
    output_prefix = str(output_dir() / run_name)
    
    print("Loading mesh")
    tet_pts, tet_ids = read_mat(f"{file_prefix}.mat", struct_name)

    print("Running tet2tri")
    tri_ids, tri_pts_old, el_map, tot_map = tet2tri(tet_pts, tet_ids)
    assert el_map is not None and tot_map is not None

    print("Running trielec")
    electrode_pts, electrode_radii = read_electrode_descr(f"{file_prefix}_elec_descr.txt")
    tri_pts, tri_ids, face_labels, warp_lines = trielec(
        tri_pts_old, tri_ids, electrode_pts, electrode_radii,
        clean_slivers=clean_slivers, min_electrode_gap=min_electrode_gap,
    )

    print("Running tetwarp")
    tet_pts, tet_ids, history, new_el_map = tetwarp(tet_pts, el_map, tot_map, tet_ids, warp_lines)

    print("Running elecpotsurf")
    # Project the surface electrode labels onto the (warped) volumetric nodes.
    electrode_pots = elecpotsurf(tri_ids, face_labels, len(tet_pts), new_el_map)
    # trielec's own spacing check can pass (its flat surface has no conflict)
    # while tetwarp's remapping still lands two different electrodes' nodes
    # on a shared triangle -- check the actual mesh, not a proxy for it.
    check_electrode_conflicts(tet_ids, electrode_pots)

    print("Running tetcor (Base Mesh)")
    tet_ids, n_corrected = tetcor(tet_pts, tet_ids)

    print("Running elecpatch (Extrusion)")
    # Extrude the electrode patches and carry the surface potentials onto the
    # newly created pad nodes (make sure to pass new_el_map here!).
    tet_pts, tet_ids, electrode_pots = elecpatch(
        tri_pts, tri_ids, face_labels, tet_pts, tet_ids, new_el_map,
        potentials = electrode_pots, height=2.0
    )
    # Extrusion adds new geometry (pad side walls/tops), so re-check rather
    # than assume the pre-extrusion result above still holds.
    check_electrode_conflicts(tet_ids, electrode_pots)

    print("Running tetcor (Extruded Geometry)")
    tet_ids, n_corrected = tetcor(tet_pts, tet_ids)

    # Flat elements survive tetcor (there is no orientation to flip), and would
    # reach the solver as a singular element Jacobian, so clear them before the
    # mesh is written rather than leaving it to whatever consumes the .mat.
    tet_ids, n_degenerate = drop_degenerate_tets(tet_pts, tet_ids)
    print(f"Dropped {n_degenerate} degenerate (zero-volume) tetrahedra")

    tet_pts, tet_ids, electrode_pots, n_orphaned = compact_nodes(
        tet_pts, tet_ids, electrode_pots
    )
    print(f"Dropped {n_orphaned} orphaned node(s) referenced by no element")

    visualize_tet_volume(tet_pts, tet_ids, name=f"{run_name}_tet_volume")
    visualize_potentials(tet_pts, tet_ids, electrode_pots, name=f"{run_name}_potentials")

    print("Saving mesh")
    write_mat(
        file_prefix, tet_pts, tet_ids, struct_name,
        potentials=electrode_pots, output_prefix=output_prefix,
    )

    print(f"Mesh saved to {output_prefix}_E.mat")
    end = time()

    print(f"Time Elapsed: {end - start:.4f}s")