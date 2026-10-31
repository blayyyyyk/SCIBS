from itertools import batched
from collections import defaultdict
from typing import Literal
from argparse import ArgumentParser
import numpy as np
from scibs.utilities.file import output_dir, read_pts, read_tet, read_el, read_tot, write_tetwarp


def parse(line: str):
    if not line.strip():
        return None
        
    if line.startswith("s"):
        parts = line.split()
        face_count = int(parts[5])
        
        face_orig = int(parts[1])
        n1 = int(parts[2])
        n2 = int(parts[3])
        n3 = int(parts[4])
        
        new_faces = []
        for f, sn1, sn2, sn3 in batched(parts[6:], 4):
            new_faces.append([int(f), int(sn1), int(sn2), int(sn3)])
            
        return "split", face_orig, n1, n2, n3, new_faces
        
    elif line.startswith("m"):
        parts = line.split()
        return "move", int(parts[1]), float(parts[2]), float(parts[3]), float(parts[4])
        
    elif line.startswith("a"):
        parts = line.split()
        return "add", int(parts[1]), float(parts[2]), float(parts[3]), float(parts[4])

    elif line.startswith("c"):
        parts = line.split()
        return "collapse", int(parts[1]), int(parts[2])

    return None


def tetwarp(pts, el_map, tot_map, tets, warp_lines: list[str]):
    if isinstance(pts, np.ndarray): pts = pts.tolist()
    if isinstance(el_map, np.ndarray): el_map = el_map.tolist()
    if isinstance(tets, np.ndarray): tets = tets.tolist()

    if isinstance(tot_map, np.ndarray):
        tot_dict = {i: [int(row[0]), int(row[1]), int(row[2]), int(row[3])] for i, row in enumerate(tot_map)}
    else:
        tot_dict = tot_map

    surface_tets_set = {v[0] for v in tot_dict.values()}
    tet_warp_history = []

    node_to_tets = defaultdict(set)
    for itet, tet in enumerate(tets):
        for node in tet:
            node_to_tets[node].add(itet)

    def split(face_orig, n1, n2, n3, new_faces):
        nsplit = len(new_faces)
        tri_orig = [n1, n2, n3]
        
        cell_orig, l0, l1, l2 = tot_dict[face_orig]
        local_nodes = [l0, l1, l2]
        tet_orig_nodes = tets[cell_orig].copy()

        tet_warp_entry = [cell_orig, nsplit, cell_orig]
        on_edge = [False, False, False]
        new_point_tvf = -1

        # part one: split the surface tetrahedron
        for k, (f_new, sn1, sn2, sn3) in enumerate(new_faces):
            tri_new = [sn1, sn2, sn3]

            if k == 0:
                jtri = face_orig
                jtet = cell_orig
            else:
                jtri = f_new
                jtet = len(tets)
                new_tet = tet_orig_nodes.copy()
                tets.append(new_tet)
                tot_dict[jtri] = [jtet, l0, l1, l2]
                surface_tets_set.add(jtet)
                tet_warp_entry.append(jtet)
                
                # register the newly appended surface tet in the adjacency map
                for node in new_tet:
                    node_to_tets[node].add(jtet)

            for l in range(3):
                if tri_new[l] != tri_orig[l]:
                    on_edge[l] = True
                    new_point_tsm = tri_new[l]
                    new_point_tvf = el_map[new_point_tsm]

                    try:
                        old_node = tets[jtet][local_nodes[l]]
                    except IndexError:
                        raise IndexError(f"jtet={jtet}, l={l}, local_nodes={local_nodes}, length={len(tets[jtet])}")
                        
                    if old_node != new_point_tvf:
                        tets[jtet][local_nodes[l]] = new_point_tvf
                        
                        # dynamically update the adjacency map logic
                        if jtet in node_to_tets[old_node]:
                            node_to_tets[old_node].remove(jtet)
                        node_to_tets[new_point_tvf].add(jtet)

        tet_warp_history.append(tet_warp_entry)

        # part two: bisecting interior tetrahedra sharing the cut edge
        def bisect_interior_tets(e0, e1, new_pt_tvf):
            affected_tets = list(node_to_tets[e0].intersection(node_to_tets[e1]))
            for itet in affected_tets:
                if itet in surface_tets_set:
                    continue
                    
                tet_nodes = tets[itet]
                new_tet = tet_nodes.copy()
                ntet = len(tets)
                tets.append(new_tet)

                idx_e0 = tet_nodes.index(e0)
                idx_e1 = tet_nodes.index(e1)

                # split the tet geometrically
                tets[itet][idx_e0] = new_pt_tvf
                tets[ntet][idx_e1] = new_pt_tvf

                # maintain adjacency map state for subsequent splits
                node_to_tets[e0].remove(itet)
                node_to_tets[new_pt_tvf].add(itet)
                
                for node in new_tet:
                    node_to_tets[node].add(ntet)

                tet_warp_history.append([itet, 2, itet, ntet])

        if nsplit == 2:
            edge_nodes = [el_map[tri_orig[l]] for l in range(3) if on_edge[l]]
            if len(edge_nodes) == 2:
                bisect_interior_tets(edge_nodes[0], edge_nodes[1], new_point_tvf)
                
        elif nsplit == 3:
            # Topologically determine which 2 original edges were bisected 
            orig_set = set(tri_orig)
            new_pts_tsm = list(set(sn for f in new_faces for sn in f[1:]) - orig_set)
            
            orig_edges = [
                tuple(sorted((tri_orig[0], tri_orig[1]))),
                tuple(sorted((tri_orig[1], tri_orig[2]))),
                tuple(sorted((tri_orig[2], tri_orig[0])))
            ]
            
            new_sub_edges = set()
            for face in new_faces:
                sn1, sn2, sn3 = face[1:]
                new_sub_edges.add(tuple(sorted((sn1, sn2))))
                new_sub_edges.add(tuple(sorted((sn2, sn3))))
                new_sub_edges.add(tuple(sorted((sn3, sn1))))
                
            # If an original edge does not exist in the new sub-triangles, it was bisected
            bisected_edges = [e for e in orig_edges if e not in new_sub_edges]
            
            # Map the new points to the edges they bisected and apply the split
            for edge in bisected_edges:
                vA, vB = edge
                for p in new_pts_tsm:
                    if tuple(sorted((vA, p))) in new_sub_edges and tuple(sorted((vB, p))) in new_sub_edges:
                        bisect_interior_tets(el_map[vA], el_map[vB], el_map[p])
                        break

    def move(node_id, x, y, z):
        tvf_idx = el_map[node_id]
        pts[tvf_idx] = [x, y, z]

    def add(node_id, x, y, z):
        if node_id != len(el_map):
            raise ValueError(f"Warp problem: Expected new node index {len(el_map)}, got {node_id}")
        pts.append([x, y, z])
        el_map.append(len(pts) - 1)

    # Volumetric-node aliasing for `collapse`. cleanup_slivers() merges surface
    # vertices in pairs (never touching electrode-boundary vertices -- see its
    # docstring), and each merge is replayed here on the corresponding
    # volumetric nodes so the tet mesh stays consistent with the surface mesh.
    # The merged-away node is left in `pts` (unreferenced, harmless) rather
    # than compacted, so every other node index -- and therefore `el_map`
    # itself -- stays stable.
    vol_alias: dict[int, int] = {}

    def resolve(tvf_idx):
        while tvf_idx in vol_alias:
            tvf_idx = vol_alias[tvf_idx]
        return tvf_idx

    def collapse(keep_node, merge_node):
        tvf_keep = resolve(el_map[keep_node])
        tvf_merge = resolve(el_map[merge_node])
        if tvf_keep == tvf_merge:
            return

        for itet in list(node_to_tets[tvf_merge]):
            tet = tets[itet]
            for k in range(4):
                if tet[k] == tvf_merge:
                    tet[k] = tvf_keep
            node_to_tets[tvf_keep].add(itet)
        node_to_tets[tvf_merge] = set()

        vol_alias[tvf_merge] = tvf_keep

    count = 0
    for line in warp_lines:
        parsed = parse(line.strip())
        if not parsed:
            continue

        func_name, *args = parsed

        func = {"split": split, "move": move, "add": add, "collapse": collapse}.get(func_name)
        if func is None:
            raise ValueError(f"Unknown function: {func_name}")

        func(*args)
        count += 1

    if vol_alias:
        # A collapse can leave a tet with two corners pointing at the same
        # (merged) node -- zero volume, and meaningless for the FEM solve.
        tets = [t for t in tets if len(set(t)) == 4]

    final_pts = np.array(pts)
    final_tets = np.array(tets)
    final_el_map = np.array(el_map)

    # Force direct extraction dump to the shared project output folder during function execution
    extraction_path = str(output_dir() / "extraction_tetwarp.txt")
    print(f"Dumping extraction data directly to: {extraction_path}")
    with open(extraction_path, "w") as f:
        f.write("=== FINAL POINTS ===\n")
        np.savetxt(f, final_pts, fmt="%.15g")

        f.write("\n=== FINAL TETS ===\n")
        np.savetxt(f, final_tets, fmt="%d")

        f.write("\n=== EL_MAP ===\n")
        np.savetxt(f, final_el_map.reshape(-1, 1), fmt="%d")

        f.write("\n=== TET_WARP_HISTORY ===\n")
        for entry in tet_warp_history:
            f.write(" ".join(str(v) for v in entry) + "\n")

    return final_pts, final_tets, tet_warp_history, final_el_map


def main(file_prefix: str):
    with open(f"{file_prefix}_E.warp", 'r') as file:
        warp_lines = [line.strip() for line in file]

    pts = read_pts(f"{file_prefix}.pts")
    tets = read_tet(f"{file_prefix}.tet")
    el_map = read_el(f"{file_prefix}.el") 
    tot_map = read_tot(f"{file_prefix}.tot")
    pts, tets, history, el_map = tetwarp(
        pts,
        el_map,
        tot_map,
        tets,
        warp_lines
    )
    write_tetwarp(f"{file_prefix}_test_out.tetwarp", history)
    

if __name__ == "__main__":
    main("out/group/MNI152")