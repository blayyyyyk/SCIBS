import argparse
from typing import Optional

import numpy as np

from scibs.utilities.file import read_tri, write_pot, write_tri
from scibs.utilities.graphics import visualize_mesh


def trielec(verts: np.ndarray, faces: np.ndarray, electrode_pts: np.ndarray, electrode_radii: np.ndarray):
    """
    Imprints electrode boundaries onto a surface mesh using a Vectorized Deterministic Cascade.
    Generates strict N=2 bisections for a .warp file without requiring a dynamic global adjacency graph.
    """
    warp_lines = []
    verts_list = verts.tolist()
    faces_list = faces.tolist()

    # --- PHASE 1: Vectorized Global Edge Hash ---
    edges = np.vstack((faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]))
    edges = np.sort(edges, axis=1)
    unique_edges = np.unique(edges, axis=0)

    edge_to_new_v = {}
    moved_nodes = set()

    for el_idx, (center, radius) in enumerate(zip(electrode_pts, electrode_radii)):
        dists = np.linalg.norm(verts - center, axis=1) - radius

        d1 = dists[unique_edges[:, 0]]
        d2 = dists[unique_edges[:, 1]]
        
        # Only evaluate edges that physically cross the boundary threshold
        crossings = (d1 < 0) != (d2 < 0)
        
        crossing_edges = unique_edges[crossings]
        crossing_d1 = d1[crossings]
        crossing_d2 = d2[crossings]

        # SLIVER PREVENTION (Epsilon Snapping)
        # Matches the MINLM2 = 0.15 threshold from the original C++ code.
        # If the cut is within 15% of an existing vertex, snap the vertex instead of splitting.
        eps = 0.15 

        for i, (vA, vB) in enumerate(crossing_edges):
            if (vA, vB) in edge_to_new_v:
                continue 

            t = crossing_d1[i] / (crossing_d1[i] - crossing_d2[i])
            p_new_calc = verts[vA] + t * (verts[vB] - verts[vA])

            if t < eps:
                if vA not in moved_nodes:
                    verts_list[vA] = p_new_calc.tolist()
                    warp_lines.append(f"m {vA} {p_new_calc[0]:.15g} {p_new_calc[1]:.15g} {p_new_calc[2]:.15g}")
                    moved_nodes.add(vA)
                # By skipping the split, the edge naturally terminates at the snapped boundary
            elif t > 1.0 - eps:
                if vB not in moved_nodes:
                    verts_list[vB] = p_new_calc.tolist()
                    warp_lines.append(f"m {vB} {p_new_calc[0]:.15g} {p_new_calc[1]:.15g} {p_new_calc[2]:.15g}")
                    moved_nodes.add(vB)
                # By skipping the split, the edge naturally terminates at the snapped boundary
            else:
                new_idx = len(verts_list)
                verts_list.append(p_new_calc.tolist())
                edge_to_new_v[(vA, vB)] = new_idx
                warp_lines.append(f"a {new_idx} {p_new_calc[0]:.15g} {p_new_calc[1]:.15g} {p_new_calc[2]:.15g}")

    # --- PHASE 2: Isolated Deterministic Face Bisection ---
    num_original_faces = len(faces)
    
    for i in range(num_original_faces):
        face = faces[i]
        v0, v1, v2 = face
        edges_of_face = [
            tuple(sorted((v0, v1))),
            tuple(sorted((v1, v2))),
            tuple(sorted((v2, v0)))
        ]

        splits_needed = [(edge, edge_to_new_v[edge]) for edge in edges_of_face if edge in edge_to_new_v]

        if not splits_needed:
            continue

        face_queue = [(i, face.tolist())]

        for (e_vA, e_vB), P_new in splits_needed:
            for q_idx, (curr_f_idx, nodes) in enumerate(face_queue):
                if e_vA in nodes and e_vB in nodes:
                    n0, n1, n2 = nodes

                    if (n0 == e_vA and n1 == e_vB) or (n0 == e_vB and n1 == e_vA):
                        t1 = [n0, P_new, n2]
                        t2 = [P_new, n1, n2]
                    elif (n1 == e_vA and n2 == e_vB) or (n1 == e_vB and n2 == e_vA):
                        t1 = [n0, n1, P_new]
                        t2 = [n0, P_new, n2]
                    else:
                        t1 = [n0, n1, P_new]
                        t2 = [P_new, n1, n2]

                    new_f_idx = len(faces_list)

                    warp_lines.append(
                        f"s {curr_f_idx} {n0} {n1} {n2} 2 "
                        f"{curr_f_idx} {t1[0]} {t1[1]} {t1[2]} "
                        f"{new_f_idx} {t2[0]} {t2[1]} {t2[2]}"
                    )

                    faces_list[curr_f_idx] = t1
                    faces_list.append(t2)
                    
                    face_queue.pop(q_idx)
                    face_queue.append((curr_f_idx, t1))
                    face_queue.append((new_f_idx, t2))
                    break

    # --- PHASE 3: Vectorized Post-Topology Centroid Labeling ---
    final_verts = np.array(verts_list)
    final_faces = np.array(faces_list)
    face_labels = np.full(len(final_faces), -1, dtype=np.int32)

    for el_idx, (center, radius) in enumerate(zip(electrode_pts, electrode_radii)):
        centroids = final_verts[final_faces].mean(axis=1)
        dists = np.linalg.norm(centroids - center, axis=1)
        
        # 1. Candidate faces
        inside_mask = dists < (radius + 1e-5)
        candidate_indices = np.where(inside_mask)[0]
        
        if len(candidate_indices) == 0:
            continue

        # 2. Local Adjacency Hash
        edge_to_faces = {}
        for f_idx in candidate_indices:
            face = final_faces[f_idx]
            for i in range(3):
                edge = tuple(sorted((face[i], face[(i+1)%3])))
                if edge not in edge_to_faces:
                    edge_to_faces[edge] = []
                edge_to_faces[edge].append(f_idx)

        # 3. Seed Identification
        seed_idx = candidate_indices[np.argmin(dists[inside_mask])]

        # 4. Region-Grow (BFS)
        visited = set([seed_idx])
        queue = [seed_idx]

        while queue:
            curr_idx = queue.pop(0)
            face = final_faces[curr_idx]

            for i in range(3):
                edge = tuple(sorted((face[i], face[(i+1)%3])))
                for neighbor_idx in edge_to_faces.get(edge, []):
                    if neighbor_idx not in visited:
                        visited.add(neighbor_idx)
                        queue.append(neighbor_idx)

        # 5. Label Assignment
        for f_idx in visited:
            face_labels[f_idx] = el_idx

    return final_verts, final_faces, face_labels, warp_lines


def main(elec_descr_in: str, tri_in: str, tri_out: str, warp_name: Optional[str] = None, plot: bool = False):
    tri_verts, tri_ids = read_tri(tri_in)

    electrode = np.loadtxt(
        elec_descr_in,
        skiprows=1,
        usecols=(1, 2, 3, 4),
        ndmin=2,
        comments=["height", "n layers"],
    )

    electrode_pts, electrode_radii = np.split(electrode, [3], axis=1)
    electrode_radii = electrode_radii.flatten()

    tri_verts, tri_ids, face_labels, warp_lines = trielec(
        tri_verts, tri_ids, electrode_pts, electrode_radii
    )

    write_tri(tri_out, tri_verts, tri_ids)

    if warp_name:
        with open(warp_name, "w") as f:
            f.write("\n".join(warp_lines) + "\n")

    if plot:
        visualize_mesh(tri_verts, tri_ids, face_labels, electrode_pts)