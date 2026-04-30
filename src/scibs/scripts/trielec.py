import argparse
from typing import Optional

import numpy as np

from scibs.utilities.file import read_tri, write_pot, write_tri
from scibs.utilities.graphics import visualize_mesh


def trielec(verts: np.ndarray, faces: np.ndarray, electrode_pts: np.ndarray, electrode_radii: np.ndarray):
    """
    Subdivides triangles that cross the radius of any electrode so boundaries are strictly preserved.
    Returns the updated vertices, faces, an array of electrode ID labels for each face, and warp lines.
    """
    face_labels = np.full(len(faces), -1, dtype=np.int32)
    warp_lines = []

    for el_idx, (center, radius) in enumerate(zip(electrode_pts, electrode_radii)):
        dists = np.linalg.norm(verts - center, axis=1) - radius

        new_verts = []
        edge_intersections = {}

        # --- PASS 1: Identify crossing edges ---
        for face in faces:
            d = dists[face]
            signs = d < 0

            if np.all(signs) or np.all(~signs):
                continue

            for i in range(3):
                v1, v2 = face[i], face[(i + 1) % 3]

                if signs[i] != signs[(i + 1) % 3]:
                    edge = tuple(sorted((v1, v2)))
                    if edge not in edge_intersections:
                        t = dists[v1] / (dists[v1] - dists[v2])
                        v_new = verts[v1] + t * (verts[v2] - verts[v1])
                        
                        new_v_idx = len(verts) + len(new_verts)
                        edge_intersections[edge] = new_v_idx
                        new_verts.append(v_new)
                        
                        warp_lines.append(f"a {new_v_idx + 1} {v_new[0]:.15g} {v_new[1]:.15g} {v_new[2]:.15g}")

        if new_verts:
            verts = np.vstack([verts, new_verts])

        # --- PASS 2: Rebuild face list ---
        new_faces = list(faces)
        new_face_labels = list(face_labels)

        original_num_faces = len(faces)
        for f_idx in range(original_num_faces):
            face = faces[f_idx]
            d = dists[face]
            signs = d < 0

            if np.all(signs):
                new_face_labels[f_idx] = el_idx
                continue

            if np.all(~signs):
                continue

            if signs[0] == signs[1]:
                lone_idx = 2
            elif signs[1] == signs[2]:
                lone_idx = 0
            else:
                lone_idx = 1

            v_lone = face[lone_idx]
            v_a = face[(lone_idx + 1) % 3]
            v_b = face[(lone_idx + 2) % 3]

            p_a = edge_intersections[tuple(sorted((v_lone, v_a)))]
            p_b = edge_intersections[tuple(sorted((v_lone, v_b)))]

            tri1 = [v_lone, p_a, p_b]
            tri2 = [p_a, v_a, v_b]
            tri3 = [p_b, p_a, v_b]

            # Overwrite original face and append new ones to preserve indexing
            new_faces[f_idx] = tri1
            
            idx_tri2 = len(new_faces)
            new_faces.append(tri2)
            
            idx_tri3 = len(new_faces)
            new_faces.append(tri3)

            s_line = (
                f"s {f_idx + 1} {face[0] + 1} {face[1] + 1} {face[2] + 1} 3 "
                f"{f_idx + 1} {tri1[0] + 1} {tri1[1] + 1} {tri1[2] + 1} "
                f"{idx_tri2 + 1} {tri2[0] + 1} {tri2[1] + 1} {tri2[2] + 1} "
                f"{idx_tri3 + 1} {tri3[0] + 1} {tri3[1] + 1} {tri3[2] + 1}"
            )
            warp_lines.append(s_line)

            lone_is_inside = signs[lone_idx]
            if lone_is_inside:
                new_face_labels[f_idx] = el_idx
                new_face_labels.extend([face_labels[f_idx], face_labels[f_idx]])
            else:
                new_face_labels[f_idx] = face_labels[f_idx]
                new_face_labels.extend([el_idx, el_idx])

        faces = np.array(new_faces)
        face_labels = np.array(new_face_labels)

    return verts, faces, face_labels, warp_lines
    

def main(elec_descr_in: str, tri_in: str, tri_out: str, warp_name: Optional[str] = None, plot: bool = False):
    # load data
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

    # process mesh
    tri_verts, tri_ids, face_labels, warp_lines = trielec(
        tri_verts, tri_ids, electrode_pts, electrode_radii
    )

    # save modified mesh
    write_tri(tri_out, tri_verts, tri_ids)

    # save warp file
    if warp_name:
        with open(warp_name, "w") as f:
            f.write("\n".join(warp_lines) + "\n")

    if plot:
        visualize_mesh(tri_verts, tri_ids, face_labels, electrode_pts)