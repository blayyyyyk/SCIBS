import argparse
from typing import Optional

import numpy as np

from scibs.utilities.file import read_tri, write_pot, write_tri
from scibs.utilities.graphics import visualize_mesh


def apply_electrodes(verts: np.ndarray, faces: np.ndarray, electrode_pts: np.ndarray, electrode_radii: np.ndarray):
    """
    Subdivides triangles that cross the radius of any electrode so boundaries are strictly preserved.
    Returns the updated vertices, faces, and an array of electrode ID labels for each face.
    """
    face_labels = np.full(len(faces), -1, dtype=np.int32)

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

                        edge_intersections[edge] = len(verts) + len(new_verts)
                        new_verts.append(v_new)

        if new_verts:
            verts = np.vstack([verts, new_verts])

        # --- PASS 2: Rebuild face list ---
        new_faces = []
        new_face_labels = []

        for f_idx, face in enumerate(faces):
            d = dists[face]
            signs = d < 0

            if np.all(signs):
                new_faces.append(face)
                new_face_labels.append(el_idx)
                continue

            if np.all(~signs):
                new_faces.append(face)
                new_face_labels.append(face_labels[f_idx])
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

            new_faces.extend([tri1, tri2, tri3])

            lone_is_inside = signs[lone_idx]
            if lone_is_inside:
                new_face_labels.extend([el_idx, face_labels[f_idx], face_labels[f_idx]])
            else:
                new_face_labels.extend([face_labels[f_idx], el_idx, el_idx])

        faces = np.array(new_faces)
        face_labels = np.array(new_face_labels)

    return verts, faces, face_labels

def trielec(elec_descr_in: str, tri_in: str, tri_out: str, elec_out: str, on_vert: bool, angle: float, warp_name: float, show_plot: bool = False):
    # load data
    tri_verts, tri_ids = read_tri(tri_in)
    tri_ids = tri_ids - 1

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
    tri_verts, tri_ids, face_labels = apply_electrodes(
        tri_verts, tri_ids, electrode_pts, electrode_radii
    )

    # save modified mesh
    write_tri(tri_out, tri_verts, tri_ids)

    # save mapping matrix to .pot file
    # TODO: fix broken file format
    if on_vert:
        vert_labels = np.full((len(tri_verts), 1), np.nan)
        for i, (center, radius) in enumerate(zip(electrode_pts, electrode_radii)):
            dists = np.linalg.norm(tri_verts - center, axis=1)
            vert_labels[dists <= radius + 1e-7] = i + 1

        write_pot(elec_out, vert_labels)
    else:
        out_labels = np.full((len(face_labels), 1), np.nan)
        mask = face_labels != -1
        out_labels[mask] = (face_labels[mask] + 1)[:, None]
        write_pot(elec_out, out_labels)

    if show_plot:
        visualize_mesh(tri_verts, tri_ids, face_labels, electrode_pts)
