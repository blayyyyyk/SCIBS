from typing import Optional
from scibs.utils import write_pot
import argparse
import numpy as np
import matplotlib.pyplot as plt
import pyvista as pv

from src.scibs.utilities.file import write_tri, read_tri

# ==========================================
# FILE IO 
# ==========================================

def load_tri(path: str) -> tuple[np.ndarray, np.ndarray]:
    """Loads a custom .tri file into vertices and face indices."""
    with open(path, "r") as f:
        vert_count = int(f.readline().split()[0])

    verts = np.loadtxt(
        path, skiprows=1, usecols=(1, 2, 3), dtype=np.float64, max_rows=vert_count
    )
    ids = np.loadtxt(
        path, skiprows=vert_count + 2, usecols=(1, 2, 3), dtype=np.int64
    )  
    return verts, ids

def save_tri(path: str, verts: np.ndarray, faces: np.ndarray):
    """Saves vertices and faces back to the custom .tri format."""
    with open(path, "w") as f:
        f.write(f"{len(verts)}\n")
        for i, v in enumerate(verts):
            f.write(f"{i+1} {v[0]:.6f} {v[1]:.6f} {v[2]:.6f}\n")
            
        f.write(f"{len(faces)}\n")
        for i, face in enumerate(faces):
            f.write(f"{i+1} {face[0]+1} {face[1]+1} {face[2]+1}\n")

def get_args():
    parser = argparse.ArgumentParser(description="Subdivides vertices of a triangulated surface to map electrode boundaries.")
    parser.add_argument("tri_in", help="Input triangulated surface file (.tri)")
    parser.add_argument("elec_descr", help="Electrode description text file")
    parser.add_argument("tri_out", help="Output triangulated surface file (.tri)")
    parser.add_argument("elec_out", help="Output single-column matrix file")
    
    # Execution flags
    parser.add_argument("-v", "--onVert", action="store_true", help="Produce elec_out on nodes rather than triangles")
    parser.add_argument("--plot", action="store_true", help="Render a 3D visualization of the subdivided mesh")
    
    # Legacy flags
    parser.add_argument("-a", "--angle", type=float, default=180.0, help="Max angle diff (legacy flag)")
    parser.add_argument("-r", "--rimName", default="", help="Store vertices of electrode rim (legacy flag)")
    parser.add_argument("-w", "--warpName", default="", help="Record changes in a warp file (legacy flag)")
    
    return parser.parse_args()


def visualize_mesh(verts: np.ndarray, faces: np.ndarray, face_labels: np.ndarray, electrode_pts: np.ndarray):
    """Renders the mesh using PyVista for GPU-accelerated 3D visualization."""
    print("Rendering hardware-accelerated mesh visualization... (Close the window to complete execution)")
    padding = np.full((len(faces), 1), 3, dtype=np.int64)
    pv_faces = np.hstack((padding, faces)).flatten()
    
    # Create the mesh object
    mesh = pv.PolyData(verts, pv_faces)
    
    # Assign our calculated electrode IDs directly to the mesh faces (cells)
    mesh.cell_data["Electrode_ID"] = face_labels
    
    # Split the mesh into two parts so we can style them differently, just like the matplotlib version
    base_mask = face_labels == -1
    base_mesh = mesh.extract_cells(base_mask)
    elec_mesh = mesh.extract_cells(~base_mask)
    
    plotter = pv.Plotter()
    
    # Plot the base brain/surface mesh: transparent, gray, no heavy wireframe to save performance
    if base_mesh.n_cells > 0:
        plotter.add_mesh(base_mesh, color='lightgray', opacity=0.3, show_edges=False)
    
    # Plot the subdivided electrode patches: opaque, colored by unique ID, with black wireframes
    if elec_mesh.n_cells > 0:
        plotter.add_mesh(
            elec_mesh, 
            scalars="Electrode_ID", 
            cmap="turbo", 
            show_edges=True, 
            edge_color="black", 
            line_width=1.5
        )
        
    # Drop the red center points into the scene
    if len(electrode_pts) > 0:
        plotter.add_points(
            electrode_pts, 
            color="red", 
            point_size=12, 
            render_points_as_spheres=True,
            label="Electrode Centers"
        )
        
    plotter.add_legend()
    plotter.show()

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
    # 1. Load Data
    tri_verts, tri_ids = load_tri(tri_in)
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


