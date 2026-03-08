import argparse
from argparse import RawTextHelpFormatter
import numpy as np
from typing import TypeVar, Union, Any, Tuple, cast
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

# ==========================================
# USER'S VECTORIZED MATH LOGIC
# ==========================================

T = TypeVar("T", bound=Union[np.ndarray, Any])

def find_crossing(
    p0: T, 
    p1: T, 
    center: T, 
    radius: T, 
    epsilon: float = 1e-8
) -> Tuple[T, T, T]:
    
    if not isinstance(p0, np.ndarray): 
        import torch as xp
        bool_dtype = xp.bool
        copy_fn = lambda x: x.clone()
    else: 
        import numpy as xp
        bool_dtype = xp.bool_
        copy_fn = lambda x: x.copy()

    assert p0.shape == p1.shape, "p0 and p1 must have the same shape"
    assert center.shape[0] == radius.shape[0], "center/radius dimension mismatch"
    assert center.ndim == radius.ndim, f"center/radius rank mismatch {center.shape} and {radius.shape}"
    
    extra_dims = (1,) * (p0.ndim - 2)
    center_reshaped = center.reshape((center.shape[0],) + extra_dims + (center.shape[-1],))
    radius_reshaped = radius.reshape((radius.shape[0],) + extra_dims)
    
    p = p0 - center_reshaped
    r = p1 - p0
    a = (r * r).sum(-1)
    b = 2 * (p * r).sum(-1)
    c = (p * p).sum(-1) - radius_reshaped**2
    
    deter = b**2 - 4 * a * c
    fix_mask = xp.zeros((*p0.shape[:-1], 2), dtype=bool_dtype)
    
    deter_safe = copy_fn(deter)
    deter_safe[deter < 0] = 0
    sqrt_deter = deter_safe**0.5
    a[a < epsilon] = epsilon
    x1 = (-b - sqrt_deter) / (2 * a)
    x2 = (-b + sqrt_deter) / (2 * a)
    
    fix_mask[(abs(x1) <= epsilon) | (abs(x2) <= epsilon), 0] = True
    fix_mask[(abs(1 - x1) <= epsilon) | (abs(1 - x2) <= epsilon), 1] = True
    intersection = xp.stack([x1, x2], -1)
    intersection_mask = xp.zeros_like(fix_mask, dtype=bool_dtype)
    intersection_mask[(epsilon < x1) & (x1 < (1 - epsilon)), 0] = True
    intersection_mask[(epsilon < x2) & (x2 < (1 - epsilon)) & (abs(x1 - x2) > epsilon), 1] = True
    
    fix_mask[deter < 0, :] = False
    intersection_mask[deter < 0, :] = False
    return cast(T, intersection), cast(T, intersection_mask), cast(T, fix_mask)

# ==========================================
# FILE IO & CLI
# ==========================================

def load_tri(path: str) -> tuple[np.ndarray, np.ndarray]:
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
    with open(path, "w") as f:
        f.write(f"{len(verts)}\n")
        for i, v in enumerate(verts):
            f.write(f"{i+1} {v[0]:.6f} {v[1]:.6f} {v[2]:.6f}\n")
        f.write(f"{len(faces)}\n")
        for i, face in enumerate(faces):
            f.write(f"{i+1} {face[0]+1} {face[1]+1} {face[2]+1}\n")

prog_desc = """Usage: trielec [options] <tri_in> <elec_descr> <tri_out> <elec_out>
\ttrielec subdivides the vertices of a triangulated surface so that no
\ttriangles edges cross the edges of the rectangular electrodes
"""

elec_descr_desc = """path to a file containing electrode position data."""

# ==========================================
# HYBRID MESH TOPOLOGY LOGIC
# ==========================================

def apply_electrodes(verts: np.ndarray, faces: np.ndarray, electrode_pts: np.ndarray, electrode_radii: np.ndarray):
    face_labels = np.full(len(faces), -1, dtype=np.int32)
    
    for el_idx, (center, radius) in enumerate(zip(electrode_pts, electrode_radii)):
        
        # 1. Spatial Culling: Find which vertices are inside the current electrode
        dists_to_center = np.linalg.norm(verts - center, axis=1)
        inside = dists_to_center < radius
        
        # Isolate faces that have mixed vertices (some inside, some outside)
        crossing_faces_mask = np.any(inside[faces], axis=1) & ~np.all(inside[faces], axis=1)
        crossing_faces = faces[crossing_faces_mask]
        
        new_verts = []
        edge_intersections = {}
        unique_edges = set()
        
        # Extract all unique edges that cross the boundary
        for face in crossing_faces:
            for i in range(3):
                v1, v2 = face[i], face[(i + 1) % 3]
                if inside[v1] != inside[v2]:
                    unique_edges.add(tuple(sorted((v1, v2))))
        
        # 2. Vectorized Intersection: Run your math on ONLY the crossing edges
        if unique_edges:
            edges_arr = np.array(list(unique_edges))
            p0 = verts[edges_arr[:, 0]]
            p1 = verts[edges_arr[:, 1]]
            
            # Use your dual-backend intersection function
            intersections, int_mask, fix_mask = find_crossing(
                p0, p1, center[None, :], np.array([radius])
            )
            
            for idx, edge in enumerate(edges_arr):
                v1, v2 = edge
                t1, t2 = intersections[idx, 0], intersections[idx, 1]
                m1, m2 = int_mask[idx, 0], int_mask[idx, 1]
                
                # Select the valid parametric 't' root between 0 and 1
                if m1: 
                    t = t1
                elif m2: 
                    t = t2
                else:
                    # Precision fallback: linear distance interpolation
                    d1, d2 = dists_to_center[v1], dists_to_center[v2]
                    t = (radius - d1) / (d2 - d1)
                
                # Generate the exact boundary coordinate
                v_new = verts[v1] + t * (verts[v2] - verts[v1])
                
                edge_intersections[tuple(edge)] = len(verts) + len(new_verts)
                new_verts.append(v_new)

        if new_verts:
            verts = np.vstack([verts, new_verts])

        # 3. Topological Stitching: Rebuild the face list
        new_faces = []
        new_face_labels = []
        
        for f_idx, face in enumerate(faces):
            is_in = inside[face]
            
            if np.all(is_in):
                new_faces.append(face)
                new_face_labels.append(el_idx)
                continue
                
            if np.all(~is_in):
                new_faces.append(face)
                new_face_labels.append(face_labels[f_idx])
                continue
                
            # Face intersects the boundary and must be split
            if is_in[0] == is_in[1]:
                lone_idx = 2
            elif is_in[1] == is_in[2]:
                lone_idx = 0
            else:
                lone_idx = 1
                
            v_lone = face[lone_idx]
            v_a = face[(lone_idx + 1) % 3]
            v_b = face[(lone_idx + 2) % 3]
            
            p_a = edge_intersections[tuple(sorted((v_lone, v_a)))]
            p_b = edge_intersections[tuple(sorted((v_lone, v_b)))]
            
            new_faces.extend([
                [v_lone, p_a, p_b],
                [p_a, v_a, v_b],
                [p_b, p_a, v_b]
            ])
            
            lone_is_inside = is_in[lone_idx]
            if lone_is_inside:
                new_face_labels.extend([el_idx, face_labels[f_idx], face_labels[f_idx]])
            else:
                new_face_labels.extend([face_labels[f_idx], el_idx, el_idx])
                
        faces = np.array(new_faces)
        face_labels = np.array(new_face_labels)
        
    return verts, faces, face_labels

def visualize_mesh(verts: np.ndarray, faces: np.ndarray, face_labels: np.ndarray, electrode_pts: np.ndarray):
    """Renders the mesh using Matplotlib to visually validate subdivisions."""
    print("Rendering mesh visualization... (Close the window to complete execution)")
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection='3d')

    unique_labels = np.unique(face_labels)
    # Generate a colormap for different electrodes
    colors = plt.cm.get_cmap('tab10', len(unique_labels))

    # Plot faces by their assigned electrode label
    for i, label in enumerate(unique_labels):
        mask = face_labels == label
        subset_faces = faces[mask]
        
        # Base mesh (label -1) is drawn faint and transparent
        if label == -1:
            face_color = (0.8, 0.8, 0.8, 0.3) 
            edge_color = (0.5, 0.5, 0.5, 0.1)
            line_width = 0.5
        # Electrode patches are drawn opaque with bold wireframes
        else:
            face_color = colors(i)[:3] + (0.9,) 
            edge_color = 'black'
            line_width = 1.0 

        # Create 3D polygon collection
        mesh_col = Poly3DCollection(verts[subset_faces], alpha=0.8)
        mesh_col.set_facecolor(face_color)
        mesh_col.set_edgecolor(edge_color)
        mesh_col.set_linewidth(line_width)
        ax.add_collection3d(mesh_col)

    # Plot the exact center points of the electrodes
    if len(electrode_pts) > 0:
        ax.scatter(electrode_pts[:, 0], electrode_pts[:, 1], electrode_pts[:, 2], 
                   color='red', s=50, label='Electrode Centers', zorder=5)

    # Auto-scale axes to fit the geometry
    max_range = np.array([verts[:,0].max()-verts[:,0].min(), 
                          verts[:,1].max()-verts[:,1].min(), 
                          verts[:,2].max()-verts[:,2].min()]).max() / 2.0

    mid_x = (verts[:,0].max()+verts[:,0].min()) * 0.5
    mid_y = (verts[:,1].max()+verts[:,1].min()) * 0.5
    mid_z = (verts[:,2].max()+verts[:,2].min()) * 0.5
    
    ax.set_xlim(mid_x - max_range, mid_x + max_range)
    ax.set_ylim(mid_y - max_range, mid_y + max_range)
    ax.set_zlim(mid_z - max_range, mid_z + max_range)

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title('Subdivided Mesh & Electrode Boundaries')
    
    # Set an isometric viewing angle
    ax.view_init(elev=30, azim=45)
    plt.legend()
    plt.tight_layout()
    plt.show()

# ==========================================
# MAIN EXECUTION
# ==========================================

def main():
    parser = argparse.ArgumentParser(
        prog="trielec", description=prog_desc, formatter_class=RawTextHelpFormatter
    )

    parser.add_argument("tri_in", help="path of file describing a surface")
    parser.add_argument("elec_descr", help=elec_descr_desc)
    parser.add_argument("tri_out", help="the output file path")
    parser.add_argument("elec_out", help="single-column matrix file")
    parser.add_argument("--plot", action="store_true", help="Render a 3D visualization of the subdivided mesh")
    
    parser.add_argument("-a", help="legacy flag", default=180)
    parser.add_argument("-v", help="produce elec_out on nodes", action="store_true")
    parser.add_argument("-r", help="legacy flag")
    parser.add_argument("-w", help="legacy flag")

    args = parser.parse_args()

    # Load Data
    electrode = np.loadtxt(
        args.elec_descr, skiprows=1, usecols=(1, 2, 3, 4), ndmin=2, comments=["height", "n layers"]
    )
    electrode_pts, electrode_radii = np.split(electrode, (3,), axis=1)
    electrode_radii = electrode_radii.flatten()

    tri_verts, tri_ids = load_tri(args.tri_in)
    tri_ids -= 1  

    # Execute Hybrid Subdivision
    tri_verts, tri_ids, face_labels = apply_electrodes(
        tri_verts, tri_ids, electrode_pts, electrode_radii
    )
    
    if args.plot:
        visualize_mesh(tri_verts, tri_ids, face_labels, electrode_pts)

    # Save Results
    save_tri(args.tri_out, tri_verts, tri_ids)

    if args.v:
        vert_labels = np.full((len(tri_verts), 1), np.nan)
        for i, (center, radius) in enumerate(zip(electrode_pts, electrode_radii)):
            dists = np.linalg.norm(tri_verts - center, axis=1)
            vert_labels[dists <= radius + 1e-7] = i + 1
        np.savetxt(args.elec_out, vert_labels, fmt='%s')
    else:
        out_labels = np.full((len(face_labels), 1), np.nan)
        mask = face_labels != -1
        out_labels[mask] = face_labels[mask] + 1  
        np.savetxt(args.elec_out, out_labels, fmt='%s')

if __name__ == "__main__":
    main()