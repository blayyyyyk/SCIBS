import argparse
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection


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

def main():
    args = get_args()

    # 1. Load Data
    tri_verts, tri_ids = load_tri(args.tri_in)
    tri_ids = tri_ids - 1  

    electrode = np.loadtxt(
        args.elec_descr,
        skiprows=1,
        usecols=(1, 2, 3, 4),
        ndmin=2,
        comments=["height", "n layers"],
    )
    electrode_pts, electrode_radii = np.split(electrode, [3], axis=1)
    electrode_radii = electrode_radii.flatten()

    tri_verts, tri_ids, face_labels = apply_electrodes(
        tri_verts, tri_ids, electrode_pts, electrode_radii
    )

    save_tri(args.tri_out, tri_verts, tri_ids)

    # save label matrix (i.e. '.pot' file)
    if args.onVert:
        vert_labels = np.full((len(tri_verts), 1), np.nan)
        for i, (center, radius) in enumerate(zip(electrode_pts, electrode_radii)):
            dists = np.linalg.norm(tri_verts - center, axis=1)
            vert_labels[dists <= radius + 1e-7] = i + 1
        np.savetxt(args.elec_out, vert_labels, fmt='%s')
    else:
        out_labels = np.full((len(face_labels), 1), np.nan)
        mask = face_labels != -1
        out_labels[mask] = (face_labels[mask] + 1)[:, None]
        np.savetxt(args.elec_out, out_labels, fmt='%s')

if __name__ == "__main__":
    main()