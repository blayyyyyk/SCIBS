import numpy as np
import trimesh
from argparse import ArgumentParser, RawTextHelpFormatter
from utils import load_tri
import struct

def write_mbf_matrix(filename, data):
    # Binary matrix writer (same as before)
    if data.ndim == 1: data = data[:, None]
    rows, cols = data.shape
    with open(filename, 'wb') as f:
        f.write(b';;mbfmat')
        f.write(struct.pack('B', 0))
        f.write(struct.pack('<I', 11))
        f.write(struct.pack('BBB', 1, 1, 8))
        f.write(struct.pack('<II', rows, cols))
        data.astype('<f8').tofile(f)

def write_tri_file(filename, vertices, faces):
    with open(filename, 'w') as f:
        f.write(f" {len(vertices)}\n")
        for i, (x, y, z) in enumerate(vertices):
            f.write(f" {i+1} {x:.6f} {y:.6f} {z:.6f}\n")
        f.write(f" {len(faces)}\n")
        for i, (v1, v2, v3) in enumerate(faces):
            f.write(f" {i+1} {v1+1} {v2+1} {v3+1}\n")

def main():
    # ... (Arg parsing same as your previous code) ...
    parser = ArgumentParser()
    parser.add_argument("tri_in")
    parser.add_argument("elec_descr")
    parser.add_argument("tri_out")
    parser.add_argument("elec_out")
    args = parser.parse_args()

    # 1. Load Data
    print("Loading data...")
    electrode_data = np.loadtxt(
        args.elec_descr,
        skiprows=1,
        usecols=(1, 2, 3, 4),
        ndmin=2,
        comments=["height", "n layers"],
    )
    electrode_pts = electrode_data[:, :3]
    electrode_radii = electrode_data[:, 3]
    
    tri_verts, tri_ids = load_tri(args.tri_in)
    tri_ids -= 1 # Adjust 1-based index if necessary
    mesh = trimesh.Trimesh(tri_verts, tri_ids)
    
    # --- FIX: MAKE MESH WATERTIGHT ---
    print(f"Original Mesh Watertight? {mesh.is_watertight}")
    
    if not mesh.is_watertight:
        print("Repairing mesh to be watertight for Boolean operations...")
        # 1. Merge vertices to ensure connectivity
        mesh.merge_vertices()
        # 2. Fill holes (e.g. the neck opening)
        # This adds faces to close the volume so 'manifold' accepts it
        mesh.fill_holes()
        # 3. Fix normals
        mesh.fix_normals()
        
        print(f"Repaired Mesh Watertight? {mesh.is_watertight}")
        
        # Fallback if fill_holes fails (rare for standard heads)
        if not mesh.is_watertight:
            print("Warning: Mesh is still not watertight. Boolean operation may fail.")
            # Last ditch attempt: use the convex hull ONLY if strictly necessary, 
            # but that ruins the shape. Ideally, fill_holes works.

    # 2. Compute Normals for Alignment
    # We need to orient each cylinder to face the scalp normal at that point
    print("Calculating surface normals...")
    closest_pts, _, tri_indices = mesh.nearest.on_surface(electrode_pts)
    normals = mesh.face_normals[tri_indices]

    # 3. Create the "Multi-Cylinder" Tool
    print(f"Generating {len(electrode_pts)} cutter cylinders...")
    cylinders = []
    
    # We make cylinders slightly longer than needed to ensure they cut through
    height = 50.0 
    
    for i, (center, radius, normal) in enumerate(zip(electrode_pts, electrode_radii, normals)):
        # Create cylinder (defaults to Z-axis)
        cyl = trimesh.creation.cylinder(radius=radius, height=height, sections=32)
        
        # Rotate Z-axis to Surface Normal
        rotation = trimesh.geometry.align_vectors([0, 0, 1], normal)
        cyl.apply_transform(rotation)
        
        # Move to position
        cyl.apply_translation(center)
        cylinders.append(cyl)

    # Combine all cylinders into one mesh object
    print(cylinders, mesh)
    tool_mesh = trimesh.util.concatenate(cylinders)

    # 4. Perform Batched Boolean Operations
    print("Running Boolean Intersection (Electrodes)...")
    # This gets ONLY the circular patches
    patches_mesh = mesh.intersection(tool_mesh, engine='manifold')
    
    print("Running Boolean Difference (Scalp)...")
    # This gets the head WITH HOLES
    scalp_mesh = mesh.difference(tool_mesh, engine='manifold')

    # 5. Labeling Logic
    # The 'patches_mesh' is currently one big object. We need to split it 
    # to identify which patch belongs to which electrode ID.
    print("Labeling electrode patches...")
    
    # Split patches into unconnected components
    patch_components = patches_mesh.split(only_watertight=False)
    
    # Map components back to electrode IDs based on distance
    patch_labels = []
    ordered_patches = []
    
    # We need to construct the final face labels
    # 0 = Scalp, 1..N = Electrodes
    # Scalp faces come first
    final_labels = [np.nan] * len(scalp_mesh.faces)
    
    for comp in patch_components:
        # Find which electrode this patch is closest to
        centroid = comp.centroid
        dists = np.linalg.norm(electrode_pts - centroid, axis=1)
        elec_id = np.argmin(dists)
        
        # Add labels for all faces in this component
        # (ID + 1 because output usually expects 1-based indexing)
        final_labels.extend([elec_id + 1] * len(comp.faces))
        ordered_patches.append(comp)

    # 6. Reassemble Full Mesh
    print("Reassembling final mesh...")
    # Order: [Scalp, Patch1, Patch2, ...]
    final_mesh = trimesh.util.concatenate([scalp_mesh] + ordered_patches)
    final_labels_arr = np.array(final_labels)

    # 7. Save
    print(f"Saving geometry to {args.tri_out}")
    write_tri_file(args.tri_out, final_mesh.vertices, final_mesh.faces)
    
    print(f"Saving labels to {args.elec_out}")
    write_mbf_matrix(args.elec_out, final_labels_arr)

if __name__ == "__main__":
    main()