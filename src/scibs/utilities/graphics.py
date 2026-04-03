import matplotlib.pyplot as plt
import numpy as np
import pyvista as pv


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
