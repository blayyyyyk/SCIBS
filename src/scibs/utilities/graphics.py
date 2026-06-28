import matplotlib.pyplot as plt
import numpy as np
import pyvista as pv

from scibs.utilities.file import read_mat


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




def visualize_tet_volume(vertices: np.ndarray, tets: np.ndarray):
    # PyVista requires a padding cell type column.
    # For tetrahedrons (4 vertices), we prepend '4' to every row.
    pad = np.full((tets.shape[0], 1), 4)
    cells = np.hstack((pad, tets)).flatten()

    # Cell type 10 corresponds to VTK_TETRA
    cell_types = np.full(tets.shape[0], 10, dtype=np.uint8)

    # Create the UnstructuredGrid
    grid = pv.UnstructuredGrid(cells, cell_types, vertices)

    # Plot it
    grid.plot(show_edges=True, opacity=1.0, color="lightgray", edge_color="green")

def visualize_potentials(vertices: np.ndarray, tets: np.ndarray, potentials: np.ndarray):
    """
    Visualizes the mapped electrode potentials on the volumetric tetrahedral mesh.
    """
    print("Rendering mapped potentials on tetrahedral volume...")
    pad = np.full((tets.shape[0], 1), 4)
    cells = np.hstack((pad, tets)).flatten()
    cell_types = np.full(tets.shape[0], 10, dtype=np.uint8)

    grid = pv.UnstructuredGrid(cells, cell_types, vertices)

    # Convert -1 (background) to NaN so PyVista can isolate and style it differently
    display_pots = potentials.astype(float)
    display_pots[potentials == -1] = np.nan

    # Assign the node-based potentials to the point data
    grid.point_data["Electrode_ID"] = display_pots

    # Extract the outer surface geometry.
    # Rendering millions of internal tetrahedral edges is computationally heavy and visually messy.
    surface = grid.extract_surface()

    plotter = pv.Plotter()

    plotter.add_mesh(
        surface,
        scalars="Electrode_ID",
        cmap="turbo",
        show_edges=True,
        edge_color="black",  # Softer edge color for dense tetrahedral surfaces
        line_width=0.5,
        nan_color="lightgray",
        nan_opacity=1.0,    # Make the non-electrode scalp transparent
        show_scalar_bar=True
    )

    plotter.show()

if __name__ == "__main__":
    pts, tets = read_mat("/Users/blakemoody/dev/SCIBS/data/MNI152.mat", "Geometry")
    visualize_tet_volume(pts, tets)
