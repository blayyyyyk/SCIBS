import matplotlib.pyplot as plt
import numpy as np
import pyvista as pv

from scibs.utilities.file import output_dir, read_mat


def _output_path(name: str) -> str:
    """Resolves the project's `out/{name}.png`, creating the folder if needed."""
    return str(output_dir() / f"{name}.png")


def visualize_mesh(
    verts: np.ndarray, faces: np.ndarray, face_labels: np.ndarray, electrode_pts: np.ndarray,
    name: str = "mesh",
):
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

        # Electrode ids are 1-indexed (see trielec.py's face_labels), so label
        # point i with i + 1 to match what's shown in the Electrode_ID scalar bar.
        electrode_ids = [str(i + 1) for i in range(len(electrode_pts))]
        plotter.add_point_labels(
            electrode_pts,
            electrode_ids,
            font_size=14,
            text_color="white",
            shape_color="black",
            shape_opacity=0.6,
            always_visible=True,
            show_points=False,
        )

    plotter.add_legend()
    screenshot_path = _output_path(name)
    plotter.show(screenshot=screenshot_path)
    print(f"Saved screenshot to {screenshot_path}")




def visualize_tet_volume(vertices: np.ndarray, tets: np.ndarray, name: str = "tet_volume"):
    # PyVista requires a padding cell type column.
    # For tetrahedrons (4 vertices), we prepend '4' to every row.
    pad = np.full((tets.shape[0], 1), 4)
    cells = np.hstack((pad, tets)).flatten()

    # Cell type 10 corresponds to VTK_TETRA
    cell_types = np.full(tets.shape[0], 10, dtype=np.uint8)

    # Create the UnstructuredGrid
    grid = pv.UnstructuredGrid(cells, cell_types, vertices)

    # Plot it
    screenshot_path = _output_path(name)
    grid.plot(show_edges=True, opacity=1.0, color="lightgray", edge_color="green", screenshot=screenshot_path)
    print(f"Saved screenshot to {screenshot_path}")

def visualize_potentials(vertices: np.ndarray, tets: np.ndarray, potentials: np.ndarray, name: str = "potentials"):
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

    # `potentials` is per-NODE, but every triangle straddling an electrode
    # boundary has one electrode-id corner and one background/NaN corner.
    # Coloring by that point data -- interpolated or not -- still blends those
    # two colors somewhere across the triangle (a smudged/soft edge at best, a
    # rainbow "spike" at worst with a high-contrast colormap), because a color
    # is still being *interpolated* across each triangle's face either way.
    # Converting to a per-triangle (cell) label first removes the ambiguity
    # entirely: each triangle gets one flat, solid color with a crisp boundary,
    # the same technique `visualize_mesh` already uses for the surface-only view.
    surf_faces = surface.faces.reshape(-1, 4)[:, 1:4]
    corner_vals = surface.point_data["Electrode_ID"][surf_faces]
    a, b, c = corner_vals[:, 0], corner_vals[:, 1], corner_vals[:, 2]

    def _eq(x, y):
        return (np.isnan(x) & np.isnan(y)) | (x == y)

    # No two corners agree: three genuinely different regions (e.g. two
    # different electrodes plus background) all touch this one triangle --
    # this happens when two electrodes sit closer together than the mesh's
    # own triangle size, so a single background triangle's footprint reaches
    # both of their boundary rings at once. Neither electrode legitimately
    # owns it (trielec's own face labels never gave it to either one), so
    # leaving it as background/NaN is the honest rendering -- picking a side
    # here previously showed up as one electrode's color bleeding into its
    # neighbor's.
    face_labels = np.where(_eq(a, b), a, np.where(_eq(b, c), b, np.where(_eq(a, c), a, np.nan)))
    surface.cell_data["Electrode_ID"] = face_labels

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
        show_scalar_bar=True,
    )

    # Label each electrode at the centroid of its potential-carrying nodes
    # (base scalp nodes + any extruded pad nodes from elecpatch).
    electrode_ids = np.unique(potentials[potentials != -1])
    if len(electrode_ids) > 0:
        label_pts = np.array([vertices[potentials == eid].mean(axis=0) for eid in electrode_ids])
        plotter.add_point_labels(
            label_pts,
            [str(int(eid)) for eid in electrode_ids],
            font_size=14,
            text_color="white",
            shape_color="black",
            shape_opacity=0.6,
            always_visible=True,
            show_points=False,
        )

    screenshot_path = _output_path(name)
    plotter.show(screenshot=screenshot_path)
    print(f"Saved screenshot to {screenshot_path}")

if __name__ == "__main__":
    pts, tets = read_mat("/Users/blakemoody/dev/SCIBS/data/MNI152.mat", "Geometry")
    visualize_tet_volume(pts, tets)
