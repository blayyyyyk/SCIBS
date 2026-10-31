"""Extrude labeled electrode patches into volumetric electrode pads.

After ``trielec``/``tetwarp`` the volumetric mesh carries the electrode
boundaries, and ``face_labels`` tells us which surface triangles belong to each
electrode.  ``elecpatch`` grows a physical electrode "pad" for every electrode
by extruding its surface patch outward along the surface normal by ``height``
and filling the resulting layer with tetrahedra.

Each surface triangle becomes a triangular prism (original triangle on the
bottom, extruded triangle on top) which is split into three tetrahedra.  The
split is chosen deterministically from the *global* node ordering so that the
diagonals of shared quad faces agree between neighbouring prisms -- this keeps
the extruded layer conforming (no cracks between adjacent electrode tets).

The newly created top nodes inherit the electrode id of the patch they were
extruded from, so the per-node ``potentials`` array is extended to cover them.
This is the "surface potentials on the output of elecpatch" step that makes the
pipeline end-to-end complete.
"""


from collections import defaultdict
import numpy as np

from scibs.utilities.file import output_dir

def _patch_vertex_normals(
    pts: np.ndarray, vol_faces: np.ndarray, patch_nodes: np.ndarray, centroid: np.ndarray
) -> np.ndarray:
    """Outward unit normal per patch node (area-weighted average of faces).

    Face normals are oriented to point away from ``centroid`` so the patch is
    extruded outward regardless of the triangles' winding order.
    """
    v0 = pts[vol_faces[:, 0]]
    v1 = pts[vol_faces[:, 1]]
    v2 = pts[vol_faces[:, 2]]

    # Non-normalised face normals (magnitude == 2 * area) so larger triangles
    # contribute more to a shared vertex.
    face_normals = np.cross(v1 - v0, v2 - v0)

    # Orient each normal outward using the vector from the mesh centroid to the
    # face centroid.
    face_centroids = (v0 + v1 + v2) / 3.0
    outward = face_centroids - centroid
    flip = np.sum(face_normals * outward, axis=1) < 0
    face_normals[flip] = -face_normals[flip]

    node_normal = defaultdict(lambda: np.zeros(3, dtype=np.float64))
    for i, face in enumerate(vol_faces):
        n = face_normals[i]
        for node in face:
            node_normal[int(node)] += n

    normals = np.array([node_normal[int(n)] for n in patch_nodes], dtype=np.float64)
    lengths = np.linalg.norm(normals, axis=1, keepdims=True)
    lengths[lengths == 0] = 1.0
    return normals / lengths


def elecpatch(
    tri_pts: np.ndarray,
    tri_ids: np.ndarray,
    face_labels: np.ndarray,
    tet_pts: np.ndarray,
    tet_ids: np.ndarray,
    el_map: np.ndarray,
    potentials: np.ndarray | None = None,
    height: float = 2.0,
):
    """Extrude electrode patches into the volumetric mesh.

    Parameters
    ----------
    tri_pts:
        Surface vertices (kept for signature/reference; extrusion uses the
        volumetric coordinates so the pad sits on the warped mesh).
    tri_ids:
        ``(F, 3)`` surface triangles, surface-indexed (``trielec`` output).
    face_labels:
        ``(F,)`` electrode id per surface triangle; ``-1`` for non-electrode.
    tet_pts:
        ``(N, 3)`` volumetric node coordinates.
    tet_ids:
        ``(M, 4)`` volumetric tetrahedra, 0-indexed into ``tet_pts``.
    el_map:
        ``(P,)`` surface-node index -> volumetric-node index (``new_el_map``).
    potentials:
        Optional ``(N,)`` per-node electrode id array (e.g. from
        :func:`elecpotsurf`).  Extended to cover the extruded nodes.  When
        ``None`` a background (``-1``) array is created.
    height:
        Extrusion distance along the outward surface normal.

    Returns
    -------
    tuple[np.ndarray, np.ndarray, np.ndarray]
        ``(tet_pts, tet_ids, potentials)`` including the extruded pads.
    """
    tet_pts = np.asarray(tet_pts, dtype=np.float64).copy()
    tet_ids = np.asarray(tet_ids, dtype=np.int64)
    face_labels = np.asarray(face_labels)
    el_map = np.asarray(el_map)

    if potentials is None:
        potentials = np.full(len(tet_pts), -1, dtype=np.int64)
    else:
        potentials = np.asarray(potentials).astype(np.int64).copy()

    centroid = tet_pts.mean(axis=0)

    extra_pts = [tet_pts]
    extra_pots = [potentials]
    new_tets: list[list[int]] = []
    next_idx = len(tet_pts)

    electrodes = np.unique(face_labels[face_labels >= 0])
    for label in electrodes:
        surf_faces = tri_ids[face_labels == label]      # (Fe, 3) surface indices
        if len(surf_faces) == 0:
            continue

        vol_faces = el_map[surf_faces]                  # (Fe, 3) volumetric indices
        patch_nodes = np.unique(vol_faces)

        normals = _patch_vertex_normals(tet_pts, vol_faces, patch_nodes, centroid)
        top_pts = tet_pts[patch_nodes] + height * normals

        # Map each base (bottom) volumetric node to its freshly created top node.
        base_to_top = {
            int(node): next_idx + i for i, node in enumerate(patch_nodes)
        }

        extra_pts.append(top_pts)
        extra_pots.append(np.full(len(patch_nodes), label, dtype=np.int64))
        next_idx += len(patch_nodes)

        for face in vol_faces:
            # Sort the base triangle by global node id so that the diagonal of
            # every side quad is anchored at the lower-indexed endpoint. Two
            # prisms sharing an edge therefore pick the same diagonal, keeping
            # the extruded layer watertight.
            a, b, c = sorted(int(n) for n in face)
            ap, bp, cp = base_to_top[a], base_to_top[b], base_to_top[c]

            # Prism (a, b, c) / (ap, bp, cp) -> three tetrahedra. Winding may be
            # inverted for some triangles; the pipeline runs `tetcor` afterwards
            # to normalise signed volumes.
            new_tets.append([a, b, c, cp])
            new_tets.append([a, b, cp, bp])
            new_tets.append([a, bp, cp, ap])

    tet_pts = np.vstack(extra_pts)
    potentials = np.concatenate(extra_pots)
    if new_tets:
        tet_ids = np.vstack([tet_ids, np.array(new_tets, dtype=np.int64)])

    # Force direct extraction dump to the shared project output folder during function execution
    extraction_path = str(output_dir() / "extraction_elecpatch.txt")
    print(f"Dumping extraction data directly to: {extraction_path}")
    with open(extraction_path, "w") as f:
        f.write("=== FINAL TET POINTS (incl. extruded pads) ===\n")
        np.savetxt(f, tet_pts, fmt="%.15g")

        f.write("\n=== FINAL TET IDS (incl. extruded pads) ===\n")
        np.savetxt(f, tet_ids, fmt="%d")

        f.write("\n=== POTENTIALS ===\n")
        np.savetxt(f, potentials.reshape(-1, 1), fmt="%d")

    return tet_pts, tet_ids, potentials





