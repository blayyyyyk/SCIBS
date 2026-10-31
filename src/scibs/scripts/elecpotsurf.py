"""Map electrode surface labels onto the volumetric mesh nodes.

``trielec`` produces a per-face electrode label (``face_labels``) over the
*surface* triangulation.  ``tetwarp`` then imprints those boundaries into the
volumetric tetrahedral mesh and returns ``el_map`` (``new_el_map`` in the
pipeline), which maps a surface-node index onto its volumetric-node index.

``elecpotsurf`` combines the two: it produces a dense per-node array over the
volumetric mesh where each node that belongs to an electrode patch carries that
electrode's id and every other node is ``-1`` (background).  This is the
"surface potential" field consumed by :func:`visualize_potentials` and written
alongside the mesh output.
"""

import numpy as np

from scibs.utilities.file import output_dir


def elecpotsurf(
    tri_ids: np.ndarray,
    face_labels: np.ndarray,
    n_nodes: int,
    el_map: np.ndarray,
    background: int = -1,
) -> np.ndarray:
    """Project surface electrode labels onto volumetric nodes.

    Parameters
    ----------
    tri_ids:
        ``(F, 3)`` surface triangles, indexed into the *surface* node set
        (i.e. the faces returned by ``trielec``).
    face_labels:
        ``(F,)`` electrode id per surface triangle; ``-1`` for non-electrode
        faces.
    n_nodes:
        Number of nodes in the volumetric mesh (``len(tet_pts)`` after warping).
    el_map:
        ``(P,)`` mapping from surface-node index to volumetric-node index
        (``new_el_map`` from ``tetwarp``).
    background:
        Value assigned to nodes that do not belong to any electrode.

    Returns
    -------
    np.ndarray
        ``(n_nodes,)`` integer array of electrode ids.
    """
    tri_ids = np.asarray(tri_ids)
    face_labels = np.asarray(face_labels)
    el_map = np.asarray(el_map)

    pots = np.full(int(n_nodes), background, dtype=np.int64)

    labeled = face_labels >= 0
    if not labeled.any():
        return pots

    # Surface-node ids of every labeled face -> volumetric-node ids.
    surf_nodes = tri_ids[labeled]              # (L, 3) surface indices
    vol_nodes = el_map[surf_nodes]             # (L, 3) volumetric indices
    labels = np.repeat(face_labels[labeled], 3)

    flat_nodes = vol_nodes.reshape(-1)

    # Guard against a malformed el_map pointing past the volumetric node set.
    in_range = flat_nodes < len(pots)
    if not in_range.all():
        dropped = int((~in_range).sum())
        print(
            f"Warning: elecpotsurf dropped {dropped} node label(s) whose "
            f"volumetric index exceeded n_nodes={n_nodes}."
        )

    pots[flat_nodes[in_range]] = labels[in_range]

    # Force direct extraction dump to the shared project output folder during function execution
    extraction_path = str(output_dir() / "extraction_elecpotsurf.txt")
    print(f"Dumping extraction data directly to: {extraction_path}")
    with open(extraction_path, "w") as f:
        f.write("=== NODE POTENTIALS (electrode id per volumetric node) ===\n")
        np.savetxt(f, pots.reshape(-1, 1), fmt="%d")

    return pots


def check_electrode_conflicts(tet_ids: np.ndarray, potentials: np.ndarray, background: int = -1):
    """Raises if any surface triangle of the volumetric mesh directly connects
    two DIFFERENT non-background electrodes (an edge with one endpoint in
    each) -- the same "no honest single-electrode owner" case
    `visualize_potentials` renders as background rather than guessing a side.

    A pre-check on `trielec`'s flat surface mesh (before tetwarp/elecpatch)
    is NOT equivalent to this: two electrodes with zero conflicting faces or
    edges there can still end up with a directly-touching triangle here,
    because tetwarp/elecpotsurf/elecpatch remap and extrude node identities
    independently per electrode with no cross-electrode awareness of their
    own. This checks the actual mesh that gets rendered/saved, not a proxy
    for it, so it has no such blind spot -- run it right after building
    `potentials` for a given mesh (once after `elecpotsurf`, again after
    `elecpatch` since extrusion adds new geometry).
    """
    tet_ids = np.asarray(tet_ids)
    faces = np.vstack([
        tet_ids[:, [0, 1, 2]], tet_ids[:, [1, 3, 2]], tet_ids[:, [2, 3, 0]], tet_ids[:, [3, 1, 0]]
    ])
    sorted_faces = np.sort(faces, axis=1)
    _, inverse_idx, counts = np.unique(sorted_faces, axis=0, return_inverse=True, return_counts=True)
    surf_faces = faces[counts[inverse_idx] == 1]

    corner_vals = potentials[surf_faces].astype(np.int64)
    conflict_pairs = []
    for i, j in ((0, 1), (1, 2), (2, 0)):
        x, y = corner_vals[:, i], corner_vals[:, j]
        mismatched = (x != background) & (y != background) & (x != y)
        if mismatched.any():
            conflict_pairs.append(np.sort(np.stack([x[mismatched], y[mismatched]], axis=1), axis=1))

    if not conflict_pairs:
        return

    all_pairs = np.concatenate(conflict_pairs, axis=0)
    uniq, cnts = np.unique(all_pairs, axis=0, return_counts=True)
    lines = [
        f"  electrode {p[0]} <-> electrode {p[1]}: {c} surface triangle edge(s) touch both directly"
        for p, c in zip(uniq.tolist(), cnts.tolist())
    ]
    raise ValueError(
        f"{len(uniq)} electrode pair(s) share at least one surface triangle with no honest "
        f"single-electrode owner:\n" + "\n".join(lines)
    )
