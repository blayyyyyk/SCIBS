import argparse
import os
from collections import defaultdict
from typing import Optional
import numpy as np
from scibs.utilities.file import desktop_output_dir, read_tri, write_pot, write_tri
from scibs.utilities.graphics import visualize_mesh


def load_electrode_file(filepath: str) -> tuple[np.ndarray, np.ndarray]:
    """Robustly parses electrode text files while skipping variable headers."""
    valid_rows = []
    with open(filepath, "r") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            # Expecting 5 numbers: [id, x, y, z, radius]
            if len(parts) == 5:
                try:
                    nums = [float(p) for p in parts]
                    valid_rows.append(nums[1:])  # Keep x, y, z, radius
                except ValueError:
                    continue

    if not valid_rows:
        raise ValueError(f"Could not parse valid electrode data from {filepath}")

    data = np.array(valid_rows)
    electrode_pts = data[:, :3]
    electrode_radii = data[:, 3]
    return electrode_pts, electrode_radii


def _triangle_quality(verts: np.ndarray, faces: np.ndarray) -> np.ndarray:
    """Normalized shape quality per triangle: 1.0 = equilateral, ->0 = sliver."""
    v0, v1, v2 = verts[faces[:, 0]], verts[faces[:, 1]], verts[faces[:, 2]]
    a = np.linalg.norm(v1 - v2, axis=1)
    b = np.linalg.norm(v0 - v2, axis=1)
    c = np.linalg.norm(v0 - v1, axis=1)
    area = 0.5 * np.linalg.norm(np.cross(v1 - v0, v2 - v0), axis=1)
    return (4 * np.sqrt(3) * area) / (a ** 2 + b ** 2 + c ** 2 + 1e-30)


def _bisect_triangle(n0, n1, n2, e_vA, e_vB, p_new):
    """Split triangle (n0, n1, n2) at `p_new` along whichever of its edges
    matches (e_vA, e_vB), returning the two resulting sub-triangles.

    The specific node arrangement in each output triple is not arbitrary:
    `tetwarp`'s face-split handling identifies which original edge was
    bisected by diffing the new sub-triangles' edges against the original
    triangle's edges, so it depends on this matching the real split exactly.
    """
    if (n0 == e_vA and n1 == e_vB) or (n0 == e_vB and n1 == e_vA):
        return [n0, p_new, n2], [p_new, n1, n2]
    elif (n1 == e_vA and n2 == e_vB) or (n1 == e_vB and n2 == e_vA):
        return [n0, n1, p_new], [n0, p_new, n2]
    else:
        return [n0, n1, p_new], [p_new, n1, n2]


def _boundary_vertex_mask(faces: np.ndarray, face_labels: np.ndarray, n_verts: int) -> np.ndarray:
    """Marks every vertex touching a label transition (electrode<->background
    or electrode<->electrode) -- i.e. a vertex that helps define the precise
    electrode-boundary curve cut in Phase 1/2."""
    vertex_labels: dict[int, set] = {}
    for face, label in zip(faces, face_labels):
        lbl = int(label)
        for v in face:
            vertex_labels.setdefault(int(v), set()).add(lbl)
    mask = np.zeros(n_verts, dtype=bool)
    for v, labels in vertex_labels.items():
        if len(labels) > 1:
            mask[v] = True
    return mask


def cleanup_slivers(
    verts: np.ndarray, faces: np.ndarray, face_labels: np.ndarray,
    quality_threshold: float = 0.15, max_passes: int = 24,
):
    """Collapses sliver triangles left behind by the electrode-boundary cuts.

    The eps-snap in Phase 1 only looks at a single edge's own crossing
    fraction; when a triangle has two boundary-crossing edges that each miss
    the snap threshold by a little, the corner they cut off can still be a
    near-degenerate sliver. This does a real topological edge collapse (merge
    the sliver's shortest edge, drop any face that becomes degenerate as a
    result) rather than just relocating a vertex, since nudging a vertex that
    is shared by several other triangles tends to turn *those* into new
    slivers instead of actually removing the bad one.

    Vertices that sit on an electrode-boundary label transition are exactly
    where Phase 1 placed them on the electrode's true circular boundary, so
    merging one into an *interior* (non-boundary) vertex would drag a point of
    that boundary inward and dent the circle right after we fixed it to be
    round -- edges that would do that are skipped. The classic sliver shape is
    actually two *adjacent* boundary vertices (both crossing points landing
    near the same mesh corner) plus one interior vertex, so boundary-boundary
    edges stay collapsible: merging two already-near-coincident points on the
    same curve doesn't move the boundary in any meaningful sense. If a sliver
    has no boundary-safe edge at all, it's left alone rather than distorting
    the boundary to remove it.

    Returns `(verts, faces, face_labels, collapse_lines)`. `collapse_lines` are
    `"c <keep> <merge>"` warp-format lines recording every vertex merge, in
    order -- `tetwarp` replays them to keep the volumetric mesh in sync, so
    (unlike a bare face/vertex edit) this output IS safe to feed back into the
    warp pipeline as long as these lines are appended to `warp_lines`.
    """
    verts = verts.copy()
    faces = faces.copy()
    face_labels = face_labels.copy()
    collapse_lines = []

    for _ in range(max_passes):
        quality = _triangle_quality(verts, faces)
        bad_idx = np.where(quality < quality_threshold)[0]
        if len(bad_idx) == 0:
            break

        boundary_mask = _boundary_vertex_mask(faces, face_labels, len(verts))

        parent = np.arange(len(verts))

        def find(x):
            while parent[x] != x:
                parent[x] = parent[parent[x]]
                x = parent[x]
            return x

        touched = set()
        n_collapsed = 0
        for f_idx in bad_idx:
            face = faces[f_idx]
            if any(int(v) in touched for v in face):
                continue  # avoid chaining multiple collapses through one vertex per pass
            pts = verts[face]
            edge_pairs = [(0, 1), (1, 2), (2, 0)]
            lengths = [np.linalg.norm(pts[i] - pts[j]) for i, j in edge_pairs]
            for k in np.argsort(lengths):
                i, j = edge_pairs[k]
                vA, vB = int(face[i]), int(face[j])
                if vA == vB:
                    continue
                if boundary_mask[vA] != boundary_mask[vB]:
                    continue  # would drag a boundary vertex into an interior one -- skip
                ra, rb = find(vA), find(vB)
                if ra == rb:
                    continue
                keep, merge = (ra, rb) if ra < rb else (rb, ra)
                parent[merge] = keep
                collapse_lines.append(f"c {keep} {merge}")
                touched.add(vA)
                touched.add(vB)
                n_collapsed += 1
                break

        if n_collapsed == 0:
            break

        remap = np.array([find(i) for i in range(len(verts))])
        new_faces = remap[faces]
        degenerate = (
            (new_faces[:, 0] == new_faces[:, 1])
            | (new_faces[:, 1] == new_faces[:, 2])
            | (new_faces[:, 2] == new_faces[:, 0])
        )
        faces = new_faces[~degenerate]
        face_labels = face_labels[~degenerate]

    return verts, faces, face_labels, collapse_lines


def _smooth_remaining_slivers(
    verts: np.ndarray, faces: np.ndarray, face_labels: np.ndarray,
    quality_threshold: float = 0.15,
):
    """Local Laplacian smoothing fallback for slivers edge collapse couldn't reach.

    `cleanup_slivers` processes at most one collapse per vertex per pass (the
    `touched` guard), so a triangle whose vertices keep getting claimed by
    *other* nearby collapses first can survive all `max_passes` untouched even
    though a legal collapse existed for it. Rather than adding more passes and
    hoping, relax its interior (non-boundary) vertex toward the centroid of its
    own 1-ring neighbors instead -- a genuinely different operation from edge
    collapse, so it doesn't compete for the same "touched" budget. As always,
    a vertex that sits on an electrode boundary is never moved.

    Returns `(verts, move_lines)`; `move_lines` are `"m <id> <x> <y> <z>"`
    warp-format lines -- `tetwarp` already supports `move`, so no changes
    there are needed for this to reach the volumetric mesh.
    """
    verts = verts.copy()
    quality = _triangle_quality(verts, faces)
    bad_idx = np.where(quality < quality_threshold)[0]
    if len(bad_idx) == 0:
        return verts, []

    boundary_mask = _boundary_vertex_mask(faces, face_labels, len(verts))

    neighbors: dict[int, set] = defaultdict(set)
    for face in faces:
        for i in range(3):
            a, b = int(face[i]), int(face[(i + 1) % 3])
            neighbors[a].add(b)
            neighbors[b].add(a)

    move_lines = []
    smoothed = set()
    for f_idx in bad_idx:
        interior = [int(v) for v in faces[f_idx] if not boundary_mask[v]]
        if not interior:
            continue  # every corner is a boundary vertex -- nothing safe to move
        for v in interior:
            if v in smoothed:
                continue
            nbrs = list(neighbors[v])
            if not nbrs:
                continue
            new_pos = verts[nbrs].mean(axis=0)
            verts[v] = new_pos
            move_lines.append(f"m {v} {new_pos[0]:.15g} {new_pos[1]:.15g} {new_pos[2]:.15g}")
            smoothed.add(v)

    return verts, move_lines


def _check_electrode_spacing(
    verts: np.ndarray, unique_edges: np.ndarray,
    electrode_pts: np.ndarray, electrode_radii: np.ndarray,
    min_gap: Optional[float] = None,
):
    """Verifies no single mesh edge has one endpoint inside electrode A and
    the other inside electrode B, for every pair (A, B).

    Phase 1/3 process each electrode independently, one at a time, purely by
    distance to that electrode's own center. Nothing checks whether two
    electrodes are competing for the same faces -- if a mesh edge (or worse,
    a whole triangle) has vertices inside two different electrodes' spheres,
    whichever electrode is processed later in the loop silently overwrites
    the earlier one's labels there, and any triangle straddling both has no
    honest single-electrode color (see `visualize_potentials`).

    This checks the real mesh topology directly rather than an analytic
    sphere-gap heuristic: edge lengths vary a lot in practice (this repo's
    test mesh has a 4.02mm median edge but real edges up to ~13mm), so a gap
    that clears the median -- or even a generous multiple of it -- can still
    be spanned by a single longer edge somewhere else on the mesh. Checking
    actual edges has no such blind spot.

    `min_gap` is an optional EXTRA distance requirement (mm) layered on top,
    for callers who want more separation than bare correctness demands (e.g.
    physical gel-bridging clearance for a real electrode cap) -- it is not
    a substitute for the topology check above, only an addition to it.
    """
    centers = np.asarray(electrode_pts)
    radii = np.asarray(electrode_radii)
    n = len(centers)

    inside = np.linalg.norm(verts[:, None, :] - centers[None, :, :], axis=2) <= radii[None, :]
    v0, v1 = unique_edges[:, 0], unique_edges[:, 1]
    in_v0, in_v1 = inside[v0], inside[v1]

    violations = []
    for i in range(n):
        bridged = (in_v0[:, i:i + 1] & in_v1[:, i + 1:]) | (in_v1[:, i:i + 1] & in_v0[:, i + 1:])
        bad = np.where(bridged.any(axis=0))[0]
        for k in bad:
            j = i + 1 + k
            violations.append((i + 1, j + 1, int(bridged[:, k].sum())))

    if min_gap is not None:
        for i in range(n):
            center_dists = np.linalg.norm(centers[i + 1:] - centers[i], axis=1)
            gaps = center_dists - radii[i] - radii[i + 1:]
            bad = np.where(gaps < min_gap)[0]
            for k in bad:
                j = i + 1 + k
                if not any(v[:2] == (i + 1, j) for v in violations):
                    violations.append((i + 1, j, gaps[k]))

    if violations:
        lines = [
            f"  electrode {a} <-> electrode {b}: {c} bridging mesh edge(s)"
            if isinstance(c, int) else
            f"  electrode {a} <-> electrode {b}: gap = {c:.4f}mm (need >= {min_gap:.4f}mm)"
            for a, b, c in violations
        ]
        raise ValueError(
            f"{len(violations)} electrode pair(s) do not maintain a safe margin before "
            f"meshing (a triangle spanning both has no honest single-electrode color):\n"
            + "\n".join(lines)
        )


def trielec(
    verts: np.ndarray, faces: np.ndarray, electrode_pts: np.ndarray, electrode_radii: np.ndarray,
    clean_slivers: bool = False, min_electrode_gap: Optional[float] = None,
):
    """
    Imprints electrode boundaries onto a surface mesh using a Vectorized Deterministic Cascade.
    Generates strict N=2 bisections for a .warp file without requiring a dynamic global adjacency graph.

    Always checks (see `_check_electrode_spacing`) that no single mesh edge
    has one endpoint inside one electrode's sphere and the other inside a
    different electrode's -- an exact, mesh-aware check, not a distance
    heuristic. An early version of this used "gap >= median edge length" as
    a proxy, but edge lengths vary a lot in practice (this repo's test mesh
    has a 4.02mm median edge yet real edges up to ~13mm), so montages that
    cleared the median heuristic still hit the real defect elsewhere on the
    head. `min_electrode_gap`, if given, is an EXTRA distance requirement
    (mm) layered on top of that exact check, for callers who want more
    separation than bare correctness demands.
    """
    warp_lines = []
    verts_list = verts.tolist()
    faces_list = faces.tolist()

    # --- PHASE 1: Vectorized Global Edge Hash ---
    edges = np.vstack((faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]))
    edges = np.sort(edges, axis=1)
    unique_edges = np.unique(edges, axis=0)

    # DEGENERATE VERTEX GUARD
    # A single corrupted/outlier vertex (e.g. a placeholder coordinate baked into
    # the source mesh) creates edges dozens of times longer than the rest of the
    # mesh. Treating those as legitimate electrode-boundary crossings makes the
    # sliver-prevention snap below drag real scalp vertices toward the bad point,
    # punching holes in otherwise round electrodes. Drop such edges from crossing
    # detection entirely so a corrupted vertex can't distort its real neighbors.
    edge_lengths = np.linalg.norm(verts[unique_edges[:, 0]] - verts[unique_edges[:, 1]], axis=1)
    median_edge_len = np.median(edge_lengths)

    degenerate_mask = edge_lengths > 20 * median_edge_len
    if degenerate_mask.any():
        # The corrupted vertex is the one shared by every bad edge (its neighbors
        # each show up on only one bad edge, since only their shared edge with the
        # corrupted vertex is abnormally long).
        touched, counts = np.unique(unique_edges[degenerate_mask].flatten(), return_counts=True)
        hub_verts = touched[counts > 1].tolist()
        print(
            f"WARNING: Ignoring {degenerate_mask.sum()} degenerate mesh edge(s) "
            f"(>20x the median edge length of {median_edge_len:.4f}). "
            f"This usually means the input mesh has corrupted vertex position(s) "
            f"-- check vertex indices {hub_verts} in the source .tri file."
        )
        unique_edges = unique_edges[~degenerate_mask]

    # SAFETY MARGIN CHECK -- see _check_electrode_spacing's docstring: a mesh
    # edge with endpoints inside two different electrodes means neither
    # labeling nor rendering can honestly attribute the triangles there to
    # just one electrode.
    _check_electrode_spacing(verts, unique_edges, electrode_pts, electrode_radii, min_gap=min_electrode_gap)

    edge_to_new_v = {}
    moved_nodes = set()

    for el_idx, (center, radius) in enumerate(zip(electrode_pts, electrode_radii)):
        print(f"\n--- Processing electrode {el_idx + 1} ---")
        print(f"Center: [{center[0]:.6f} {center[1]:.6f} {center[2]:.6f}]")
        print(f"Radius: {radius}")
        print("Calculating vertex distances...")

        dists = np.linalg.norm(verts - center, axis=1) - radius
        print(f"Distances calculated. Shape: {dists.shape}")
        print("Edge distances calculated")

        d1 = dists[unique_edges[:, 0]]
        d2 = dists[unique_edges[:, 1]]
        
        # Only evaluate edges that physically cross the boundary threshold
        crossings = (d1 < 0) != (d2 < 0)
        
        crossing_edges = unique_edges[crossings]
        crossing_d1 = d1[crossings]
        crossing_d2 = d2[crossings]

        num_crossings = len(crossing_edges)
        print(f"Crossing edges: {num_crossings}")
        print(f"Processing {num_crossings} crossing edges...")
        if num_crossings > 0:
            print(f"  Crossing edge 0/{num_crossings}")

        # SLIVER PREVENTION (Epsilon Snapping)
        # Matches the MINLM2 = 0.15 threshold from the original C++ code.
        eps = 0.15 

        for i, (vA, vB) in enumerate(crossing_edges):
            if (vA, vB) in edge_to_new_v:
                continue

            t = crossing_d1[i] / (crossing_d1[i] - crossing_d2[i])
            p_new_calc = verts[vA] + t * (verts[vB] - verts[vA])

            if t < eps:
                if vA not in moved_nodes:
                    verts_list[vA] = p_new_calc.tolist()
                    warp_lines.append(f"m {vA} {p_new_calc[0]:.15g} {p_new_calc[1]:.15g} {p_new_calc[2]:.15g}")
                    moved_nodes.add(vA)
            elif t > 1.0 - eps:
                if vB not in moved_nodes:
                    verts_list[vB] = p_new_calc.tolist()
                    warp_lines.append(f"m {vB} {p_new_calc[0]:.15g} {p_new_calc[1]:.15g} {p_new_calc[2]:.15g}")
                    moved_nodes.add(vB)
            else:
                new_idx = len(verts_list)
                verts_list.append(p_new_calc.tolist())
                edge_to_new_v[(vA, vB)] = new_idx
                warp_lines.append(f"a {new_idx} {p_new_calc[0]:.15g} {p_new_calc[1]:.15g} {p_new_calc[2]:.15g}")

        print(f"Finished electrode {el_idx + 1}")
        print(f"Total new vertices so far: {len(edge_to_new_v)}")

    # --- PHASE 2: Isolated Deterministic Face Bisection ---
    num_original_faces = len(faces)
    print("\n=== PHASE 2: FACE BISECTION ===")
    print(f"what is this: {num_original_faces}")
    print(f"Original faces: {num_original_faces}")
    print(f"Edges with new vertices: {len(edge_to_new_v)}")

    for i in range(num_original_faces):
        if i == 0 or i == 10000:
            print(f"Processing face {i}/{num_original_faces}")

        face = faces[i]
        v0, v1, v2 = face
        edges_of_face = [
            tuple(sorted((v0, v1))),
            tuple(sorted((v1, v2))),
            tuple(sorted((v2, v0)))
        ]

        splits_needed = [(edge, edge_to_new_v[edge]) for edge in edges_of_face if edge in edge_to_new_v]

        if not splits_needed:
            continue

        face_queue = [(i, face.tolist())]

        for (e_vA, e_vB), P_new in splits_needed:
            for q_idx, (curr_f_idx, nodes) in enumerate(face_queue):
                if e_vA in nodes and e_vB in nodes:
                    n0, n1, n2 = nodes
                    t1, t2 = _bisect_triangle(n0, n1, n2, e_vA, e_vB, P_new)

                    new_f_idx = len(faces_list)

                    warp_lines.append(
                        f"s {curr_f_idx} {n0} {n1} {n2} 2 "
                        f"{curr_f_idx} {t1[0]} {t1[1]} {t1[2]} "
                        f"{new_f_idx} {t2[0]} {t2[1]} {t2[2]}"
                    )

                    faces_list[curr_f_idx] = t1
                    faces_list.append(t2)
                    
                    face_queue.pop(q_idx)
                    face_queue.append((curr_f_idx, t1))
                    face_queue.append((new_f_idx, t2))
                    break

    print("phase 2 ends here **__**__")
    print(f"total faces: {len(faces_list)}")

    # --- PHASE 3: Vectorized Post-Topology Centroid Labeling ---
    print("\n centroid labeling third phase")
    final_verts = np.array(verts_list)
    final_faces = np.array(faces_list)
    print(f"Final vertices: {final_verts.shape}")
    print(f"Final faces: {final_faces.shape}")

    face_labels = np.full(len(final_faces), -1, dtype=np.int32)

    centroids = final_verts[final_faces].mean(axis=1)
    print(f"Centroids calculated: {centroids.shape}")

    for el_idx, (center, radius) in enumerate(zip(electrode_pts, electrode_radii)):
        electrode_label = el_idx + 1
        print(f"Labeling electrode {electrode_label}")
        dists = np.linalg.norm(centroids - center, axis=1)

        # 1. Candidate faces
        # Phase 1/2 already cut every boundary-crossing edge exactly at the
        # sphere, so a correctly-split face should never straddle it. Checking
        # the CENTROID distance instead of every vertex lets a triangle with one
        # genuinely-outside vertex sneak in whenever the other two vertices sit
        # close enough to the boundary to pull the average back under the
        # threshold -- that produces a thin spike poking out of the disc.
        vert_dists = np.linalg.norm(final_verts - center, axis=1)
        inside_mask = np.all(vert_dists[final_faces] < (radius + 1e-5), axis=1)
        candidate_indices = np.where(inside_mask)[0]
        print(f"Candidate faces: {len(candidate_indices)}")
        
        if len(candidate_indices) == 0:
            continue

        # 2. Local Adjacency Hash
        edge_to_faces = {}
        for f_idx in candidate_indices:
            face = final_faces[f_idx]
            for i in range(3):
                edge = tuple(sorted((face[i], face[(i+1)%3])))
                if edge not in edge_to_faces:
                    edge_to_faces[edge] = []
                edge_to_faces[edge].append(f_idx)

        print(f"Adjacency edges: {len(edge_to_faces)}")

        # 3. Seed Identification
        seed_idx = candidate_indices[np.argmin(dists[inside_mask])]
        print(f"Seed face: {seed_idx}")

        # 4. Region-Grow (BFS)
        visited = set([seed_idx])
        queue = [seed_idx]

        while queue:
            curr_idx = queue.pop(0)
            face = final_faces[curr_idx]

            for i in range(3):
                edge = tuple(sorted((face[i], face[(i+1)%3])))
                for neighbor_idx in edge_to_faces.get(edge, []):
                    if neighbor_idx not in visited:
                        visited.add(neighbor_idx)
                        queue.append(neighbor_idx)

        print(f"Region size: {len(visited)} faces")

        # 5. Label Assignment
        for f_idx in visited:
            face_labels[f_idx] = electrode_label

    labeled_count = np.sum(face_labels != -1)
    print("\n=== TRIELEC FINISHED ===")
    print(f"Final vertices: {len(final_verts)}")
    print(f"Final faces: {len(final_faces)}")
    print(f"Warp commands: {len(warp_lines)}")
    print(f"Labeled faces: {labeled_count}")

    if clean_slivers:
        print("\n=== PHASE 4: SLIVER CLEANUP ===")
        n_before = len(final_faces)
        final_verts, final_faces, face_labels, collapse_lines = cleanup_slivers(final_verts, final_faces, face_labels)
        warp_lines.extend(collapse_lines)
        print(f"Faces before: {n_before}  after: {len(final_faces)}  ({len(collapse_lines)} vertex collapses)")

        # Fallback for any sliver edge collapse couldn't reach this run (e.g. its
        # vertices kept losing the per-pass "touched" race to other collapses).
        final_verts, smooth_lines = _smooth_remaining_slivers(final_verts, final_faces, face_labels)
        warp_lines.extend(smooth_lines)
        if smooth_lines:
            print(f"Smoothed {len(smooth_lines)} remaining sliver-adjacent vertex/vertices")

    # Force direct extraction dump to the shared Desktop output folder during function execution
    extraction_path = str(desktop_output_dir() / "extraction_trielec.txt")
    print(f"Dumping extraction data directly to: {extraction_path}")
    with open(extraction_path, "w") as f:
        f.write("=== FINAL VERTICES ===\n")
        np.savetxt(f, final_verts, fmt="%.15g")

        f.write("\n=== FINAL FACES ===\n")
        np.savetxt(f, final_faces, fmt="%d")

        f.write("\n=== FACE LABELS ===\n")
        np.savetxt(f, face_labels.reshape(-1, 1), fmt="%d")

        f.write("\n=== WARP LINES ===\n")
        for line in warp_lines:
            f.write(line + "\n")

    return final_verts, final_faces, face_labels, warp_lines


def main(
    elec_descr_in: str, tri_in: str, tri_out: str, warp_name: Optional[str] = None,
    plot: bool = False, clean_slivers: bool = False, min_electrode_gap: Optional[float] = None,
):
    tri_verts, tri_ids = read_tri(tri_in)
    electrode_pts, electrode_radii = load_electrode_file(elec_descr_in)

    tri_verts, tri_ids, face_labels, warp_lines = trielec(
        tri_verts, tri_ids, electrode_pts, electrode_radii,
        clean_slivers=clean_slivers, min_electrode_gap=min_electrode_gap,
    )

    write_tri(tri_out, tri_verts, tri_ids)

    if warp_name:
        with open(warp_name, "w") as f:
            f.write("\n".join(warp_lines) + "\n")

    if plot:
        visualize_mesh(tri_verts, tri_ids, face_labels, electrode_pts, name=os.path.splitext(os.path.basename(tri_out))[0])


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Imprint electrode geometry onto surface mesh.")
    parser.add_argument("elec_descr_in", type=str, help="Path to electrode description file")
    parser.add_argument("tri_in", type=str, help="Path to input .tri surface mesh")
    parser.add_argument("tri_out", type=str, help="Path for output .tri surface mesh")
    parser.add_argument("--warp_name", type=str, default=None, help="Path to output .warp file")
    parser.add_argument("--plot", "-p", action="store_true", help="Plot interactive 3D mesh")
    parser.add_argument(
        "--clean-slivers", action="store_true",
        help="Collapse sliver triangles left by the electrode cuts, without moving the "
             "electrode boundary itself. The vertex collapses are recorded in warp_name too, "
             "so tetwarp can replay them on the volumetric mesh."
    )
    parser.add_argument(
        "--min-electrode-gap", type=float, default=None,
        help="Extra minimum center-to-center gap (mm) required between any two electrode "
             "spheres, layered on top of the always-on exact mesh-edge check. Unset by "
             "default (the exact check alone is the real correctness requirement)."
    )

    args = parser.parse_args()
    main(
        elec_descr_in=args.elec_descr_in,
        tri_in=args.tri_in,
        tri_out=args.tri_out,
        warp_name=args.warp_name,
        plot=args.plot,
        clean_slivers=args.clean_slivers,
        min_electrode_gap=args.min_electrode_gap,
    )