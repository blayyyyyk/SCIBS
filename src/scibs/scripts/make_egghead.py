"""Generates the synthetic `egghead` test head: a clean, convex egg-shaped mesh.

`testhead` is real source data, so its mesh quirks come along for the ride (it
shipped with a corrupted placeholder vertex at the apex, for instance). This
builds a purely synthetic stand-in instead: a convex ovoid, uniformly sampled,
Delaunay-tetrahedralized, carrying the same 64-electrode angular layout. Because
it is generated rather than measured, it is reproducible anywhere -- which is why
this script lives in the repo while the `.mat` it produces stays gitignored.

Run it from the repo root:

    python -m scibs.scripts.make_egghead

Writes `data/egghead.mat` (volumetric), `data/egghead.tri` (surface) and
`data/egghead_elec_descr.txt` (electrode layout).
"""

from pathlib import Path

import numpy as np
import scipy.io as sio
from scipy.spatial import Delaunay

from scibs.scripts.tet2tri import tet2tri
from scibs.utilities.file import read_electrode_descr, write_tri

# Ovoid semi-axes. The z-axis differs above and below the equator, which is what
# makes this an egg rather than an ellipsoid: a rounded dome on top, a more
# tapered bottom. Both halves share the equatorial circle, so the surface stays
# convex and the Delaunay hull of the point set is exactly the solid we want.
A_EQ = 75.0      # equatorial (x, y) radius
C_TOP = 95.0     # top pole
C_BOTTOM = 65.0  # bottom pole

SURFACE_POINTS = 6000
# (radial scale, point count) per interior shell -- counts grow roughly with the
# shell's surface area so tetrahedra stay reasonably uniform through the volume.
INTERIOR_SHELLS = [(0.35, 350), (0.55, 800), (0.72, 1500), (0.87, 2600)]
SEED = 42

# Deliberately smaller than testhead's 9.0mm. This egg is smaller than testhead
# (mean radius ~77 vs ~81) and testhead's angular layout packs centers only
# ~17.1mm apart here, so 9.0mm discs physically overlap and `trielec`'s spacing
# guard rejects the mesh. 6.5mm leaves a rim-to-rim gap of ~1.09 median mesh
# edges -- slightly better than the ~1.00 testhead itself clears the guard by.
ELECTRODE_RADIUS = 6.5


def egg_scale(unit_pts: np.ndarray) -> np.ndarray:
    """Maps points on the unit sphere onto the egg surface."""
    out = np.asarray(unit_pts, dtype=np.float64).copy()
    out[:, 0] *= A_EQ
    out[:, 1] *= A_EQ
    top = out[:, 2] >= 0
    out[top, 2] *= C_TOP
    out[~top, 2] *= C_BOTTOM
    return out


def fibonacci_sphere(n: int) -> np.ndarray:
    """`n` near-uniformly spaced points on the unit sphere (golden-angle spiral).

    Preferred over a lat/long grid because it has no pole singularity, so the
    apex -- where electrode 1 sits -- is sampled no differently to anywhere else.
    """
    i = np.arange(n)
    z = 1 - 2 * (i + 0.5) / n
    phi = np.arccos(np.clip(z, -1, 1))
    theta = np.pi * (3 - np.sqrt(5)) * i
    return np.stack([np.sin(phi) * np.cos(theta), np.sin(phi) * np.sin(theta), z], axis=1)


def build_mesh() -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(SEED)

    points = [egg_scale(fibonacci_sphere(SURFACE_POINTS))]
    for scale, count in INTERIOR_SHELLS:
        # Jitter each shell's radius slightly so shells don't line up into
        # perfectly concentric layers, which Delaunay turns into flat slivers.
        jitter = 1.0 + rng.uniform(-0.02, 0.02, size=count)
        points.append(egg_scale(fibonacci_sphere(count)) * (scale * jitter)[:, None])

    all_pts = np.vstack(points)
    tets = Delaunay(all_pts).simplices

    v0, v1, v2, v3 = (all_pts[tets[:, i]] for i in range(4))
    volume = np.abs(np.einsum("ij,ij->i", np.cross(v1 - v0, v2 - v0), v3 - v0)) / 6.0
    tets = tets[volume > 1e-6]

    return all_pts, tets


def build_electrodes(reference_descr: str) -> tuple[np.ndarray, np.ndarray]:
    """Refits a reference electrode layout onto the egg surface.

    Keeps the reference's angular arrangement (so the two heads are comparable
    electrode-for-electrode) but projects each center onto this surface, the
    same way a real cap is fitted to an individual head.
    """
    ref_pts, _ = read_electrode_descr(reference_descr)
    unit_dirs = ref_pts / np.linalg.norm(ref_pts, axis=1)[:, None]
    centers = egg_scale(unit_dirs)
    radii = np.full(len(centers), ELECTRODE_RADIUS)
    return centers, radii


def write_electrode_descr(path: str, centers: np.ndarray, radii: np.ndarray):
    with open(path, "w") as f:
        f.write(f"{len(centers)} electrodes\n")
        f.write("n layers 1\n")
        f.write("height 2.0\n")
        for i, (p, r) in enumerate(zip(centers, radii)):
            f.write(f"{i + 1} {p[0]:.6f} {p[1]:.6f} {p[2]:.6f} {r:.6f}\n")


def main(data_dir: str = "data", reference: str = "testhead"):
    data = Path(data_dir)

    pts, tets = build_mesh()
    print(f"Built egg mesh: {len(pts)} nodes, {len(tets)} tets")

    geometry = np.empty((1, 1), dtype=np.dtype([("node", "O"), ("cell", "O")]))
    geometry["node"][0, 0] = pts
    geometry["cell"][0, 0] = (tets + 1).astype(np.int32)  # MATLAB is 1-indexed
    sio.savemat(str(data / "egghead.mat"), {"Geometry": geometry})
    print(f"Wrote {data / 'egghead.mat'}")

    tri_ids, tri_pts, _, _ = tet2tri(pts, tets)
    write_tri(str(data / "egghead.tri"), tri_pts, tri_ids)
    print(f"Wrote {data / 'egghead.tri'} ({len(tri_pts)} verts, {len(tri_ids)} faces)")

    centers, radii = build_electrodes(str(data / f"{reference}_elec_descr.txt"))
    write_electrode_descr(str(data / "egghead_elec_descr.txt"), centers, radii)
    print(f"Wrote {data / 'egghead_elec_descr.txt'} ({len(centers)} electrodes @ r={ELECTRODE_RADIUS})")


if __name__ == "__main__":
    main()
