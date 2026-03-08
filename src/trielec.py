import scipy.io as sio
from scipy.spatial.distance import cdist
import numpy as np
from argparse import ArgumentParser, RawTextHelpFormatter
from utils import load_tri
import trimesh
import torch, time
from typing import TypeVar, Union, Any, Tuple, cast
import igl
import scipy.sparse as sp
from scipy.sparse.csgraph import dijkstra
from itertools import batched



prog_desc = """Usage: trielec [options] <tri_in> <elec_descr> <tri_out> <elec_out>
\ttrielec subdivides the vertices of a triangulated surface so that no
\ttriangles edges cross the edges of the rectangular electrodes
"""

elec_descr_desc = """path to a file containing electrode position data.

<elec_descr> is in an ascii file in this format:
\t<nel>
\t1 <x1> <y1> <z1> <w1> <h1> <a1>
\t...
<nel>: number of electrodes
(<x1> <y1> <z1>): center of electrode 1
<w1>: width of electrode 1
<h1>: height of electrode 1
<a1>: angle with horizon of electrode 1\n
if <h1> and <a1> are not present, the electrode is considered
to be circular with radius <w1>
"""

elec_out_desc = """single-column matrix containing value <i> if
triangle is at electode <i>, and NaN if it is not on an electrode.
"""

flag_desc_a = """only triangles whose normal differ less than <angle>
degrees with the normal of the triangle at the center
of the electrode can become part of the electrode
(default 180).
"""

flag_desc_r = """store vertices of electrode rim. If name=hup.pnt
the name for electrode 1 will be hup1.pnt.
Only the rims of rectangular electrodes are stored.
"""

flag_desc_w = """record changes in a warp file, that can be used to
morph a matching tetrahordon file in the same way.");
"""


def trielec(electrode_pts: np.ndarray, mesh_pts: np.ndarray):
    pass

T = TypeVar("T", bound=Union[np.ndarray, Any])

# p0: [E, T, V, C], p1: [E, T, V, C]
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
        bool_dtype = xp.bool
        copy_fn = lambda x: x.copy()

    assert p0.shape == p1.shape, "p0 and p1 must have the same shape"
    assert center.shape[0] == radius.shape[0], "center/radius dimension mismatch"
    assert center.ndim == radius.ndim, "center/radius rank mismatch"
    
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
    print(((epsilon < x1) & (x1 < (1 - epsilon))).shape)
    fix_mask[deter < 0, :] = False
    intersection_mask[deter < 0, :] = False
    return cast(T, intersection), cast(T, intersection_mask), cast(T, fix_mask)
    


def main():

    parser = ArgumentParser(
        prog="trielec", description=prog_desc, formatter_class=RawTextHelpFormatter
    )

    # Positional Arguments #
    parser.add_argument(
        "tri_in", help="path of file describing a surface consisting of triangles"
    )
    parser.add_argument("elec_descr", help=elec_descr_desc)
    parser.add_argument(
        "tri_out", help="the output file path to write the adjusted triangles to"
    )
    parser.add_argument("elec_out", help=elec_out_desc)

    # Flag Arguments #
    parser.add_argument("-a", help=flag_desc_a, default=180)
    parser.add_argument(
        "-v",
        help="produce elec_out on nodes rather than on triangles",
        action="store_true",
    )
    parser.add_argument("-r", help=flag_desc_r)
    parser.add_argument("-w", help=flag_desc_w)

    args = parser.parse_args()

    electrode = np.loadtxt(
        args.elec_descr,
        skiprows=1,
        usecols=(1, 2, 3, 4),
        ndmin=2,
        comments=["height", "n layers"],
    )
    electrode_pts: np.ndarray
    electrode_radii: np.ndarray
    electrode_pts, electrode_radii = np.split(electrode, (3,), axis=1)
    

    tri_verts, tri_ids = load_tri(args.tri_in)
    #dist = cdist(tri_verts, electrode_pts)
    #mask = (dist < electrode_radii.T).any(axis=1)
    tri_ids -= 1
    mesh = trimesh.Trimesh(tri_verts, tri_ids)
    
    closest_points, distances, triangle_id = mesh.nearest.on_surface(electrode_pts)
    closest_tri_pts: np.ndarray = mesh.triangles[triangle_id]
    closest_tri_pts_dist = np.linalg.norm(closest_tri_pts - electrode_pts[:, None, :], axis=-1)
    if (closest_tri_pts_dist > electrode_radii).any():
        print(
            f"""
            Oops! Projection triangle of electrode {(closest_tri_pts_dist > electrode_radii).sum(axis=-1).argmax()} is not "
                   "located\n",
                   iel + 1);
                   completely within the electrode. This is a show "
                   "stopper.\n");
                   Either make the electrode larger or the triangles "
                   "smaller.\n\n");
            """
        )
        exit()
     
    #pot = sio.loadmat("data/MNI152_E.pot")
    
    
    
    B1 = electrode_pts.shape[0]
    B2, T, C = mesh.triangles.shape
    
    # t0 = time.time()
    triangles: np.ndarray = mesh.triangles
    p1 = triangles[None, :, (0, 1, 2)].repeat(B1, axis=0)
    p0 = triangles[None, :, (1, 2, 0)].repeat(B1, axis=0)
    new_intersections = None
    new_intersection_masks = None
    new_fix_masks = None
    
    intersection, intersection_mask, fix_mask = find_crossing(p0, p1, electrode_pts, electrode_radii)
    exit()
    
    # GPU Test #
    # t1 = time.time()
    # device = torch.device("mps")
    # triangles = torch.tensor(triangles, device=device, dtype=torch.float32)
    # electrode_pts = torch.tensor(electrode_pts, device=device, dtype=torch.float32)
    # electrode_radii = torch.tensor(electrode_radii, device=device, dtype=torch.float32)
    # p0 = torch.tensor(p0, device=device, dtype=torch.float32)
    # p1 = torch.tensor(p1, device=device, dtype=torch.float32)
    # intersection, intersection_mask, fix_mask = find_crossing(p0, p1, electrode_pts, electrode_radii)
    # t2 = time.time()
    # 
    # print(f"CPU: {t1 - t0}")
    # print(f"GPU: {t2 - t1}")
    
    
    
    
    
    

if __name__ == "__main__":
    main()
