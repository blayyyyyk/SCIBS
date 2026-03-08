import scipy.io as sio
import numpy as np
from argparse import ArgumentParser
import os
from utils import is_model

def calculate_signed_volumes(pts, ids):
    # check if ids are 1-indexed
    if ids.min() == 1 and ids.max() == pts.shape[0]:
        ids -= 1 # shift to be 0-indexed

    tet_verts = pts[ids]
    e0 = tet_verts[:, 1] - tet_verts[:, 0]
    e1 = tet_verts[:, 2] - tet_verts[:, 0]
    e2 = tet_verts[:, 3] - tet_verts[:, 0]
    cross = np.cross(e0, e1)
    jacobian_det = np.sum(e2 * cross, axis=1)
    return jacobian_det / 6.0 # volume is 1/6 of jacobian

def load_tetrahedrons(mat_contents, struct_name) -> tuple[np.ndarray, np.ndarray]:
    pts = mat_contents[struct_name]['node'][0, 0]
    ids = mat_contents[struct_name]['cell'][0, 0]
    return pts, ids





def save_tetrahedrons(mat_contents, output_matlab_file_name: str, new_pts: np.ndarray, new_ids: np.ndarray, struct_name: str):
    mat_contents[struct_name]['node'][0, 0] = new_pts
    mat_contents[struct_name]['cell'][0, 0] = new_ids

    # remove metadata to avoid warnings
    try:
        for key in ['__header__', '__version__', '__globals__']:
            mat_contents.pop(key, None)
    except:
        pass

    sio.savemat(output_matlab_file_name, mat_contents)

def correct_tetrahedrons(pts: np.ndarray, ids: np.ndarray) -> tuple[np.ndarray, int]:
    volumes = calculate_signed_volumes(pts, ids)
    negative_ids = np.where(volumes < 0)[0]
    n_corrected = len(negative_ids)
    if n_corrected > 0:
        # swap indices to invert volume of tetrahedron
        temp = ids[negative_ids, 2].copy()
        ids[negative_ids, 2] = ids[negative_ids, 3]
        ids[negative_ids, 3] = temp

    return ids, n_corrected

def main():
    parser = ArgumentParser(
        prog='tetcor',
        description="Usage: Corrects a matlab file to assure all\ntetrahedron are oriented similarly."
    )
    parser.add_argument('file_name', help="The matlab tetrahedron file to read from (.mat).")
    parser.add_argument('-s', '--struct-name', help="Matlab struct name (i.e. HeadModel, Geometry).")
    parser.add_argument('-o', '--output', help="Output file to write to. Defaults to `{file_name}_corrected.mat`")
    args = parser.parse_args()

    struct_names = []
    mat_contents = sio.loadmat(args.file_name)

    for i, (struct_name, struct_content) in enumerate(mat_contents.values()):
        if args.struct_name is not None and args.struct_name != struct_name: continue
        if not is_model(struct_content): continue
        print(f"Correct struct {struct_name} ({i+1}, {len(struct_names)})")
        output_file_name = args.output if args.output else f"{os.path.dirname(args.file_name)}.{args.file_name.split('.')[-1]}"
        pts, ids = load_tetrahedrons(mat_contents, struct_name)
        print(f"Number of total volumes: {len(ids)}")
        new_ids, n_corrected = correct_tetrahedrons(pts, ids)
        print(f"Number of corrected volumes: {n_corrected}")
        struct_name +=
        save_tetrahedrons(mat_contents, output_file_name, pts, new_ids, struct_name)
        print(f"Output saved at {output_file_name}")


if __name__ == "__main__":
    main()
