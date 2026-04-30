from scibs.scripts.pipeline import pipeline
from argparse import ArgumentParser
from src.scibs.scripts import tet2tri, trielec, tetcor
from functools import partial

parser = ArgumentParser(prog="scibs")

subparsers = parser.add_subparsers(title="subcommands")

# tet2tri.py #
tettri_parser = subparsers.add_parser("tet2tri", usage="tet2tri [-options] ptsName tetName triName [elName [totName]]\n\ttet2tri [-options] name")
tettri_parser.add_argument('-s', "--surface-only", help="output surface triangles only", action="store_true")
tettri_parser.add_argument("pts_name", type=str)
tettri_parser.add_argument("tet_name", nargs="?", type=str)
tettri_parser.add_argument("tri_name", nargs="?", type=str)
tettri_parser.add_argument("el_name", nargs="?", type=str)
tettri_parser.add_argument("tot_name", nargs="?", type=str)
tettri_parser.set_defaults(func=tet2tri.main)

# trielec.py #
trielec_parser = subparsers.add_parser("trielec", description="Subdivides vertices of a triangulated surface to map electrode boundaries.")
trielec_parser.add_argument("tri_in", help="Input triangulated surface file (.tri)")
trielec_parser.add_argument("elec_descr_in", help="Electrode description text file")
trielec_parser.add_argument("tri_out", help="Output triangulated surface file (.tri)")
trielec_parser.add_argument("--plot", action="store_true", help="Render a 3D visualization of the subdivided mesh")
trielec_parser.add_argument("-w", "--warp_name", default="", help="Record changes in a warp file (legacy flag)")
trielec_parser.set_defaults(func=trielec.main)

# tetcor.py #
tetcor_parser = subparsers.add_parser("tetcor", description="Correct tetrahedron volumes in a matlab file.")
tetcor_parser.add_argument('file_name', help="The matlab tetrahedron file to read from (.mat).")
tetcor_parser.add_argument('-s', '--struct-name', help="Matlab struct name (i.e. HeadModel, Geometry).")
tetcor_parser.add_argument('-o', '--output', help="Output file to write to. Defaults to `{file_name}_corrected.mat`")
tetcor_parser.set_defaults(func=tetcor.main)

# pipeline.py #
pipeline_parser = subparsers.add_parser("pipeline", description="")
pipeline_parser.add_argument('file_prefix', help="run the full pipeline")
pipeline_parser.add_argument('-s', '--struct-name', help="Matlab struct name (i.e. HeadModel, Geometry).", default="Geometry")
pipeline_parser.set_defaults(func=pipeline)



args = parser.parse_args()
if hasattr(args, "func"):
    locals = vars(args)
    func = locals.pop('func')
    func(**locals)
else:
    parser.print_help() # print help if no/invalid mode specified
