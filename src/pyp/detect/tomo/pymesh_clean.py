#!/usr/bin/env -S python -B
import argparse

# from pymesh import form_mesh, remove_duplicated_vertices
import os

import numpy as np
import pymesh

parser = argparse.ArgumentParser()

parser.add_argument(
    "--d1", help="Original vertices array first dimension shape", type=str
)
parser.add_argument(
    "--d2", help="Original vertices array second dimension shape.", type=str
)
parser.add_argument("--f1", help="Original faces array first dimension shape", type=str)
parser.add_argument(
    "--f2", help="Original faces array second dimension shape.", type=str
)
parser.add_argument(
    "--virion_name", help="virion name from pyp virion processing.", type=str
)
parser.add_argument(
    "--distance", help="picking distance between each point (in pixel).", type=str
)
args = parser.parse_args()


print("Current working directory is " + str(os.getcwd()))

loaded_verts = np.loadtxt("{0}_raw_verts.txt".format(args.virion_name)).reshape(
    int(args.d1), int(args.d2)
)
loaded_verts = loaded_verts.astype(int)

loaded_faces = np.loadtxt("{0}_raw_faces.txt".format(args.virion_name)).reshape(
    int(args.f1), int(args.f2)
)
loaded_faces = loaded_faces.astype(int)

mesh = pymesh.form_mesh(loaded_verts, loaded_faces)
mesh, info = pymesh.remove_duplicated_vertices(mesh, float(args.distance))

print("Mesh coordinates number: " + str(len(mesh.vertices)))

clean_verts = mesh.vertices

with open("{}_auto.cmm".format(args.virion_name), "w") as cmm:
    cmm.write('<marker_sets>\n<marker_set name="grid_picking">\n')
    id = 1
    for xyz in clean_verts:
        x = float(xyz[2])
        y = float(xyz[1])
        z = float(xyz[0])
        cmm.write(
            """<marker id="{0}" x="{1}" y="{2}" z="{3}" r="1" g="0.8" b="0.2" radius="1"/>\n""".format(
                str(id), str(x), str(y), str(z)
            )
        )
        id += 1
    cmm.write("</marker_set>\n</marker_sets>")
