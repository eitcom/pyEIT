# coding: utf-8
""" demo forward 3D (computation on tetrahedrons) """
# Copyright (c) Benyuan Liu. All Rights Reserved.
# Distributed under the (new) BSD License. See LICENSE.txt for more info.
from __future__ import division, absolute_import, print_function

import numpy as np

import pyeit.mesh as mesh
from pyeit.mesh import quality
import pyeit.mesh.plot as mplot
from pyeit.eit.fem import Forward
from pyeit.eit.interp2d import sim2pts
from pyeit.eit.utils import eit_scan_lines

# build tetrahedron
# 3D tetrahedron must have a bbox
bbox = [[-1, -1, -1], [1, 1, 1]]
# save calling convention as distmesh 2D
mesh_obj, el_pos = mesh.create(h0=0.15, bbox=bbox)

pts = mesh_obj['node']
tri = mesh_obj['element']

# report the status of the 2D mesh
quality.stats(pts, tri)

""" 1. FEM forward simulations """
# setup EIT scan conditions
el_dist, step = 7, 1
ex_mat = eit_scan_lines(16, el_dist)

# calculate simulated data
fwd = Forward(mesh_obj, el_pos)

# in python, index start from 0
ex_line = ex_mat[1].ravel()

# change alpha
anomaly = [{'x': 0.40, 'y': 0.40, 'z': 0.0, 'd': 0.30, 'perm': 100.0}]
mesh_new = mesh.set_perm(mesh_obj, anomaly=anomaly, background=1.0)
tri_perm = mesh_new['perm']
node_perm = sim2pts(pts, tri, np.real(tri_perm))

# solving once using fem
f, _ = fwd.solve(ex_line, perm=tri_perm)
f = np.real(f)

# mplot.tetplot(p, t, edge_color=(0.2, 0.2, 1.0, 1.0), alpha=0.01)
mplot.tetplot(pts, tri, vertex_color=f, alpha=1.0)
