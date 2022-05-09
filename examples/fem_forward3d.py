# coding: utf-8
""" demo forward 3D (computation on tetrahedrons) """
# Copyright (c) Benyuan Liu. All Rights Reserved.
# Distributed under the (new) BSD License. See LICENSE.txt for more info.
from __future__ import division, absolute_import, print_function
import numpy as np

import pyeit.mesh as mesh
import pyeit.mesh.plot as mplot
from pyeit.eit.fem import Forward
from pyeit.eit.interp2d import sim2pts
import pyeit.eit.protocol as protocol
from pyeit.mesh.wrapper import PyEITAnomaly_Ball

# tetrahedron meshing in a 3D bbox
bbox = [[-1, -1, -1], [1, 1, 1]]
# 3D Mesh shape is specified with fd parameter in the instantiation, e.g : fd=ball , Default in 3D :fd=ball
n_el = 16  # nb of electrodes
mesh_obj = mesh.create(n_el, h0=0.15, bbox=bbox)

# report the status of the 2D mesh
pts = mesh_obj.node
tri = mesh_obj.element
mesh_obj.print_stats()

""" 1. FEM forward simulations """
# setup EIT scan conditions
protocol_obj = protocol.create(n_el, dist_exc=4, step_meas=1, parser_meas="std")


# calculate simulated data
fwd = Forward(mesh_obj)

# in python, index start from 0
ex_line = protocol_obj.ex_mat[1].ravel()

# change alpha
anomaly = PyEITAnomaly_Ball(center=[0.4, 0.4, 0], r=0.3, perm=100.0)
mesh_new = mesh.set_perm(mesh_obj, anomaly=anomaly, background=1.0)
tri_perm = mesh_new.perm
node_perm = sim2pts(pts, tri, np.real(tri_perm))

# solving once using fem
f = fwd.solve(ex_line)
f = np.real(f)

# mplot.tetplot(p, t, edge_color=(0.2, 0.2, 1.0, 1.0), alpha=0.01)
mplot.tetplot(pts, tri, vertex_color=f, alpha=0.8)
