# coding: utf-8
""" demo on JAC 3D, extremely slow """
# Copyright (c) Benyuan Liu. All Rights Reserved.
# Distributed under the (new) BSD License. See LICENSE.txt for more info.
from __future__ import division, absolute_import, print_function

import numpy as np

import pyeit.mesh as mesh
import pyeit.mesh.plot as mplot
from pyeit.eit.fem import EITForward
from pyeit.eit.interp2d import sim2pts
import pyeit.eit.protocol as protocol
import pyeit.eit.jac as jac
from pyeit.mesh.wrapper import PyEITAnomaly_Ball

# build tetrahedron
# 3D tetrahedron must have a bbox
bbox = [[-1, -1, -1], [1, 1, 1]]
# save calling convention as distmesh 2D
# 3D Mesh shape is specified with fd parameter in the instantiation, e.g : fd=ball , Default in 3D :fd=ball
n_el = 16  # nb of electrodes
mesh_obj = mesh.create(n_el, h0=0.2, bbox=bbox)
pts = mesh_obj.node
tri = mesh_obj.element

# report the status of the 2D mesh
mesh_obj.print_stats()

""" 1. FEM forward simulations """
protocol_obj = protocol.create(n_el, dist_exc=7, step_meas=1, parser_meas="std")

# calculate simulated data
fwd = EITForward(mesh_obj, protocol_obj)

# change alpha
anomaly = PyEITAnomaly_Ball(center=[0.4, 0.4, 0.0], r=0.3, perm=100.0)
mesh_new = mesh.set_perm(mesh_obj, anomaly=anomaly, background=1.0)
tri_perm = mesh_new.perm
node_perm = sim2pts(pts, tri, np.real(tri_perm))

# solving once using fem
# f, _ = fwd.solve(ex_line, tri_perm)
# f = np.real(f)

# calculate simulated data
v0 = fwd.solve_eit()
v1 = fwd.solve_eit(perm=mesh_new.perm)

"""  Static GN Solver"""
# number of stimulation lines/patterns
eit = jac.JAC(mesh_obj, protocol_obj)
eit.setup(p=0.25, lamb=1.0, method="lm")
# lamb = lamb * lamb_decay
ds = eit.gn(v1, lamb_decay=0.1, lamb_min=1e-5, maxiter=20, verbose=True)
node_ds = sim2pts(pts, tri, np.real(ds))

# mplot.tetplot(p, t, edge_color=(0.2, 0.2, 1.0, 1.0), alpha=0.01)
mplot.tetplot(pts, tri, vertex_color=node_ds, alpha=1.0)
