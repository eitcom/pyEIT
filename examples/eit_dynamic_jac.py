# coding: utf-8
""" demo on dynamic eit using JAC method """
# Copyright (c) Benyuan Liu. All Rights Reserved.
# Distributed under the (new) BSD License. See LICENSE.txt for more info.
from __future__ import division, absolute_import, print_function

import numpy as np
import matplotlib.pyplot as plt

import pyeit.mesh as mesh
from pyeit.eit.fem import Forward
from pyeit.eit.utils import eit_scan_lines

import pyeit.eit.jac as jac
from pyeit.eit.interp2d import sim2pts

""" 0. construct mesh """
mesh_obj, el_pos = mesh.create(16, h0=0.1)
# mesh_obj, el_pos = mesh.layer_circle()

# extract node, element, alpha
pts = mesh_obj["node"]
tri = mesh_obj["element"]
x, y = pts[:, 0], pts[:, 1]

""" 1. problem setup """
mesh_obj["alpha"] = np.random.rand(tri.shape[0]) * 200 + 100
anomaly = [{"x": 0.5, "y": 0.5, "d": 0.1, "perm": 1000.0}]
mesh_new = mesh.set_perm(mesh_obj, anomaly=anomaly)

""" 2. FEM simulation """
el_dist, step = 8, 1
ex_mat = eit_scan_lines(16, el_dist)

# calculate simulated data
fwd = Forward(mesh_obj, el_pos)
f0 = fwd.solve_eit(ex_mat, step=step, perm=mesh_obj["perm"])
f1 = fwd.solve_eit(ex_mat, step=step, perm=mesh_new["perm"])

""" 3. JAC solver """
# Note: if the jac and the real-problem are generated using the same mesh,
# then, data normalization in solve are not needed.
# However, when you generate jac from a known mesh, but in real-problem
# (mostly) the shape and the electrode positions are not exactly the same
# as in mesh generating the jac, then data must be normalized.
eit = jac.JAC(
    mesh_obj,
    el_pos,
    ex_mat=ex_mat,
    step=step,
    perm=1.0,
    parser="std",
)
eit.setup(p=0.5, lamb=0.01, method="kotre")
ds = eit.solve(f1.v, f0.v, normalize=True)
ds_n = sim2pts(pts, tri, np.real(ds))

# plot ground truth
fig, axes = plt.subplots(1,2, constrained_layout=True)
fig.set_size_inches(6, 4)

ax = axes[0]
delta_perm = mesh_new["perm"] - mesh_obj["perm"]
im = ax.tripcolor(x, y, tri, np.real(delta_perm), shading="flat")
ax.set_aspect("equal")

# plot EIT reconstruction
ax = axes[1]
im = ax.tripcolor(x, y, tri, ds_n, shading="flat")
# for i, e in enumerate(el_pos):
#     ax.annotate(str(i + 1), xy=(x[e], y[e]), color="r")
ax.set_aspect("equal")

fig.colorbar(im, ax=axes.ravel().tolist())
#plt.savefig('../doc/images/demo_jac.png', dpi=96)
plt.show()
