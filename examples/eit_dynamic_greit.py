# coding: utf-8
""" demo using GREIT """
# Copyright (c) Benyuan Liu. All Rights Reserved.
# Distributed under the (new) BSD License. See LICENSE.txt for more info.
from __future__ import division, absolute_import, print_function

import numpy as np
import matplotlib.pyplot as plt

import pyeit.mesh as mesh
from pyeit.eit.fem import Forward
from pyeit.eit.utils import eit_scan_lines
from pyeit.mesh.shape import thorax
import pyeit.eit.greit as greit

""" 0. construct mesh """
# Mesh shape is specified with fd parameter in the instantiation, e.g : fd=thorax , Default :fd=circle
mesh_obj, el_pos = mesh.create(16, h0=0.1, fd=thorax)

# extract node, element, alpha
pts = mesh_obj["node"]
tri = mesh_obj["element"]

""" 1. problem setup """
# this step is not needed, actually
# mesh_0 = mesh.set_perm(mesh_obj, background=1.0)

# test function for altering the 'permittivity' in mesh
anomaly = [
    {"x": 0.4, "y": 0, "d": 0.1, "perm": 10},
    {"x": -0.4, "y": 0, "d": 0.1, "perm": 10},
    {"x": 0, "y": 0.5, "d": 0.1, "perm": 0.1},
    {"x": 0, "y": -0.5, "d": 0.1, "perm": 0.1},
]
mesh_new = mesh.set_perm(mesh_obj, anomaly=anomaly, background=1.0)
delta_perm = np.real(mesh_new["perm"] - mesh_obj["perm"])

# show alpha
fig, axes = plt.subplots(2, 1, constrained_layout=True)
fig.set_size_inches(6, 4)

ax = axes[0]
im = ax.tripcolor(
    pts[:, 0], pts[:, 1], tri, delta_perm, shading="flat", cmap=plt.cm.viridis
)
ax.axis("equal")
ax.set_xlim([-1.2, 1.2])
ax.set_ylim([-1.2, 1.2])
ax.set_title(r"$\Delta$ Conductivity")
# fig.set_size_inches(6, 4)

""" 2. FEM forward simulations """
# setup EIT scan conditions
el_dist, step = 1, 1
ex_mat = eit_scan_lines(16, el_dist)

# calculate simulated data
fwd = Forward(mesh_obj, el_pos)
f0 = fwd.solve_eit(ex_mat, step=step, perm=mesh_obj["perm"])
f1 = fwd.solve_eit(ex_mat, step=step, perm=mesh_new["perm"])

""" 3. Construct using GREIT """
eit = greit.GREIT(mesh_obj, el_pos, ex_mat=ex_mat, step=step, parser="std")
eit.setup(p=0.50, lamb=0.001)
ds = eit.solve(f1.v, f0.v)
x, y, ds = eit.mask_value(ds, mask_value=np.NAN)

# plot
"""
imshow will automatically set NaN (bad values) to 'w',
if you want to manually do so

import matplotlib.cm as cm
cmap = cm.gray
cmap.set_bad('w', 1.)
plt.imshow(np.real(ds), interpolation='nearest', cmap=cmap)
"""
ax = axes[1]
im = ax.imshow(np.real(ds), interpolation="none", cmap=plt.cm.viridis)
ax.axis("equal")

fig.colorbar(im, ax=axes.ravel().tolist())
# fig.savefig('../doc/images/demo_greit.png', dpi=96)
plt.show()
