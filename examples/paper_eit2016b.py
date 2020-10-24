# coding: utf-8
""" reproducible code for EIT2016 """
# Copyright (c) Benyuan Liu. All Rights Reserved.
# Distributed under the (new) BSD License. See LICENSE.txt for more info.
from __future__ import division, absolute_import, print_function

import numpy as np
import matplotlib.pyplot as plt

# import matplotlib
# matplotlib.rcParams.update({'font.size': 3})
import matplotlib.gridspec as gridspec

# pyEIT 2D algorithm modules
import pyeit.mesh as mesh
from pyeit.eit.fem import Forward
from pyeit.eit.interp2d import sim2pts
from pyeit.eit.utils import eit_scan_lines

import pyeit.eit.greit as greit
import pyeit.eit.bp as bp
import pyeit.eit.jac as jac

""" 0. construct mesh structure """
mesh_obj, el_pos = mesh.create(16, h0=0.08)

# extract node, element, permittivity
pts = mesh_obj["node"]
tri = mesh_obj["element"]

""" 1. problem setup """
# test function for altering the permittivity in mesh
anomaly = [
    {"x": 0.4, "y": 0, "d": 0.1, "perm": 5},
    {"x": -0.4, "y": 0, "d": 0.1, "perm": 5},
    {"x": 0, "y": 0.5, "d": 0.1, "perm": 0.1},
    {"x": 0, "y": -0.5, "d": 0.1, "perm": 0.1},
]
mesh_new = mesh.set_perm(mesh_obj, anomaly=anomaly, background=1.0)
delta_perm = np.real(mesh_new["perm"] - mesh_obj["perm"])

""" ax1. FEM forward simulations """
# setup EIT scan conditions
el_dist, step = 1, 1
ex_mat = eit_scan_lines(16, el_dist)

# calculate simulated data
fwd = Forward(mesh_obj, el_pos)
f0 = fwd.solve_eit(ex_mat, step=step, perm=mesh_obj["perm"])
f1 = fwd.solve_eit(ex_mat, step=step, perm=mesh_new["perm"])

""" ax2. BP """
eit = bp.BP(mesh_obj, el_pos, ex_mat=ex_mat, step=1, parser="std")
ds = eit.solve(f1.v, f0.v, normalize=True)
ds_bp = ds

""" ax3. JAC """
eit = jac.JAC(mesh_obj, el_pos, ex_mat=ex_mat, step=step, perm=1.0, parser="std")
# parameter tuning is needed for better EIT images
eit.setup(p=0.5, lamb=0.1, method="kotre")
# if the jacobian is not normalized, data may not to be normalized too.
ds = eit.solve(f1.v, f0.v, normalize=False)
ds_jac = sim2pts(pts, tri, ds)

""" ax4. GREIT """
eit = greit.GREIT(mesh_obj, el_pos, ex_mat=ex_mat, step=step, parser="std")
# parameter tuning is needed for better EIT images
eit.setup(p=0.5, lamb=0.01)
ds = eit.solve(f1.v, f0.v, normalize=False)
x, y, ds_greit = eit.mask_value(ds, mask_value=np.NAN)

""" build for EIT2016b (orig: 300p x 300p, 150dpi) """
size = (8, 6)
axis_size = [-1.2, 1.2, -1.2, 1.2]
im_size = [-2, 34, -2, 34]
fig = plt.figure(figsize=size)
gs = gridspec.GridSpec(2, 2)

# simulation
pmax = np.max(np.abs(delta_perm))
ax1 = fig.add_subplot(gs[0, 0])
im1 = ax1.tripcolor(
    pts[:, 0],
    pts[:, 1],
    tri,
    delta_perm,
    shading="flat",
    cmap=plt.cm.RdBu,
    vmin=-pmax,
    vmax=pmax,
)
ax1.set_title(r"(a) $\Delta$ Conductivity")
ax1.axis(axis_size)
ax1.set_aspect("equal")
fig.colorbar(im1)
ax1.axis("off")

# Filtered BP
bp_max = np.max(np.abs(ds_bp))
ax2 = fig.add_subplot(gs[0, 1])
im2 = ax2.tripcolor(
    pts[:, 0],
    pts[:, 1],
    tri,
    np.real(ds_bp),
    cmap=plt.cm.RdBu,
    vmin=-bp_max,
    vmax=bp_max,
)
ax2.set_title(r"(b) BP")
ax2.axis(axis_size)
ax2.set_aspect("equal")
fig.colorbar(im2)
ax2.axis("off")

# JAC
jac_max = np.max(np.abs(ds_jac))
ax3 = fig.add_subplot(gs[1, 0])
im3 = ax3.tripcolor(
    pts[:, 0],
    pts[:, 1],
    tri,
    np.real(ds_jac),
    cmap=plt.cm.RdBu,
    vmin=-jac_max,
    vmax=jac_max,
)
ax3.set_title(r"(c) JAC")
ax3.axis(axis_size)
ax3.set_aspect("equal")
fig.colorbar(im3)
ax3.axis("off")

# GREIT
gr_max = np.max(np.abs(ds_greit))
ax4 = fig.add_subplot(gs[1, 1])
im4 = ax4.imshow(
    np.real(ds_greit),
    interpolation="nearest",
    cmap=plt.cm.RdBu,
    vmin=-gr_max,
    vmax=gr_max,
)
ax4.set_title(r"(d) GREIT")
ax4.axis(im_size)
ax4.set_aspect("equal")
fig.colorbar(im4)
ax4.axis("off")

# save
plt.show()
# fig.tight_layout()
# fig.subplots_adjust(top=0.875, bottom=0.01)
# fig.set_size_inches(1, 1)
# fig.savefig('eit2016b.png', dpi=300)
