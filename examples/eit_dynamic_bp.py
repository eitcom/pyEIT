# coding: utf-8
""" demo code for back-projection """
# Copyright (c) Benyuan Liu. All Rights Reserved.
# Distributed under the (new) BSD License. See LICENSE.txt for more info.
from __future__ import division, absolute_import, print_function

import numpy as np
import matplotlib.pyplot as plt

import pyeit.mesh as mesh
from pyeit.eit.fem import EITForward
from pyeit.eit.utils import eit_scan_lines
from pyeit.mesh.shape import thorax
import pyeit.eit.bp as bp

""" 0. build mesh """
use_customize_shape = False
if use_customize_shape:
    # Mesh shape is specified with fd parameter in the instantiation, e.g : fd=thorax
    mesh_obj = mesh.create(16, h0=0.1, fd=thorax)
else:
    mesh_obj = mesh.create(16, h0=0.1)

""" 1. problem setup """
anomaly = [{"x": 0.5, "y": 0.5, "d": 0.1, "perm": 10.0}]
mesh_new = mesh.set_perm(mesh_obj, anomaly=anomaly, background=1.0)

""" 2. FEM forward simulations """
# setup EIT scan conditions
# adjacent stimulation (el_dist=1), adjacent measures (step=1)
el_dist, step = 1, 1
ex_mat = eit_scan_lines(16, el_dist)
protocol = {"ex_mat": ex_mat, "step": step, "parser": "std"}

# calculate simulated data
fwd = EITForward(mesh_obj, protocol)
v0 = fwd.solve_eit()
v1 = fwd.solve_eit(perm=mesh_new.perm, init=True)

""" 3. naive inverse solver using back-projection """
eit = bp.BP(mesh_obj, protocol)
eit.setup(weight="none")
ds = 192.0 * eit.solve(v1, v0, normalize=False)

# extract node, element, alpha
pts = mesh_obj.node
tri = mesh_obj.element

# draw
fig, axes = plt.subplots(2, 1, constrained_layout=True, figsize=(6, 9))
# original
ax = axes[0]
ax.axis("equal")
ax.set_title(r"Input $\Delta$ Conductivities")
delta_perm = np.real(mesh_new.perm - mesh_obj.perm)
im = ax.tripcolor(pts[:, 0], pts[:, 1], tri, delta_perm, shading="flat")
# reconstructed
ax1 = axes[1]
im = ax1.tripcolor(pts[:, 0], pts[:, 1], tri, ds)
ax1.set_title(r"Reconstituted $\Delta$ Conductivities")
ax1.axis("equal")
fig.colorbar(im, ax=axes.ravel().tolist())
# fig.savefig('../doc/images/demo_bp.png', dpi=96)
plt.show()
