# coding: utf-8
""" demo on static solving using JAC (experimental) """
# Copyright (c) Benyuan Liu. All Rights Reserved.
# Distributed under the (new) BSD License. See LICENSE.txt for more info.
from __future__ import division, absolute_import, print_function

import numpy as np
import matplotlib.pyplot as plt

# pyEIT 2D algorithms modules
from pyeit.mesh import create, set_perm
from pyeit.eit.fem import Forward
from pyeit.eit.utils import eit_scan_lines
from pyeit.mesh.shape import thorax
import pyeit.eit.jac as jac

""" 1. setup """
n_el = 16
# Mesh shape is specified with fd parameter in the instantiation, e.g : fd=thorax , Default :fd=circle
mesh_obj, el_pos = create(n_el, h0=0.05, fd=thorax)
# test function for altering the permittivity in mesh
anomaly = [
    {"x": 0.4, "y": 0.4, "d": 0.2, "perm": 10},
    {"x": -0.4, "y": -0.4, "d": 0.2, "perm": 0.1},
]
# background changed to values other than 1.0 requires more iterations
mesh_new = set_perm(mesh_obj, anomaly=anomaly, background=2.0)

# extract node, element, perm
pts = mesh_obj["node"]
tri = mesh_obj["element"]
perm = mesh_new["perm"]

# show
fig, axes = plt.subplots(1, 2, constrained_layout=True)
fig.set_size_inches(6, 4)

ax = axes[0]
im = ax.tripcolor(
    pts[:, 0], pts[:, 1], tri, np.real(perm), shading="flat", cmap=plt.cm.viridis
)
ax.axis("equal")
ax.set_title(r"$\Delta$ Conductivities")

""" 2. calculate simulated data """
el_dist, step = 1, 1
ex_mat = eit_scan_lines(n_el, el_dist)
fwd = Forward(mesh_obj, el_pos)
f1 = fwd.solve_eit(ex_mat, step, perm=mesh_new["perm"], parser="std")

""" 3. solve_eit using gaussian-newton (with regularization) """
# number of stimulation lines/patterns
eit = jac.JAC(mesh_obj, el_pos, ex_mat, step, perm=1.0, parser="std")
eit.setup(p=0.25, lamb=1.0, method="lm")
# lamb = lamb * lamb_decay
ds = eit.gn(f1.v, lamb_decay=0.1, lamb_min=1e-5, maxiter=20, verbose=True)

# plot
ax = axes[1]
im = ax.tripcolor(
    pts[:, 0],
    pts[:, 1],
    tri,
    np.real(ds),
    shading="flat",
    alpha=1.0,
    cmap=plt.cm.viridis,
)
ax.axis("equal")
ax.set_title("Conductivities Reconstructed")

fig.colorbar(im, ax=axes.ravel().tolist())
# fig.savefig('../doc/images/demo_static.png', dpi=96)
plt.show()
