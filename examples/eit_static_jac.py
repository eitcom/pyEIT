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
import pyeit.eit.jac as jac

# Mesh shape is specified with fd parameter in the instantiation, e.g:
# from pyeit.mesh.shape import thorax
# mesh_obj, el_pos = create(n_el, h0=0.05, fd=thorax)  # Default : fd=circle
n_el = 64  # test fem_vectorize
mesh_obj, el_pos = create(n_el, h0=0.05)
# set anomaly (altering the permittivity in the mesh)
anomaly = [
    {"x": 0.4, "y": 0.4, "d": 0.2, "perm": 10},
    {"x": -0.4, "y": -0.4, "d": 0.2, "perm": 0.1},
]
# background changed to values other than 1.0 requires more iterations
mesh_new = set_perm(mesh_obj, anomaly=anomaly, background=2.0)
# extract node, element, perm
xx, yy = mesh_obj["node"][:, 0], mesh_obj["node"][:, 1]
tri = mesh_obj["element"]
perm = mesh_new["perm"]

# %% calculate simulated data
el_dist, step = 1, 1
ex_mat = eit_scan_lines(n_el, el_dist)
fwd = Forward(mesh_obj, el_pos)
f1 = fwd.solve_eit(ex_mat, step, perm=mesh_new["perm"], parser="std", vector=True)

# plot
fig, ax = plt.subplots(figsize=(9, 6))
im = ax.tripcolor(xx, yy, tri, np.real(perm), cmap="viridis")
for el in el_pos:
    ax.plot(xx[el], yy[el], "ro")
ax.axis("equal")
ax.set_title(r"$\Delta$ Conductivities")

# %% solve_eit using gaussian-newton (with regularization)
# number of stimulation lines/patterns
eit = jac.JAC(mesh_obj, el_pos, ex_mat, step, perm=1.0, parser="std")
eit.setup(p=0.25, lamb=1.0, method="lm")
# lamb = lamb * lamb_decay
ds = eit.gn(f1.v, lamb_decay=0.1, lamb_min=1e-5, maxiter=20, verbose=True, vector=True)

# plot
fig, ax = plt.subplots(figsize=(9, 6))
im = ax.tripcolor(xx, yy, tri, np.real(ds), alpha=1.0, cmap="viridis")
ax.axis("equal")
ax.set_title("Conductivities Reconstructed")
fig.colorbar(im)
# fig.savefig('../doc/images/demo_static.png', dpi=96)
plt.show()
