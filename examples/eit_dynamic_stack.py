# coding: utf-8
""" demo using stacked ex_mat (the devil is in the details) """
# Copyright (c) Benyuan Liu. All Rights Reserved.
# Distributed under the (new) BSD License. See LICENSE.txt for more info.
from __future__ import division, absolute_import, print_function

import numpy as np
import matplotlib.pyplot as plt

# pyEIT 2D algorithm modules
import pyeit.mesh as mesh
from pyeit.eit.fem import Forward
from pyeit.eit.utils import eit_scan_lines
import pyeit.eit.jac as jac

""" 1. setup """
mesh_obj, el_pos = mesh.create(16)

# test function for altering the permittivity in mesh
anomaly = [{"x": 0.4, "y": 0.4, "d": 0.2, "perm": 100}]
mesh_new = mesh.set_perm(mesh_obj, anomaly=anomaly, background=1.0)

# extract node, element, alpha
pts = mesh_obj["node"]
tri = mesh_obj["element"]
delta_perm = mesh_new["perm"] - mesh_obj["perm"]

# show alpha
fig, ax = plt.subplots(figsize=(6, 4))
im = ax.tripcolor(pts[:, 0], pts[:, 1], tri, np.real(delta_perm), shading="flat")
fig.colorbar(im)
ax.set_aspect("equal")
ax.set_title(r"$\Delta$ Permittivity")

""" 2. calculate simulated data using stack ex_mat """
el_dist, step = 7, 1
n_el = len(el_pos)
ex_mat1 = eit_scan_lines(n_el, el_dist)
ex_mat2 = eit_scan_lines(n_el, 1)
ex_mat = np.vstack([ex_mat1, ex_mat2])

# forward solver
fwd = Forward(mesh_obj, el_pos)
f0 = fwd.solve_eit(ex_mat, step, perm=mesh_obj["perm"])
f1 = fwd.solve_eit(ex_mat, step, perm=mesh_new["perm"])

""" 3. solving using dynamic EIT """
# number of stimulation lines/patterns
eit = jac.JAC(mesh_obj, el_pos, ex_mat=ex_mat, step=step, parser="std")
eit.setup(p=0.40, lamb=1e-3, method="kotre")
ds = eit.solve(f1.v, f0.v, normalize=False)

""" 4. plot """
fig, ax = plt.subplots(figsize=(6, 4))
im = ax.tripcolor(
    pts[:, 0],
    pts[:, 1],
    tri,
    np.real(ds),
    shading="flat",
    alpha=0.90,
    cmap=plt.cm.viridis,
)
fig.colorbar(im)
ax.set_aspect("equal")
ax.set_title(r"$\Delta$ Permittivity Reconstructed")
# plt.savefig('quasi-demo-eit.pdf')
