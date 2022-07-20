# coding: utf-8
""" demo using stacked ex_mat (the devil is in the details) """
# Copyright (c) Benyuan Liu. All Rights Reserved.
# Distributed under the (new) BSD License. See LICENSE.txt for more info.
from __future__ import absolute_import, division, print_function

import matplotlib.pyplot as plt
import numpy as np
import pyeit.eit.jac as jac
import pyeit.eit.protocol as protocol

# pyEIT 2D algorithm modules
import pyeit.mesh as mesh
from pyeit.eit.fem import EITForward
from pyeit.mesh.shape import thorax
from pyeit.mesh.wrapper import PyEITAnomaly_Circle

""" 1. setup """
n_el = 16  # nb of electrodes
use_customize_shape = False
if use_customize_shape:
    # Mesh shape is specified with fd parameter in the instantiation, e.g : fd=thorax
    mesh_obj = mesh.create(n_el, h0=0.1, fd=thorax)
else:
    mesh_obj = mesh.create(n_el, h0=0.1)

# test function for altering the permittivity in mesh
anomaly = PyEITAnomaly_Circle(center=[0.4, 0.4], r=0.2, perm=100.0)
mesh_new = mesh.set_perm(mesh_obj, anomaly=anomaly, background=1.0)

""" 2. calculate simulated data using stack ex_mat """
protocol_obj = protocol.create(n_el, dist_exc=[7, 3], step_meas=1, parser_meas="std")

# forward solver
fwd = EITForward(mesh_obj, protocol_obj)
v0 = fwd.solve_eit()
v1 = fwd.solve_eit(perm=mesh_new.perm)

""" 3. solving using dynamic EIT """
# number of stimulation lines/patterns
eit = jac.JAC(mesh_obj, protocol_obj)
eit.setup(p=0.40, lamb=1e-3, method="kotre", jac_normalized=False)
ds = eit.solve(v1, v0, normalize=False)

# extract node, element, alpha
pts = mesh_obj.node
tri = mesh_obj.element
delta_perm = mesh_new.perm - mesh_obj.perm

# show alpha
fig, ax = plt.subplots(figsize=(6, 4))
im = ax.tripcolor(pts[:, 0], pts[:, 1], tri, np.real(delta_perm), shading="flat")
fig.colorbar(im)
ax.set_aspect("equal")
ax.set_title(r"$\Delta$ Permittivity")

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
plt.show()
