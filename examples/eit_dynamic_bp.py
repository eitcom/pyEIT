# coding: utf-8
""" demo code for back-projection """
# Copyright (c) Benyuan Liu. All Rights Reserved.
# Distributed under the (new) BSD License. See LICENSE.txt for more info.
from __future__ import absolute_import, division, print_function

import matplotlib.pyplot as plt
import numpy as np
import pyeit.eit.bp as bp
import pyeit.eit.protocol as protocol
import pyeit.mesh as mesh
from pyeit.eit.fem import EITForward
from pyeit.mesh.shape import thorax
from pyeit.mesh.wrapper import PyEITAnomaly_Circle

""" 0. build mesh """
n_el = 16  # nb of electrodes
use_customize_shape = False
if use_customize_shape:
    # Mesh shape is specified with fd parameter in the instantiation, e.g : fd=thorax
    mesh_obj = mesh.create(n_el, h0=0.1, fd=thorax)
else:
    mesh_obj = mesh.create(n_el, h0=0.1)

""" 1. problem setup """
anomaly = PyEITAnomaly_Circle(center=[0.5, 0.5], r=0.1, perm=10.0)
mesh_new = mesh.set_perm(mesh_obj, anomaly=anomaly, background=1.0)

""" 2. FEM forward simulations """
# setup EIT scan conditions
# adjacent stimulation (dist_exc=1), adjacent measures (step_meas=1)
protocol_obj = protocol.create(n_el, dist_exc=1, step_meas=1, parser_meas="std")

# calculate simulated data
fwd = EITForward(mesh_obj, protocol_obj)
v0 = fwd.solve_eit()
v1 = fwd.solve_eit(perm=mesh_new.perm)

""" 3. naive inverse solver using back-projection """
eit = bp.BP(mesh_obj, protocol_obj)
eit.setup(weight="none")
# the normalize for BP when dist_exc>4 should always be True
ds = 192.0 * eit.solve(v1, v0, normalize=True)

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
