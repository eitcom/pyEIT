# coding: utf-8
""" demo on creating triangle meshes using mesh2d in EIT """
# Copyright (c) Benyuan Liu. All Rights Reserved.
# Distributed under the (new) BSD License. See LICENSE.txt for more info.
from __future__ import division, absolute_import, print_function

import numpy as np
import matplotlib.pyplot as plt

from pyeit.mesh import wrapper
from pyeit.eit.interp2d import sim2pts

""" 0. create mesh """
mesh_obj, el_pos = wrapper.create(16, h0=0.1)

# extract nodes and triangles (truss)
pts = mesh_obj["node"]
tri = mesh_obj["element"]

# plot the mesh
fig, ax = plt.subplots(figsize=(6, 4))
ax.triplot(pts[:, 0], pts[:, 1], tri, linewidth=1)
ax.plot(pts[el_pos, 0], pts[el_pos, 1], "ro")
ax.axis("equal")
ax.axis([-1.2, 1.2, -1.2, 1.2])
ax.set_xlabel("x")
ax.set_ylabel("y")
title_src = (
    "number of triangles = "
    + str(np.size(tri, 0))
    + ", "
    + "number of nodes = "
    + str(np.size(pts, 0))
)
ax.set_title(title_src)
plt.show()

""" 1. a simple function for adding anomaly regions """
anomaly = [
    {"x": 0.5, "y": 0.5, "d": 0.2, "perm": 10},
    {"x": -0.2, "y": -0.2, "d": 0.4, "perm": 20},
]
ms0 = wrapper.set_perm(mesh_obj, anomaly=anomaly, background=1.0)

anomaly = [
    {"x": 0.5, "y": 0.5, "d": 0.2, "perm": 20},
    {"x": -0.2, "y": -0.2, "d": 0.4, "perm": 10},
]
ms1 = wrapper.set_perm(mesh_obj, anomaly=anomaly, background=1.0)

# show delta permittivity on nodes (reverse interp)
ele_ds = ms1["perm"] - ms0["perm"]
node_ds = sim2pts(pts, tri, ele_ds)

# plot
fig, ax = plt.subplots(figsize=(6, 4))
# tripcolor shows values on nodes (shading='flat' or 'gouraud')
im = ax.tripcolor(
    pts[:, 0],
    pts[:, 1],
    tri,
    np.real(node_ds),
    edgecolor="k",
    shading="flat",
    alpha=0.8,
    cmap=plt.cm.RdBu,
)
# 'tricontour' interpolates values on nodes, for example
# ax.tricontour(pts[:, 0], pts[:, 1], tri, np.real(node_ds),
#               shading='flat', alpha=1.0, linewidths=1,
#               cmap=plt.cm.RdBu)
fig.colorbar(im)
ax.axis("equal")
ax.axis([-1.2, 1.2, -1.2, 1.2])
plt.show()
