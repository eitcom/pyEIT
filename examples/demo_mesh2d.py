# coding: utf-8
# author: benyuan liu
""" demo on creating triangle meshes using mesh2d in EIT """
from __future__ import division, absolute_import, print_function

import numpy as np
import matplotlib.pyplot as plt

from pyeit.mesh import wrapper
from pyeit.eit.pde import pdeprtni

""" 0. create mesh """
ms, el_pos = wrapper.create(16, h0=0.1)

# extract nodes and triangles (truss)
no2xy = ms['node']
el2no = ms['element']

# plot the mesh
fig, ax = plt.subplots(figsize=(6, 4))
ax.triplot(no2xy[:, 0], no2xy[:, 1], el2no, linewidth=1)
ax.plot(no2xy[el_pos, 0], no2xy[el_pos, 1], 'ro')
ax.axis('equal')
ax.axis([-1.2, 1.2, -1.2, 1.2])
ax.set_xlabel('x')
ax.set_ylabel('y')
title_src = 'number of triangles = ' + str(np.size(el2no, 0)) + ', ' + \
            'number of nodes = ' + str(np.size(no2xy, 0))
ax.set_title(title_src)
plt.show()

""" 1. a simple function for adding anomaly regions """
anomaly = [{'x': 0.5, 'y': 0.5, 'd': 0.2, 'alpha': 10},
           {'x': -0.2, 'y': -0.2, 'd': 0.4, 'alpha': 20}]
ms0 = wrapper.set_alpha(ms, anomaly=anomaly, background=1.)

anomaly = [{'x': 0.5, 'y': 0.5, 'd': 0.2, 'alpha': 20},
           {'x': -0.2, 'y': -0.2, 'd': 0.4, 'alpha': 10}]
ms1 = wrapper.set_alpha(ms, anomaly=anomaly, background=1.)

# show alpha on nodes (reverse interp)
ele_ds = (ms1['alpha'] - ms0['alpha'])
node_ds = pdeprtni(no2xy, el2no, ele_ds)

# plot
fig, ax = plt.subplots(figsize=(6, 4))
# tripcolor shows element (shading='flat') or nodes (shading='gouraud')
ax.tripcolor(no2xy[:, 0], no2xy[:, 1], el2no, np.real(node_ds),
             shading='gouraud', alpha=0.8, cmap=plt.cm.viridis)
# tricontour only interpolates values on nodes
ax.tricontour(no2xy[:, 0], no2xy[:, 1], el2no, np.real(node_ds),
              shading='flat', alpha=1.0, linewidths=1,
              cmap=plt.cm.viridis)
ax.axis('equal')
ax.axis([-1.2, 1.2, -1.2, 1.2])
plt.show()
