""" demo on using mesh2d in EIT """

import numpy as np
import matplotlib.pyplot as plt

from pyeit.mesh import distmesh2d
from pyeit.eit.fem import pdeprtni

""" 0. create mesh """
ms, elPos = distmesh2d.create(16)

# extract nodes and triangles (truss)
no2xy = ms['node']
el2no = ms['element']

# plot the mesh
plt.figure()
plt.triplot(no2xy[:, 0], no2xy[:, 1], el2no)
plt.plot(no2xy[elPos, 0], no2xy[elPos, 1], 'ro')
plt.axis('equal')
plt.axis([-1.2, 1.2, -1.2, 1.2])
plt.xlabel('x')
plt.ylabel('y')
title_src = 'number of triangles = ' + str(np.size(el2no, 0)) + ', ' + \
            'number of nodes = ' + str(np.size(no2xy, 0))
plt.title(title_src)
plt.show()

""" 1. a simple function for adding anomaly regions """
anomaly = [{'x': 0.5, 'y': 0.5, 'd': 0.2, 'alpha': 100},
           {'x': -0.2, 'y': -0.2, 'd': 0.4, 'alpha': 200}]
ms0 = distmesh2d.set_alpha(ms, anom=anomaly, background=1.)

anomaly = [{'x': 0.5, 'y': 0.5, 'd': 0.2, 'alpha': 200},
           {'x': -0.2, 'y': -0.2, 'd': 0.4, 'alpha': 100}]
ms1 = distmesh2d.set_alpha(ms, anom=anomaly, background=1.)

# show alpha on nodes (reverse interp)
ele_ds = (ms1['alpha'] - ms0['alpha'])
node_ds = pdeprtni(no2xy, el2no, ele_ds)

# show
plt.figure()
# tripcolor shows values on element (if shading='flat')
#           shows values on nodes   (if shading='gouraud')
plt.tripcolor(no2xy[:, 0], no2xy[:, 1], el2no, np.real(ele_ds),
              shading='flat', alpha=0.50, cmap=plt.cm.viridis)
# tricontour only shows values on nodes
plt.tricontour(no2xy[:, 0], no2xy[:, 1], el2no, np.real(node_ds),
               shading='flat', alpha=0.80, linewidths=2)
plt.axis('equal')
plt.axis([-1.2, 1.2, -1.2, 1.2])
plt.show()
