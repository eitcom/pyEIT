""" demo on bp """

import numpy as np

import pyeit.mesh as mesh
from pyeit.mesh import quality
import pyeit.mesh.plot as mplot
from pyeit.eit.fem import forward, pdeprtni
from pyeit.eit.utils import eit_scan_lines

# build tetrahedron
# 3D tetrahedron must have a bbox
bbox = [[-1, -1, -1], [1, 1, 1]]
# save calling convention as distmesh 2D
ms, elPos = mesh.create(h0=0.15, bbox=bbox)

no2xy = ms['node']
el2no = ms['element']

# report the status of the 2D mesh
quality.fstats(no2xy, el2no)

""" 1. FEM forward simulations """
# setup EIT scan conditions
elDist, step = 7, 1
exMtx = eit_scan_lines(16, elDist)

# calculate simulated data
fwd = forward(ms, elPos)

# in python, index start from 0
exLine = exMtx[2].ravel()

# change alpha
anomaly = [{'x': 0.20, 'y': 0.40, 'z': 0.0, 'd': 0.30, 'alpha': 100.0}]
ms_test = mesh.set_alpha(ms, anom=anomaly, background=1.0)
tri_perm = ms_test['alpha']
node_perm = pdeprtni(no2xy, el2no, np.real(tri_perm))

# solving once using fem
f, _ = fwd.solve_once(exLine, tri_perm)
f = np.real(f)

# mplot.tetplot(p, t, edge_color=(0.2, 0.2, 1.0, 1.0), alpha=0.01)
mplot.tetplot(no2xy, el2no, vertex_color=f, alpha=0.1)
