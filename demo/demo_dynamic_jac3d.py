""" demo on JAC 3D, extremely slow """

import numpy as np

import pyeit.mesh as mesh
from pyeit.mesh import quality
import pyeit.mesh.plot as mplot
from pyeit.eit.fem import Forward
from pyeit.eit.pde import pdeprtni
from pyeit.eit.utils import eit_scan_lines
import pyeit.eit.jac as jac

# build tetrahedron
# 3D tetrahedron must have a bbox
bbox = [[-1, -1, -1], [1, 1, 1]]
# save calling convention as distmesh 2D
ms, el_pos = mesh.create(h0=0.2, bbox=bbox)

no2xy = ms['node']
el2no = ms['element']

# report the status of the 2D mesh
quality.stats(no2xy, el2no)

""" 1. FEM forward simulations """
# setup EIT scan conditions
el_dist, step = 7, 1
ex_mat = eit_scan_lines(16, el_dist)

# calculate simulated data
fwd = Forward(ms, el_pos)

# in python, index start from 0
ex_line = ex_mat[2].ravel()

# change alpha
anomaly = [{'x': 0.40, 'y': 0.40, 'z': 0.0, 'd': 0.30, 'alpha': 100.0}]
ms_test = mesh.set_alpha(ms, anomaly=anomaly, background=1.0)
tri_perm = ms_test['alpha']
node_perm = pdeprtni(no2xy, el2no, np.real(tri_perm))

# solving once using fem
# f, _ = fwd.solve_once(ex_line, tri_perm)
# f = np.real(f)

# calculate simulated data
f0 = fwd.solve(ex_mat, step=step, perm=ms['alpha'])
f1 = fwd.solve(ex_mat, step=step, perm=ms_test['alpha'])

""" 3. JAC solver """
# number of excitation lines & excitation patterns
eit = jac.JAC(ms, el_pos, ex_mat=ex_mat, step=step, perm=1., parser='std')
eit.setup(p=0.50, lamb=1e-4, method='kotre')
ds = eit.solve(f1.v, f0.v)
node_ds = pdeprtni(no2xy, el2no, np.real(ds))

# mplot.tetplot(p, t, edge_color=(0.2, 0.2, 1.0, 1.0), alpha=0.01)
mplot.tetplot(no2xy, el2no, vertex_color=node_ds, alpha=1.0)
