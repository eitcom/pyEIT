""" demo on dynamic eit using JAC method """

import numpy as np
import matplotlib.pyplot as plt

import pyeit.mesh as mesh
from pyeit.eit.fem import Forward
from pyeit.eit.utils import eit_scan_lines
import pyeit.eit.jac as jac

""" 0. construct mesh """
ms, el_pos = mesh.create(16, h0=0.1)

# extract node, element, alpha
no2xy = ms['node']
el2no = ms['element']

""" 1. problem setup """
anomaly = [{'x': 0.5, 'y': 0.5, 'd': 0.1, 'alpha': 10.0}]
ms1 = mesh.set_alpha(ms, anomaly=anomaly, background=1.0)

""" 2. FEM simulation """
el_dist, step = 1, 1
ex_mat = eit_scan_lines(16, el_dist)

# calculate simulated data
fwd = Forward(ms, el_pos)
f0 = fwd.solve(ex_mat, step=step, perm=ms['alpha'])
f1 = fwd.solve(ex_mat, step=step, perm=ms1['alpha'])

""" 3. JAC solver """
# number of excitation lines & excitation patterns
eit = jac.JAC(ms, el_pos, ex_mat=ex_mat, step=step,
              perm=1., parser='std')
eit.setup(p=0.30, lamb=1e-4, method='kotre')
ds = eit.solve(f1.v, f0.v)

# plot
fig, ax = plt.subplots(figsize=(6, 4))
im = ax.tripcolor(no2xy[:, 0], no2xy[:, 1], el2no, np.real(ds),
                  shading='flat', cmap=plt.cm.viridis)
fig.colorbar(im)
ax.axis('equal')
# fig.set_size_inches(6, 4)
# plt.savefig('../figs/demo_jac.png', dpi=96)
plt.show()
