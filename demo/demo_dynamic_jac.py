""" demo on dynamic eit using JAC method """

import numpy as np
import matplotlib.pyplot as plt

import pyeit.mesh as mesh
from pyeit.eit.fem import forward
from pyeit.eit.utils import eit_scan_lines
import pyeit.eit.jac as jac

""" 0. construct mesh """
ms, elPos = mesh.create(16, h0=0.1)

# extract node, element, alpha
no2xy = ms['node']
el2no = ms['element']

""" 1. problem setup """
anomaly = [{'x': 0.5, 'y': 0.5, 'd': 0.1, 'alpha': 10.0}]
ms1 = mesh.set_alpha(ms, anom=anomaly, background=1.0)

""" 2. FEM simulation """
elDist, step = 1, 1
exMtx = eit_scan_lines(16, elDist)

# calculate simulated data
fwd = forward(ms, elPos)
f0 = fwd.solve(exMtx, step=step, perm=ms['alpha'])
f1 = fwd.solve(exMtx, step=step, perm=ms1['alpha'])

""" 3. JAC solver """
# number of excitation lines & excitation patterns
eit = jac.JAC(ms, elPos, exMtx=exMtx, step=step,
              perm=1., parser='std',
              p=0.25, lamb=1e-4, method='kotre')
ds = eit.solve(f1.v, f0.v)
# static EIT
# ds = eit.gn_solve(f0.v, maxiter=5)

# plot
fig = plt.figure()
plt.tripcolor(no2xy[:, 0], no2xy[:, 1], el2no, np.real(ds),
              shading='flat', cmap=plt.cm.viridis)
plt.colorbar()
plt.axis('equal')
fig.set_size_inches(6, 4)
# plt.savefig('../figs/demo_jac.png', dpi=96)
plt.show()
