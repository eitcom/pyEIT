""" demo on dynamic eit using JAC method """

import numpy as np
import matplotlib.pyplot as plt

from pyeit.mesh import distmesh2d
from pyeit.eit.fem import forward
from pyeit.eit.utils import eit_scan_lines
import pyeit.eit.jac as jac

""" 0. construct mesh """
ms, elPos = distmesh2d.create(16, h0=0.1)

# extract node, element, alpha
no2xy = ms['node']
el2no = ms['element']

""" 1. problem setup """
anomaly = [{'x': 0.5, 'y': 0.5, 'd': 0.1, 'alpha': 100.0}]
ms1 = distmesh2d.set_alpha(ms, anom=anomaly, background=1.0)

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
              perm=0.01, parser='std',
              p=0.10, lamb=0.01, method='kotre')
ds = eit.solve(f1.v, f0.v)
# static EIT
# ds = eit.gn_solve(f0.v, maxiter=5)

# plot
fig = plt.figure()
plt.tripcolor(no2xy[:, 0], no2xy[:, 1], el2no, np.real(ds),
              shading='flat', cmap=plt.cm.Blues)
plt.colorbar()
plt.axis('tight')
# fig.set_size_inches(4, 3)
# plt.savefig('demo_jac.png', dpi=96)
