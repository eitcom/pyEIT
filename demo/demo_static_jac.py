""" demo on static solving using JAC (experimental) """

import numpy as np
import matplotlib.pyplot as plt

# pyEIT 2D algo modules
from pyeit.mesh.distmesh2d import create, set_alpha
from pyeit.eit.fem import forward
from pyeit.eit.utils import eit_scan_lines
import pyeit.eit.jac as jac

""" 1. setup """
numEl = 16
ms, elPos = create(numEl)

# test function for altering the 'alpha' in mesh dictionary
anomaly = [{'x': 0.2, 'y': 0.2, 'd': 0.2, 'alpha': 10},
           {'x': -0.3, 'y': -0.3, 'd': 0.2, 'alpha': 0.1}]
# TODO: even if background changed to values other than 1.0 will fail
ms1 = set_alpha(ms, anom=anomaly, background=1.)

# extract node, element, alpha
no2xy = ms['node']
el2no = ms['element']
alpha = ms1['alpha']

# show alpha
fig = plt.figure()
plt.tripcolor(no2xy[:, 0], no2xy[:, 1], el2no,
              np.real(alpha), shading='flat')
plt.colorbar()
plt.axis('equal')
plt.axis([-1.2, 1.2, -1.2, 1.2])
plt.title(r'\Delta Conductivities')

""" 2. calculate simulated data """
elDist, step = 1, 1
exMtx = eit_scan_lines(numEl, elDist)
fwd = forward(ms, elPos)
f1 = fwd.solve(exMtx, step, perm=ms1['alpha'])

""" 3. solve using gaussian-newton """
# number of excitation lines & excitation patterns
eit = jac.JAC(ms, elPos, exMtx, step,
              perm=1., parser='std',
              p=0.20, lamb=0.001, method='kotre')
ds = eit.gn_solve(f1.v, maxiter=6, verbose=True)

# plot
fig = plt.figure()
plt.tripcolor(no2xy[:, 0], no2xy[:, 1], el2no, np.real(ds),
              shading='flat', alpha=1.0, cmap=plt.cm.viridis)
plt.colorbar()
plt.axis('equal')
plt.axis([-1.2, 1.2, -1.2, 1.2])
plt.title(r'Conductivities Reconstructed')
# fig.set_size_inches(4, 3)
# fig.savefig('demo_static.png', dpi=96)
