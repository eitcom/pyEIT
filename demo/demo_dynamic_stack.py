# demo using stacked exMtx (the devil is in the details)
# liubenyuan@gmail.com
# 2015-07-23

import numpy as np
import matplotlib.pyplot as plt

# pyEIT 2D algo modules
import pyeit.mesh as mesh
from pyeit.eit.fem import forward
from pyeit.eit.utils import eit_scan_lines
import pyeit.eit.jac as jac

""" 1. setup """
ms, elPos = mesh.create(16)

# test function for altering the 'alpha' in mesh dictionary
anomaly = [{'x': 0.4, 'y': 0.4, 'd': 0.2, 'alpha': 100}]
ms1 = mesh.set_alpha(ms, anom=anomaly, background=1.)

# extract node, element, alpha
no2xy = ms['node']
el2no = ms['element']
alpha = ms1['alpha'] - ms['alpha']

# show alpha
fig = plt.figure()
plt.tripcolor(no2xy[:, 0], no2xy[:, 1], el2no,
              np.real(alpha), shading='flat')
plt.colorbar()
plt.title(r'$\Delta$ Permitivity')
fig.set_size_inches(6, 4.5)

""" 2. calculate simulated data using stack exMtx """
elDist, step = 7, 1
numEl = len(elPos)
exMtx1 = eit_scan_lines(numEl, elDist)
exMtx2 = eit_scan_lines(numEl, 1)
exMtx = np.vstack([exMtx1, exMtx2])

# forward solver
fwd = forward(ms, elPos)
f0 = fwd.solve(exMtx, step, perm=ms['alpha'])
f1 = fwd.solve(exMtx, step, perm=ms1['alpha'])

""" 3. solving using dynamic EIT """
# number of excitation lines & excitation patterns
eit = jac.JAC(ms, elPos, exMtx=exMtx, step=step,
              p=0.40, lamb=1e-3,
              parser='std', method='kotre')
ds = eit.solve(f1.v, f0.v)

""" 4. plot """
fig = plt.figure()
plt.tripcolor(no2xy[:, 0], no2xy[:, 1], el2no, np.real(ds),
              shading='flat', alpha=0.90, cmap=plt.cm.viridis)
plt.colorbar()
plt.axis('tight')
plt.title(r'$\Delta$ Permitivity Reconstructed')
fig.set_size_inches(6, 4)
# plt.savefig('quasi-demo-eit.pdf')
