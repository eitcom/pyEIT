# pylint: disable=invalid-name, no-member
"""demo files of reading/processing TAAR/DHCA data"""
# Copyright (c) Benyuan Liu. All Rights Reserved.
# Distributed under the (new) BSD License. See LICENSE.txt for more info.
import json
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.pyplot import cm

from pyeit.io import ET3, mes
from pyeit.feature_extraction import ati
from pyeit.eit import jac
from pyeit.eit.interp2d import sim2pts, meshgrid, weight_idw, tri_area
from pyeit.mesh import layer_circle

from pyeit.app.plot import mesh_plot, ts_plot
import pkg_resources


# load mesh from .mes
mesh_file = pkg_resources.resource_filename('pyeit', 'data/model/DLS2.mes')
mesh_obj, el_pos = mes.load(mesh_file)

# build mesh
mesh_obj, el_pos = layer_circle(n_layer=8, n_fan=6)

pts = mesh_obj['node']
tri = mesh_obj['element']

# load data and convert to pandas.DataFrame
data_file = '../../../datasets/dhca/DATA.et3'
et3 = ET3(data_file)
df = et3.to_df()

# convert complex values to impedance
# df.update(np.abs(df.values), overwrite=True)

# load information of person (json)
info_file = '../../../datasets/dhca/info.json'
with open(info_file, encoding='utf-8') as json_data_file:
    info = json.load(json_data_file)

# dynamic eit imaging
t1 = info['eit_view'][1]
t0 = info['eit_view'][0]

v1 = df.loc[t1]
v0 = df.loc[t0]
# v1 = df.iloc[9987]
# v0 = df.iloc[8245]

v1 = np.real(v1)
v0 = np.real(v0)
# see boundary measurements
# fig, ax = plt.subplots(figsize=(12, 3))
# ax.plot(v1, 'r-')
# ax.plot(v0, 'b-')
# axt = ax.twinx()
# axt.plot(v1 - v0, 'm-')
# ax.grid('on')

# reconstruct using Jacobian without jac_normalization
eit = jac.JAC(mesh_obj, el_pos, jac_normalized=True, parser='fmmu')
eit.setup(p=0.5, lamb=1.0, method='lm')
ds = eit.solve(v1, v0, normalize=True)

a = np.abs(tri_area(pts, tri))

# plot the real-part
ds = np.real(ds)
ds_n = sim2pts(pts, tri, ds)

ds_max = np.max(np.abs(ds_n)) * 1.01

# plot
fig, ax = mesh_plot(mesh_obj, el_pos, show_number=True)
# colormap can be bwr_r
im = ax.tripcolor(pts[:, 0], pts[:, 1], tri, ds_n,
                  shading='flat', cmap=cm.jet_r, edgecolors='none',
                  linewidth=0,
                  alpha=0.90, vmax=ds_max, vmin=-ds_max)
plt.colorbar(im)

# re-weight on uniform grids
fig, ax = plt.subplots(figsize=(9, 6))
# ax.triplot(pts[:, 0], pts[:, 1], tri, color='k', alpha=0.2)
ax.set_aspect('equal')
ax.invert_yaxis()

# Note: below are the code that mapping values on elements to regular grids
xg, yg, mask = meshgrid(pts, n=32)
im = np.ones_like(mask)
# weight
xy = np.mean(pts[tri], axis=1)
xyi = np.vstack((xg.flatten(), yg.flatten())).T
w_mat = weight_idw(xy, xyi)
# w_mat = weight_sigmod(xy, xyi)
im = np.dot(w_mat.T, ds)
# im = weight_linear_rbf(xy, xyi, mesh_new['perm'])
im[mask] = np.NaN

# imshow
im = im.reshape(xg.shape)
im_max = np.nanmax(np.abs(im)) * 1.01
cmap = cm.RdBu
cmap.set_bad(color='black')
# aim = ax.pcolor(xg, yg, im, edgecolors=None, linewidth=0,
#                 cmap=cm.RdBu, alpha=0.99, vmax=im_max, vmin=-im_max)
aim = ax.imshow(im, cmap=cmap, vmax=im_max, vmin=-im_max)
plt.colorbar(aim)

# annotate
n = im.shape[0]
ax.text(n-3, n-5, 'L', size=24, color='w')
ax.axis('off')

# plot time series data (ATI)
a = df.apply(ati, axis=1)
fig2, ax2 = ts_plot(a, figsize=(9, 6), ylim=[7.5, 11])

# done
plt.show()
