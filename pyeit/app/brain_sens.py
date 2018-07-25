# coding: utf-8
# byliu@fmmu.edu.cn
#
# theme:
# 1. sensitivity analysis
# 2. fused image augmentation
# 3. symmetric preserving quasi-static imaging and its application
#
""" demo on sensitivity analysis of 2D mesh"""
# Copyright (c) Benyuan Liu. All Rights Reserved.
# Distributed under the (new) BSD License. See LICENSE.txt for more info.
from __future__ import division, absolute_import, print_function

# numeric
import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

# pyEIT
import pyeit.mesh as mesh
from pyeit.eit.interp2d import tri_area, sim2pts
from pyeit.mesh import quality
from pyeit.eit.fem import Forward
from pyeit.eit.utils import eit_scan_lines

# fmmu-repo
from pyeit.io import mes
import pkg_resources
mes_file = pkg_resources.resource_filename('pyeit', 'data/model/I0007.mes')


""" 0. build mesh """
# mesh_obj, el_pos = mesh.layer_circle(n_layer=10, n_fan=6)
# mesh_obj, el_pos = mesh.create(h0=0.08)
mesh_obj, el_pos = mes.load(fstr=mes_file)

# extract node, element, alpha
pts = mesh_obj['node']
tri = mesh_obj['element']
x, y = pts[:, 0], pts[:, 1]
quality.stats(pts, tri)
cx, cy = np.mean(x), np.mean(y)
r = np.sqrt((x - cx)**2 + (y - cy)**2)
r_c1 = np.max(r) * 0.3
r_c2 = np.max(r) * 0.675


def calc_sens(fwd, ex_mat, keep=None):
    """
    see Adler2017 on IEEE TBME, pp 5, figure 6,
    Electrical Impedance Tomography: Tissue Properties to Image Measures
    """
    # solving EIT problem
    p = fwd.solve_eit(ex_mat=ex_mat, parser='fmmu', step=1)
    v0 = p.v

    # normalized jacobian (note: normalize affect sensitivity)
    v0 = v0[:, np.newaxis]
    jac = p.jac  # / np.abs(v0)

    # conditional number of pseudo inverse J
    JtJ = jac.transpose().dot(jac)
    print('rcond of JtJ = %g' % np.linalg.cond(JtJ))

    # keep jacobians
    if keep is not None:
        jac = jac[keep]
    print(jac.shape)

    # calculate sensitivity matrix
    s = np.linalg.norm(jac, axis=0)
    ae = tri_area(pts, tri)
    s = np.sqrt(s) / np.abs(ae)
    assert(any(s >= 0))

    se = np.log10(s)
    sn = sim2pts(pts, tri, se)
    return sn


def _voltage_index(ex_line, el_index, n_el=16, step=1, parser=None):
    """ extract voltage indexes of electrodes pairs """
    # local node
    drv_a = ex_line[0]
    drv_b = ex_line[1]
    i0 = drv_a if parser == 'fmmu' else 0

    # build differential pairs
    v = []
    for a in range(i0, i0 + n_el):
        m = a % n_el
        n = (m + step) % n_el
        val_cd = m in el_index
        # val_ab = a in el_index
        val = val_cd
        # if any of the electrodes is the stimulation electrodes
        if not(m == drv_a or m == drv_b or n == drv_a or n == drv_b):
            v.append(val)

    return np.array(v)


def voltage_index(ex_mat, el_index):
    """ build keep-list for electrodes """
    ind = np.array([], dtype=np.bool)
    for ex_line in ex_mat:
        ind_s = _voltage_index(ex_line, el_index, parser='fmmu')
        ind = np.hstack([ind, ind_s])

    return ind


""" 1. FEM forward setup """
# calculate simulated data using FEM
fwd = Forward(mesh_obj, el_pos)

# loop over EIT scan settings: vary the distance of stimulation nodes, AB
ex_list = [1, 2, 4, 8]
s = []
for ex_dist in ex_list:
    ex_mat = eit_scan_lines(16, ex_dist)
    # TODO: ex_mat can also be stacked, see demo_dynamic_stack.py
    s0 = calc_sens(fwd, ex_mat)
    s.append(s0)

# mixed
ex_list.append(0)
ex_mat1 = eit_scan_lines(16, 4)
ex_mat2 = eit_scan_lines(16, 7)
ex_mat = np.vstack([ex_mat1, ex_mat2])
vind = voltage_index(ex_mat, el_index=[0, 1, 2, 3])
print(vind.shape)
# vind[:192] = True
s0 = calc_sens(fwd, ex_mat, keep=vind)
s.append(s0)

""" 2. Plot (elements) sensitivity """
N = len(ex_list)
vmax = np.max(s)
# vmin = np.min(s)
vmin = vmax - 1.0
print('\n')

# plot
fig = plt.figure(figsize=(12, 2))
gs = gridspec.GridSpec(1, N)
for ix in range(N):
    ax = fig.add_subplot(gs[ix])
    sn = s[ix]
    ex_dist = ex_list[ix]

    # statistics, it seems like ex_dist=4 yields the minimal std
    s_exp10 = 10**(sn)
    mean, std = sp.mean(s_exp10), sp.std(s_exp10)
    print("(ex_dist=%d) mean, std, ratio = %f, %f, %f" %
          (ex_dist, mean, std, std/mean))

    # extract layered information
    s_c0 = s_exp10[r <= r_c1]
    s_c1 = s_exp10[(r_c1 < r) & (r <= r_c2)]
    s_c2 = s_exp10[r > r_c2]
    sr_c0 = np.mean(s_c0)
    sr_c1 = np.mean(s_c1)
    sr_c2 = np.mean(s_c2)
    st = (sr_c0 + sr_c1 + sr_c2)
    print("ratio: %f, %f, %f" % (sr_c0/st, sr_c1/st, sr_c2/st))

    # plot bmp
    # image_name = mes_file.replace('mes', 'bmp')
    # im_bmp = plt.imread(image_name)
    # ax.imshow(im_bmp)
    # ax.set_aspect('equal')

    # plot inner circles
    cir1 = plt.Circle((cx, cy), r_c1, linewidth=2, color='k',
                      fill=False, alpha=0.6)
    cir2 = plt.Circle((cx, cy), r_c2, linewidth=2, color='b',
                      fill=False, alpha=0.6)
    ax.add_artist(cir1)
    ax.add_artist(cir2)
    im = ax.tripcolor(x, y, tri, sn,
                      edgecolors='none', shading='gouraud', cmap=plt.cm.Reds,
                      antialiased=True, vmin=vmin, vmax=vmax, alpha=0.8)
    # annotate
    if ix != (N-1):
        tstr = 'skip=' + str(ex_dist)
    else:
        tstr = 'Mixed Mode'
    ax.set_title(tstr)
    ax.set_aspect('equal')
    # ax.set_ylim([-1.2, 1.2])
    # ax.set_xlim([-1.2, 1.2])
    ax.axis('off')
    plt.colorbar(im)

# fig.savefig('demo_sens.png', dpi=96)
plt.show()
