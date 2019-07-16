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
# mesh_obj, el_pos = mesh.layer_circle(n_layer=8, n_fan=6)
# mesh_obj, el_pos = mesh.create()
mesh_obj, el_pos = mes.load(fstr=mes_file)

# extract node, element, alpha
pts = mesh_obj['node']
tri = mesh_obj['element']
x, y = pts[:, 0], pts[:, 1]
quality.stats(pts, tri)
cx, cy = np.mean(x), np.mean(y)
r = np.sqrt((x - cx)**2 + (y - cy)**2)
rp = np.max(r) * 0.6


def calc_sens(fwd, ex_mat):
    """
    see Adler2017 on IEEE TBME, pp 5, figure 6,
    Electrical Impedance Tomography: Tissue Properties to Image Measures
    """
    # solving EIT problem
    p = fwd.solve_eit(ex_mat=ex_mat, parser='fmmu')
    v0 = p.v
    # normalized jacobian (note: normalize affect sensitivity)
    v0 = v0[:, np.newaxis]
    jac = p.jac  # / np.abs(v0)
    print('rcond of jac = %g' % np.linalg.cond(jac))
    # calculate sensitivity matrix
    s = np.linalg.norm(jac, axis=0)
    ae = tri_area(pts, tri)
    s = np.sqrt(s) / np.abs(ae)
    assert(any(s >= 0))

    se = np.log10(s)
    sn = sim2pts(pts, tri, se)
    return sn


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

# fused
N = len(ex_list) + 1
ex_list.append(0)
ex_mat1 = eit_scan_lines(16, 7)
ex_mat2 = eit_scan_lines(16, 8)
# ex_mat3 = eit_scan_lines(16, 7)
# ex_mat4 = eit_scan_lines(16, -7)
ex_mat = np.vstack([ex_mat1, ex_mat2])
s0 = calc_sens(fwd, ex_mat)
s.append(s0)

""" 2. Plot (elements) sensitivity """
vmin = np.min(s)
vmax = np.max(s)
fig = plt.figure(figsize=(15, 3))
gs = gridspec.GridSpec(1, N)
# art
print('\n')
for ix in range(N):
    ax = fig.add_subplot(gs[ix])
    sn = s[ix]
    ex_dist = ex_list[ix]
    # statistics, it seems like ex_dist=4 yields the minimal std
    s_exp10 = 10**(sn)
    mean, std = sp.mean(s_exp10), sp.std(s_exp10)
    print("std (ex_dist=%d) = %f, %f, %f" % (ex_dist, mean, std, std/mean))
    s_inner = s_exp10[r > rp]
    s_outer = s_exp10[r <= rp]
    s_ratio = np.mean(s_inner) / np.mean(s_outer)
    print("ratio of s = %f" % s_ratio)
    # plot bmp
    # image_name = mes_file.replace('mes', 'bmp')
    # im_bmp = plt.imread(image_name)
    # ax.imshow(im_bmp)
    # ax.set_aspect('equal')
    # plot s
    # plot inner
    circle1 = plt.Circle((cx, cy), rp, linewidth=2, color='b',
                         fill=False, alpha=0.6)
    ax.add_artist(circle1)
    im = ax.tripcolor(x, y, tri, sn,
                      edgecolors='none', shading='gouraud', cmap=plt.cm.Reds,
                      antialiased=True, vmin=vmin, vmax=vmax, alpha=0.8)
    # annotate
    ax.set_title('skip=' + str(ex_dist))
    ax.set_aspect('equal')
    # ax.set_ylim([-1.2, 1.2])
    # ax.set_xlim([-1.2, 1.2])
    ax.axis('off')
    plt.colorbar(im)

# fig.savefig('demo_sens.png', dpi=96)
plt.show()
