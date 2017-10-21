# coding: utf-8
# pylint: disable=invalid-name
""" demo for distmesh 3D """
# Copyright (c) Benyuan Liu. All Rights Reserved.
# Distributed under the (new) BSD License. See LICENSE.txt for more info.
from __future__ import division, absolute_import, print_function

import numpy as np

import pyeit.mesh as mesh
import pyeit.mesh.plot as mplot

# build tetrahedron
# 3D tetrahedron must have a bbox
bbox = [[-1, -1, -1], [1, 1, 1]]

# save calling convention as distmesh 2D
ms, el_pos = mesh.create(h0=0.2, bbox=bbox)
p = ms['node']
t = ms['element']

# print mesh quality
print('points =', p.shape)
print('simplices =', t.shape)

# create random color
f = np.random.randn(p.shape[0])
# mplot.tetplot(p, t, edge_color=(0.2, 0.2, 1.0, 1.0), alpha=0.01)
mplot.tetplot(p, t, f, alpha=0.25)
