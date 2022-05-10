# coding: utf-8
# pylint: disable=invalid-name
""" demo for distmesh 3D """
# Copyright (c) Benyuan Liu. All Rights Reserved.
# Distributed under the (new) BSD License. See LICENSE.txt for more info.
from __future__ import division, absolute_import, print_function
import numpy as np

import pyeit.mesh as mesh
import pyeit.mesh.plot as mplot

# tetrahedron meshing in a 3D bbox
bbox = [[-1.2, -1.2, -1.2], [1.2, 1.2, 1.2]]
# 3D Mesh shape is specified with fd parameter in the instantiation, e.g : fd=ball , Default in 3D :fd=ball
ms = mesh.create(h0=0.15, bbox=bbox)

# print mesh quality
p = ms.node
t = ms.element
ms.print_stats()

# plot
mplot.tetplot(p, t, edge_color=(0.2, 0.2, 1.0, 1.0), alpha=0.01)
# create random color
f = np.random.randn(p.shape[0])
# mplot.tetplot(p, t, f, alpha=0.25)
