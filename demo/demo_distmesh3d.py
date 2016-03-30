# test
# pylint: disable=no-member
""" demo for distmesh """
from __future__ import absolute_import

from pyeit.mesh import shape
from pyeit.mesh import distmesh
import pyeit.mesh.plot as mplot

# build tetrahedron
# 3D tetrahedron must have a bbox
bbox = [[-1, -1, -1], [1, 1, 1]]
# save calling convention as distmesh 2D
p, t = distmesh.build(shape.unit_ball, shape.huniform, bbox=bbox,
                      h0=0.125, verbose=False)

# print mesh quality
print('points = ', p.shape)
print('simplices = ', t.shape)

# mplot.tetplot(p, t, edge_color=(0.2, 0.2, 1.0, 1.0), alpha=0.01)
mplot.tetplot(p, t, alpha=0.01)
