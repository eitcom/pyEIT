# coding: utf-8
# pylint: disable=no-member
""" plot function based on vispy.visuals for tetrahedral plots """
from __future__ import absolute_import

from itertools import combinations
import numpy as np
import sys

from vispy.visuals import CompoundVisual
from vispy.visuals.mesh import MeshVisual
from vispy.visuals.line import LineVisual
from vispy.visuals.markers import MarkersVisual
from vispy.color import Color
from vispy.gloo import set_state


def sim_conv(simplices, N=3):
    """ simplices to any dimension """
    v = [list(combinations(sim, N)) for sim in simplices]
    # change to (num_of_points x N)
    t = np.sort(np.array(v).reshape(-1, N), axis=1)
    # delete duplicated entries
    t_unique = np.unique(t.view([('', t.dtype)]*N)).view(np.uint32)
    return t_unique


def sim2tri(simplices):
    """ convert simplices of high dimension to indices of triangles """
    return sim_conv(simplices, 3)


def sim2edge(simplices):
    """ convert simplices of high dimension to indices of edges """
    return sim_conv(simplices, 2)
    
    
# build your visual
class TetVisual(CompoundVisual):
    """ display a 3D mesh """
    
    def __init__(self,
                 vertices=None, simplices=None, vertex_colors=None,
                 edge_color=None, edge_width=1,
                 markers=None, marker_colors=None, marker_size=1,
                 **kwargs):
        """
        a mesh visualization toolkit that can also plot edges or markers
        
        Parameters
        ----------
        
        Notes
        -----
        """
        self._mesh = MeshVisual()
        self._edge = LineVisual()
        self._edge_color = Color(edge_color)
        self._marker = MarkersVisual()
        #
        self._vertex_colors = vertex_colors
        
        self._update()
        # initialize visuals
        CompoundVisual.__init__(self,
                                [self._mesh, self._edge, self._marker],
                                **kwargs)
        # set default state, 'opaque', 'translucent' or 'additive'
        self._mesh.set_gl_state(preset='translucent',
                                blend=True,
                                depth_test=False,
                                cull_face=False,
                                polygon_offset_fill=True,
                                polygon_offset=(1, 1))
        # end
        self.freeze()
        
        def _update(self):
            """
            update parameters to visuals
            """
            pass
        
        @property
        def edge_color(self):
            return self._edge_color
            
        @edge_color.setter
        def edge_color(self, edge_color):
            self._edge_color = Color(edge_color)
            self._update()


def demo():
    """ test run """
    pts = np.array([(0.0, 0.0, 0.0),
                    (1.0, 0.0, 0.0),
                    (0.0, 1.0, 0.0),
                    (0.0, 0.0, 1.0),
                    (1.0, 1.0, 1.0)], dtype=np.float32)

    sim = np.array([(0, 1, 2, 3),
                    (1, 3, 2, 4)], dtype=np.uint32)

    # tetplot(pts, sim, edge_color=[0.2, 0.2, 1.0, 0.2],
    #         alpha=0.1, axis=False)
            
            
if __name__ == '__main__':
    if sys.flags.interactive != 1:
        demo()
