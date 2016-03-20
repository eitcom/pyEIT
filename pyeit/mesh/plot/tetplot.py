# coding: utf-8
# pylint: disable=no-member
""" plot function based on vispy for tetrahedral plots """
from __future__ import absolute_import

from itertools import combinations
import numpy as np
import sys

from vispy import app, gloo, visuals, scene


# build vertex shader for tetplot
vert = """
uniform vec4 u_color;
attribute vec4 a_color;
varying vec4 v_color;

void main()
{
    vec4 visual_pos = vec4($position, 1);
    vec4 doc_pos = $visual_to_doc(visual_pos);
    gl_Position = $doc_to_render(doc_pos);

    v_color = a_color * u_color;
}
"""

# build fragment shader for tetplot
frag = """
varying vec4 v_color;

void main()
{
    gl_FragColor = v_color;
}
"""


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


class TetPlotVisual(visuals.Visual):
    """ template """

    def __init__(self, points, simplices, vertex_color=None,
                 color=None, alpha=1.0,
                 mode='triangles'):
        """ initialize tetrahedra face plot

        Parameters
        ----------
        points : NDArray of float32
            N x 3 points coordinates
        simplices : NDArray of uint32
            N x 4 connectivity matrix

        Note
        ----
        initialize triangles structure
        """
        visuals.Visual.__init__(self, vcode=vert, fcode=frag)

        # set data
        self.shared_program.vert['position'] = gloo.VertexBuffer(points)
        if vertex_color is None:
            vertex_color = np.ones((points.shape[0], 4), dtype=np.float32)
        else:
            assert(vertex_color.shape[0] == points.shape[0])
        self.shared_program['a_color'] = vertex_color

        # currently, do not support color parsing
        if color is None:
            color = [1.0, 1.0, 1.0, 1.0]
        else:
            assert(len(color) == 4)
        color[-1] *= alpha
        self.shared_program['u_color'] = color

        # build buffer
        if mode is 'triangles':
            vbo = sim2tri(simplices)
        elif mode is 'lines':
            vbo = sim2edge(simplices)
        else:
            raise ValueError('Drawing mode = ' + mode + ' not supported')
        self._index_buffer = gloo.IndexBuffer(vbo)

        # config OpenGL
        self.set_gl_state('additive',
                          blend=True,
                          depth_test=False,
                          polygon_offset_fill=True)
        self._draw_mode = mode

    def _prepare_transforms(self, view):
        """ This method is called when the user or the scenegraph has assigned
        new transforms to this visual """
        # Note we use the "additive" GL blending settings so that we do not
        # have to sort the mesh triangles back-to-front before each draw.
        tr = view.transforms
        view_vert = view.view_program.vert
        view_vert['visual_to_doc'] = tr.get_transform('visual', 'document')
        view_vert['doc_to_render'] = tr.get_transform('document', 'render')


def tetplot(points, simplices, vertex_color=None,
            edge_color=None, alpha=1.0, axis=True):
    """ main function for tetplot """
    TetPlot = scene.visuals.create_visual_node(TetPlotVisual)

    # convert data types for OpenGL
    pts_float32 = points.astype(np.float32)
    sim_uint32 = simplices.astype(np.uint32)

    # The real-things : plot using scene
    # build canvas
    canvas = scene.SceneCanvas(keys='interactive', show=True)

    # Add a ViewBox to let the user zoom/rotate
    view = canvas.central_widget.add_view()
    view.camera = 'turntable'
    view.camera.fov = 50
    view.camera.distance = 5

    # toggle drawing mode
    TetPlot(pts_float32, sim_uint32, vertex_color,
            color=None, alpha=alpha, mode='triangles', parent=view.scene)
    if edge_color is not None:
        TetPlot(pts_float32, sim_uint32, vertex_color,
                color=edge_color, alpha=alpha, mode='lines',
                parent=view.scene)

    # show axis
    if axis:
        scene.visuals.XYZAxis(parent=view.scene)

    # run
    if sys.flags.interactive != 1:
        app.run()

# run
if __name__ == '__main__':
    # data
    pts = np.array([(0.0, 0.0, 0.0),
                    (1.0, 0.0, 0.0),
                    (0.0, 1.0, 0.0),
                    (0.0, 0.0, 1.0),
                    (1.0, 1.0, 1.0)], dtype=np.float32)

    sim = np.array([(0, 1, 2, 3),
                    (1, 3, 2, 4)], dtype=np.uint32)

    tetplot(pts, sim, edge_color=[0.2, 0.2, 1.0, 1.0], alpha=0.1)
