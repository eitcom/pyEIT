# coding: utf-8
# pylint: disable=no-member
""" plot function based on vispy for tetrahedral plots """
from __future__ import absolute_import

from itertools import combinations
import numpy as np
import matplotlib
import matplotlib.cm as cm
from matplotlib.colors import LinearSegmentedColormap    
import sys

from vispy import app, gloo, scene
from vispy.visuals import Visual


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


class TetPlotVisual(Visual):
    """ template """

    def __init__(self, points, simplices, vertex_color=None,
                 mask_color=None, alpha=1.0,
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
        Visual.__init__(self, vcode=vert, fcode=frag)

        # set data
        self.shared_program.vert['position'] = gloo.VertexBuffer(points)
        if vertex_color is None:
            vertex_color = np.ones((points.shape[0], 4), dtype=np.float32)
        else:
            assert(vertex_color.shape[0] == points.shape[0])
            # vertex color may be grayscale
            if np.ndim(vertex_color) == 1:
                f = vertex_color[:, np.newaxis]
                v = np.repeat(f, 4, axis=1)
                v[:, -1] = 1.0
                vertex_color = v.astype(np.float32)
        self.shared_program['a_color'] = vertex_color

        # mask colors, alpha channel is not used when mask_color is given.
        if mask_color is None:
            mask_color = [1.0, 1.0, 1.0, alpha]
        self.shared_program['u_color'] = mask_color

        # build buffer
        if mode is 'triangles':
            vbo = sim2tri(simplices)
        elif mode is 'lines':
            vbo = sim2edge(simplices)
        else:
            raise ValueError('Drawing mode = ' + mode + ' not supported')
        self._index_buffer = gloo.IndexBuffer(vbo)

        # config OpenGL, 'translucent' or 'additive'
        self.set_gl_state('additive',
                          blend=True,
                          depth_test=False,
                          cull_face=False,
                          polygon_offset_fill=False,
                          polygon_offset=(1, 1))
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
    view.camera.distance = 3
    
    # convert vertex_color
    cdict1 = {'red':   ((0.0, 0.0, 0.0),
                       (0.5, 0.0, 0.0),
                       (1.0, 1.0, 1.0)),
              'green': ((0.0, 0.0, 0.0),
                       (1.0, 0.0, 0.0)),
              'blue':  ((0.0, 0.0, 1.0),
                       (0.5, 0.0, 0.0),
                       (1.0, 0.0, 0.0))
             }
         
    def blue_red():
        return LinearSegmentedColormap('BlueRed', cdict1)
        
    if vertex_color is not None and vertex_color.ndim == 1:
        maxima = np.max(np.abs(vertex_color))
        minima = -maxima
        print(maxima)
        brcmap = blue_red()
        norm = matplotlib.colors.Normalize(vmin=minima, vmax=maxima, clip=True)
        mapper = cm.ScalarMappable(norm=norm, cmap=brcmap)
        vertex_color = mapper.to_rgba(vertex_color)
        vertex_color = vertex_color.astype(np.float32)

    # drawing only triangles
    # 1. turn off mask_color, default = [1.0, 1.0, 1.0, alpha]
    # 2. mode = 'triangles'
    TetPlot(pts_float32, sim_uint32, vertex_color,
            mask_color=None, alpha=alpha, mode='triangles',
            parent=view.scene)

    # drawing only lines
    # 1. turn off vertex_color, default = [[1.0, 1.0, 1.0, 1.0]*N]
    # 2. mode = 'lines'
    # 3. alpha channel is specified instead of mask_color
    if edge_color is not None:
        TetPlot(pts_float32, sim_uint32, vertex_color=None,
                mask_color=edge_color, alpha=alpha, mode='lines',
                parent=view.scene)

    # show axis
    if axis:
        scene.visuals.XYZAxis(parent=view.scene)

    # run
    app.run()

# demo
if __name__ == '__main__':
    if sys.flags.interactive != 1:
        # data
        pts = np.array([(0.0, 0.0, 0.0),
                        (1.0, 0.0, 0.0),
                        (0.0, 1.0, 0.0),
                        (0.0, 0.0, 1.0),
                        (1.0, 1.0, 1.0)], dtype=np.float32)

        sim = np.array([(0, 1, 2, 3),
                        (1, 3, 2, 4)], dtype=np.uint32)

        tetplot(pts, sim, edge_color=[0.2, 0.2, 1.0, 0.2],
                alpha=0.1, axis=False)
