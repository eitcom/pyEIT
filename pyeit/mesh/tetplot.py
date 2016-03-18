# coding: utf-8
# pylint: disable=no-member
""" plot function based on vispy for tetrahedral plots """
from __future__ import absolute_import

from itertools import combinations
import numpy as np
from vispy import app, gloo
from vispy.util.transforms import translate, perspective, rotate

# build vertex shader for tetplot
vertex = """
uniform mat4   u_model;         // Model matrix
uniform mat4   u_view;          // View matrix
uniform mat4   u_projection;    // Projection matrix
uniform vec4   u_color;         // mask color for edge plotting
attribute vec3 a_position;
attribute vec4 a_color;
varying vec4   v_color;

void main()
{
    gl_Position = u_projection * u_view * u_model * vec4(a_position, 1.0);
    v_color = a_color * u_color;
}
"""

# build fragment shader for tetplot
fragment = """
varying vec4 v_color;

void main()
{
    gl_FragColor = v_color;
}
"""


class Canvas(app.Canvas):
    """ build canvas class for this demo """

    def __init__(self, V, C, I, E,
                 figsize=(512, 512), title='tetplot'):
        """ initialize the canvas """
        app.Canvas.__init__(self, size=figsize, title=title,
                            keys='interactive')

        # shader program
        tet = gloo.Program(vert=vertex, frag=fragment)

        # bind to data
        tet['a_position'] = V
        tet['a_color'] = C
        self.I = gloo.IndexBuffer(I)
        self.E = gloo.IndexBuffer(E)

        # intialize transformation matrix
        view = np.eye(4, dtype=np.float32)
        model = np.eye(4, dtype=np.float32)
        projection = np.eye(4, dtype=np.float32)

        tet['u_model'] = model
        tet['u_view'] = view
        tet['u_projection'] = projection

        # bind your program
        self.program = tet

        # config and set viewport
        gloo.set_viewport(0, 0, *self.physical_size)
        gloo.set_clear_color('white')
        gloo.set_state('translucent')
        gloo.set_polygon_offset(1.0, 1.0)

        # update parameters
        self.theta = 0.0
        self.phi = 0.0
        self.z = 5.0

        # bind a timer
        self.timer = app.Timer('auto', self.on_timer)
        self.timer.start()

        # control plots
        gloo.set_line_width(1.0)

        # show the canvas
        self.show()

    def on_resize(self, event):
        """ canvas resize callback """
        ratio = event.physical_size[0] / float(event.physical_size[1])
        self.program['u_projection'] = perspective(45.0, ratio, 2.0, 10.0)
        gloo.set_viewport(0, 0, *event.physical_size)

    def on_draw(self, event):
        """ canvas update callback """
        gloo.clear()

        # Filled cube
        gloo.set_state(blend=True, depth_test=False,
                       polygon_offset_fill=True)
        self.program['u_color'] = [1.0, 1.0, 1.0, 0.8]
        self.program.draw('triangles', self.I)

        # draw outline
        # gloo.set_state(blend=True, depth_test=False,
        #                polygon_offset_fill=True)
        # self.program['u_color'] = [0.0, 0.0, 0.0, 0.2]
        # self.program.draw('lines', self.E)

    def on_timer(self, event):
        self.theta += 0.5
        self.phi += 0.5
        self.view(theta=self.theta, phi=self.phi)

    def tetplot(self, V, C=None, I=None, E=None):
        """ plot tetrahedron """
        self.program['a_position'] = V
        self.program['a_color'] = C
        self.I = gloo.IndexBuffer(I)
        self.E = gloo.IndexBuffer(E)
        self.update()

    def view(self, z=5, theta=0.0, phi=0.0):
        """ change the zoom factor and view point """
        self.program['u_view'] = translate((0, 0, -self.z))
        model = np.dot(rotate(self.theta, (0, 1, 0)),
                       rotate(self.phi, (0, 0, 1)))
        self.program['u_model'] = model
        self.update()


def tetplot(points, simplices):
    """ main function for tetplot """
    colors = np.random.rand(points.shape[0], 4)
    colors[:, -1] = 1.0
    colors = colors.astype(np.float32)

    # extract triangles and edges
    triangles = sim2tri(simplices)
    edges = sim2edge(simplices)

    # plot
    Canvas(points, colors, triangles, edges)
    app.run()


def sim_conv(simplices, N=3):
    """ simplices to any dimension """
    v = [list(combinations(sim, N)) for sim in simplices]
    return np.array(v, dtype=np.uint32).reshape(-1, N)


def sim2tri(simplices):
    """ convert simplices of high dimension to indices of triangles """
    return sim_conv(simplices, 3)


def sim2edge(simplices):
    """ convert simplices of high dimension to indices of edges """
    return sim_conv(simplices, 2)


if __name__ == "__main__":
    pts = np.array([(0.0, 0.0, 0.0),
                    (1.0, 0.0, 0.0),
                    (0.0, 1.0, 0.0),
                    (0.0, 0.0, 1.0),
                    (1.0, 1.0, 1.0)], dtype=np.float32)

    sim = np.array([(0, 1, 2, 3),
                    (1, 3, 2, 4)], dtype=np.uint32)

    tetplot(pts, sim)
