#
""" create multi-layered mesh on a unit circle """
import numpy as np
import matplotlib.pyplot as plt


class MeshCircle(object):
    """ create meshes on uniform circle """

    def __init__(self, n_fan=8, n_layer=6, n_el=16):
        """
        Parameters
        ----------
        n_fan : int
            number of fans (see the inner most layer)
        n_layer : int
            number of layers
        n_el : int
            number of boundary electrodes
        """
        self.n_fan = n_fan
        self.n_layer = n_layer
        self.n_el = n_el

        # number of points per-layer
        nl = self.n_fan * np.arange(self.n_layer+1)
        # '0' is the number of points of the initial layer
        nl[0] = 1
        self.points_per_layer = nl

        # starting point of each layer
        index = np.cumsum(nl)
        # '-1' is the initial layer start with 0
        index[-1] = 0
        self.index_per_layer = index

    def create(self):
        """ create no2xy and el2no """
        p = self._spawn_points()
        e = self._spawn_elements()
        el_pos = self._get_electrodes()
        return p, e, el_pos

    def _get_electrodes(self):
        """ return the numbering of electrodes """
        el_start = self.index_per_layer[self.n_layer-1]
        el_len = self.points_per_layer[self.n_layer]

        # place electrodes uniformly on the boundary
        n = np.linspace(el_start, el_start + el_len, self.n_el,
                        endpoint=False, dtype=np.int)
        return n

    def _spawn_points(self):
        """ generate points """
        # init points
        p = [0, 0]

        # divide r uniformly axial
        delta_r = 1. / self.n_layer

        for i in range(1, self.n_layer+1):
            # increment points per-layer by fans
            n = i*self.n_fan
            r = i*delta_r
            # generate points on a layer
            pts = r * self._points_on_circle(n, offset=i)
            p = np.vstack([p, pts])

        return p

    @staticmethod
    def _points_on_circle(n, offset=0, offset_enabled=False):
        """ generate points on unit circle """
        fan_angle = 2*np.pi / n
        a = np.array([i*fan_angle for i in range(n)])
        if offset_enabled:
            a += offset * (fan_angle / 2.)
        pts = np.array([np.cos(a), np.sin(a)]).T

        return pts

    def _spawn_elements(self):
        """ connect points fan-by-fan using a fixed pattern """

        # element connections
        e = []
        for i in range(self.n_layer):
            e_layer = self._connect_layer(i)
            e.append(e_layer)

        return np.vstack(e)

    def _connect_layer(self, i):
        """ generate connections on the i-th layer using points
        on the i-th and (i-1)-the layers """

        # number of points in current and previous layer
        nl_now = self.points_per_layer[i+1]
        nl_old = self.points_per_layer[i]

        # starting index of current and previous layer
        index_now = self.index_per_layer[i]
        index_old = self.index_per_layer[i-1]

        e = []
        # k is the pattern per-fan
        points_per_fan = i + 1
        k = 0
        for j in range(nl_now):

            # numbering in current layer
            out_prefetch = (j + 1) % nl_now
            out_now = index_now + j
            out_next = index_now + out_prefetch

            # numbering in previous layer
            in_prefetch = (k + 1) % nl_old
            in_now = index_old + k
            in_next = index_old + in_prefetch

            # every (nl_now/n_fan) points
            mode = (j % points_per_fan)
            if mode == 0:
                ei = [out_now, in_now, out_next]
                e.append(ei)
            else:
                ei = [out_now, in_now, in_next]
                e.append(ei)
                ei = [out_now, in_next, out_next]
                e.append(ei)
                k += 1

        return e


def demo():
    """ demo using unit_circle_mesh """
    model = MeshCircle()
    p, e, el_pos = model.create()

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.plot(p[:, 0], p[:, 1], 'ro', markersize=5)
    for i in range(p.shape[0]):
        ax.text(p[i, 0], p[i, 1], str(i))
    ax.set_xlim([-1.2, 1.2])
    ax.set_ylim([-1.2, 1.2])
    ax.grid('on')

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.triplot(p[:, 0], p[:, 1], e)
    ax.plot(p[el_pos, 0], p[el_pos, 1], 'ro')
    for i, el in enumerate(el_pos):
        ax.text(p[el, 0], p[el, 1], str(i))
    ax.set_xlim([-1.2, 1.2])
    ax.set_ylim([-1.2, 1.2])
    ax.grid('on')

    plt.show()

if __name__ == "__main__":
    """ demo """
    demo()
