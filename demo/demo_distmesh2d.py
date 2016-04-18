# coding: utf-8
# pylint: disable=invalid-name
""" demo for distmesh """
from __future__ import absolute_import

import numpy as np
import matplotlib.pyplot as plt

from pyeit.mesh import shape
from pyeit.mesh import distmesh
import pyeit.mesh.plot as mplot


def example1():
    """ unit circle mesh """

    def _fd(pts):
        """ shape function """
        return shape.circle(pts, pc=[0, 0], r=1.)

    def _fh(pts):
        """ distance function """
        r2 = np.sum(pts**2, axis=1)
        return 0.2*(2.0 - r2)

    # build fix points, may be used as the position for electrodes
    num = 16
    pfix = shape.pfix_circle(numEl=num)

    # firs num nodes are the positions for electrodes
    epos = np.arange(num)

    # build triangle
    p, t = distmesh.build(_fd, _fh, pfix=pfix, h0=0.05)

    # plot
    fig, ax = plt.subplots()
    ax.triplot(p[:, 0], p[:, 1], t)
    ax.plot(p[epos, 0], p[epos, 1], 'ro')
    plt.axis('equal')
    plt.axis([-1.5, 1.5, -1.1, 1.1])
    plt.show()


def example2():
    """unit circle with a whole at the center"""

    def _fd(pts):
        return shape.ddiff(shape.circle(pts, r=0.7),
                           shape.circle(pts, r=0.3))

    # build triangle
    p, t = distmesh.build(_fd, shape.huniform, h0=0.1)

    # plot
    fig, ax = plt.subplots()
    ax.triplot(p[:, 0], p[:, 1], t)
    plt.axis('equal')
    plt.show()


def example3():
    """rectangle with a whole at the center"""

    # interior
    def _fd(pts):
        return shape.ddiff(shape.rectangle(pts, p1=[-1, -0.6], p2=[1, 0.6]),
                           shape.circle(pts, r=0.3))

    # constraints
    def _fh(pts):
        return 0.05 + 0.05 * shape.circle(pts, r=0.3)

    # build triangle
    p, t = distmesh.build(_fd, _fh, h0=0.05)

    # plot
    fig, ax = plt.subplots()
    ax.triplot(p[:, 0], p[:, 1], t)
    ax.set_xlim([-1.2, 1.2])
    ax.set_ylim([-1, 1])
    plt.show()


def example4():
    """ellipse"""

    def _fd(pts):
        if pts.ndim == 1:
            pts = pts[np.newaxis]
        a, b = 2.0, 1.0
        return np.sum((pts/[a, b])**2, axis=1) - 1.0

    # build triangle
    p, t = distmesh.build(_fd, shape.huniform,
                          bbox=[[-2, -1], [2, 1]], h0=0.15)

    # plot
    fig, ax = plt.subplots()
    ax.triplot(p[:, 0], p[:, 1], t)
    plt.axis('equal')
    plt.show()


def example5():
    """ L-shaped domain from 'Finite Elements and Fast Iterative Solvers'
    by Elman, Silvester, and Wathen. """

    # set fixed points
    pfix = [[1, 0],  [1, -1], [0, -1], [-1, -1],
            [-1, 0], [-1, 1], [0, 1],  [0, 0]]
    pfix = np.array(pfix)

    def _fd(pts):
        return shape.ddiff(shape.rectangle(pts, p1=[-1, -1], p2=[1, 1]),
                           shape.rectangle(pts, p1=[0, 0], p2=[1, 1]))

    # build
    p, t = distmesh.build(_fd, shape.huniform, pfix=pfix, h0=0.15)

    # plot
    fig, ax = plt.subplots()
    ax.triplot(p[:, 0], p[:, 1], t)
    ax.plot(pfix[:, 0], pfix[:, 1], 'ro')
    ax.set_xlim([-1.2, 1.2])
    ax.set_ylim([-1.2, 1.2])
    plt.show()


def example_voronoi():
    """draw voronoi plots for triangle elements"""

    def _fd(pts):
        # return d2d.dcircle(pts, pc=[0, 0], r=1.)
        return shape.ddiff(shape.circle(pts, r=0.7),
                           shape.circle(pts, r=0.3))

    # build triangle
    p, t = distmesh.build(_fd, shape.huniform, h0=0.1)

    # plot using customized voronoi function
    mplot.voronoi_plot(p, t)


if __name__ == "__main__":
    example1()
    # example2()
    # example3()
    # example4()
    # example5()
    # example_voronoi()
