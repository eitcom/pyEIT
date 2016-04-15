# coding: utf-8
# pylint: disable=invalid-name, no-member, too-many-locals
""" plot function for distmesh 2d and 3d """
from __future__ import absolute_import

import matplotlib
import matplotlib.cm as cm
import matplotlib.pyplot as plt
import numpy as np

from ..utils import edge_project, edge_list


def circumcircle(P1, P2, P3):
    """
    calculate circumcircle of a triangle, returns (x, y, r) of circum-center

    Parameters
    ----------
    P1, P2, P3 : array_like
        points

    Note
    ----
    http://www.labri.fr/perso/nrougier/coding/gallery/
    """
    dp1 = P1 - P2
    dp2 = P3 - P1

    mid1 = (P1 + P2)/2.
    mid2 = (P3 + P1)/2.

    A = np.array([[-dp1[1], dp2[1]],
                  [dp1[0], -dp2[0]]])
    b = -mid1 + mid2
    s = np.linalg.solve(A, b)
    # extract circum pc and radius
    cpc = mid1 + s[0]*np.array([-dp1[1], dp1[0]])
    cr = np.linalg.norm(P1 - cpc)

    return cpc[0], cpc[1], cr


def voronoi(pts, tri, fd=None):
    """
    build voronoi cells using delaunay tessellation

    Parameters
    ----------
    pts : array_like
        points on 2D
    tri : array_like
        triangle structure
    fd : str
        function handler of distances

    Returns
    -------
    array_like
        voronoi cells of lists

    Note
    ----
    byliu adds 'edge-list using signed distance function'
    http://www.labri.fr/perso/nrougier/coding/gallery/
    """
    n = tri.shape[0]

    # Get circle for each triangle, center will be a voronoi cell point
    cells = []
    for i in range(pts.shape[0]):
        cells.append(list())

    def extract_xy(i):
        """ append center (x,y) of triangle-circumcircle to the cell list """
        x, y, _ = circumcircle(pts[tri[i, 0]], pts[tri[i, 1]], pts[tri[i, 2]])
        return [x, y]

    # list(map(extract_xy, range(n)))
    pc = np.array([extract_xy(i) for i in range(n)])

    # peoject point on the boundary if it is outside, where fd(p) > 0
    # this happens when low-quality mesh is generated.
    if fd is not None:
        d = fd(pc)
        ix = d > 0
        pc[ix] -= edge_project(pc[ix], fd)

    # build cells enclosing points
    for i in range(n):
        pc_tuple = tuple(pc[i])
        cells[tri[i, 0]].append(pc_tuple)
        cells[tri[i, 1]].append(pc_tuple)
        cells[tri[i, 2]].append(pc_tuple)

    # append middle (x, y) of edge-bars to the cells,
    # make a closed patch of the voronoi tessellation.
    # note : it may be better if you peoject this point on fd
    edge_bars = edge_list(tri)
    hbars = np.mean(pts[edge_bars], axis=1)
    if fd is not None:
        hbars -= edge_project(hbars, fd)
    for i, bars in enumerate(edge_bars):
        cells[bars[0]].append(tuple(hbars[i]))
        cells[bars[1]].append(tuple(hbars[i]))

    X = pts[:, 0]
    Y = pts[:, 1]
    # Reordering cell points in trigonometric way
    for i, cell in enumerate(cells):
        xy = np.array(cell)
        angles = np.arctan2(xy[:, 1]-Y[i], xy[:, 0]-X[i])
        I = np.argsort(angles)
        cell = xy[I].tolist()
        cell.append(cell[0])
        cells[i] = cell

    return cells


def voronoi_plot(pts, tri, val=None, fd=None):
    """ plot voronoi diagrams on bounded shape

    Parameters
    ----------
    pts : array_like
        points on 2D
    tri : array_like
        triangle structure
    val : array_like, optional
        values on nodes
    fd : str, optional
        function handler

    Returns
    fig : str
        figure handler
    ax : str
        axis handler

    Note
    ----
    byliu adds 'maps value to colormap', see
    http://www.labri.fr/perso/nrougier/coding/gallery/
    """
    cells = voronoi(pts, tri, fd)

    # map values on nodes to colors
    if val is None:
        val = np.random.rand(pts.shape[0])
    norm = matplotlib.colors.Normalize(vmin=min(val),
                                       vmax=max(val), clip=True)
    mapper = cm.ScalarMappable(norm=norm, cmap=cm.Greens)

    fig, ax = plt.subplots()
    # draw mesh (optional)
    # ax.triplot(pts[:, 0], pts[:, 1], tri, color='b', alpha=0.50, lw=0.5)
    # ax.scatter(pts[:, 0], pts[:, 1], s=3, color='r', zorder=1)

    # draw voronoi tessellation
    for i, cell in enumerate(cells):
        codes = [matplotlib.path.Path.MOVETO] \
              + [matplotlib.path.Path.LINETO] * (len(cell)-2) \
              + [matplotlib.path.Path.CLOSEPOLY]
        path = matplotlib.path.Path(cell, codes)
        # map values on nodes to colormap
        # e.g., color = np.random.uniform(.4, .9, 3)
        color = mapper.to_rgba(val[i])
        patch = matplotlib.patches.PathPatch(path, facecolor=color,
                                             edgecolor='w', zorder=-1,
                                             lw=0.4)
        ax.add_patch(patch)
    plt.axis('equal')
    plt.show()

    return fig, ax

