# coding: utf-8
# pylint: disable=invalid-name, no-member, no-name-in-module
# pylint: disable=too-many-arguments, too-many-locals
""" implement a 2D distmesh """
from __future__ import absolute_import

import numpy as np
from numpy import sqrt
import matplotlib
import matplotlib.cm as cm
import matplotlib.pyplot as plt
from scipy.spatial import Delaunay
from scipy.sparse import csr_matrix


def delaunay(pts):
    """ simplices : triangles where the points are arranged counterclockwise

    Parameters
    ----------
    pts : array_like

    Returns
    -------
    triangles : array_like
        simplices from Delaunay
    """
    return Delaunay(pts).simplices


def dist(p):
    """ distances to origin of nodes

    Parameters
    ----------
    p : array_like
        points in 2D

    Returns
    -------
    array_like
        distances of points to origin
    """
    return np.array([sqrt(row[0]**2 + row[1]**2) for row in p])


def bbox2p(h0, bbox):
    """
    convert bbox to p (not including the ending point of bbox)
    shift every second row h0/2 to the right, therefore,
    all points will be a distance h0 from their closest neighbors

    Parameters
    ----------
    h0 : float
        minimal distance of points
    bbox : array_like
        [x0, y0, x1, y1]

    Returns
    -------
    array_like
        points in bbox
    """
    x, y = np.meshgrid(np.arange(bbox[0], bbox[2], h0),
                       np.arange(bbox[1], bbox[3], h0*sqrt(3)/2.),
                       indexing='xy')
    # shift even rows of x
    x[1::2, :] += h0/2.
    # p : Nx2 ndarray
    p = np.array([x.ravel(), y.ravel()]).T
    return p


def remove_duplicate_nodes(p, pfix, geps):
    """ remove duplicate points in p from pfix

    Parameters
    ----------
    p : array_like
        points in 2D
    pfix : array_like
        points that are fixed (can not be moved in distmesh)
    geps : float, optional (default=0.01*h0)
        minimal distance that two points are assumed to be identical

    Returns
    -------
    array_like
        non-duplicated points in 2D
    """
    for row in pfix:
        pdist = dist(p - row)
        p = p[pdist > geps]
    return p


def triangulate(pts, fd, geps):
    """
    Compute the Delaunay triangulation and remove trianges with
    centroids outside the domain (with a geps gap)

    Parameters
    ----------
    pts : array_like
        points on 2D
    fd : str
        distance function handler
    geps : float
        tol on the gap of distances compared to zero

    Returns
    -------
    array_like
        triangles
    """
    tri = delaunay(pts)
    pmid = np.mean(pts[tri], axis=1)
    tri = tri[fd(pmid) < -geps]
    return tri


def build(fd, fh, pfix=None,
          bbox=None, h0=0.1, densityctrlfreq=30,
          dptol=0.001, ttol=0.1, Fscale=1.2, deltat=0.2,
          maxiter=1000):
    """ main function for distmesh2d

    Parameters
    ----------
    fd : str
        function handle for distance of boundary
    fh : str
        function handle for distance distributions
    pfix : array_like, optional
        fixed points, default=[]
    bbox : array_like, optional
        bounding box for region, bbox=[xmin, ymin, xmax, ymax].
        default=[-1, -1, 1, 1]
    h0 : float, optional
        Distance between points in the initial distribution p0, default=0.1
        For uniform meshes, h(x,y) = constant,
        the element size in the final mesh will usually be
        a little larger than this input.
    densityctrlfreq : int, optional
        cycles of iterations of density control, default=20
    dptol : float, optional
        exit criterion for minimal distance all points moved, default=0.001
    ttol : float, optional
        enter criterion for re-delaunay the lattices, default=0.1
    Fscale : float, optional
        rescaled string forces, default=1.2
        if set too small, points near boundary will be pushed back
        if set too large, points will be pushed towards boundary
    deltat : float, optional
        mapping forces to distances, default=0.2
    maxiter : int, optional
        maximum iteration numbers, default=1000

    Returns
    -------
    p : array_like
        points on 2D bbox
    t : array_like
        triangles describe the mesh structure

    Note
    ----
    there are many python or hybrid python + C implementations in github,
    this implementation is merely implemented from scratch
    using PER-OLOF PERSSON's Ph.D thesis and SIAM paper.

    .. [1] P.-O. Persson, G. Strang, "A Simple Mesh Generator in MATLAB".
       SIAM Review, Volume 46 (2), pp. 329-345, June 2004

    """
    geps = 0.001 * h0

    # p : Nx2 coordinates (x,y) of meshes
    if bbox is None:
        bbox = [-1, -1, 1, 1]
    p = bbox2p(h0, bbox)

    # discard points out of the distance function fd
    p = p[fd(p) < geps]

    # rejection sampling on fh
    r0 = 1. / fh(p)**2
    selection = np.random.rand(p.shape[0]) < (r0 / np.max(r0))
    p = p[selection]

    # pre-pend fixed points (warning : avoid overlap mesh points)
    if pfix is None:
        pfix = []
    nfix = len(pfix)
    if len(pfix) > 0:
        p = remove_duplicate_nodes(p, pfix, geps)
        p = np.vstack([pfix, p])
    N = p.shape[0]

    # now iterate to push to equilibrium
    pold = np.inf * np.ones((N, 2))
    for i in range(maxiter):
        if np.max(dist(p - pold)/h0) > ttol:
            # retriangle by delaunay
            # pnew[:] = pold[:] makes a new copy, not reference
            pold[:] = p[:]
            t = triangulate(p, fd, geps)
            # build edges or bars
            bars = t[:, [[0, 1], [1, 2], [2, 0]]].reshape((-1, 2))
            # sort and remove duplicated edges, eg (1,2) and (2,1)
            # note : for all edges, non-duplicated edge is boundary edge
            bars = np.sort(bars, axis=1)
            bars = np.unique(bars.view('i, i')).view('i').reshape((-1, 2))

        # the forces of bars (python is by-default row-wise operation)
        barvec = p[bars[:, 0]] - p[bars[:, 1]]
        # L : length of bars, must be column ndarray (2D)
        L = dist(barvec).reshape((-1, 1))
        hbars = fh((p[bars[:, 0]] + p[bars[:, 1]])/2.0).reshape((-1, 1))
        # L0 : desired lengths (Fscale matters)
        L0 = hbars * Fscale * sqrt(np.sum(L**2) / np.sum(hbars**2))

        if (i % densityctrlfreq) == 0 and (L0 > 2*L).any():
            # Density control - remove points that are too close
            # L0 : Kx1, L : Kx1, bars : Kx2
            # bars[L0 > 2*L] only returns bar[:, 0] where L0 > 2L
            ixout = (L0 > 2*L).ravel()
            ixdel = np.setdiff1d(bars[ixout, :].reshape(-1), np.arange(nfix))
            p = p[np.setdiff1d(np.arange(N), ixdel)]
            # Nold = N
            N = p.shape[0]
            pold = np.inf * np.ones((N, 2))
            # print('density control ratio : %f' % (float(N)/Nold))
            # continue to triangulate
            continue

        # forces on bars
        F = np.maximum(L0 - L, 0)
        # normalized and vectorized forces
        Fvec = F * (barvec / L)

        # using sparse matrix to perform automatic summation
        # rows : left, left, right, right
        # cols : x, y, x, y
        data = np.hstack([Fvec, -Fvec])
        rows = bars[:, [0, 0, 1, 1]]
        cols = np.dot(np.ones(np.shape(F)), np.array([[0, 1, 0, 1]]))
        # sum nodes at duplicated locations using sparse matrices
        Ftot = csr_matrix((data.reshape(-1),
                           [rows.reshape(-1), cols.reshape(-1)]),
                          shape=(N, 2))
        Ftot = Ftot.toarray()

        # zero out forces at fixed points:
        Ftot[0:len(pfix)] = 0

        # update p
        p += deltat * Ftot

        # if a point ends up outside, move it back to the closest
        # on the boundary using the distance function
        d = fd(p)
        ix = d > 0
        p[ix] -= edge_project(p[ix], fd)

        # the stopping ctriterion (movements interior are small)
        delta_move = deltat * Ftot[d < -geps]
        if np.max(dist(delta_move)/h0) < dptol:
            break

    # at the end of iteration, (p - pold) is small, so we recreate delaunay
    t = triangulate(p, fd, geps)

    # you should remove duplicate nodes and triangles
    return p, t


def edge_project(pts, fd, h0=1.0):
    """
    project points back on the boundary (where fd=0) using numerical gradient

    Parameters
    ----------
    pts : array_like
        points on 2D
    fd : str
        function handler of distances
    h0 : float
        minimal distance

    Returns
    -------
    array_like
        gradients of points on the boundary

    Note
    ----
    you should specify h0 according to your actual mesh size
    """
    deps = sqrt(np.finfo(float).eps)*h0
    d = fd(pts)
    dgradx = (fd(pts + [deps, 0]) - d) / deps
    dgrady = (fd(pts + [0, deps]) - d) / deps
    dgrad2 = dgradx**2 + dgrady**2
    dgrad2[dgrad2 == 0] = 1.
    # calculate gradient vector (minus)
    pgrad = np.vstack([d*dgradx/dgrad2, d*dgrady/dgrad2]).T
    return pgrad


def edge_list(tri):
    """
    edge of delaunay triangles are unique bars, O(n^2)

    besides this duplication test, you may also use fd to test edge bars,
    where the two nodes of any edge both satisfy fd=0 (or with a geps gap)

    Parameters
    ----------
    tri : array_like
        triangles list
    """
    bars = tri[:, [[0, 1], [1, 2], [2, 0]]].reshape((-1, 2))
    bars = np.sort(bars, axis=1)
    bars = bars.view('i, i')
    n = bars.shape[0]

    ix = [True] * n
    for i in range(n-1):
        # skip if already marked as duplicated
        if ix[i] is not True:
            continue
        # mark duplicate entries, at most 2-duplicate bars, if found, break
        for j in range(i+1, n):
            if bars[i] == bars[j]:
                ix[i], ix[j] = False, False
                break

    return bars[np.array(ix)].view('i')


def dcircle(pts, pc=None, r=1.0):
    """ Distance function for the circle centered at pc = [xc, yc]

    Parameters
    ----------
    pts : array_like
        points on 2D
    pc : array_like, optional
        center of points
    r : float, optional
        radius

    Returns
    -------
    array_like
        distance of (points - pc) - r

    Note
    ----
    copied and modified from https://github.com/ckhroulev/py_distmesh2d
    """
    if pc is None:
        pc = [0, 0]
    return dist(pts - pc) - r


def drectangle(pts, p1=None, p2=None):
    """
    Distance function for the rectangle p1=[x1, y1] and p2=[x2, y2]

    Note
    ----
    p1 should be bottom-left, p2 should be top-right
    if p in rect(p1, p2), then (p-p1)_x and (p-p2)_x must have opposite sign

    Parameters
    ----------
    pts : array_like
    p1 : array_like, optional
        bottom left coordinates
    p2 : array_like, optional
        top tight coordinates

    Returns
    -------
    array_like
        distance
    """
    if p1 is None:
        p1 = [0, 0]
    if p2 is None:
        p2 = [1, 1]
    pd_left = [-min(row) for row in pts - p1]
    pd_right = [max(row) for row in pts - p2]

    return np.maximum(pd_left, pd_right)


def ddiff(d1, d2):
    """ Distance function for the difference of two sets.

    Parameters
    ----------
    d1 : array_like
    d2 : array_like
        distance of two functions

    Returns
    -------
    array_like
        maximum difference

    Note
    ----
    boundary is denoted by d=0
    copied and modified from https://github.com/ckhroulev/py_distmesh2d
    """
    return np.maximum(d1, -d2)


def dintersect(d1, d2):
    """ Distance function for the intersection of two sets.

    Parameters
    ----------
    d1 : array_like
    d2 : array_like
        distance of two functions

    Returns
    -------
    array_like

    Note
    ----
    boundary is denoted by d=0
    copied and modified from https://github.com/ckhroulev/py_distmesh2d
    """
    return np.maximum(d1, d2)


def dunion(d1, d2):
    """ Distance function for the union of two sets.

    Parameters
    ----------
    d1 : array_like
    d2 : array_like
        distance of two functions

    Returns
    -------
    array_like

    Note
    ----
    boundary is denoted by d=0
    copied and modified from https://github.com/ckhroulev/py_distmesh2d
    """
    return np.minimum(d1, d2)


def huniform(p):
    """ uniform mesh distribution

    Parameters
    ----------
    p : array_like
        points coordinates

    Returns
    -------
    array_like
        ones

    """
    return np.array([1. for _ in p])


def dunitCircle(pts):
    """ unit circle at (0,0)

    Parameters
    ----------
    pts : array_like
        points coordinates

    Returns
    -------
    array_like
    """
    return dist(pts) - 1.


def dboxCircle(pts):
    """ unit circle at (0.5,0.5) with r=0.5 """
    return dcircle(pts, pc=[0.5, 0.5], r=0.5)


def pcircle(pc=None, r=1., numEl=16):
    """
    return fixed and uniformly distributed points on
    a circle with radius r

    Parameters
    ----------
    pc : array_like, optional
        center of points
    r : float, optional
        radius
    numEl : number of electrodes, optional

    Returns
    -------
    array_like
        coordinates of fixed points

    """
    if pc is None:
        pc = [0, 0]

    theta = 2. * np.pi * np.arange(numEl)/float(numEl)
    pfix = [[r*np.sin(th), r*np.cos(th)] for th in theta]
    return np.array(pfix) + pc


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
    vmin = min(val)
    vmax = max(val)
    norm = matplotlib.colors.Normalize(vmin=vmin, vmax=vmax, clip=True)
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
    return fig, ax


def create(numEl=16, h0=0.1):
    """ wrapper for pyEIT interface

    Parameters
    ----------
    numEl : int, optional
        number of electrodes
    h0 : float, optional
        initial mesh size

    Returns
    -------
    dict
        {'element', 'node', 'alpha'}
    """
    pfix = pcircle(numEl=numEl)
    p, t = build(dunitCircle, huniform, pfix=pfix, h0=h0, Fscale=1.2)
    # electrodes are the same as pfix (top numEl)
    elPos = np.arange(numEl)
    # build output dictionary, uniform element sigma
    alpha = 1. * np.ones(t.shape[0])
    mesh = {'element': t,
            'node': p,
            'alpha': alpha}
    return mesh, elPos


def set_alpha(mesh, anom=None, background=None):
    """ wrapper for pyEIT interface

    Note
    ----
    update alphas of mesh structure, if specified,

    Parameters
    ----------
    mesh : dict
        mesh structure
    anom : dict, optional
        anom is a dictionary (or arrays of dictionary) contains,
        {'x': val, 'y': val, 'd': val, 'alpha': val}
        all alphas on triangles whose distance to (x,y) are less than (d)
        will be replaced with a new alpha, alpha can have a complex dtype
    background : float, optional
        set background permitivities

    Returns
    -------
    dict
        updated mesh structure
    """
    el2no = mesh['element']
    no2xy = mesh['node']
    alpha = mesh['alpha']
    tri_centers = np.mean(no2xy[el2no], axis=1)

    # this code is equivalent to:
    # >>> N = np.shape(el2no)[0]
    # >>> for i in range(N):
    # >>>     tri_centers[i] = np.mean(no2xy[el2no[i]], axis=0)
    # >>> plt.plot(tri_centers[:,0], tri_centers[:,1], 'kx')
    N = np.size(mesh['alpha'])

    # reset background if needed
    if background is not None:
        alpha = background * np.ones(N, dtype='complex')

    if anom is not None:
        for _, attr in enumerate(anom):
            cx = attr['x']
            cy = attr['y']
            diameter = attr['d']
            alpha_anomaly = attr['alpha']
            # find elements whose distance to (cx,cy) is smaller than d
            indice = np.sqrt((tri_centers[:, 0] - cx)**2 +
                             (tri_centers[:, 1] - cy)**2) < diameter
            alpha[indice] = alpha_anomaly

    mesh_new = {'node': no2xy,
                'element': el2no,
                'alpha': alpha}
    return mesh_new


def demo():
    """ show you a demo """
    # number of electrodes
    numEl = 16
    pfix = pcircle(numEl=numEl)
    pts, tri = build(dunitCircle, huniform, pfix=pfix, h0=0.16, Fscale=1.2)
    elPos = np.arange(numEl)

    # 1. show nods using pylab.plot
    _, ax = plt.subplots()
    ax.plot(pts[:, 0], pts[:, 1], 'ro')
    plt.axis('equal')

    # 2. show meshes (delaunay) using triplot
    _, ax = plt.subplots()
    ax.triplot(pts[:, 0], pts[:, 1], tri)
    ax.plot(pts[elPos, 0], pts[elPos, 1], 'ro')
    # c = np.mean(p[tri], axis=1)
    plt.axis('equal')

    # 3. show voronoi tessellation
    vals = np.random.rand(pts.shape[0]) - 0.5
    _, ax = voronoi_plot(pts, tri, vals, fd=dunitCircle)
    ax.plot(pts[elPos, 0], pts[elPos, 1], 'ro')
    ax.triplot(pts[:, 0], pts[:, 1], tri, color='k', alpha=0.4)


if __name__ == "__main__":
    demo()
