# coding: utf-8
# pylint: disable=invalid-name, no-member, no-name-in-module
# pylint: disable=too-many-arguments, too-many-locals
# pylint: disable=too-many-instance-attributes
""" implement 2D/3D distmesh """
# Copyright (c) Benyuan Liu. All rights reserved.
# Distributed under the (new) BSD License. See LICENSE.txt for more info.
from __future__ import division, absolute_import, print_function

from itertools import combinations
import numpy as np
from numpy import sqrt
from scipy.spatial import Delaunay
from scipy.sparse import csr_matrix

from .utils import dist, edge_project

from .shape import thorax


class DISTMESH:
    """class for distmesh"""

    def __init__(
        self,
        fd,
        fh,
        h0=0.1,
        p_fix=None,
        bbox=None,
        density_ctrl_freq=30,
        deltat=0.1,
        dptol=0.001,
        ttol=0.1,
        Fscale=1.2,
        verbose=False,
    ):
        """initial distmesh class

        Parameters
        ----------
        fd : str
            function handle for distance of boundary
        fh : str
            function handle for distance distributions
        h0 : float, optional
            Distance between points in the initial distribution p0,
            default=0.1 For uniform meshes, h(x,y) = constant,
            the element size in the final mesh will usually be
            a little larger than this input.
        p_fix : array_like, optional
            fixed points, default=[]
        bbox : array_like, optional
            bounding box for region, bbox=[xmin, ymin, xmax, ymax].
            default=[-1, -1, 1, 1]
        density_ctrl_freq : int, optional
            cycles of iterations of density control, default=20
        deltat : float, optional
            mapping forces to distances, default=0.2
        dptol : float, optional
            exit criterion for minimal distance all points moved, default=0.01
        ttol : float, optional
            enter criterion for re-delaunay the lattices, default=0.1
        Fscale : float, optional
            rescaled string forces, default=1.2
            if set too small, points near boundary will be pushed back
            if set too large, points will be pushed towards boundary

        Notes
        -----
        """
        # shape description
        self.fd = fd
        self.fh = fh
        self.h0 = h0

        # a small gap, allow points who are slightly outside of the region
        self.deps = np.sqrt(np.finfo(np.double).eps) * h0
        self.geps = 1e-1 * h0

        # control the distmesh computation flow
        self.densityctrlfreq = density_ctrl_freq
        self.dptol = dptol
        self.ttol = ttol
        self.Fscale = Fscale
        self.deltat = deltat

        # default bbox is 2D
        if bbox is None:
            bbox = [[-1, -1], [1, 1]]
        # p : coordinates (x,y) or (x,y,z) of meshes
        self.n_dim = np.shape(bbox)[1]
        if self.n_dim == 2:
            p = bbox2d_init(h0, bbox)
        else:
            p = bbox3d_init(h0, bbox)

        # control debug messages
        self.verbose = verbose
        self.num_triangulate = 0
        self.num_density = 0
        self.num_move = 0

        """
        keep points that are inside the thorax shape using a function that returns a matrix containing
        True if the corresponing point is inside the shape, False if not.
        """
        if fd == thorax:
            p = p[fd(p)]
        else:
            # keep points inside (minus distance) with a small gap (geps)
            p = p[fd(p) < self.geps]  # pylint: disable=E1136

        # rejection points by sampling on fh
        r0 = 1.0 / fh(p) ** self.n_dim
        selection = np.random.rand(p.shape[0]) < (r0 / np.max(r0))
        p = p[selection]

        # specify fixed points
        if p_fix is None:
            p_fix = []
        self.pfix = p_fix
        self.nfix = len(p_fix)

        # convert boolean array to 2D to be compatible with Delaunay pts paramater (must be 2D)
        if fd == thorax:
            p = np.reshape(p, (-1, 2))

        # remove duplicated points of p and p_fix
        # avoid overlapping of mesh points
        if self.nfix > 0:
            p = remove_duplicate_nodes(p, p_fix, self.geps)
            p = np.vstack([p_fix, p])

        if fd == thorax:
            p = np.reshape(
                p, (-1, 2)
            )  # convert boolean array to 2D to be compatible with Delaunay pts paramater (must be 2D)

        # store p and N
        self.N = p.shape[0]
        self.p = p

        # initialize pold with inf: it will be re-triangulate at start
        self.pold = np.inf * np.ones((self.N, self.n_dim))

        # build edges list for triangle or tetrahedral. i.e., in 2D triangle
        # edge_combinations is [[0, 1], [1, 2], [2, 0]]
        self.edge_combinations = list(combinations(range(self.n_dim + 1), 2))

        # triangulate, generate simplices and bars
        self.triangulate()

    def is_retriangulate(self):
        """test whether re-triangulate is needed"""
        return np.max(dist(self.p - self.pold)) > (self.h0 * self.ttol)

    def triangulate(self):
        """retriangle by delaunay"""
        self.debug("enter triangulate = ", self.num_triangulate)
        self.num_triangulate += 1
        # pnew[:] = pold[:] makes a new copy, not reference
        self.pold = self.p.copy()

        # triangles where the points are arranged counterclockwise
        if self.fd != thorax:
            tri = Delaunay(self.p).simplices
        else:
            tri = Delaunay(
                self.p, qhull_options="QJ"
            ).simplices  # QJ parameter so tuples don't exceed boundary

        pmid = np.mean(self.p[tri], axis=1)

        if self.fd != thorax:
            # keeps only interior points
            t = tri[self.fd(pmid) < -self.geps]
        else:
            # adapting returned triangles matrix with the thorax integrated fd
            tri_pmid = [p[0] for p in self.fd(pmid)]
            tri_pmid = np.array(tri_pmid)
            t = tri[tri_pmid]
        # extract edges (bars)
        bars = t[:, self.edge_combinations].reshape((-1, 2))
        # sort and remove duplicated edges, eg (1,2) and (2,1)
        # note : for all edges, non-duplicated edge is boundary edge
        bars = np.sort(bars, axis=1)
        bars_tuple = bars.view([("", bars.dtype)] * bars.shape[1])

        self.bars = np.unique(bars_tuple).view(bars.dtype).reshape((-1, 2))
        self.t = t

    def bar_length(self):
        """the forces of bars (python is by-default row-wise operation)"""
        # two node of a bar
        bars_a, bars_b = self.p[self.bars[:, 0]], self.p[self.bars[:, 1]]
        barvec = bars_a - bars_b

        # L : length of bars, must be column ndarray (2D)
        L = dist(barvec).reshape((-1, 1))
        # density control on bars
        hbars = self.fh((bars_a + bars_b) / 2.0).reshape((-1, 1))
        # L0 : desired lengths (Fscale matters!)
        L0 = hbars * self.Fscale * sqrt(np.sum(L**2) / np.sum(hbars**2))

        return L, L0, barvec

    def bar_force(self, L, L0, barvec):
        """forces on bars"""
        # abs(forces)
        F = np.maximum(L0 - L, 0)
        # normalized and vectorized forces
        Fvec = F * (barvec / L)
        # now, we get forces and sum them up on nodes
        # using sparse matrix to perform automatic summation
        # rows : left, left, right, right (2D)
        # : left, left, left, right, right, right (3D)
        # cols : x, y, x, y (2D)
        # : x, y, z, x, y, z (3D)
        data = np.hstack([Fvec, -Fvec])
        if self.n_dim == 2:
            rows = self.bars[:, [0, 0, 1, 1]]
            cols = np.dot(np.ones(np.shape(F)), np.array([[0, 1, 0, 1]]))
        else:
            rows = self.bars[:, [0, 0, 0, 1, 1, 1]]
            cols = np.dot(np.ones(np.shape(F)), np.array([[0, 1, 2, 0, 1, 2]]))
        # sum nodes at duplicated locations using sparse matrices
        Ftot = csr_matrix(
            (data.reshape(-1), [rows.reshape(-1), cols.reshape(-1)]),
            shape=(self.N, self.n_dim),
        )
        Ftot = Ftot.toarray()
        # zero out forces at fixed points, as they do not move
        Ftot[0 : len(self.pfix)] = 0

        return Ftot

    def density_control(self, L, L0, dscale=3.0):
        """
        Density control - remove points that are too close
        L0 : Kx1, L : Kx1, bars : Kx2
        bars[L0 > 2*L] only returns bar[:, 0] where L0 > 2L
        """
        self.debug("enter density control = ", self.num_density)
        self.num_density += 1
        # print(self.num_density, self.p.shape)
        # quality control
        ixout = (L0 > dscale * L).ravel()
        ixdel = np.setdiff1d(self.bars[ixout, :].reshape(-1), np.arange(self.nfix))
        self.p = self.p[np.setdiff1d(np.arange(self.N), ixdel)]
        # Nold = N
        self.N = self.p.shape[0]
        self.pold = np.inf * np.ones((self.N, self.n_dim))
        # print('density control ratio : %f' % (float(N)/Nold))

    def move_p(self, Ftot):
        """update p"""
        self.debug("  number of moves = ", self.num_move)
        self.num_move += 1
        # move p along forces
        self.p += self.deltat * Ftot

        # if there is any point ends up outside
        # move it back to the closest point on the boundary
        # using the numerical gradient of distance function
        d = self.fd(self.p)
        ix = d > 0
        if ix.any():
            self.p[ix] = edge_project(self.p[ix], self.fd, self.geps)

        # check whether convergence : no big movements
        delta_move = self.deltat * np.max(dist(Ftot[d < -self.geps]))
        self.debug("  delta_move = ", delta_move)
        score = delta_move < self.dptol * self.h0

        return score

    def debug(self, *args):
        """print debug messages"""
        if self.verbose:
            print(*args)


def bbox2d_init(h0, bbox):
    """
    generate points in 2D bbox (not including the ending point of bbox)

    Parameters
    ----------
    h0 : float
        minimal distance of points
    bbox : array_like
        [[x0, y0],
         [x1, y1]]

    Returns
    -------
    array_like
        points in bbox
    """
    x, y = np.meshgrid(
        np.arange(bbox[0][0], bbox[1][0], h0),
        np.arange(bbox[0][1], bbox[1][1], h0 * sqrt(3) / 2.0),
        indexing="xy",
    )
    # shift even rows of x
    # shift every second row h0/2 to the right, therefore,
    # all points will be a distance h0 from their closest neighbors
    x[1::2, :] += h0 / 2.0
    # p : Nx2 ndarray
    p = np.array([x.ravel(), y.ravel()]).T
    return p


def bbox3d_init(h0, bbox):
    """
    generate nodes in 3D bbox

    See Also
    --------
    bbox2d : converting bbox to 2D points
    """
    xspace = h0
    yspace = h0 * sqrt(3) / 2.0
    zspace = h0 * sqrt(3 / 2.0)
    # build meshgrid with Cartesian indexing
    x, y, z = np.meshgrid(
        np.arange(bbox[0][0], bbox[1][0], xspace),
        np.arange(bbox[0][1], bbox[1][1], yspace),
        np.arange(bbox[0][2], bbox[1][2], zspace),
        indexing="xy",
    )

    # shift every second row of x h0/2 to the right, therefore,
    # all points on xy-plane will have equal distance h0
    # from their closest neighbors
    x[1::2, :, :] += h0 / 2.0
    # shift every second z, where x += h0/2, y += h0/(2*sqrt(3))
    # note : in tetrahedral, the distance to all neighbores of a point
    # is not equilength, aka, equi-tetrahedral can not fill space.
    x[:, :, 1::2] += h0 / 2.0
    y[:, :, 1::2] += h0 / (2.0 * sqrt(3))
    # p : Nx3 ndarray
    p = np.array([x.ravel(), y.ravel(), z.ravel()]).T
    return p


def remove_duplicate_nodes(p, pfix, geps):
    """remove duplicate points in p who are closed to pfix. 3D, ND compatible

    Parameters
    ----------
    p : array_like
        points in 2D, 3D, ND
    pfix : array_like
        points that are fixed (can not be moved in distmesh)
    geps : float, optional (default=0.01*h0)
        minimal distance that two points are assumed to be identical

    Returns
    -------
    array_like
        non-duplicated points
    """
    for row in pfix:
        pdist = dist(p - row)
        # extract non-duplicated row slices
        p = p[pdist > geps]
    return p


def build(
    fd, fh, pfix=None, bbox=None, h0=0.1, densityctrlfreq=10, maxiter=500, verbose=False
):
    """main function for distmesh

    See Also
    --------
    DISTMESH : main class for distmesh

    Parameters
    ----------
    maxiter : int, optional
        maximum iteration numbers, default=1000

    Returns
    -------
    p : array_like
        points on 2D bbox
    t : array_like
        triangles describe the mesh structure

    Notes
    -----
    there are many python or hybrid python + C implementations in github,
    this implementation is merely implemented from scratch
    using PER-OLOF PERSSON's Ph.D thesis and SIAM paper.

    .. [1] P.-O. Persson, G. Strang, "A Simple Mesh Generator in MATLAB".
       SIAM Review, Volume 46 (2), pp. 329-345, June 2004

    Also, the user should be aware that, equal-edged tetrahedron cannot fill
    space without gaps. So, in 3D, you can lower dptol, or limit the maximum
    iteration steps.

    """
    # parsing arguments
    # make sure : g_Fscale < 1.5
    mode_3D = False
    if bbox is None:
        g_dptol, g_ttol, g_Fscale, g_deltat = 0.001, 0.1, 1.2, 0.2
    else:
        # perform error check on bbox
        bbox = np.array(bbox)
        if (bbox.ndim == 1) or (bbox.shape[1] not in [2, 3]):
            raise TypeError("only 2D, 3D are supported, bbox = ", bbox)
        if bbox.shape[0] != 2:
            raise TypeError("please specify lower and upper bound of bbox")
        if bbox.shape[1] == 2:
            # default parameters for 2D
            g_dptol, g_ttol, g_Fscale, g_deltat = 0.001, 0.1, 1.3, 0.2
        else:
            # default parameters for 3D
            # g_dptol, g_ttol, g_Fscale = 0.045, 0.150, 1.125
            mode_3D = True
            g_dptol, g_ttol, g_Fscale, g_deltat = 0.001, 0.1, 1.1, 0.1

    # initialize distmesh
    dm = DISTMESH(
        fd,
        fh,
        h0=h0,
        p_fix=pfix,
        bbox=bbox,
        density_ctrl_freq=densityctrlfreq,
        deltat=g_deltat,
        dptol=g_dptol,
        ttol=g_ttol,
        Fscale=g_Fscale,
        verbose=verbose,
    )

    # now iterate to push to equilibrium
    for i in range(maxiter):
        if dm.is_retriangulate():
            # print("triangulate = %d" % dm.num_triangulate)
            dm.triangulate()

        # calculate bar forces
        L, L0, barvec = dm.bar_length()

        # density control
        if not mode_3D:
            if (i % densityctrlfreq) == 0 and (L0 > 2 * L).any():
                dm.density_control(L, L0)
                # continue to triangulate
                continue

        # calculate bar forces
        Ftot = dm.bar_force(L, L0, barvec)

        if fd != thorax:
            # update p
            converge = dm.move_p(Ftot)
            # the stopping ctriterion (movements interior are small)
            if converge:
                break
        else:  # Thorax mesh is created so far without iteration process (to be updated)
            break

    # at the end of iteration, (p - pold) is small, so we recreate delaunay
    dm.triangulate()

    # you should remove duplicate nodes and triangles
    return dm.p, dm.t
