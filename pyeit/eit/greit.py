# coding: utf-8
# pylint: disable=invalid-name, no-member, too-many-instance-attributes
# pylint: disable=too-many-arguments
"""
GREIT (using distribution method)

Note, that, the advantages of greit is NOT on simulated data, but
1. construct RM using real-life data with a stick move in the cylinder
2. construct RM on finer mesh, and use coarse-to-fine map for visualization
3. more robust to noise by adding noise via (JJ^T + lamb*Sigma_N)^{-1}

liubenyuan <liubenyuan@gmail.com>, 2016-01-27
"""
from __future__ import absolute_import

import numpy as np
import scipy.linalg as la
from matplotlib.path import Path

from .fem import forward
from .utils import eit_scan_lines


class GREIT(object):
    """ the GREIT algorithm """

    def __init__(self, mesh, elPos, method='dist',
                 w=None, p=0.20, lamb=1e-2, N=32, s=20., ratio=0.1,
                 exMtx=None, step=1, perm=None, parser='std'):
        """ GREIT algorithm

        Parameters
        ----------
        mesh : dict
            mesh structure
        elPos : NDArray
            numbering of electrodes
        method : str, optional
            'set' or 'dist'
        w : NDArray, optional
            weight on each element
        lamb : float, optional
            noise covariance
        N : int, optional
            grid size
        s : float, optional
            control the blur
        ratio : float, optional
            desired ratio
        exMtx : NDArray, optional
            excitation matrix
        step : int, optional
            measurement step (method)
        perm : NDArray, optional
            initial perm used to calculate Jacobian
        parser : str, optional
            data file (data structure parser)

        Note
        ----
        .. [1] Bartlomiej Grychtol, Beat Muller, Andy Adler
           "3D EIT image reconstruction with GREIT"
        .. [2] Adler, Andy, et al. "GREIT: a unified approach to
           2D linear EIT reconstruction of lung images."
           Physiological measurement 30.6 (2009): S35.
        """
        # 1. update mesh and initialize forward solver
        self.mesh = mesh
        self.no2xy = mesh['node']
        self.el2no = mesh['element']
        self.elPos = elPos
        self.fwd = forward(mesh, elPos)
        # 2. parameters for forward solver
        if exMtx is None:
            exMtx = eit_scan_lines(len(elPos), 8)
        if perm is None:
            perm = np.ones_like(mesh['alpha'])
        self.exMtx = exMtx
        self.step = step
        self.perm = perm
        self.parser = parser
        # 3. parameters for GREIT projection
        if w is None:
            self.w = np.ones_like(mesh['alpha'])
        self.p = p
        self.lamb = lamb
        self.N = N
        self.s = s
        self.ratio = ratio
        # action (currently only support set method)
        if method is 'dist':
            self.RM = self._build_dist()
        else:
            raise ValueError('method ' + method + ' not supported yet')

    def _build_dist(self):
        """ generate R using distribution method. """
        f = self.fwd.solve(self.exMtx, step=self.step, perm=self.perm,
                           parser=self.parser)
        J = f.Jac
        # build D on grids
        self.xg, self.yg, self.mask = self._build_grid()
        rmax = self._get_rmax()
        D = self._psf_grid(rmax)
        # E[yy^T]
        JJW = np.dot(J, J.transpose())
        R = np.diag(np.diag(JJW) ** self.p)
        Jinv = la.inv(JJW + self.lamb*R)
        # RM = E[xx^T] / E[yy^T]
        RM = np.dot(np.dot(D, J.transpose()), Jinv)
        return RM

    def _build_grid(self):
        """ building grid from mesh in 2D. grid size should be small """
        xmin, xmax = min(self.no2xy[:, 0]), max(self.no2xy[:, 0])
        ymin, ymax = min(self.no2xy[:, 1]), max(self.no2xy[:, 1])
        xv = np.linspace(xmin, xmax, num=self.N, endpoint=True)
        yv = np.linspace(ymin, ymax, num=self.N, endpoint=True)
        xg, yg = np.meshgrid(xv, yv, sparse=False, indexing='xy')
        # 1. create mask based on meshes
        x, y = xg.flatten(), yg.flatten()
        points = np.vstack((x, y)).T
        # 2. extract edge point using el_pos
        edge_points = self.no2xy[self.elPos]
        path = Path(edge_points, closed=False)
        mask = path.contains_points(points)

        return xg, yg, mask

    @staticmethod
    def _distance2d(x, y, center=None):
        """ Calculate radius given center. This function can be OPTIMIZED """
        if center is None:
            xc, yc = np.mean(x), np.mean(y)
        else:
            xc, yc = center[0], center[1]
        r = np.sqrt((x-xc)**2 + (y-yc)**2).ravel()
        return r

    def _get_rmax(self):
        """
        calculate max radius using mask and xy

        Notes
        -----
        mask is a 1d boolean array, xg and yg are 2D meshgrids,
        so mask should be reshaped before indexing (numpy 1.10)
        """
        mask = self.mask.reshape(self.xg.shape)
        r = self._distance2d(self.xg[mask], self.yg[mask])
        return np.max(r)

    def _psf_grid(self, rmax=1.):
        """ point spread function (psf) mapping (convolve)
        values of elements on grid.
        """
        ne = self.el2no.shape[0]
        D = np.zeros((self.N**2, ne))
        R = rmax * self.ratio
        # loop over all elements
        for i in range(ne):
            ei = self.el2no[i, :]
            xy = np.mean(self.no2xy[ei], axis=0)
            # there may be bias between grid and center of a element
            r = self._distance2d(self.xg, self.yg, center=xy)
            f = 1./(1.+np.exp(self.s*np.abs(r))/np.exp(self.s*R))
            D[:, i] = f
        return D

    def mask_value(self, ds, mask_value=0.0):
        """ mask values on nodes outside 2D mesh. """
        ds[self.mask == 0] = mask_value
        ds = ds.reshape(self.xg.shape)
        return self.xg, self.yg, ds

    @staticmethod
    def build_set(X, Y):
        """ generate R from a set of training sets (deprecate). """
        # E_w[yy^T]
        YYT = la.inv(np.dot(Y, Y.transpose()))
        RM = np.dot(np.dot(X, Y), YYT)
        return RM

    def solve(self, v1, v0, normalize=False):
        """ solving and interpolating (psf convolve) on grids. """
        if normalize:
            dv = -(v1 - v0)/v0
        else:
            dv = v1 - v0
        return -np.dot(self.RM, dv)

    def map_h(self, X):
        """ return RM*X """
        return -np.dot(self.RM, X)


# pylint: disable=too-few-public-methods
class GREIT3D(object):
    """ 3D GREIT algorithm. """

    def __init__(self):
        pass
