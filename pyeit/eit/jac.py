# coding: utf-8
# pylint: disable=invalid-name
""" dynamic EIT solver using JAC """
from __future__ import absolute_import

import numpy as np
import scipy.linalg as la

from .fem import forward, CmpAoE
from .utils import eit_scan_lines


class JAC(object):
    """ implementing a JAC class """

    def __init__(self, mesh, elPos,
                 exMtx=None, step=1, perm=1., parser='et3',
                 p=0.20, lamb=0.001, method='kotre'):
        """
        JAC, default file parser is 'std'

        Parameters
        ----------
        mesh : dict
            mesh structure
        elPos : array_like
            position (numbering) of electrodes
        exMtx : array_like, optional
            2D array, each row is one excitation pattern
        step : int, optional
            measurement method
        perm : array_like, optional
            initial permitivities in generating Jacobian
        parser : str, optional
            parsing file format
        p,lamb : float
            JAC parameters
        method : str
            regularization methods
        """
        # store configuration values
        self.no2xy = mesh['node']
        self.el2no = mesh['element']
        self.elPos = elPos
        # extract structural elements, calculate area of element
        self.ae = CmpAoE(self.no2xy, self.el2no)

        # generate excitation patterns
        if exMtx is None:
            self.exMtx = eit_scan_lines(16, 8)
        else:
            self.exMtx = exMtx
        self.step = step

        # background (init, x0) perm
        n_e = np.size(self.el2no, 0)
        if np.size(perm) == n_e:
            perm_init = perm
        else:
            perm_init = perm * np.ones(n_e)

        # generate Jacobian
        self.fwd = forward(mesh, elPos)
        fs = self.fwd.solve(exMtx=self.exMtx, step=self.step,
                            perm=perm_init, parser=parser)
        self.Jac = fs.Jac
        self.v = fs.v
        self.normv = la.norm(self.v)
        self.x0 = perm_init
        self.parser = parser

        # pre-compute H0 for dynamical imaging
        # H = (J.T*J + R)^(-1) * J.T
        self.H = h_matrix(self.Jac, p, lamb, method)
        self.p = p
        self.lamb = lamb
        self.method = method

    def proj(self, ds):
        """ project ds using spatial difference filter (deprecated)

        Parameters
        ----------
        ds : NDArray
            delta sigma (conductivities)

        Returns
        -------
        NDArray
        """
        L = sar(self.el2no)
        return np.dot(L, ds)

    def solve(self, v1, v0, normalize=False):
        """ dynamic solve

        Parameters
        ----------
        v1 : NDArray
        v0 : NDArray, optional
            d = H(v1 - v0)
        normalize : Boolean
            true for conducting normalization

        Returns
        -------
        NDArray
            complex-valued NDArray, changes of conductivities
        """
        # normalize is not required for JAC
        dv = (v1 - v0)
        # s = -Hv
        ds = - np.dot(self.H, dv)
        # return average epsilon on element
        self.ds = ds / self.ae
        return self.ds

    def bp_solve(self, v1, v0, normalize=False):
        """ solve via a 'naive' back projection. """
        # normalize is not required for JAC
        dv = (v1 - v0)
        # s_r = J^Tv_r
        ds = - np.dot(self.Jac.T.conjugate(), dv)
        # return average epsilon on element
        self.ds = ds / self.ae
        return self.ds

    def gn_solve(self, v,
                 x0=None, maxiter=1,
                 p=None, lamb=None, method='kotre',
                 verbose=False):
        """
        Gaussian Newton Static Solver
        You can use a different lamb, p other than the default ones in JAC

        Parameters
        ----------
        v : NDArray
        x0 : NDArray, optional
            initial guess
        maxiter : int, optional
        p, lamb : float, optional
        method : str, optional
        verbose : bool, optional

        Returns
        -------
        NDArray
            Complex-valued conductivities
        """
        if x0 is None:
            x0 = self.x0
        if p is None:
            p = self.p
        if lamb is None:
            lamb = self.lamb
        if method is None:
            method = self.method

        """
        Gauss-Newton Iterative solver,
            x1 = x0 - (J^TJ + lamb*R)^(-1) * r0
        where:
            R = diag(J^TJ)**p
            r0 (residual) = real_measure - forward_v
        """
        for i in range(maxiter):
            if verbose:
                print('iter = ', i)
            # forward solver
            fs = self.fwd.solve(self.exMtx, step=self.step,
                                perm=x0, parser=self.parser)
            # Residual
            r0 = v - fs.v
            Jac = fs.Jac
            Jr = np.dot(Jac.T.conjugate(), r0)
            # Gaussian-Newton
            JWJ = np.dot(Jac.T.conjugate(), Jac)
            if method is 'kotre':
                R = np.diag(np.diag(JWJ) ** p)
            else:
                R = np.eye(Jac.shape[1])
            H = (JWJ + lamb*R)
            # update
            d_k = la.solve(H, Jr)
            x0 = x0 - d_k

        return x0


def h_matrix(Jac, p, lamb, method='kotre'):
    """
    JAC method of dynamic EIT solver:
        H = (J.T*J + lamb*R)^(-1) * J.T

    Parameters
    ----------
    Jac : NDArray
        Jacobian
    p, lamb : float
        regularization parameters
    method : str, optional
        regularization method

    Returns
    -------
    NDArray
        pseudo-inverse matrix of JAC
    """
    JWJ = np.dot(Jac.transpose(), Jac)
    if method is 'kotre':
        """
        see adler-dai-lionheart-2007, when
        p=0   : noise distribute on the boundary
        p=0.5 : noise distribute on the middle
        p=1   : noise distribute on the center
        """
        R = np.diag(np.diag(JWJ) ** p)
    else:
        """
        Marquardt–Levenberg, 'lm'
        """
        R = np.eye(Jac.shape[1])

    # build H
    H = np.dot(la.inv(JWJ + lamb*R), Jac.transpose())
    return H


def sar(el2no):
    """
    extract spatial difference matrix on the neighbores of each element
    in 2D fem using triangular mesh.

    Parameters
    ----------
    el2no : NDArray
        triangle structures

    Returns
    -------
    NDArray
        SAR matrix
    """
    ne = el2no.shape[0]
    L = np.eye(ne)
    for i in range(ne):
        ei = el2no[i, :]
        #
        i0 = np.argwhere(el2no == ei[0])[:, 0]
        i1 = np.argwhere(el2no == ei[1])[:, 0]
        i2 = np.argwhere(el2no == ei[2])[:, 0]
        idx = np.unique(np.hstack([i0, i1, i2]))
        # build row-i
        for j in idx:
            L[i, j] = -1
        nn = idx.size - 1
        L[i, i] = nn
    return L
