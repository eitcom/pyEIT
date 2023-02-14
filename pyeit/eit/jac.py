# coding: utf-8
# pylint: disable=invalid-name, no-member, too-many-arguments
# pylint: disable=too-many-instance-attributes, too-many-locals
# pylint: disable=arguments-differ
""" dynamic EIT solver using JAC """
# Copyright (c) Benyuan Liu. All Rights Reserved.
# Distributed under the (new) BSD License. See LICENSE.txt for more info.
from __future__ import division, absolute_import, print_function, annotations

from typing import Union, Optional
import numpy as np
import scipy.linalg as la
from .base import EitBase


class JAC(EitBase):
    """A sensitivity-based EIT imaging class"""

    def setup(
        self,
        p: float = 0.20,
        lamb: float = 0.001,
        method: str = "kotre",
        perm: Optional[Union[int, float, complex, np.ndarray]] = None,
        jac_normalized: bool = False,
    ) -> None:
        """
        Setup JAC solver

        Jacobian matrix based reconstruction.

        Parameters
        ----------
        p : float, optional
            JAC parameters, by default 0.20
        lamb : float, optional
            JAC parameters, by default 0.001
        method : str, optional
            regularization methods ("kotre", "lm", "dgn" ), by default "kotre"
        perm : Union[int, float, np.ndarray], optional
            If perm is not None, a prior of perm distribution is used to build jac
        jac_normalized : bool, optional
            normalize the jacobian using f0 computed from input perm, by
            default False
        """
        # passing imaging parameters
        self.params: dict = {
            "p": p,
            "lamb": lamb,
            "method": method,
            "jac_normalize": jac_normalized,
        }
        # pre-compute H0 for dynamical imaging
        # H = (J.T*J + R)^(-1) * J.T
        self.J, self.v0 = self.fwd.compute_jac(perm=perm, normalize=jac_normalized)
        self.H = self._compute_h(self.J, p, lamb, method)
        self.is_ready = True

    def _compute_h(  # type: ignore[override]
        self, jac: np.ndarray, p: float, lamb: float, method: str = "kotre"
    ):
        """
        Compute self.H matrix for JAC solver

        JAC method of dynamic EIT solver:
            H = (J.T*J + lamb*R)^(-1) * J.T

        Parameters
        ----------
        jac : np.ndarray
            Jacobian
        p : float
            Regularization parameter, the p in R=diag(diag(JtJ) ** p)
        lamb : float
            Regularization parameter, the lambda in (JtJ + lambda*R)^{-1}
        method : str, optional
            Regularization method, ("kotre", "lm", "dgn" ), by default "kotre".
            Note that the name method="kotre" uses regularization alike the one
            in adler-dai-lionheart-2007 (pp4):
            "Temporal Image Reconstruction in Electrical Impedance Tomography",
            it regularize the diagonal of JtJ by an exponential parameter p.

        Returns
        -------
        np.ndarray
            H matrix, pseudo-inverse matrix of JAC
        """
        j_w_j = np.dot(jac.transpose(), jac)
        if method == "kotre":
            # p=0   : noise distribute on the boundary ('dgn')
            # p=0.5 : noise distribute on the middle
            # p=1   : noise distribute on the center ('lm')
            r_mat = np.diag(np.diag(j_w_j) ** p)
        elif method == "lm":
            # Marquardt–Levenberg, 'lm' for short
            # or can be called NOSER, DLS
            r_mat = np.diag(np.diag(j_w_j))
        else:
            # Damped Gauss Newton, 'dgn' for short
            r_mat = np.eye(jac.shape[1])

        # build H
        return np.dot(la.inv(j_w_j + lamb * r_mat), jac.transpose())

    def solve_gs(self, v1: np.ndarray, v0: np.ndarray):
        """
        Solving by weighted frequency

        Parameters
        ----------
        v1: np.ndarray
            current frame
        v0: np.ndarray
            referenced frame

        Raises
        ------
        SolverNotReadyError
            raised if solver not ready (see self._check_solver_is_ready())

        Returns
        -------
        np.ndarray
            complex-valued np.ndarray, changes of conductivities
        """
        self._check_solver_is_ready()
        a = np.dot(v1, v0) / np.dot(v0, v0)
        dv = v1 - a * v0
        # return ds average epsilon on element
        return -np.dot(self.H, dv.transpose())

    def jt_solve(self, v1: np.ndarray, v0: np.ndarray, normalize: bool = True):
        """
        a 'naive' back projection using the transpose of Jac.
        This scheme is the one published by kotre (1989), see note [1].

        Parameters
        ----------
        v1: np.ndarray
            current frame
        v0: np.ndarray
            referenced frame
        normalize : bool, optional
            flag to log-normalize the current frame difference dv, by default
            True. The input (dv) and output (ds) is log-normalized.

        Raises
        ------
        SolverNotReadyError
            raised if solver not ready (see self._check_solver_is_ready())

        Returns
        -------
        np.ndarray
            complex-valued np.ndarray, changes of conductivities

        Notes
        -----
            [1] Kotre, C. J. (1989).
                A sensitivity coefficient method for the reconstruction of
                electrical impedance tomograms.
                Clinical Physics and Physiological Measurement,
                10(3), 275--281. doi:10.1088/0143-0815/10/3/008

        """
        self._check_solver_is_ready()
        if normalize:
            dv = np.log(np.abs(v1) / np.abs(v0)) * np.sign(v0.real)
        else:
            dv = (v1 - v0) * np.sign(v0.real)
        # s_r = J^Tv_r
        ds = -np.dot(self.J.conj().T, dv)
        return np.exp(ds) - 1.0

    def gn(
        self,
        v: np.ndarray,
        x0: Optional[Union[int, float, complex, np.ndarray]] = None,
        maxiter: int = 1,
        gtol: float = 1e-4,
        p: Optional[float] = None,
        lamb: Optional[float] = None,
        lamb_decay: float = 1.0,
        lamb_min: float = 0.0,
        method: str = "kotre",
        verbose: bool = False,
        generator: bool = False,
        **kwargs,
    ):
        """
        Gaussian Newton Static Solver
        You can use a different p, lamb other than the default ones in setup

        Parameters
        ----------
        v : np.ndarray
            boundary measurement
        x0 : Union[int, float, np.ndarray], optional
            initial permittivity guess, by default None
            (see Foward._get_perm for more details, in fem.py)
        maxiter : int, optional
            number of maximum iterations, by default 1
        gtol : float, optional
            convergence threshold, by default 1e-4
        p : float, optional
            JAC parameters (can be overridden), by default None
        lamb : float, optional
            JAC parameters (can be overridden), by default None
        lamb_decay : float, optional
            decay of lamb0, i.e., lamb0 = lamb0 * lamb_delay of each iteration,
            by default 1.0
        lamb_min : float, optional
            minimal value of lamb, by default 0.0
        method : str, optional
            regularization methods ("kotre", "lm", "dgn" ), by default "kotre"
        verbose : bool, optional
            verbose flag, by default False

        Raises
        ------
        SolverNotReadyError
            raised if solver not ready (see self._check_solver_is_ready())

        Returns
        -------
        np.ndarray
            Complex-valued conductivities, sigma

        Note
        ----
        Gauss-Newton Iterative solver,
            x1 = x0 - (J^TJ + lamb*R)^(-1) * r0
        where:
            R = diag(J^TJ)**p
            r0 (residual) = real_measure - forward_v
        """
        self._check_solver_is_ready()
        if x0 is None:
            x0 = self.mesh.perm
        if p is None:
            p = self.params["p"]
        if lamb is None:
            lamb = self.params["lamb"]
        if method is None:
            method = self.params["method"]

        # convergence test
        x0_norm = np.linalg.norm(x0)

        def generator_gn():
            nonlocal x0, lamb
            for i in range(maxiter):
                # forward solver,
                jac, v0 = self.fwd.compute_jac(x0)
                # Residual
                r0 = v - v0

                # Damped Gaussian-Newton
                h_mat = self._compute_h(jac, p, lamb, method)

                # update
                d_k = np.dot(h_mat, r0)
                x0 = x0 - d_k

                # convergence test
                c = np.linalg.norm(d_k) / x0_norm
                if c < gtol:
                    break

                if verbose:
                    print("iter = %d, lamb = %f, gtol = %f" % (i, lamb, c))

                # update regularization parameter
                # lambda can be given in user defined decreasing lists
                lamb *= lamb_decay
                lamb = max(lamb, lamb_min)
                yield x0

        real_gen = generator_gn
        if not generator:
            item = None
            for item in real_gen():
                pass
            return item
        else:
            return real_gen()

    def project(self, ds: np.ndarray):
        """
        Project ds using spatial difference filter (deprecated)

        Parameters
        ----------
        ds : np.ndarray
            delta sigma (conductivities)

        Returns
        -------
        np.ndarray
            _description_
        """
        """project ds using spatial difference filter (deprecated)

        Parameters
        ----------
        ds: np.ndarray
            delta sigma (conductivities)

        Returns
        -------
        np.ndarray
        """
        d_mat = sar(self.mesh.element)
        return np.dot(d_mat, ds)


def h_matrix(jac: np.ndarray, p: float, lamb: float, method: str = "kotre"):
    """
    (NOT USED in JAC solver)
    JAC method of dynamic EIT solver:
        H = (J.T*J + lamb*R)^(-1) * J.T

    Parameters
    ----------
    jac : np.ndarray
        Jacobian
    p : float
        regularization parameter
    lamb : float
        regularization parameter
    method : str, optional
        regularization method, ("kotre", "lm", "dgn" ), by default "kotre"

    Returns
    -------
    np.ndarray
        H matrix, pseudo-inverse matrix of JAC
    """
    j_w_j = np.dot(jac.transpose(), jac)
    if method == "kotre":
        # see adler-dai-lionheart-2007
        # p=0   : noise distribute on the boundary ('dgn')
        # p=0.5 : noise distribute on the middle
        # p=1   : noise distribute on the center ('lm')
        r_mat = np.diag(np.diag(j_w_j)) ** p
    elif method == "lm":
        # Marquardt–Levenberg, 'lm' for short
        # or can be called NOSER, DLS
        r_mat = np.diag(np.diag(j_w_j))
    else:
        # Damped Gauss Newton, 'dgn' for short
        r_mat = np.eye(jac.shape[1])

    # build H
    return np.dot(la.inv(j_w_j + lamb * r_mat), jac.transpose())


def sar(el2no: np.ndarray) -> np.ndarray:
    """
    Extract spatial difference matrix on the neighbors of each element
    in 2D fem using triangular mesh.

    Parameters
    ----------
    el2no : np.ndarray
        triangle structures

    Returns
    -------
    np.ndarray
        SAR matrix
    """
    ne = el2no.shape[0]
    d_mat = np.eye(ne)
    for i in range(ne):
        ei = el2no[i, :]
        #
        i0 = np.argwhere(el2no == ei[0])[:, 0]
        i1 = np.argwhere(el2no == ei[1])[:, 0]
        i2 = np.argwhere(el2no == ei[2])[:, 0]
        idx = np.unique(np.hstack([i0, i1, i2]))
        # build row-i
        for j in idx:
            d_mat[i, j] = -1
        nn = idx.size - 1
        d_mat[i, i] = nn
    return d_mat
