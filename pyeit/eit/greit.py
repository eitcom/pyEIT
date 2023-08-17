# coding: utf-8
# pylint: disable=invalid-name, no-member, too-many-instance-attributes
# pylint: disable=too-many-arguments, arguments-differ
"""
GREIT (using distribution method)

Note, that, the advantages of greit is NOT on simulated data, but
1. construct RM using real-life data with a stick move in the cylinder
2. construct RM on finer mesh, and use coarse-to-fine map for visualization
3. more robust to noise by adding noise via (JJ^T + lamb*Sigma_N)^{-1}
"""
# Copyright (c) Benyuan Liu. All Rights Reserved.
# Distributed under the (new) BSD License. See LICENSE.txt for more info.
from __future__ import absolute_import, division, print_function, annotations

from typing import Tuple, Union, Optional
import numpy as np
import scipy.linalg as la
from .base import EitBase
from .interp2d import rasterize, weight_sigmod


class GREIT(EitBase):
    """The GREIT algorithm"""

    def setup(
        self,
        method: str = "dist",
        w: Optional[np.ndarray] = None,
        p: float = 0.20,
        lamb: float = 1e-2,
        n: int = 32,
        s: float = 20.0,
        ratio: float = 0.1,
        perm: Optional[Union[int, float, complex, np.ndarray]] = None,
        jac_normalized: bool = False,
    ) -> None:
        """
        Setup GREIT solver

        Parameters
        ----------
        method : str, optional
            only 'dist' accepted, by default "dist"
        w : np.ndarray, optional
            weight on each element, by default None
        p : float, optional
            noise covariance, by default 0.20
        lamb : float, optional
            regularization parameters, by default 1e-2
        n : int, optional
            grid size, by default 32
        s : float, optional
            control the blur, by default 20.0
        ratio : float, optional
            desired ratio, by default 0.1
        perm : Union[int, float, np.ndarray], optional
            If perm is not None, a prior of perm distribution is used to build Jacobian
        jac_normalized : bool, optional
            normalize the jacobian using f0 computed from input perm, by
            default False

        Raises
        ------
        ValueError
            raised if method != "dist"

        References
        ----------
        [1] Bartlomiej Grychtol, Beat Muller, Andy Adler
            "3D EIT image reconstruction with GREIT"
        [2] Adler, Andy, et al.
            "GREIT: a unified approach to 2D linear EIT reconstruction of
            lung images." Physiological measurement 30.6 (2009): S35.
        """

        if method != "dist":
            raise ValueError(f"method {method} not supported yet")

        # parameters for GREIT projection
        if w is None:
            w = np.ones_like(self.mesh.perm)
        self.params = {
            "w": w,
            "p": p,
            "lamb": lamb,
            "n": n,
            "s": s,
            "ratio": ratio,
            "jac_normalize": jac_normalized,
        }

        # Build grids and mask
        self.xg, self.yg, self.mask = rasterize(self.mesh.node, self.mesh.element, n=n)

        w_mat = self._compute_grid_weights(self.xg, self.yg)
        self.J, self.v0 = self.fwd.compute_jac(perm=perm, normalize=jac_normalized)
        self.H = self._compute_h(jac=self.J, w_mat=w_mat)
        self.is_ready = True

    def _compute_h(self, jac: np.ndarray, w_mat: np.ndarray):  # type: ignore[override]
        """
        Generate H (or R) using distribution method for GREIT solver

        Args:
            jac (np.ndarray): Jacobian matrix
            w_mat (np.ndarray): meights matrix

        Returns:
            np.ndarray: H
        """
        lamb, p = self.params["lamb"], self.params["p"]
        # E[yy^T], it is more efficient to use left pinv than right pinv
        j_j_w = np.dot(jac, jac.T)
        r_mat = np.diag(np.diag(j_j_w) ** p)
        jac_inv = la.inv(j_j_w + lamb * r_mat)
        # RM = E[xx^T] / E[yy^T]
        return np.dot(np.dot(w_mat.T, jac.T), jac_inv)

    def get_grid(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Return masking grid data

        Raises
        ------
        SolverNotReadyError
            raised if solver not ready (see self._check_solver_is_ready())

        Returns
        -------
        Tuple[np.ndarray, np.ndarray, np.ndarray]
            x grid, y grid and masking data, which denotes nodes outside
            2D mesh
        """
        self._check_solver_is_ready()
        return self.xg, self.yg, self.mask

    def mask_value(
        self, ds: np.ndarray, mask_value: float = 0
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Set mask values on nodes outside 2D mesh. (for plot only)

        Parameters
        ----------
        ds : np.ndarray
            conductivity data on nodes
        mask_value : float, optional
            mask conductivity value to set on nodes outside 2D mesh, by
            default 0

        Raises
        ------
        SolverNotReadyError
            raised if solver not ready (see self._check_solver_is_ready())

        Returns
        -------
        Tuple[np.ndarray, np.ndarray, np.ndarray]
            x grid, y grid and "masked" conductivity data on nodes
        """
        self._check_solver_is_ready()
        ds[self.mask] = mask_value
        ds = ds.reshape(self.xg.shape)
        return self.xg, self.yg, ds

    def _compute_grid_weights(self, xg: np.ndarray, yg: np.ndarray):
        """
        Compute weights for given grid (xg,yg)

        Parameters
        ----------
        xg : np.ndarray
            x grid
        yg : np.ndarray
            y grid

        Returns
        -------
        np.ndarray
            weights
        """
        # mapping from values on triangles to values on grids
        xy = (
            self.mesh.elem_centers
        )  # np.mean(self.mesh.node[self.mesh.element], axis=1)
        xyi = np.vstack((xg.flatten(), yg.flatten())).T
        # GREIT is using sigmod as weighting function (global)
        ratio, s = self.params["ratio"], self.params["s"]
        return weight_sigmod(xy, xyi, ratio=ratio, s=s)

    @staticmethod
    def build_set(x: np.ndarray, y: np.ndarray):
        """generate R from a set of training sets (deprecate)."""
        # E_w[yy^T]
        y_y_t = la.inv(np.dot(y, y.transpose()))
        return np.dot(np.dot(x, y), y_y_t)
