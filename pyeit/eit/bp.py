# coding: utf-8
# pylint: disable=invalid-name, no-member, arguments-differ
""" bp (back-projection) and f(filtered)-bp module """
# Copyright (c) Benyuan Liu. All Rights Reserved.
# Distributed under the (new) BSD License. See LICENSE.txt for more info.
from __future__ import division, absolute_import, print_function, annotations

from typing import Union, Optional
import numpy as np
from .base import EitBase


class BP(EitBase):
    """A naive inversion of (Euclidean) back projection."""

    def setup(
        self,
        weight: str = "none",
        perm: Optional[Union[int, float, complex, np.ndarray]] = None,
    ) -> None:
        """
        Setup BP solver

        Parameters
        ----------
        weight : str, optional
            BP parameter, by default "none"
        perm : Union[int, float, np.ndarray], optional
            If perm is not None, a prior of perm distribution is used to build the smear matrix
        """
        self.params = {"weight": weight}

        # build the weighting matrix
        # BP: in node imaging, H is the smear matrix (transpose of B)
        self.B = self.fwd.compute_b_matrix(perm=perm)
        self.H = self._compute_h(b_matrix=self.B)
        self.is_ready = True

    def _compute_h(self, b_matrix: np.ndarray) -> np.ndarray:  # type: ignore[override]
        """
        Compute H matrix for BP solver

        Parameters
        ----------
        b_matrix : np.ndarray
            BP matrix

        Returns
        -------
        np.ndarray
            H matrix
        """
        if self.params["weight"] == "simple":
            weights = self._simple_weight(b_matrix.shape[0])
            b_matrix = weights * b_matrix
        return b_matrix.T

    def solve_gs(self, v1: np.ndarray, v0: np.ndarray):
        """
        Solving using gram-schmidt orthogonalization

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
        vn = -(v1 - a * v0) / np.sign(v0.real)
        return np.dot(self.H, vn.transpose())

    def _normalize(self, v1: np.ndarray, v0: np.ndarray):
        """
        redefine normalize for BP (without amplitude normalization) using
        only the sign of v0.real. [experimental]

        Normalize current frame using the amplitude of the reference frame.
        Boundary measurements v are complex-valued

        Parameters
        ----------
        v1: np.ndarray
            current frame
        v0: np.ndarray
            referenced frame

        Returns
        -------
        np.ndarray
            Normalized current frame difference dv
        """
        return (v1 - v0) / np.sign(v0.real)

    def _simple_weight(self, num_voltages: int):
        """
        Build weighting matrix : simple, normalize by radius.

        Parameters
        ----------
        num_voltages : int
            number of equal-potential lines

        Returns
        -------
        np.ndarray
            weighting matrix

        Notes
        -----
        as in fem.py, we could either smear at,

        (1) elements, using the center co-ordinates (x,y) of each element
            >> center_e = np.mean(self.pts[self.tri], axis=1)
        (2) nodes.
        """
        d = np.sqrt(np.sum(self.mesh.node**2, axis=1))
        r = np.max(d)
        w = (1.01 * r - d) / (1.01 * r)
        # weighting by element-wise multiplication W with B
        return np.dot(np.ones((num_voltages, 1)), w.reshape(1, -1))
