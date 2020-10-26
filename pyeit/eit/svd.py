# coding: utf-8
# pylint: disable=invalid-name, no-member, arguments-differ
""" dynamic EIT solver using SVD """
# Copyright (c) Benyuan Liu. All Rights Reserved.
# Distributed under the (new) BSD License. See LICENSE.txt for more info.
from __future__ import division, absolute_import, print_function

import numpy as np

from .jac import JAC


class SVD(JAC):
    """ implementing a sensitivity-based EIT imaging class """

    def setup(self, n=25, rcond=1e-2, method="svd"):
        """
        SVD, singular value decomposition based reconstruction.

        Parameters
        ----------
        n: int
            largest n eigenvalues to be kept
        rcond: double
            r-condition number of pinv
        method: string
            'svd': SVD truncation,
            'pinv': pseudo inverse
        """
        # correct n_ord
        nm, ne = self.J.shape
        n_ord = np.min([nm, ne, n])

        # passing imaging parameters
        self.params = {"n": n_ord, "rcond": rcond, "method": method}

        # pre-compute H0 for dynamical imaging
        if method == "svd":
            JtJ = np.dot(self.J.T, self.J)

            # using svd
            # U, s, Ut = np.linalg.svd(JtJ)
            # U = U[:, :n_ord]
            # s = s[:n_ord]

            # using eigh (more faster for large, symmetric matrix)
            s, U = np.linalg.eigh(JtJ)
            idx = np.argsort(s)[::-1]
            s = s[idx[:n_ord]]
            U = U[:, idx[:n_ord]]

            # pseudo inverse
            JtJ_inv = np.dot(U, np.dot(np.diag(s ** -1), U.T))
            self.H = np.dot(JtJ_inv, self.J.T)
        elif method == "pinv":
            self.H = np.linalg.pinv(self.J, rcond=rcond)
