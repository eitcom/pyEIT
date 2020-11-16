# coding: utf-8
# pylint: disable=invalid-name, no-member, arguments-differ
""" bp (back-projection) and f(filtered)-bp module """
# Copyright (c) Benyuan Liu. All Rights Reserved.
# Distributed under the (new) BSD License. See LICENSE.txt for more info.
from __future__ import division, absolute_import, print_function

import numpy as np
from .base import EitBase


class BP(EitBase):
    """ A naive inversion of (Euclidean) back projection. """

    def setup(self, weight="none"):
        """ setup BP """
        self.params = {"weight": weight}

        # build the weighting matrix
        if weight == "simple":
            weights = self.simple_weight(self.B.shape[0])
            self.H = weights * self.B

        # BP: H is the smear matrix B, which must be transposed for node imaging.
        self.H = self.H.T

    def map(self, dv):
        """ return Hx """
        x = -dv / self.v0_sign
        return np.dot(self.H, x)

    def solve_gs(self, v1, v0):
        """ solving using gram-schmidt """
        a = np.dot(v1, v0) / np.dot(v0, v0)
        vn = -(v1 - a * v0) / self.v0_sign
        ds = np.dot(self.H, vn)
        return ds

    def simple_weight(self, num_voltages):
        """
        building weighting matrix : simple, normalize by radius.

        Note
        ----
        as in fem.py, we could either smear at,

        (1) elements, using the center co-ordinates (x,y) of each element
            >> center_e = np.mean(self.pts[self.tri], axis=1)
        (2) nodes.

        Parameters
        ----------
        num_voltages: int
            number of equal-potential lines

        Returns
        -------
        w: NDArray
            weighting matrix
        """
        d = np.sqrt(np.sum(self.pts ** 2, axis=1))
        r = np.max(d)
        w = (1.01 * r - d) / (1.01 * r)
        # weighting by element-wise multiplication W with B
        weights = np.dot(np.ones((num_voltages, 1)), w.reshape(1, -1))
        return weights
