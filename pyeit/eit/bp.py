# coding: utf-8
# pylint: disable=invalid-name, no-member
""" bp (back-projection) and f(filtered)-bp module """
from __future__ import absolute_import

import numpy as np
from .fem import Forward
from .base import EitBase
from .utils import eit_scan_lines


class BP(EitBase):
    """ implement a naive inversion of (Euclidean) back projection. """

    def setup(self, weight='none'):
        # build the weighting matrix
        if weight is 'simple':
            weights = self.simple_weight(self.B.shape[0])
            self.H = weights * self.B

    def solve(self, v1, v0=None, normalize=True):
        """
        back projection : mapping boundary data on element
        (note) normalize method affect the shape (resolution) of bp

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
            real-valued NDArray, changes of conductivities
        """
        # without specifying any reference frame
        if v0 is None:
            v0 = self.v0
        # choose normalize method, we use sign by default
        if normalize:
            vn = -(v1 - v0) / np.sign(self.v0)
        else:
            vn = (v1 - v0)
        # smearing
        ds = np.dot(self.H.transpose(), vn)
        return np.real(ds)

    def map(self, v):
        """ return Hx """
        x = -v / np.sign(self.v0)
        return np.dot(self.H.transpose(), x)

    def solve_gs(self, v1, v0):
        """ solving using gram-schmidt """
        a = np.dot(v1, v0) / np.dot(v0, v0)
        vn = - (v1 - a*v0) / np.sign(self.v0)
        ds = np.dot(self.H.transpose(), vn)
        return ds

    def simple_weight(self, num_voltages):
        """
        building weighting matrix : simple, normalize by radius.

        Note
        ----
        as in fem.py, we could either smear at
        (1) elements using the center co-ordinates (x,y) of each element
            >> center_e = np.mean(self.no2xy[self.el2no], axis=1)
        (2) smearing at the nodes.

        Parameters
        ----------
        num_voltages : int
            number of equal-potential lines

        Returns
        -------
        NDArray
            weighting matrix
        """
        # center co-ordinates of elements
        center_e = np.mean(self.no2xy[self.el2no], axis=1)
        dis_e = np.sqrt(np.sum(center_e**2, axis=1))
        # infer r
        dis_node = np.sqrt(np.sum(self.no2xy**2, axis=1))
        r = np.max(dis_node)
        w = (1.01*r - dis_e) / (1.01*r)
        # weighting by element-wise multiplication W with B
        W = np.dot(np.ones((num_voltages, 1)), w.reshape(1, -1))
        return W
