# coding: utf-8
# pylint: disable=invalid-name, no-member
""" bp (back-projection) and f(filtered)-bp module """
from __future__ import absolute_import

import numpy as np
from .fem import forward
from .utils import eit_scan_lines


class BP(object):
    """ implement a naive inversion of (Euclidean) back projection. """

    # pylint: disable=too-many-arguments
    def __init__(self, mesh, elPos,
                 exMtx=None, step=1, perm=None, parser='std',
                 weight='none'):
        """
        bp projection initialization, default file parser is 'std'

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
        weight : str, optional
            weighting method in BP
        """
        if exMtx is None:
            exMtx = eit_scan_lines(len(elPos), 8)
        if perm is None:
            perm = np.ones_like(mesh['alpha'])

        # build forward solver
        fwd = forward(mesh, elPos)
        self.no2xy = mesh['node']
        self.el2no = mesh['element']
        self.elPos = elPos

        # solving Jacobina using uniform sigma distribution
        self.parser = parser
        f = fwd.solve(exMtx, step=step, perm=perm, parser=self.parser)
        self.J, self.v0, B = f.Jac, f.v, f.B

        # build the weighting matrix
        if weight is 'simple':
            W = self.simple_weight(B.shape[0])
            self.WB = W * B
        else:
            self.WB = B

    # pylint: enable=too-many-arguments
    def solve(self, v1, v0=None, normalize=False):
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
        # if the user do not specify any reference frame
        if v0 is None:
            v0 = self.v0
        # choose normalize method, we use sign
        if normalize:
            vn = - (v1 - v0) / np.sign(v0)
        else:
            vn = (v1 - v0)
        # smearing
        ds = np.dot(self.WB.transpose(), vn)
        return np.real(ds)

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
        w = (1.01*r - dis_e) / (1.0*r)
        # weighting by element-wise multiplication W with B
        W = np.dot(np.ones((num_voltages, 1)), w.reshape(1, -1))
        return W
