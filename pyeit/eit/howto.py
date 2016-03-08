# coding: utf-8
# pylint: disable=invalid-name, too-many-arguments
"""
This is a python code template that guide you through writing your own
reconstruction algorithm.
"""
from __future__ import absolute_import

# import numpy as np

# from .fem import forward
# from .utils import eit_scan_lines


class EIT(object):
    """
    documentation.
    """

    def __init__(self, mesh, elPos,
                 exMtx=None, step=1, perm=None, parser='std'):
        """
        documentation

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

        Note
        ----
        parser is required for your code to be compatible with
        (a) simulation dataset or (b) FMMU dataset

        Example
        -------
        if exMtx is None:
            exMtx = eit_scan_lines(len(elPos), 8)
        if perm is None:
            perm = np.ones_like(mesh['alpha'])

        # build forward solver
        fwd = forward(mesh, elPos)
        self.no2xy = mesh['node']
        self.el2no = mesh['element']
        self.elPos = elPos

        # solving Jacobian using uniform sigma distribution
        self.parser = parser
        f = fwd.solve(exMtx, step=step, perm=perm, parser=self.parser)
        self.J, self.v0, self.B = f.Jac, f.v, f.B
        """
        pass

    def solve(self, v1, v0):
        """
        documentation.

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
        pass

    def static_solve(self, v1):
        """ static solvers """
        pass
