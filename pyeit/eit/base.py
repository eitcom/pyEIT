# coding: utf-8
# pylint: disable=invalid-name, too-many-arguments
"""
This is a python code template that guide you through
writing your own reconstruction algorithms.
"""
from __future__ import absolute_import

import numpy as np

from .fem import Forward
from .utils import eit_scan_lines


class EitBase(object):
    """
    A base EIT solver.
    """

    def __init__(self, mesh, el_pos,
                 ex_mat=None, step=1, perm=None, parser='std'):
        """
        Parameters
        ----------
        mesh : dict
            mesh structure
        el_pos : array_like
            position (numbering) of electrodes
        ex_mat : array_like, optional
            2D array, each row is one excitation pattern
        step : int, optional
            measurement method
        perm : array_like, optional
            initial permittivity in generating Jacobian
        parser : str, optional
            parsing file format

        Notes
        -----
        parser is required for your code to be compatible with
        (a) simulation data set or (b) FMMU data set
        """
        if ex_mat is None:
            ex_mat = eit_scan_lines(len(el_pos), 8)
        if perm is None:
            perm = np.ones_like(mesh['alpha'])

        # build forward solver
        fwd = Forward(mesh, el_pos)
        self.no2xy = mesh['node']
        self.el2no = mesh['element']
        self.elPos = el_pos

        # solving Jacobian using uniform sigma distribution
        self.parser = parser
        f = fwd.solve(ex_mat, step=step, perm=perm, parser=self.parser)
        self.J, self.v0, self.B = f.Jac, f.v, f.B

        # mapping matrix
        self.H = self.B

        # call other parameters
        self.setup()

    def setup(self):
        """ setup bp """
        raise NotImplementedError

    def solve(self):
        """ solving """
        raise NotImplementedError

    def map(self):
        """ simple mat using projection matrix """
        raise NotImplementedError
