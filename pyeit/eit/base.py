# coding: utf-8
# pylint: disable=invalid-name, too-many-arguments
# pylint: disable=too-many-instance-attributes
"""
This is a python code template that guide you through
writing your own reconstruction algorithms.
"""
# author: benyuan liu
from __future__ import division, absolute_import, print_function

import numpy as np

from .fem import Forward
from .utils import eit_scan_lines


class EitBase(object):
    """
    A base EIT solver.
    """

    def __init__(self, mesh, el_pos,
                 ex_mat=None, step=1, perm=1., parser='std'):
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
            perm = mesh['alpha']

        # build forward solver
        fwd = Forward(mesh, el_pos)
        self.fwd = fwd

        # solving mesh structure
        self.mesh = mesh
        self.no2xy = mesh['node']
        self.el2no = mesh['element']
        # shape of the mesh
        self.no_num, self.dim = self.no2xy.shape
        self.el_num, self.n_vertices = self.el2no.shape
        self.elPos = el_pos
        self.parser = parser

        # solving configurations
        self.ex_mat = ex_mat
        self.step = step
        # user may specify a scalar for uniform permittivity
        if np.size(perm) == 1:
            self.perm = perm * np.ones(self.el_num)
        else:
            self.perm = perm

        # solving Jacobian using uniform sigma distribution
        res = fwd.solve(ex_mat, step=step, perm=self.perm, parser=self.parser)
        self.J, self.v0, self.B = res.jac, res.v, res.b_matrix

        # mapping matrix
        self.H = self.B

        # initialize other parameters
        self.params = {}
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
