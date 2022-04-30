# coding: utf-8
# pylint: disable=invalid-name, too-many-arguments
# pylint: disable=too-many-instance-attributes
"""
This is a python code template that guide you through
writing your own reconstruction algorithms.
"""
# Copyright (c) Benyuan Liu. All Rights Reserved.
# Distributed under the (new) BSD License. See LICENSE.txt for more info.
from __future__ import division, absolute_import, print_function
from abc import ABC, abstractmethod
from ctypes import Union
from typing import Tuple

import numpy as np

from .fem import Forward
from .utils import eit_scan_lines


class SolverNotReadyError(BaseException):
    """Is raised if solver.setup() not called before using solver"""


class EitBase(ABC):
    """
    Base EIT solver.
    """

    def __init__(
        self,
        mesh: dict,
        el_pos: np.ndarray,
        ex_mat: np.ndarray = None,
        step: int = 1,
        perm: np.ndarray = None,
        jac_normalized: bool = False,
        parser: Tuple[str, list[str]] = "std",
        **kwargs,
    ) -> None:
        """
        An EIT solver.

        WARNING: Before using it run solver.setup() get set the solver ready!

        Parameters
        ----------
        mesh : dict
            mesh structure
        el_pos : np.ndarray
            position (numbering) of electrodes
        ex_mat : np.ndarray, optional
            2D array, each row is one stimulation pattern/line, by default None
        step : int, optional
            measurement method, by default 1
        perm : np.ndarray, optional
            initial permittivity in generating Jacobian, by default None
        jac_normalized : bool, optional
            normalize the jacobian using f0 computed from input perm, by
            default False
        parser : Tuple[str, list[str]], optional
            parsing the format of each frame in measurement/file, by
            default "std"

        Notes
        -----
        parser is required for your code to be compatible with
        (a) simulation data set or (b) FMMU data set
        """
        if ex_mat is None:
            ex_mat = eit_scan_lines(len(el_pos), 8)
        if perm is None:
            perm = mesh["perm"]

        # build forward solver
        self.fwd = Forward(mesh, el_pos)

        # solving mesh structure
        self.mesh = mesh
        self.pts = mesh["node"]
        self.tri = mesh["element"]

        # shape of the mesh
        self.no_num, self.n_dim = self.pts.shape
        self.el_num, self.n_vertices = self.tri.shape
        self.el_pos = el_pos
        self.parser = parser

        # user may specify a scalar for uniform permittivity
        self.perm = perm * np.ones(self.el_num) if np.size(perm) == 1 else perm
        # solving configurations
        self.ex_mat = ex_mat
        self.step = step
        self.jac_normalized = jac_normalized

        # initialize other parameters
        self.params = None
        self.xg = None
        self.yg = None
        self.mask = None
        # user must run solver.setup() manually to get correct H
        self.H = None
        self.is_ready = False

    @abstractmethod
    def setup(self) -> None:
        """
        Setup EIT solver

        1. memory parameters in self.params
        2. compute some other stuff needed for 3.
        3. compute self.H used for solving inv problem by using
            >> self.H=self._compute_h()
        4. set flag self.is_ready to `True`
        """

    @abstractmethod
    def _compute_h(self) -> np.ndarray:
        """
        Compute H matrix for solving inv problem

        To be used in self.setup()
        >> self.H=self._compute_h()

        Returns
        -------
        np.ndarray
            H matrix
        """

    def solve(
        self,
        v1: np.ndarray,
        v0: np.ndarray,
        normalize: bool = False,
        log_scale: bool = False,
    ) -> np.ndarray:
        """
        Dynamic imaging (conductivities imaging)

        Parameters
        ----------
        v1: np.ndarray
            current frame
        v0: np.ndarray
            referenced frame, d = H(v1 - v0)
        normalize: Bool, optional
            true for conducting normalization, by default False
        log_scale: Bool, optional
            remap reconstructions in log scale,by default False

        Raises
        ------
            SolverNotReadyError: raised if solver not ready
            (see self._check_solver_is_ready())

        Returns
        -------
        ds: np.ndarray
            complex-valued np.ndarray, changes of conductivities
        """
        self._check_solver_is_ready()
        dv = self._normalize(v1, v0) if normalize else v1 - v0
        ds = -np.dot(self.H, dv.transpose())  # s = -Hv
        if log_scale:
            ds = np.exp(ds) - 1.0
        return ds

    def map(self, dv: np.ndarray) -> np.ndarray:
        """
        (NOT USED, Deprecated?) simple mat using projection matrix

        return -H*dv, dv should be normalized.

        Parameters
        ----------
        dv : np.ndarray
            voltage measurement frame difference (reference frame - current frame)

        Raises
        ------
        SolverNotReadyError
            raised if solver not ready (see self._check_solver_is_ready())

        Returns
        -------
        np.ndarray
            -H*dv
        """
        self._check_solver_is_ready()
        return -np.dot(self.H, dv.transpose())

    def _compute_jac_matrix(self, perm:Union[int, float, np.ndarray]=None, allow_jac_norm: bool = True) -> np.ndarray:
        """
        Return Jacobian matrix correspoding to the fwd 

        Parameters
        ----------
        perm : Union[int, float, np.ndarray], optional
            permittivity, by default None
            (see Foward._get_perm for more details, in fem.py)
        allow_jac_norm : bool, optional
            flag allowing the Jacobian to be normalized according to
            `self.jac_normalized` intern flag, by default True
            (e.g. for `jac.gn` or `greit` no normalization is needed!)

        Returns
        -------
        np.ndarray
            Jacobian matrix
        """
       
        return self.fwd.compute_jac(
            ex_mat = self.ex_mat,
            step = self.step, 
            perm = perm or self.perm, 
            parser = self.parser, 
            normalize = self.jac_normalized and allow_jac_norm,
        )

    def _compute_b_matrix(self) -> np.ndarray:
        """
        Return BP matrix correspoding to the fwd

        Returns
        -------
        np.ndarray
            BP matrix
        """
        return self.fwd.compute_b_matrix(
           ex_mat= self.ex_mat, step=self.step, perm=self.perm, parser=self.parser
        )

    def _check_solver_is_ready(self) -> None:
        """
        Check if solver is ready for solving

        Addtionaly test also if self.H not `None`

        Raises
        ------
        SolverNotReadyError
            raised if solver not ready
        """
        if not self.is_ready or self.H is None:
            msg = "User must first run solver.setup() before using solver for solving purpose"
            raise SolverNotReadyError(msg)

    def _normalize(self, v1: np.ndarray, v0: np.ndarray) -> np.ndarray:
        """
        Normalize current frame using the amplitude of the reference frame.
        Boundary measurements v are complex-valued, we can use the real part of v,
        np.real(v), or the absolute values of v, np.abs(v).

        Parameters
        ----------
        v1 : np.ndarray
            current frame, can be a Nx192 matrix where N is the number of frames
        v0 : np.ndarray
            referenced frame, which is a row vector

        Returns
        -------
        np.ndarray
            Normalized current frame difference dv
        """
        return (v1 - v0) / np.abs(v0)
