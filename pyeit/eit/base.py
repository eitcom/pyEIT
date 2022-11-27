# coding: utf-8
# pylint: disable=invalid-name, too-many-arguments
# pylint: disable=too-many-instance-attributes
"""
This is a python code template that guide you through
writing your own reconstruction algorithms.
"""
# Copyright (c) Benyuan Liu. All Rights Reserved.
# Distributed under the (new) BSD License. See LICENSE.txt for more info.
from __future__ import absolute_import, division, print_function
from abc import ABC, abstractmethod
import numpy as np
from pyeit.eit.protocol import PyEITProtocol
from pyeit.mesh import PyEITMesh

from .fem import EITForward


class SolverNotReadyError(BaseException):
    """Is raised if solver.setup() not called before using solver"""


class EitBase(ABC):
    """
    Base EIT solver.
    """

    def __init__(
        self,
        mesh: PyEITMesh,
        protocol: PyEITProtocol,
    ) -> None:
        """
        An EIT solver.

        WARNING: Before using it run solver.setup() to set the solver ready!

        Parameters
        ----------
        mesh: PyEITMesh
            mesh object
        protocol: PyEITProtocol
            measurement object
        """
        # build forward solver
        self.fwd = EITForward(mesh=mesh, protocol=protocol)

        # initialize other parameters
        self.params: dict = {}
        self.xg: np.ndarray = np.zeros(mesh.n_elems)
        self.yg: np.ndarray = np.zeros(mesh.n_elems)
        self.mask: np.ndarray = np.zeros(mesh.n_elems)
        # user must run solver.setup() manually to get correct H
        self.H: np.ndarray = np.zeros((mesh.n_elems, protocol.n_meas), dtype=mesh.dtype)
        self.is_ready = False

    @property
    def mesh(self) -> PyEITMesh:
        return self.fwd.mesh

    # # if needed protocol attributes can be accessed by using self.protocol
    # # instead of self.fwd.protocol
    # @property
    # def protocol(self)->PyEITProtocol:
    #     return self.fwd.protocol

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
    def _compute_h(self):
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
    ):
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
            remap reconstructions in log scale, by default False

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

    def map(self, dv: np.ndarray):
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
            msg = "User must first run {type(self).__name__}.setup() before imaging."
            raise SolverNotReadyError(msg)

    def _normalize(self, v1: np.ndarray, v0: np.ndarray):
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
