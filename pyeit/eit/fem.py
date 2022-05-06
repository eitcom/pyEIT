# coding: utf-8
# pylint: disable=invalid-name, no-member, too-many-locals
# pylint: disable=too-many-instance-attributes
""" 2D/3D FEM routines """
# Copyright (c) Benyuan Liu. All Rights Reserved.
# Distributed under the (new) BSD License. See LICENSE.txt for more info.
from __future__ import division, absolute_import, print_function

from dataclasses import dataclass
from typing import Union
import numpy as np
import numpy.linalg as la
from scipy import sparse
import scipy.linalg

from .utils import eit_scan_lines


@dataclass
class FwdResult:
    """Summarize the results from solving the eit fwd problem

    Attributes
    ----------
    v: np.ndarray
        number of measures x 1 array, simulated boundary measures; shape(n_exc, n_el)
    """

    v: np.ndarray  # number of measures x 1 array, simulated boundary measures


class Forward:
    """FEM forward computing code"""

    def __init__(self, mesh: dict[str, np.ndarray], el_pos: np.ndarray) -> None:
        """
        FEM forward solver

        A good FEM forward solver should only depend on
        mesh structure and the position of electrodes

        Parameters
        ----------
        mesh: dict
            mesh structure, {'node', 'element', 'perm'}
        el_pos: np.ndarray
            numbering of electrodes positions

        Note
        ----
        1, The nodes are continuous numbered, the numbering of an element is
        CCW (counter-clock-wise).
        2, The Jacobian and the boundary voltages used the SIGN information,
        for example, V56 = V6 - V5 = -V65. If you are using absolute boundary
        voltages for imaging, you MUST normalize it with the signs of v0
        under each current-injecting pattern.
        """
        self.pts = mesh["node"]
        self.tri = mesh["element"]
        self.tri_perm = mesh["perm"]
        self.el_pos = el_pos

        # reference electrodes [ref node should not be on electrodes]
        self.set_ref_el()

        # infer dimensions from mesh
        self.n_pts, self.n_dim = self.pts.shape
        self.n_tri, self.n_vertices = self.tri.shape
        self.n_el = el_pos.size

        # temporary memory attributes for computation (e.g. jac)
        self._r_matrix = None
        self._ke = None

    def solve(
        self,
        ex_mat: np.ndarray = None,
        perm: np.ndarray = None,
    ) -> np.ndarray:
        """
        Calculate and compute the potential distribution (complex-valued)
        corresponding to the permittivity distribution `perm ` for all
        excitations contained in the excitation pattern `ex_mat`

        Currently, only simple electrode model is supported,
        CEM (complete electrode model) is under development.

        Parameters
        ----------
        ex_mat : np.ndarray, optional
            stimulation/excitation matrix, of shape (n_exc, 2), by default `None`.
            (see _get_ex_mat for more details)
        perm : Union[int, float, np.ndarray], optional
            permittivity on elements ; shape (n_tri,), by default `None`.
            Must be the same size with self.tri_perm
            If `None`, `self.tri_perm` will be used
            If perm is int or float, uniform permittivity on elements will be used
            (see _get_perm for more details)

        Returns
        -------
        np.ndarray
            potential on nodes ; shape (n_exc, n_pts)

        Notes
        -------
        For compatibility with some scripts in /examples a single excitation
        line can be passed instead of the whole excitation pattern `ex_mat`
        (e.g. [0,7] or np.array([0,7]) or ex_mat[0].ravel). In that case a
        simplified version of `f` with shape (n_pts,)
        """
        ex_mat = self._check_ex_mat(ex_mat)  # check/init stimulation
        perm = self._check_perm(perm)  # check/init permitivity
        f = self._compute_potential_distribution(ex_mat=ex_mat, perm=perm)
        # case ex_line has been passed instead of ex_mat
        # we return simplified version of f with shape (n_pts,)
        if f.shape[0] == 1:
            return f[0, :].ravel()
        return f

    def solve_eit(
        self,
        ex_mat: np.ndarray = None,
        step: int = 1,
        perm: Union[int, float, np.ndarray] = None,
        parser: Union[str, list[str]] = None,
    ) -> FwdResult:
        """
        EIT simulation, generate forward v measurement

        Parameters
        ----------
        ex_mat : np.ndarray, optional
            stimulation/excitation matrix, of shape (n_exc, 2), by default `None`.
            (see _get_ex_mat for more details)
        step: int, optional
            the configuration of measurement electrodes, by default 1 (adjacent).
        perm : Union[int, float, np.ndarray], optional
            permittivity on elements ; shape (n_tri,), by default `None`.
            Must be the same size with self.tri_perm
            If `None`, `self.tri_perm` will be used
            If perm is int or float, uniform permittivity on elements will be used
            (see _get_perm for more details)
        parser: Union[str, list[str]], optional
            see voltage_meter for more details, by default `None`.

        Returns
        -------
        FwdResult
            Foward results comprising
                v: np.ndarray
                    simulated boundary voltage measurements; shape(n_exc, n_el)
        """
        ex_mat = self._check_ex_mat(ex_mat)  # check/init stimulation
        perm = self._check_perm(perm)  # check/init permitivity
        f = self._compute_potential_distribution(ex_mat, perm)
        # boundary measurements, subtract_row-voltages on electrodes
        diff_op = voltage_meter(ex_mat, n_el=self.n_el, step=step, parser=parser)
        return FwdResult(v=self._get_boundary_voltages(f, diff_op))

    def _get_boundary_voltages(self, f: np.ndarray, diff_op: np.ndarray) -> np.ndarray:
        """
        Compute boundary voltages from potential distribution

        Parameters
        ----------
        f : np.ndarray
            potential on nodes ; shape (n_exc, n_pts)
        diff_op : np.ndarray
            measurements pattern / subtract_row pairs [N, M]; shape (n_exc, n_meas_per_exc, 2)

        Returns
        -------
        np.ndarray
            simulated boundary voltage measurements; shape(n_exc, n_el)
        """
        f_el = f[:, self.el_pos]
        v = subtract_row(f_el, diff_op)
        return np.hstack(v)

    def compute_jac(
        self,
        ex_mat: np.ndarray = None,
        step: int = 1,
        perm: Union[int, float, np.ndarray] = None,
        parser: Union[str, list[str]] = None,
        normalize: bool = False,
    ) -> np.ndarray:
        """
        Compute the Jacobian matrix

        Parameters
        ----------
        ex_mat : np.ndarray, optional
            stimulation/excitation matrix, of shape (n_exc, 2), by default `None`.
            (see _get_ex_mat for more details)
        step: int, optional
            the configuration of measurement electrodes, by default 1 (adjacent).
        perm : Union[int, float, np.ndarray], optional
            permittivity on elements ; shape (n_tri,), by default `None`.
            Must be the same size with self.tri_perm
            If `None`, `self.tri_perm` will be used
            If perm is int or float, uniform permittivity on elements will be used
            (see _get_perm for more details)
        parser: Union[str, list[str]], optional
            see voltage_meter for more details, by default `None`.
        normalize : bool, optional
            flag for Jacobian normalization, by default False.
            If True the Jacobian is normalized

        Returns
        -------
        np.ndarray
            Jacobian matrix

        Notes
        -----
            - initial boundary voltage meas. extimation v0 can be accessed
            after computation through call fwd.v0

        """
        ex_mat = self._check_ex_mat(ex_mat)  # check/init stimulation
        perm = self._check_perm(perm)  # check/init permitivity
        f = self._compute_potential_distribution(
            ex_mat=ex_mat, perm=perm, memory_4_jac=True
        )

        # Build Jacobian matrix column wise (element wise)
        #    Je = Re*Ke*Ve = (nex3) * (3x3) * (3x1)
        jac_i = np.zeros((ex_mat.shape[0], self.n_el, self.n_tri), dtype=perm.dtype)

        r_el = self._r_matrix[self.el_pos]

        def jac_init(jac, k):
            for (i, e) in enumerate(self.tri):
                jac[:, i] = np.dot(np.dot(r_el[:, e], self._ke[i]), f[k, e])
            return jac

        jac_i = np.array(list(map(jac_init, jac_i, np.arange(ex_mat.shape[0]))))

        self._r_matrix = None  # clear memory
        self._ke = None  # clear memory

        diff_op = voltage_meter(ex_mat, n_el=self.n_el, step=step, parser=parser)
        jac = subtract_row(jac_i, diff_op)
        self.v0 = self._get_boundary_voltages(f, diff_op)
        jac = np.vstack(jac)

        # Jacobian normalization: divide each row of J (J[i]) by abs(v0[i])

        return jac / np.abs(self.v0[:, None]) if normalize else jac

    def compute_b_matrix(
        self,
        ex_mat: np.ndarray = None,
        step: int = 1,
        perm: Union[int, float, np.ndarray] = None,
        parser: Union[str, list[str]] = None,
    ) -> np.ndarray:
        """
        Compute back-projection mappings (smear matrix)

        Parameters
        ----------
        ex_mat : np.ndarray, optional
            stimulation/excitation matrix, of shape (n_exc, 2), by default `None`.
            (see _get_ex_mat for more details)
        step: int, optional
            the configuration of measurement electrodes, by default 1 (adjacent).
        perm : Union[int, float, np.ndarray], optional
            permittivity on elements ; shape (n_tri,), by default `None`.
            Must be the same size with self.tri_perm
            If `None`, `self.tri_perm` will be used
            If perm is int or float, uniform permittivity on elements will be used
            (see _get_perm for more details)
        parser: Union[str, list[str]], optional
            see voltage_meter for more details, by default `None`.

        Returns
        -------
        np.ndarray
            back-projection mappings (smear matrix); shape(n_exc, n_pts, 1), dtype= bool
        """
        ex_mat = self._check_ex_mat(ex_mat)  # check/init stimulation
        perm = self._check_perm(perm)  # check/init permitivity

        f = self._compute_potential_distribution(ex_mat=ex_mat, perm=perm)
        f_el = f[:, self.el_pos]
        # build bp projection matrix
        # 1. we can either smear at the center of elements, using
        #    >> fe = np.mean(f[:, self.tri], axis=1)
        # 2. or, simply smear at the nodes using f
        diff_op = voltage_meter(ex_mat, n_el=self.n_el, step=step, parser=parser)
        # set new to `False` to get smear-computation from ChabaneAmaury
        b_matrix = smear(f, f_el, diff_op, new=True)
        return np.vstack(b_matrix)

    def set_ref_el(self, val: int = None) -> None:
        """
        Set reference electrode node

        Parameters
        ----------
        val : int, optional
            node number of reference electrode, by default None

        """
        self.ref_el = (
            val if val is not None and val not in self.el_pos else max(self.el_pos) + 1
        )

    def _compute_potential_distribution(
        self, ex_mat: np.ndarray, perm: np.ndarray, memory_4_jac: bool = False
    ) -> np.ndarray:
        """
        Calculate and compute the potential distribution (complex-valued)
        corresponding to the permittivity distribution `perm ` for all
        excitations contained in the excitation pattern `ex_mat`

        Currently, only simple electrode model is supported,
        CEM (complete electrode model) is under development.

        Parameters
        ----------
        ex_mat: np.ndarray
            stimulation/excitation matrix ; shape (n_exc, 2)
        perm: np.ndarray
            permittivity on elements ; shape (n_tri,)
        memory_4_jac : bool, optional
            flag to memory r_matrix to self._r_matrix and ke to self._ke,
            by default False.

        Returns
        -------
        np.ndarray
            potential on nodes ; shape (n_exc, n_pts)

        """
        # 1. calculate local stiffness matrix (on each element)
        ke = calculate_ke(self.pts, self.tri)
        # 2. assemble to global K
        kg = assemble(ke, self.tri, perm, self.n_pts, ref=self.ref_el)

        if memory_4_jac:
            # save
            # 3. calculate electrode impedance matrix R = K^{-1}
            self._r_matrix = la.inv(kg)
            self._ke = ke

        # 4. solving nodes potential using boundary conditions
        b = self._natural_boundary(ex_mat)

        return (
            scipy.linalg.solve(kg, b.swapaxes(0, 1))
            .swapaxes(0, 1)
            .reshape(b.shape[0:2])
        )

    def _check_perm(self, perm: Union[int, float, np.ndarray] = None) -> np.ndarray:
        """
        Check/init the permittivity on element

        Parameters
        ----------
        perm : Union[int, float, np.ndarray], optional
            permittivity on elements ; shape (n_tri,), by default `None`.
            Must be the same size with self.tri_perm
            If `None`, `self.tri_perm` will be used
            If perm is int or float, uniform permittivity on elements will be used

        Returns
        -------
        np.ndarray
            permittivity on elements ; shape (n_tri,)

        Raises
        ------
        TypeError
            raised if perm is not ndarray and of shape (n_tri,)
        """

        if perm is None:
            return self.tri_perm
        elif isinstance(perm, (int, float)):
            return np.ones(self.n_tri, dtype=float) * perm

        if not isinstance(perm, np.ndarray) or perm.shape != (self.n_tri,):
            raise TypeError(
                f"Wrong type/shape of {perm=}, expected an ndarray; shape (n_tri, )"
            )
        return perm

    def _check_ex_mat(self, ex_mat: np.ndarray = None) -> np.ndarray:
        """
        Check/init stimulation

        Parameters
        ----------
        ex_mat : np.ndarray, optional
            stimulation/excitation matrix, of shape (n_exc, 2), by default `None`.
            If `None` initialize stimulation matrix for 16 electrode and
            apposition mode (see function `eit_scan_lines(16, 8)`)
            If single stimulation (ex_line) is passed only a list of length 2
            and np.ndarray of size 2 will be treated.

        Returns
        -------
        np.ndarray
            stimulation matrix

        Raises
        ------
        TypeError
            Only accept, `None`, list of length 2, np.ndarray of size 2,
            or np.ndarray of shape (n_exc, 2)
        """
        if ex_mat is None:
            # initialize the scan lines for 16 electrodes (default: apposition)
            ex_mat = eit_scan_lines(16, 8)
        elif isinstance(ex_mat, list) and len(ex_mat) == 2:
            # case ex_line has been passed instead of ex_mat
            ex_mat = np.array([ex_mat]).reshape((1, 2))  # build a 2D array
        elif isinstance(ex_mat, np.ndarray) and ex_mat.size == 2:
            #     case ex_line np.ndarray has been passed instead of ex_mat
            ex_mat = ex_mat.reshape((-1, 2))

        if (
            not isinstance(ex_mat, np.ndarray)
            or ex_mat.ndim != 2
            or ex_mat.shape[1] != 2
        ):
            raise TypeError(
                f"Wrong shape of {ex_mat=} expected an ndarray ; shape (n_exc, 2)"
            )

        return ex_mat

    def _natural_boundary(self, ex_mat: np.ndarray) -> np.ndarray:
        """
        Generate the Neumann boundary condition.

        In utils.py, you should note that ex_mat is local indexed from 0...15,
        which need to be converted to global node number using el_pos.

        Parameters
        ----------
        ex_mat: np.ndarray
            stimulation/excitation matrix ; shape (n_exc, 2)

        Returns
        ----------
        np.ndarray
            Global boundary condition on pts ; shape (n_exc, n_pts, 1)
        """
        drv_a_global = self.el_pos[ex_mat[:, 0]]
        drv_b_global = self.el_pos[ex_mat[:, 1]]

        # global boundary condition
        b = np.zeros((ex_mat.shape[0], self.n_pts, 1))
        b[np.arange(drv_a_global.shape[0]), drv_a_global] = 1.0
        b[np.arange(drv_b_global.shape[0]), drv_b_global] = -1.0

        return b


def _smear(f: np.ndarray, fb: np.ndarray, pairs: np.ndarray) -> np.ndarray:
    """
    Build smear matrix B for bp for one exitation

    used for the smear matrix computation by @ChabaneAmaury

    Parameters
    ----------
    f: np.ndarray
        potential on nodes
    fb: np.ndarray
        potential on adjacent electrodes
    pairs: np.ndarray
        electrodes numbering pairs

    Returns
    -------
    B: np.ndarray
        back-projection matrix
    """
    # Replacing the code below by a faster implementation in Numpy
    f_min = np.minimum(fb[pairs[:, 0]], fb[pairs[:, 1]]).reshape((-1, 1))
    f_max = np.maximum(fb[pairs[:, 0]], fb[pairs[:, 1]]).reshape((-1, 1))
    return (f_min < f) & (f <= f_max)


def smear(
    f: np.ndarray, fb: np.ndarray, meas_pattern: np.ndarray, new: bool = False
) -> np.ndarray:
    """
    Build smear matrix B for bp

    Parameters
    ----------
    f: np.ndarray
        potential on nodes; shape (n_exc, n_pts)
    fb: np.ndarray
        potential on adjacent electrodes; shape (n_exc, n_el)
    meas_pattern: np.ndarray
        electrodes numbering pairs; shape (n_exc, n_meas, 2)
    new : bool, optional
        flag to use new matrices based computation, by default False.
        If `False` to smear-computation from ChabaneAmaury is used

    Returns
    -------
    np.ndarray
        back-projection (smear) matrix; shape (n_exc, n_meas, n_pts), dtype= bool
    """
    if new:
        # new implementation not much faster! :(
        idx_meas_0 = meas_pattern[:, :, 0]
        idx_meas_1 = meas_pattern[:, :, 1]
        n_exc = meas_pattern.shape[0]  # number of excitations
        n_meas = meas_pattern.shape[1]  # number of measurements per excitations
        n_pts = f.shape[1]  # number of nodes
        idx_exc = np.ones_like(idx_meas_0, dtype=int) * np.arange(n_exc).reshape(
            n_exc, 1
        )
        f_min = np.minimum(fb[idx_exc, idx_meas_0], fb[idx_exc, idx_meas_1])
        f_max = np.maximum(fb[idx_exc, idx_meas_0], fb[idx_exc, idx_meas_1])
        # contruct matrices of shapes (n_exc, n_meas, n_pts) for comparison
        f_min = np.repeat(f_min[:, :, np.newaxis], n_pts, axis=2)
        f_max = np.repeat(f_max[:, :, np.newaxis], n_pts, axis=2)
        f0 = np.repeat(f[:, :, np.newaxis], n_meas, axis=2)
        f0 = f0.swapaxes(1, 2)
        return (f_min < f0) & (f0 <= f_max)
    else:
        # Replacing the below code by a faster implementation in Numpy
        def b_matrix_init(k):
            return _smear(f[k], fb[k], meas_pattern[k])

        return np.array(list(map(b_matrix_init, np.arange(f.shape[0]))))


def subtract_row(v: np.ndarray, meas_pattern: np.ndarray) -> np.ndarray:
    """
    Build the voltage differences using the meas_pattern.
    v_diff[k] = v[i, :] - v[j, :]

    New implementation 33% less computation time

    Parameters
    ----------
    v: np.ndarray
        Nx1 boundary measurements vector or NxM matrix; shape (n_exc,n_el,1)
    meas_pattern: np.ndarray
        Nx2 subtract_row pairs; shape (n_exc, n_meas, 2)

    Returns
    -------
    np.ndarray
        difference measurements v_diff
    """

    if v.shape[:1] != meas_pattern.shape[:1]:
        raise ValueError(
            f"Measurements vector v ({v.shape=}) should have same 1stand 2nd dim as meas_pattern ({meas_pattern.shape=})"
        )

    # creation of excitation indexe for each idx_meas
    idx_meas_0 = meas_pattern[:, :, 0]
    idx_meas_1 = meas_pattern[:, :, 1]
    n_exc = meas_pattern.shape[0]
    idx_exc = np.ones_like(idx_meas_0, dtype=int) * np.arange(n_exc).reshape(n_exc, 1)

    return v[idx_exc, idx_meas_0] - v[idx_exc, idx_meas_1]


def voltage_meter(
    ex_mat: np.ndarray,
    n_el: int = 16,
    step: int = 1,
    parser: Union[str, list[str]] = None,
) -> np.ndarray:
    """
    Build the measurement pattern (subtract_row-voltage pairs [N, M])
    for all excitations on boundary electrodes.

    we direct operate on measurements or Jacobian on electrodes,
    so, we can use LOCAL index in this module, do not require el_pos.

    Notes
    -----
    ABMN Model.
    A: current driving electrode,
    B: current sink,
    M, N: boundary electrodes, where v_diff = v_n - v_m.

    Parameters
    ----------
    ex_mat : np.ndarray
        Nx2 array, [positive electrode, negative electrode]. ; shape (n_exc, 2)
    n_el : int, optional
        number of total electrodes, by default 16
    step : int, optional
        measurement method (two adjacent electrodes are used for measuring), by default 1 (adjacent)
    parser : Union[str, list[str]], optional
        parsing the format of each frame in measurement/file, by default None
        if parser contains 'fmmu', or 'rotate_meas' then data are trimmed,
        boundary voltage measurements are re-indexed and rotated,
        start from the positive stimulus electrode start index 'A'.
        if parser contains 'std', or 'no_rotate_meas' then data are trimmed,
        the start index (i) of boundary voltage measurements is always 0.
        if parser contains 'meas_current', the measurements on current carrying
        electrodes are allowed. Otherwise the measurements on current carrying
        electrodes are discarded (like 'no_meas_current' option in EIDORS3D).

    Returns
    -------
    np.ndarray
        measurements pattern / subtract_row pairs [N, M]; shape (n_exc, n_meas_per_exc, 2)
    """
    # local node
    ex_mat = ex_mat.astype(int)
    n_exc = ex_mat.shape[0]
    drv_a = np.ones((n_exc, n_exc), dtype=int) * ex_mat[:, 0].reshape(n_exc, 1)
    drv_b = np.ones((n_exc, n_exc), dtype=int) * ex_mat[:, 1].reshape(n_exc, 1)

    if not isinstance(parser, list):  # transform parser in list
        parser = [parser]

    meas_current = "meas_current" in parser
    fmmu_rotate = any(p in ("fmmu", "rotate_meas") for p in parser)
    i0 = drv_a if fmmu_rotate else np.zeros_like(drv_a)

    idx_el = np.ones((n_exc, n_el), dtype=int) * np.arange(n_el)
    m = (i0 + idx_el) % n_el
    n = (m + step) % n_el
    meas_pattern = np.concatenate((n[:, :, np.newaxis], m[:, :, np.newaxis]), 2)

    if meas_current:
        return meas_pattern

    diff_pairs_mask = np.logical_and.reduce(
        (m != drv_a, m != drv_b, n != drv_a, n != drv_b)
    )
    return meas_pattern[diff_pairs_mask].reshape(n_exc, -1, 2)


def assemble(
    ke: np.ndarray, tri: np.ndarray, perm: np.ndarray, n_pts: int, ref: int = 0
) -> np.ndarray:
    """
    Assemble the stiffness matrix (using sparse matrix)

    Parameters
    ----------
    ke: np.ndarray
        n_tri x (n_dim x n_dim) 3d matrix
    tri: np.ndarray
        the structure of mesh
    perm: np.ndarray
        n_tri x 1 conductivities on elements
    n_pts: int
        number of nodes
    ref: int, optional
        reference electrode, by default 0

    Returns
    -------
    np.ndarray
        NxN array of complex stiffness matrix

    Notes
    -----
    you may use sparse matrix (IJV) format to automatically add the local
    stiffness matrix to the global matrix.
    """
    n_tri, n_vertices = tri.shape

    # New: use IJV indexed sparse matrix to assemble K (fast, prefer)
    # index = np.array([np.meshgrid(no, no, indexing='ij') for no in tri])
    # note: meshgrid is slow, using handcraft sparse index, for example
    # let tri=[[1, 2, 3], [4, 5, 6]], then indexing='ij' is equivalent to
    # row = [1, 1, 1, 2, 2, 2, ...]
    # col = [1, 2, 3, 1, 2, 3, ...]
    row = np.repeat(tri, n_vertices).ravel()
    col = np.repeat(tri, n_vertices, axis=0).ravel()
    data = np.array([ke[i] * perm[i] for i in range(n_tri)]).ravel()

    # set reference nodes before constructing sparse matrix, where
    # K[ref, :] = 0, K[:, ref] = 0, K[ref, ref] = 1.
    # write your own mask code to set the corresponding locations of data
    # before building the sparse matrix, for example,
    # data = mask_ref_node(data, row, col, ref)

    # for efficient sparse inverse (csc)
    k_matrix = sparse.csr_matrix(
        (data, (row, col)), shape=(n_pts, n_pts), dtype=perm.dtype
    )

    # the stiffness matrix may not be sparse
    k_matrix = k_matrix.toarray()

    # place reference electrode
    if 0 <= ref < n_pts:
        k_matrix[ref, :] = 0.0
        k_matrix[:, ref] = 0.0
        k_matrix[ref, ref] = 1.0

    return k_matrix


def calculate_ke(pts: np.ndarray, tri: np.ndarray) -> np.ndarray:
    """
    Calculate local stiffness matrix on all elements.

    Parameters
    ----------
    pts: np.ndarray
        Nx2 (x,y) or Nx3 (x,y,z) coordinates of points
    tri: np.ndarray
        Mx3 (triangle) or Mx4 (tetrahedron) connectivity of elements

    Returns
    -------
    np.ndarray
        n_tri x (n_dim x n_dim) 3d matrix
    """
    n_tri, n_vertices = tri.shape

    # check dimension
    # '3' : triangles
    # '4' : tetrahedrons
    if n_vertices == 3:
        _k_local = _k_triangle
    elif n_vertices == 4:
        _k_local = _k_tetrahedron
    else:
        raise TypeError("The num of vertices of elements must be 3 or 4")

    # default data types for ke
    ke_array = np.zeros((n_tri, n_vertices, n_vertices))
    for ei in range(n_tri):
        no = tri[ei, :]
        xy = pts[no]

        # compute the KIJ (permittivity=1.)
        ke = _k_local(xy)
        ke_array[ei] = ke

    return ke_array


def _k_triangle(xy: np.ndarray) -> np.ndarray:
    """
    Given a point-matrix of an element, solving for Kij analytically
    using barycentric coordinates (simplex coordinates)

    Parameters
    ----------
    xy: np.ndarray
        (x,y) of nodes 1,2,3 given in counterclockwise manner

    Returns
    -------
    np.ndarray
        local stiffness matrix
    """
    # edges (vector) of triangles
    s = xy[[2, 0, 1]] - xy[[1, 2, 0]]

    # area of triangles. Note, abs is removed since version 2020,
    # user must make sure all triangles are CCW (conter clock wised).
    at = 0.5 * det2x2(s[0], s[1])
    # TODO maybe replace with:
    # at= 0.5 * np.linalg.det(s)

    # Local stiffness matrix (e for element)
    return np.dot(s, s.T) / (4.0 * at)


def det2x2(s1: np.ndarray, s2: np.ndarray) -> float:
    """Calculate the determinant of a 2x2 matrix"""
    return s1[0] * s2[1] - s1[1] * s2[0]


def _k_tetrahedron(xy: np.ndarray) -> np.ndarray:
    """
    Given a point-matrix of an element, solving for Kij analytically
    using barycentric coordinates (simplex coordinates)

    Parameters
    ----------
    xy: np.ndarray
        (x,y) of nodes 1, 2, 3, 4 given in counterclockwise manner, see notes.

    Returns
    -------
    np.ndarray
        local stiffness matrix

    Notes
    -----
    A tetrahedron is described using [0, 1, 2, 3] (local node index) or
    [171, 27, 9, 53] (global index). Counterclockwise (CCW) is defined
    such that the barycentric coordinate of face (1->2->3) is positive.
    """
    s = xy[[2, 3, 0, 1]] - xy[[1, 2, 3, 0]]

    # volume of the tetrahedron, Note abs is removed since version 2020,
    # user must make sure all tetrahedrons are CCW (counter clock wised).
    vt = 1.0 / 6 * la.det(s[[0, 1, 2]])

    # calculate area (vector) of triangle faces
    # re-normalize using alternative (+,-) signs
    ij_pairs = [[0, 1], [1, 2], [2, 3], [3, 0]]
    signs = [1, -1, 1, -1]
    a = np.array([sign * np.cross(s[i], s[j]) for (i, j), sign in zip(ij_pairs, signs)])

    # local (e for element) stiffness matrix
    return np.dot(a, a.transpose()) / (36.0 * vt)


if __name__ == "__main__":
    """"""
