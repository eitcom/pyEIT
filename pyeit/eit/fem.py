# coding: utf-8
# pylint: disable=invalid-name, no-member, too-many-locals
# pylint: disable=too-many-instance-attributes
""" 2D/3D FEM routines """
# Copyright (c) Benyuan Liu. All Rights Reserved.
# Distributed under the (new) BSD License. See LICENSE.txt for more info.
from __future__ import division, absolute_import, print_function, annotations

from typing import Tuple, Union, Optional
import numpy as np
import numpy.linalg as la
from scipy import sparse
import warnings
import scipy.sparse.linalg
from pyeit.eit.protocol import PyEITProtocol
from pyeit.mesh import PyEITMesh


class Forward:
    """FEM forward computing code"""

    def __init__(self, mesh: PyEITMesh) -> None:
        """
        FEM forward solver.
        A good FEM forward solver should only depend on
        mesh structure and the position of electrodes.

        Parameters
        ----------
        mesh: PyEITMesh
            mesh object

        Note
        ----
        The nodes are continuous numbered, the numbering of an element is
        CCW (counter-clock-wise).
        """
        self.mesh = mesh
        # coefficient matrix [initialize]
        self.se = calculate_ke(self.mesh.node, self.mesh.element)
        self.assemble_pde(self.mesh.perm)

    def assemble_pde(
        self, perm: Optional[Union[int, float, complex, np.ndarray]] = None
    ) -> None:
        """
        assemble PDE

        Parameters
        ----------
        perm : Union[int, float, np.ndarray]
            permittivity on elements ; shape (n_tri,).
            if `None`, assemble_pde is aborded

        """
        if perm is None:
            return
        perm_array = self.mesh.get_valid_perm_array(perm)
        self.kg = assemble(
            self.se,
            self.mesh.element,
            perm_array,
            self.mesh.n_nodes,
            ref=self.mesh.ref_node,
        )

    def solve(self, ex_line: np.ndarray = np.array([0, 1])):
        """
        Calculate and compute the potential distribution (complex-valued)
        corresponding to the permittivity distribution `perm ` for a
        excitation contained specified by `ex_line` (Neumann BC)

        Parameters
        ----------
        ex_line : np.ndarray, optional
            stimulation/excitation matrix, of shape (2,)

        Returns
        -------
        np.ndarray
            potential on nodes ; shape (n_pts,)

        Notes
        -----
        Currently, only simple electrode model is supported,
        CEM (complete electrode model) is under development.
        """
        # using natural boundary conditions
        b = np.zeros(self.mesh.n_nodes)
        b[self.mesh.el_pos[ex_line]] = [1, -1]

        # solve
        return scipy.sparse.linalg.spsolve(self.kg, b)

    def solve_vectorized(self, ex_mat: np.ndarray) -> np.ndarray:
        """
        Calculate and compute the potential distribution (complex-valued)
        corresponding to the permittivity distribution `perm ` for a
        excitation contained specified by `ex_line` (Neumann BC)

        Parameters
        ----------
        ex_line : np.ndarray, optional
            stimulation/excitation matrix, of shape (2,)

        Returns
        -------
        np.ndarray
            potential on nodes ; shape (n_pts,)

        Notes
        -----
        Currently, only simple electrode model is supported,
        CEM (complete electrode model) is under development.
        """

        # using natural boundary conditions
        b = np.zeros((ex_mat.shape[0], self.mesh.n_nodes))
        b[np.arange(b.shape[0])[:, None], self.mesh.el_pos[ex_mat]] = [1, -1]
        result = np.empty((ex_mat.shape[0], self.kg.shape[0]))

        # TODO Need to inspect this deeper
        for i in range(result.shape[0]):
            result[i] = sparse.linalg.spsolve(self.kg, b[i])

        # solve
        return result


class EITForward(Forward):
    """EIT Forward simulation, depends on mesh and protocol"""

    def __init__(self, mesh: PyEITMesh, protocol: PyEITProtocol) -> None:
        """
        EIT Forward Solver

        Parameters
        ----------
        mesh: PyEITMesh
            mesh object
        protocol: PyEITProtocol
            measurement object

        Notes
        -----
        The Jacobian and the boundary voltages used the SIGN information,
        for example, V56 = V6 - V5 = -V65. If you are using absolute boundary
        voltages for imaging, you MUST normalize it with the signs of v0
        under each current-injecting pattern.
        """
        self._check_mesh_protocol_compatibility(mesh, protocol)

        # FEM solver
        super().__init__(mesh=mesh)

        # EIT measurement protocol
        self.protocol = protocol

    def _check_mesh_protocol_compatibility(
        self, mesh: PyEITMesh, protocol: PyEITProtocol
    ) -> None:
        """
        Check if mesh and protocol are compatible

        - #1 n_el in mesh >=  n_el in protocol
        - #2 .., TODO if necessary

        Raises
        ------
        ValueError
            if protocol is not compatible to the mesh
        """
        # n_el in mesh should be >=  n_el in protocol
        m_n_el = mesh.n_el
        p_n_el = protocol.n_el

        if m_n_el != p_n_el:
            warnings.warn(
                f"The mesh use {m_n_el} electrodes, and the protocol use only {p_n_el} electrodes",
                stacklevel=2,
            )

    def solve_eit(
        self,
        perm: Optional[Union[int, float, complex, np.ndarray]] = None,
    ):
        """
        EIT simulation, generate forward v measurements

        Parameters
        ----------
        perm : Union[int, float, np.ndarray], optional
            permittivity on elements ; shape (n_tri,), by default `None`.
            if perm is `None`, the computation of forward v measurements will be
            based on the permittivity of the mesh, self.mesh.perm
        Returns
        -------
        v: np.ndarray
            simulated boundary voltage measurements; shape(n_exe*n_el,)
        """
        self.assemble_pde(perm)
        # v = np.zeros((self.protocol.n_exc, self.protocol.n_meas), dtype=self.mesh.dtype)
        # for i, ex_line in enumerate(self.protocol.ex_mat):
        #     f = self.solve(ex_line)
        #     v[i] = subtract_row(f[self.mesh.el_pos], self.protocol.meas_mat[i])
        f = self.solve_vectorized(self.protocol.ex_mat)
        v = subtract_row_vectorized(f[:, self.mesh.el_pos], self.protocol.meas_mat)

        return v.reshape(-1)

    def compute_jac(
        self,
        perm: Optional[Union[int, float, complex, np.ndarray]] = None,
        normalize: bool = False,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute the Jacobian matrix and initial boundary voltage meas.
        extimation v0

        Parameters
        ----------
        perm : Union[int, float, np.ndarray], optional
            permittivity on elements ; shape (n_tri,), by default `None`.
            if perm is `None`, the computation of Jacobian matrix will be based
            on the permittivity of the mesh, self.mesh.perm
        normalize : bool, optional
            flag for Jacobian normalization, by default False.
            If True the Jacobian is normalized

        Returns
        -------
        Tuple[np.ndarray, np.ndarray]
            Jacobian matrix, initial boundary voltage meas. extimation v0

        """
        # update k if necessary and calculate r=inv(k), dense matrix, slow
        self.assemble_pde(perm)
        r_mat = la.inv(self.kg.toarray())[self.mesh.el_pos]
        r_el = np.full((self.protocol.ex_mat.shape[0],) + r_mat.shape, r_mat)
        # nodes potential
        f = self.solve_vectorized(self.protocol.ex_mat)
        f_el = f[:, self.mesh.el_pos]
        # build measurements and node resistance
        v = subtract_row_vectorized(f_el, self.protocol.meas_mat)
        ri = subtract_row_vectorized(r_el, self.protocol.meas_mat)
        v0 = v.reshape(-1)

        ## calculate v, jac per excitation (ex_line)
        # _jac = np.zeros((self.protocol.n_meas, self.mesh.n_elems), dtype=self.mesh.dtype)
        # for i, ex_line in enumerate(self.protocol.ex_mat):
        #     f = self.solve(ex_line)
        #     v[i] = subtract_row(f[self.mesh.el_pos], self.protocol.meas_mat[i])
        #     ri = subtract_row(r_el, self.protocol.meas_mat[i])
        #     for (e, ijk) in enumerate(self.mesh.element):
        #         _jac[i, :, e] = np.dot(np.dot(ri[:, ijk], self.se[e]), f[ijk])
        # jac = np.concatenate(_jac)

        # Build Jacobian matrix element wise (column wise)
        # Je = Re*Ke*Ve = (n_measx3) * (3x3) * (3x1)
        jac = np.zeros((self.protocol.n_meas, self.mesh.n_elems), dtype=self.mesh.dtype)
        indices = self.protocol.meas_mat[:, 2]
        f_n = f[indices]  # replica of potential on nodes of difference excitations
        for e, ijk in enumerate(self.mesh.element):
            jac[:, e] = np.sum(np.dot(ri[:, ijk], self.se[e]) * f_n[:, ijk], axis=1)

        # Jacobian normalization: divide each row of J (J[i]) by abs(v0[i])
        if normalize:
            jac = jac / np.abs(v0[:, None])
        return jac, v0

    def compute_b_matrix(
        self,
        perm: Optional[Union[int, float, complex, np.ndarray]] = None,
    ):
        """
        Compute back-projection mappings (smear matrix)

        Parameters
        ----------
        perm : Union[int, float, np.ndarray], optional
            permittivity on elements ; shape (n_tri,), by default `None`.
            if perm is `None`, the computation of smear matrix will be based
            on the permittivity of the mesh, self.mesh.perm

        Returns
        -------
        np.ndarray
            back-projection mappings (smear matrix); shape(n_exc, n_pts, 1), dtype= bool
        """
        self.assemble_pde(perm)
        f = self.solve_vectorized(self.protocol.ex_mat)
        f_el = f[:, self.mesh.el_pos]
        return _smear_nd(f, f_el, self.protocol.meas_mat)

    def compute_b_matrix_iter(
        self,
        perm: Optional[Union[int, float, complex, np.ndarray]] = None,
    ):
        """Compute back-projection mappings (smear matrix) [obsolete]"""
        self.assemble_pde(perm)
        b_mat = np.zeros((self.protocol.n_exc, self.protocol.n_meas, self.mesh.n_nodes))

        for i in range(self.protocol.n_exc):
            ex_line = self.protocol.ex_mat[i]
            f = self.solve(ex_line=ex_line)
            f_el = f[self.mesh.el_pos]
            # build bp projection matrix
            # 1. we can either smear at the center of elements, using
            #    >> fe = np.mean(f[:, self.tri], axis=1)
            # 2. or, simply smear at the nodes using f
            b_mat[i] = _smear(f, f_el, self.protocol.meas_mat[i])

        return np.concatenate(b_mat)


def _smear(f: np.ndarray, fb: np.ndarray, pairs: np.ndarray):
    """
    Build smear matrix B for bp for one exitation [obsolete]

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


def _smear_nd(f: np.ndarray, fb: np.ndarray, meas_pattern: np.ndarray) -> np.ndarray:
    """
    Build smear matrix B for bp (vectorized version using exc_idx from meas_pattern)

    Parameters
    ----------
    f: np.ndarray
        potential on nodes; shape (n_exc, n_pts)
    fb: np.ndarray
        potential on adjacent electrodes; shape (n_exc, n_el)
    meas_pattern: np.ndarray
        electrodes numbering pairs; shape (n_meas_tot, 3)

    Returns
    -------
    np.ndarray
        back-projection (smear) matrix; shape (n_meas_tot, n_pts), dtype= bool
    """
    n = meas_pattern[:, 0]
    m = meas_pattern[:, 1]
    exc_id = meas_pattern[:, 2]
    # (n_meas_tot,) voltages on electrodes
    f_min = np.minimum(fb[exc_id, n], fb[exc_id, m])
    f_max = np.maximum(fb[exc_id, n], fb[exc_id, m])
    # contruct matrix of shapes (n_meas_tot, n_pts) for comparison
    n_pts = f.shape[1]
    f_min = np.repeat(f_min[:, np.newaxis], n_pts, axis=1)
    f_max = np.repeat(f_max[:, np.newaxis], n_pts, axis=1)
    f_pts = f[exc_id]  # voltages on nodes of all excitations

    return np.array((f_min < f_pts) & (f_pts <= f_max))


def subtract_row(v: np.ndarray, meas_pattern: np.ndarray):
    """
    Build the voltage differences on axis=1 using the meas_pattern.
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
    return v[meas_pattern[:, 0]] - v[meas_pattern[:, 1]]


def subtract_row_vectorized(v: np.ndarray, meas_pattern: np.ndarray):
    """
    Build the voltage differences on axis=1 using the meas_pattern.
    v_diff[k] = v[exc_id, i] - v[exc_id, j]

    New implementation 33% less computation time

    Parameters
    ----------
    v: np.ndarray
        (n_exc, n_el) boundary measurements or (n_exc, (n_el, n_element)) nodes resistance
    meas_pattern: np.ndarray
        Nx2 subtract_row pairs; shape (n_meas_tot, 3)

    Returns
    -------
    np.ndarray
        difference measurements v_diff
    """
    idx = meas_pattern[:, 2]
    return v[idx, meas_pattern[:, 0]] - v[idx, meas_pattern[:, 1]]


def assemble(
    ke: np.ndarray, tri: np.ndarray, perm: np.ndarray, n_pts: int, ref: int = 0
):
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

    # set reference nodes before constructing sparse matrix
    if 0 <= ref < n_pts:
        dirichlet_ind = np.logical_or(row == ref, col == ref)
        # K[ref, :] = 0, K[:, ref] = 0
        row = row[~dirichlet_ind]
        col = col[~dirichlet_ind]
        data = data[~dirichlet_ind]
        # K[ref, ref] = 1.0
        row = np.append(row, ref)
        col = np.append(col, ref)
        data = np.append(data, 1.0)

    # for efficient sparse inverse (csc)
    return sparse.csr_matrix((data, (row, col)), shape=(n_pts, n_pts))


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


def _k_triangle(xy: np.ndarray):
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
    # at = 0.5 * np.linalg.det(s[[0, 1]])
    at = 0.5 * det2x2(s[0], s[1])

    # Local stiffness matrix (e for element)
    return np.dot(s, s.T) / (4.0 * at)


def det2x2(s1: np.ndarray, s2: np.ndarray):
    """Calculate the determinant of a 2x2 matrix"""
    return s1[0] * s2[1] - s1[1] * s2[0]


def _k_tetrahedron(xy: np.ndarray):
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
