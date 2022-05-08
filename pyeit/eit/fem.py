# coding: utf-8
# pylint: disable=invalid-name, no-member, too-many-locals
# pylint: disable=too-many-instance-attributes
""" 2D/3D FEM routines """
# Copyright (c) Benyuan Liu. All Rights Reserved.
# Distributed under the (new) BSD License. See LICENSE.txt for more info.
from __future__ import division, absolute_import, print_function

from typing import Union
import numpy as np
import numpy.linalg as la
from scipy import sparse
import scipy.sparse.linalg


class Forward:
    """FEM forward computing code"""

    def __init__(self, mesh: dict[str, np.ndarray]) -> None:
        """
        FEM forward solver.
        A good FEM forward solver should only depend on
        mesh structure and the position of electrodes.

        Parameters
        ----------
        mesh: dict or dataset
            mesh structure, {'node', 'element', 'perm', 'el_pos', 'ref'}

        Note
        ----
        The nodes are continuous numbered, the numbering of an element is
        CCW (counter-clock-wise).
        """
        self.pts = mesh["node"]
        self.tri = mesh["element"]
        self.tri_perm = mesh["perm"]
        self.el_pos = mesh["el_pos"]
        # ref node should not be on electrodes, it is up to the user to decide
        self.ref_el = mesh["ref"]

        # infer dimensions from mesh
        self.n_pts, self.n_dim = self.pts.shape
        self.n_tri, self.n_vertices = self.tri.shape
        self.n_el = self.el_pos.size
        self.user_perm = self.tri_perm

        # coefficient matrix [initialize]
        self.se = calculate_ke(self.pts, self.tri)
        self.assemble_pde(self.tri_perm, init=True)

    def assemble_pde(self, perm, init: bool = True):
        """
        assemble PDE

        Parameters
        ----------
        perm : Union[int, float, np.ndarray], optional
            permittivity on elements ; shape (n_tri,), by default `None`.
        kinit : bool, optional
            re-calculate kg
        """
        # if self.user_perm != perm and kinit = False, a warning message should
        # be raised, telling a user that it should pass kinit = True
        p = self._check_perm(perm)
        if init:
            self.user_perm = p
            self.kg = assemble(self.se, self.tri, p, self.n_pts, ref=self.ref_el)

    def solve(self, ex_line: np.ndarray = None) -> np.ndarray:
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
        b = np.zeros(self.n_pts)
        b[self.el_pos[ex_line]] = [1, -1]

        # solve
        f = scipy.sparse.linalg.spsolve(self.kg, b)

        return f

    def _check_perm(self, perm: Union[int, float, np.ndarray] = None) -> np.ndarray:
        """
        Check/init the permittivity on element

        Parameters
        ----------
        perm : Union[int, float, np.ndarray], optional, default None
            permittivity on elements, Must be the same size with self.tri_perm.
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
            raise TypeError(f"Wrong type/shape of {perm=}, expected an ndarray(n_tri,)")

        return perm


class EITForward(Forward):
    """EIT Forward simulation, depends on mesh and protocol"""

    def __init__(
        self, mesh: dict[str, np.ndarray], protocol: dict[str, np.ndarray]
    ) -> None:
        """
        EIT Forward Solver

        Parameters
        ----------
        mesh: dict or dataset
            mesh structure, {'node', 'element', 'perm', 'el_pos', 'ref'}
        protocol: dict or dataset
            measurement protocol, {'ex_mat', 'step', 'parser'}

        Notes
        -----
        The Jacobian and the boundary voltages used the SIGN information,
        for example, V56 = V6 - V5 = -V65. If you are using absolute boundary
        voltages for imaging, you MUST normalize it with the signs of v0
        under each current-injecting pattern.
        """
        # FEM solver
        super().__init__(mesh=mesh)

        # EIT measurement protocol
        self.ex_mat = self._check_ex_mat(protocol["ex_mat"])
        self.step = protocol["step"]
        self.parser = protocol["parser"]

        # setup boundary voltage measurement protocol
        self.n_exe = self.ex_mat.shape[0]
        self.diff_op = self.build_meas_pattern()
        self.n_meas = self.diff_op[0].shape[0]

    def build_meas_pattern(self) -> np.ndarray:
        """
        Build the measurement pattern (voltage pairs [N, M])
        for all excitations on boundary electrodes.

        We direct operate on measurements or Jacobian on electrodes,
        so, we can use LOCAL index in this module, do not require el_pos.

        This function runs once, so we favor clearity over speed (vectorization)

        Notes
        -----
        ABMN Model.
        A: current driving electrode,
        B: current sink,
        M, N: boundary electrodes, where v_diff = v_n - v_m.

        Returns
        -------
        np.ndarray
            measurements pattern / subtract_row pairs [N, M]; shape (n_exc, n_meas, 2)
        """
        if not isinstance(self.parser, list):  # transform parser into list
            parser = [self.parser]
        meas_current = "meas_current" in parser
        fmmu_rotate = any(p in ("fmmu", "rotate_meas") for p in parser)

        diff_op = []
        for ex_line in self.ex_mat:
            a, b = ex_line[0], ex_line[1]
            i0 = a if fmmu_rotate else 0
            m = (i0 + np.arange(self.n_el)) % self.n_el
            n = (m + self.step) % self.n_el
            meas_pattern = np.vstack([n, m]).T

            if not meas_current:
                diff_keep = np.logical_and.reduce((m != a, m != b, n != a, n != b))
                meas_pattern = meas_pattern[diff_keep]

            diff_op.append(meas_pattern)

        return diff_op

    def _check_ex_mat(self, ex_mat: np.ndarray = None) -> np.ndarray:
        """
        Check/init stimulation

        Parameters
        ----------
        ex_mat : np.ndarray, optional
            stimulation/excitation matrix, of shape (n_exc, 2), by default `None`.
            If `None` initialize stimulation matrix for n_el electrode and
            adjacent mode (see function `eit_scan_lines`)
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
            # initialize the scan lines for 16 electrodes (default: adjacent)
            ex_mat = np.array([[i, np.mod(i + 1, self.n_el)] for i in range(self.n_el)])
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

    def _check_meas_pattern(
        self, n_exc: int, meas_pattern: np.ndarray = None
    ) -> np.ndarray:
        """
        Check measurement pattern

        Parameters
        ----------
        n_exc : int
            number of excitations/stimulations
        meas_pattern : np.ndarray, optional
           measurements pattern / subtract_row pairs [N, M] to check; shape (n_exc, n_meas_per_exc, 2), by default None
           if None (no meas_pattern has been passed) None is returned

        Returns
        -------
        np.ndarray
            measurements pattern / subtract_row pairs [N, M]; shape (n_exc, n_meas_per_exc, 2)

        Raises
        ------
        TypeError
            raised if meas_pattern is not a nd.array of shape (n_exc, : , 2)
        """
        if meas_pattern is None:
            return None

        if not isinstance(meas_pattern, np.ndarray):
            raise TypeError(
                f"Wrong type of {meas_pattern=}, expected an ndarray; shape ({n_exc}, n_meas_per_exc, 2)"
            )
        # test shape is something like (n_exc, :, 2)
        if meas_pattern.ndim != 3 or meas_pattern.shape[::2] != (n_exc, 2):
            raise TypeError(
                f"Wrong shape of {meas_pattern=}: {meas_pattern.shape=}, expected an ndarray; shape ({n_exc}, n_meas_per_exc, 2)"
            )

        return meas_pattern

    def solve_eit(
        self,
        perm: Union[int, float, np.ndarray] = None,
        init: bool = False,
    ) -> np.ndarray:
        """
        EIT simulation, generate forward v measurement

        Parameters
        ----------
        perm : Union[int, float, np.ndarray], optional
            permittivity on elements ; shape (n_tri,), by default `None`.
        init : bool, optional
            re-calculate kg

        Returns
        -------
        v: np.ndarray
            simulated boundary voltage measurements; shape(n_exe*n_el,)
        """
        self.assemble_pde(perm=perm, init=init)
        v = np.zeros((self.n_exe, self.n_meas))
        for i, ex_line in enumerate(self.ex_mat):
            f = self.solve(ex_line)
            v[i] = subtract_row(f[self.el_pos], self.diff_op[i])

        return v.reshape(-1)

    def compute_jac(
        self,
        perm: Union[int, float, np.ndarray] = None,
        init: bool = False,
        normalize: bool = False,
    ) -> np.ndarray:
        """
        Compute the Jacobian matrix

        Parameters
        ----------
        perm : Union[int, float, np.ndarray], optional
            permittivity on elements ; shape (n_tri,), by default `None`.
        kinit : bool, optional
            re-calculate kg
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
        # update k if necessary and calculate r=inv(k)
        self.assemble_pde(perm=self._check_perm(perm), init=init)
        r_el = la.inv(self.kg.toarray())[self.el_pos]

        # calculate v, jac per excitation pattern (ex_line)
        jac = np.zeros(
            (self.n_exe, self.n_meas, self.n_tri), dtype=self.user_perm.dtype
        )
        v = np.zeros((self.n_exe, self.n_meas))
        for i, ex_line in enumerate(self.ex_mat):
            f = self.solve(ex_line)
            v[i] = subtract_row(f[self.el_pos], self.diff_op[i])
            ri = subtract_row(r_el, self.diff_op[i])
            # Build Jacobian matrix column wise (element wise)
            #    Je = Re*Ke*Ve = (nex3) * (3x3) * (3x1)
            for (e, ijk) in enumerate(self.tri):
                jac[i, :, e] = np.dot(np.dot(ri[:, ijk], self.se[e]), f[ijk])

        # measurement protocol
        J = np.vstack(jac)
        v0 = v.reshape(-1)

        # Jacobian normalization: divide each row of J (J[i]) by abs(v0[i])
        if normalize:
            J = J / np.abs(v0[:, None])
        return J, v0

    def compute_b_matrix(
        self,
        perm: Union[int, float, np.ndarray] = None,
        init: bool = False,
    ) -> np.ndarray:
        """
        Compute back-projection mappings (smear matrix)

        Parameters
        ----------
        perm : Union[int, float, np.ndarray], optional
            permittivity on elements ; shape (n_tri,), by default `None`.
        init : bool, optional
            re-calculate kg

        Returns
        -------
        np.ndarray
            back-projection mappings (smear matrix); shape(n_exc, n_pts, 1), dtype= bool
        """
        self.assemble_pde(self._check_perm(perm), init=init)
        b_mat = np.zeros((self.n_exe, self.n_meas, self.n_pts))

        for i, ex_line in enumerate(self.ex_mat):
            f = self.solve(ex_line=ex_line)
            f_el = f[self.el_pos]
            # build bp projection matrix
            # 1. we can either smear at the center of elements, using
            #    >> fe = np.mean(f[:, self.tri], axis=1)
            # 2. or, simply smear at the nodes using f
            b_mat[i] = _smear(f, f_el, self.diff_op[i])

        return np.vstack(b_mat)


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


def smear_nd(
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
    k_matrix = sparse.csr_matrix((data, (row, col)), shape=(n_pts, n_pts))

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
    # at = 0.5 * np.linalg.det(s[[0, 1]])
    at = 0.5 * det2x2(s[0], s[1])

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
