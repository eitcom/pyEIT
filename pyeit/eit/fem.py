# coding: utf-8
# pylint: disable=invalid-name, no-member, too-many-locals
""" 2D/3D FEM routines """
from __future__ import absolute_import

from collections import namedtuple
import numpy as np
import scipy.linalg as la

from .utils import eit_scan_lines


class Forward(object):
    """ FEM forward computing code """

    def __init__(self, mesh, el_pos):
        """
        a good FEM forward solver should only depend on
        mesh structure and the position of electrodes

        Parameters
        ----------
        mesh : dict
            mesh structure
        el_pos : NDArray
            numbering of electrodes positions
        """
        self.no2xy = mesh['node']
        self.el2no = mesh['element']
        self.tri_perm = mesh['alpha']
        self.el_pos = el_pos
        self.no_num, self.dim = self.no2xy.shape
        self.el_num, self.n_vertices = self.el2no.shape

    def solve(self, ex_mat=None, step=1, perm=None, parser=None):
        """
        generate perturbation matrix and forward v

        Parameters
        ----------
        ex_mat : NDArray
            numLines x n_el array, excitation matrix
        step : int
            the configuration of the measurement electrodes (default: adjacent)
        perm : NDArray
            Mx1 array, initial x0
        parser : str
            if parser is 'fmmu', diff_pairs are re-indexed
            if parser is 'std', diff always start from the 1st electrode

        Returns
        -------
        jac : NDArray
            number of measures x n_E complex array, the Jacobian
        v : NDArray
            number of measures x 1 array, simulated boundary measures
        b_matrix : NDArray
            back-projection mappings (smear matrix)
        """
        # initialize permittivity on elements
        if perm is not None:
            tri_perm = perm
        else:
            tri_perm = self.tri_perm

        # extract scan lines (typical: apposition)
        if ex_mat is None:
            ex_mat = eit_scan_lines(16, 8)
        num_lines = np.shape(ex_mat)[0]

        # calculate f and Jacobian loop over all excitation lines
        jac, v, b_matrix = [], [], []
        for i in range(num_lines):
            # FEM solver
            ex_line = ex_mat[i]
            f, jac_i = self.solve_once(ex_line, tri_perm)

            # diff on electrodes
            diff_array = diff_pairs(ex_line, step, parser)
            v_diff = diff(f[self.el_pos], diff_array)
            jac_diff = diff(jac_i, diff_array)

            # build bp projection matrix
            # 1. we can either smear at the center of elements, using
            #    >> fe = np.mean(f[self.el2no], axis=1)
            # 2. or, simply smear at the nodes using f
            f_el = f[self.el_pos]
            b = smear(f, f_el, diff_array)

            # append
            v.append(v_diff)
            jac.append(jac_diff)
            b_matrix.append(b)

        # update output
        pde_result = namedtuple("pde_result", ['jac', 'v', 'b_matrix'])
        p = pde_result(jac=np.vstack(jac),
                       v=np.hstack(v),
                       b_matrix=np.vstack(b_matrix))
        return p

    def solve_once(self, ex_line, tri_perm=None):
        """
        with one pos (A), neg(B) driven pairs, calculate and
        compute the potential distribution (complex-valued)

        Parameters
        ex_line : NDArray
            excitation pattern/scan line
        tri_perm : NDArray
            permittivity on elements (initial)

        Returns
        -------
        f : NDArray
            potential on nodes
        J : NDArray
            Jacobian
        """
        # boundary conditions
        b = self.natural_boundary(ex_line)

        # assemble
        ref_el = self.el_pos[0]
        k_global, k_element = assemble(self.no2xy,
                                       self.el2no,
                                       perm=tri_perm,
                                       ref=ref_el)

        # electrode impedance
        r_matrix = la.inv(k_global)
        # nodes potential
        f = np.dot(r_matrix, b).ravel()

        # build perturbation on each element, Je = R*J*Ve
        ne = len(self.el_pos)
        jac = np.zeros((ne, self.el_num), dtype='complex')
        r_el = r_matrix[self.el_pos]

        # build jacobian matrix column-by-column (element wise)
        for i in range(self.el_num):
            ei = self.el2no[i, :]
            jac[:, i] = np.dot(np.dot(r_el[:, ei], k_element[i]), f[ei])

        return f, jac

    def natural_boundary(self, ex_line):
        """ generate Neumann boundary conditions """
        b = np.zeros((self.no_num, 1))
        a_pos = self.el_pos[np.where(ex_line == 1)]
        b_pos = self.el_pos[np.where(ex_line == -1)]
        b[a_pos] = 1.
        b[b_pos] = -1.

        return b


def smear(f, fb, pairs):
    """
    build smear matrix B for bp

    Parameters
    ----------
    f : NDArray
        potential on nodes
    fb : NDArray
        potential on adjacent electrodes
    pairs : NDArray
        electrodes numbering pairs

    Returns
    -------
    NDArray
        back-projection matrix
    """
    b_matrix = []
    for i, j in pairs:
        f_min, f_max = min(fb[i], fb[j]), max(fb[i], fb[j])
        b_matrix.append((f_min < f) & (f <= f_max))
    return np.array(b_matrix)


def diff(v, pairs):
    """
    v_diff[k] = v[i, :] - v[j, :]

    Parameters
    ----------
    v : NDArray
        boundary measurements
    pairs : NDArray
        diff pairs

    Returns
    -------
    NDArray
        difference measurements
    """
    i = pairs[:, 0]
    j = pairs[:, 1]
    # row-wise operation
    v_diff = v[i] - v[j]

    return v_diff


def diff_pairs(ex_line, m_step=1, parser=None):
    """
    extract diff-voltage measurements on boundary electrodes

    Notes
    -----
    A : current driving electrode
    B : current sink
    M, N : boundary electrodes, where v_diff = v_n - v_m

    Parameters
    ----------
    ex_line : NDArray
        nEx1 array, 1 for positive, -1 for negative, 0 otherwise
    m_step : int
        measurement method (which two electrodes are used for measuring)
    parser : str
        if parser is 'fmmu', data are trimmed, start index (i) is always 'A'.

    Returns
    -------
    v : NDArray
        (N-1)*2 arrays of diff pairs
    """
    drv_a = np.where(ex_line == 1)[0][0]
    drv_b = np.where(ex_line == -1)[0][0]
    l = len(ex_line)
    i0 = drv_a if parser is 'fmmu' else 0

    # build
    v = []
    for a in range(i0, i0 + l):
        m = a % l
        n = (m + m_step) % l
        if not(m == drv_a or m == drv_b or n == drv_a or n == drv_b):
            # the order of m, n matters
            v.append([n, m])

    return np.array(v)


def assemble(no2xy, el2no, perm=None, ref=0):
    """
    assemble the stiffness matrix (do not build into class)

    Parameters
    ----------
    no2xy : NDArray
        Nx2 (x,y) or Nx3 (x,y,z) coordinates of points
    el2no : NDArray
        Mx3 (triangle) or Mx4 (tetrahedron) connectivity of elements
    perm : NDArray
        conductivities on elements
    ref : int
        reference electrode

    Returns
    -------
    NDArray
        k_matrix, NxN array of complex stiffness matrix

    Notes
    -----
    you can use sparse matrix (IJV) format to automatically add the local
    stiffness matrix to the global matrix.
    """
    no_num, _ = no2xy.shape
    el_num, n_vertices = el2no.shape

    # initialize the permittivity on element
    if perm is None:
        perm = np.ones(el_num, dtype=np.float)

    # check dimension
    if n_vertices == 3:
        # triangle
        _k_local = _k_triangle
    elif n_vertices == 4:
        # tetrahedron
        _k_local = _k_tetrahedron
    else:
        # TODO: only triangles or tetrahedrons
        raise TypeError('num of vertices of the element must be 3 or 4')

    # assemble the global matrix A and local stiffness matrix K
    k_global = np.zeros((no_num, no_num), dtype='complex')
    k_element = np.zeros((el_num, n_vertices, n_vertices), dtype='complex')

    # for each element, calculate local stiffness matrix and add to the global
    for ei in range(el_num):
        # get the nodes and their coordinates for element ei
        no = el2no[ei, :]
        xy = no2xy[no, :]
        pe = perm[ei]

        # compute the KIJ (permittivity=1)
        ke = _k_local(xy)
        k_element[ei] = ke

        # add the contribution to the global matrix.
        # NOTE: in python A[no, no] returns a 3x1 array,
        # use np.ix_ to construct an open mesh from multiple sequences.
        ij = np.ix_(no, no)

        # TODO: use IJV indexed sparse matrix
        # row[ei], col[ei] = np.meshgrid(no, no)
        k_global[ij] += (ke * pe)

    # TODO: use sparse matrix (preferred)
    # sparse matrix automatically adds duplicated entries to global matrix.
    # >> import scipy.sparse as sparse
    # >> K = np.array([k_element[i]*perm[i] for i in range(elNum)])
    # >> A = sparse.coo_matrix((K.ravel(), (row.ravel(), col.ravel())),
    #                          shape=(noNum, noNum),
    #                          dtype='complex')

    # place reference electrode
    if 0 <= ref < no_num:
        k_global[ref, :] = 0.
        k_global[:, ref] = 0.
        k_global[ref, ref] = 1.

    return k_global, k_element


def _k_triangle(xy):
    """
    given a point-matrix of an element, solving for Kij analytically
    using barycentric coordinates (simplex coordinates)

    Parameters
    ----------
    xy : NDArray
        (x,y) of nodes 1,2,3 given in counterclockwise manner

    Returns
    -------
    NDArray
        k_matrix, local stiffness matrix
    """
    # s1 = xy[2, :] - xy[1, :]
    # s2 = xy[0, :] - xy[2, :]
    # s3 = xy[1, :] - xy[0, :]
    s = xy[[2, 0, 1]] - xy[[1, 2, 0]]

    #
    at = 0.5 * la.det(s[[0, 1]])

    # vectorized
    k_matrix = np.dot(s, s.transpose()) / (4. * at)

    return k_matrix


def _k_tetrahedron(xy):
    """
    given a point-matrix of an element, solving for Kij analytically
    using barycentric coordinates (simplex coordinates)

    Parameters
    ----------
    xy : NDArray
        (x,y) of nodes 1, 2, 3, 4 given in counterclockwise manner,
        see notes.

    Returns
    -------
    NDArray
        Ae, local stiffness matrix

    Notes
    -----
    counterclockwise is defined such that the barycentric coordinate
    of face (1->2->3) is positive.
    """
    s = xy[[2, 3, 0, 1]] - xy[[1, 2, 3, 0]]

    # volume
    vt = 1./6 * la.det(s[[0, 1, 2]])

    # calculate area vector of triangles
    # re-weighted using alternative (+,-) signs
    ij_pairs = [[0, 1], [1, 2], [2, 3], [3, 0]]
    signs = [1, -1, 1, -1]
    a = [p*np.cross(s[i], s[j]) for (i, j), p in zip(ij_pairs, signs)]
    a = np.array(a)

    # vectorized
    k_matrix = np.dot(a, a.transpose()) / (36. * vt)

    return k_matrix
