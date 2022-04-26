# coding: utf-8
# pylint: disable=invalid-name, no-member, too-many-locals
# pylint: disable=too-many-instance-attributes
""" 2D/3D FEM routines """
# Copyright (c) Benyuan Liu. All Rights Reserved.
# Distributed under the (new) BSD License. See LICENSE.txt for more info.
from __future__ import division, absolute_import, print_function

from collections import namedtuple
from dataclasses import dataclass
import timeit
from typing import Any, Union
import numpy as np
import numpy.linalg as la
from scipy import sparse

from pyeit.eit.utils import eit_scan_lines


@dataclass
class PdeResult:
    v:np.ndarray
    jac:np.ndarray
    b_matrix:np.ndarray


class Forward:
    """FEM forward computing code"""

    def __init__(self, mesh, el_pos):
        """
        A good FEM forward solver should only depend on
        mesh structure and the position of electrodes

        Parameters
        ----------
        mesh: dict
            mesh structure, {'node', 'element', 'perm'}
        el_pos: NDArray
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
        ref_el = 0
        while ref_el in self.el_pos:
            ref_el += 1
        self.ref = ref_el

        # infer dimensions from mesh
        self.n_pts, self.n_dim = self.pts.shape
        self.n_tri, self.n_vertices = self.tri.shape
        self.ne = el_pos.size # TODO n_el would be more consistent

    def solve_eit(self, ex_mat=None, step=1, perm=None, parser=None, **kwargs):
        """
        EIT simulation, generate perturbation matrix and forward v

        Parameters
        ----------
        ex_mat: NDArray
            numLines x n_el array, stimulation matrix
        step: int
            the configuration of measurement electrodes (default: apposition)
        perm: NDArray
            Mx1 array, initial x0. must be the same size with self.tri_perm
        parser: str
            see voltage_meter for more details.
        vector: bool, optional
            Use vectorized methods or regular methods, for compatibility.

        Returns
        -------
        jac: NDArray
            number of measures x n_E complex array, the Jacobian of shape(n_exc, n_el, n_tri)
        v: NDArray
            number of measures x 1 array, simulated boundary measures of shape(n_exc, n_el)
        b_matrix: NDArray(bool)
            back-projection mappings (smear matrix) of shape(n_exc, n_pts, 1)
        """
        # initialize/extract the scan lines (default: apposition)
        if ex_mat is None:
            ex_mat = eit_scan_lines(16, 8)

        # initialize the permittivity on element
        if perm is None:
            perm0 = self.tri_perm
        elif np.isscalar(perm):
            perm0 = np.ones(self.n_tri, dtype=np.float)
        else:
            assert perm.shape == (self.n_tri,)
            perm0 = perm
        
        f, jac_i = self.solve_nd(ex_mat, perm0)
        f_el = f[:, self.el_pos]
        # boundary measurements, subtract_row-voltages on electrodes
        diff_op = voltage_meter(ex_mat, n_el=self.ne, step=step, parser=parser)
        print(f'{diff_op=}, {diff_op.shape=}')
        print(f'{f_el=}, {f_el.shape=}')
        v = subtract_row(f_el, diff_op)
        print(f'{v=}, {v.shape=}')
        print(f'{jac_i=}, {jac_i.shape=}')
        jac = subtract_row(jac_i, diff_op)
        # build bp projection matrix
        # 1. we can either smear at the center of elements, using
        #    >> fe = np.mean(f[:, self.tri], axis=1)
        # 2. or, simply smear at the nodes using f
        b_matrix = smear_nd(f, f_el, diff_op)

        # update output, now you can call p.jac, p.v, p.b_matrix
        return  PdeResult(jac, v, b_matrix)

    def solve(self, ex_line, perm):
        """
        with one pos (A), neg(B) driven pairs, calculate and
        compute the potential distribution (complex-valued)

        The calculation of Jacobian can be skipped.
        Currently, only simple electrode model is supported,
        CEM (complete electrode model) is under development.

        Parameters
        ----------
        ex_line: NDArray
            stimulation (scan) patterns/lines
        perm: NDArray
            permittivity on elements (initial)

        Returns
        -------
        f: NDArray
            potential on nodes
        J: NDArray
            Jacobian
        """
        # 1. calculate local stiffness matrix (on each element)
        ke = calculate_ke(self.pts, self.tri)

        # 2. assemble to global K
        kg = assemble(ke, self.tri, perm, self.n_pts, ref=self.ref)

        # 3. calculate electrode impedance matrix R = K^{-1}
        r_matrix = la.inv(kg)
        r_el = r_matrix[self.el_pos]

        # 4. solving nodes potential using boundary conditions
        b = self._natural_boundary(ex_line)
        f = np.dot(r_matrix, b).ravel()

        # 5. build Jacobian matrix column wise (element wise)
        #    Je = Re*Ke*Ve = (nex3) * (3x3) * (3x1)
        jac = np.zeros((self.ne, self.n_tri), dtype=perm.dtype)
        for (i, e) in enumerate(self.tri):
            jac[:, i] = np.dot(np.dot(r_el[:, e], ke[i]), f[e])

        # print(f'{b=}')
        # print(f'{f=}')
        # print(f'{jac=}')
        print(f'{self.pts.shape=}')
        print(f'{self.tri.shape=}')
        print(f'{b.shape=}')
        print(f'{f.shape=}')
        print(f'{jac.shape=}')
        return f, jac

    def solve_nd(self, ex_mat:np.ndarray, perm:np.ndarray)->tuple[np.ndarray, np.ndarray]:
        """
        Vectorized version of solve. It take the full ex_mat
        instead of lines.

        with one pos (A), neg(B) driven pairs, calculate and
        compute the potential distribution (complex-valued)

        The calculation of Jacobian can be skipped.
        Currently, only simple electrode model is supported,
        CEM (complete electrode model) is under development.

        Parameters
        ----------
        ex_mat: NDArray
            stimulation/excitation matrix of shape (n_exc, 2)
        perm: NDArray
            permittivity on elements (initial) of shape (n_tri,)

        Returns
        -------
        f: NDArray
            potential on nodes of shape (ne, n_pts)
        J: NDArray
            Jacobian of shape (n_exc,ne, n_tri)
        """
        # 1. calculate local stiffness matrix (on each element)
        ke = calculate_ke(self.pts, self.tri)

        # 2. assemble to global K
        kg = assemble(ke, self.tri, perm, self.n_pts, ref=self.ref)

        # 3. calculate electrode impedance matrix R = K^{-1}
        r_matrix = la.inv(kg)
        r_el = r_matrix[self.el_pos]

        # 4. solving nodes potential using boundary conditions
        b = self._natural_boundary_nd(ex_mat)
        
        f = np.dot(r_matrix, b[:, None]).T.reshape(b.shape[:-1])

        # 5. build Jacobian matrix column wise (element wise)
        #    Je = Re*Ke*Ve = (nex3) * (3x3) * (3x1)
        jac = np.zeros((ex_mat.shape[0], self.ne, self.n_tri), dtype=perm.dtype)

        def jac_init(jac, k):
            for (i, e) in enumerate(self.tri):
                jac[:, i] = np.dot(np.dot(r_el[:, e], ke[i]), f[k, e])
            return jac

        jac = np.array(list(map(jac_init, jac, np.arange(ex_mat.shape[0]))))
        # print(f'{b=}')
        # print(f'{f=}')
        # print(f'{jac=}')
        print(f'{self.pts.shape=}')
        print(f'{self.tri.shape=}')
        print(f'{ex_mat.shape=}')
        print(f'{perm.shape=}')
        print(f'{b.shape=}')
        print(f'{f.shape=}')
        print(f'{jac.shape=}')
        return f, jac

    def _natural_boundary(self, ex_line):
        """
        Notes
        -----
        Generate the Neumann boundary condition. In utils.py,
        you should note that ex_line is local indexed from 0...15,
        which need to be converted to global node number using el_pos.
        """
        drv_a_global = self.el_pos[ex_line[0]]
        drv_b_global = self.el_pos[ex_line[1]]

        # global boundary condition
        b = np.zeros((self.n_pts, 1))
        b[drv_a_global] = 1.0
        b[drv_b_global] = -1.0

        return b

    def _natural_boundary_nd(self, ex_mat:np.ndarray)->np.ndarray:
        """ Generate the Neumann boundary condition.

        In utils.py, you should note that ex_mat is local indexed from 0...15,
        which need to be converted to global node number using el_pos.
        
        Parameters
        ----------
            ex_mat (np.ndarray): Excitation matrix shape=(n_exc, 2)

        Returns
        ----------
            np.ndarray: global boundary condition on pts of shape (n_pts, 1)
        """

        drv_a_global = self.el_pos[ex_mat[:, 0]]
        drv_b_global = self.el_pos[ex_mat[:, 1]]

        # global boundary condition
        b = np.zeros((ex_mat.shape[0], self.n_pts, 1))
        b[np.arange(drv_a_global.shape[0]), drv_a_global] = 1.0
        b[np.arange(drv_b_global.shape[0]), drv_b_global] = -1.0

        return b


def smear(f, fb, pairs):
    """
    build smear matrix B for bp

    Parameters
    ----------
    f: NDArray
        potential on nodes
    fb: NDArray
        potential on adjacent electrodes
    pairs: NDArray
        electrodes numbering pairs

    Returns
    -------
    B: NDArray
        back-projection matrix
    """

    # Replacing the code below by a faster implementation in Numpy
    f_min, f_max = np.minimum(fb[pairs[:, 0]], fb[pairs[:, 1]]).reshape(
        (-1, 1)
    ), np.maximum(fb[pairs[:, 0]], fb[pairs[:, 1]]).reshape((-1, 1))
    # b_matrix = []
    # for i, j in pairs:
    #     f_min, f_max = min(fb[i], fb[j]), max(fb[i], fb[j])
    #     b_matrix.append((f_min < f) & (f <= f_max))
    # return np.array(b_matrix)
    return (f_min < f) & (f <= f_max)


def smear_nd(f, fb, pairs):
    """
    Same as smear, except it takes advantage of
    Numpy's vectorization capacities.
    build smear matrix B for bp

    Parameters
    ----------
    f: NDArray
        potential on nodes
    fb: NDArray
        potential on adjacent electrodes
    pairs: NDArray
        electrodes numbering pairs

    Returns
    -------
    B: NDArray
        back-projection matrix
    """

    # Replacing the below code by a faster implementation in Numpy
    def b_matrix_init(k):
        return smear(f[k], fb[k], pairs[k])

    return np.array(list(map(b_matrix_init, np.arange(f.shape[0]))))

def smear_nd_new(f:np.ndarray, fb:np.ndarray, meas_pattern:np.ndarray)->np.ndarray:
    """
    Same as smear, except it takes advantage of
    Numpy's vectorization capacities.
    build smear matrix B for bp

    Parameters
    ----------
    f: NDArray
        potential on nodes
    fb: NDArray
        potential on adjacent electrodes
    meas_pattern: NDArray
        electrodes numbering pairs of shape (n_exc, n_meas, 2)

    Returns
    -------
    B: NDArray
        back-projection matrix
    """

    # Replacing the below code by a faster implementation in Numpy
    def b_matrix_init(k):
        return smear(f[k], fb[k], meas_pattern[k])

    idx_meas_0=meas_pattern[:,:,0]
    idx_meas_1=meas_pattern[:,:,1]
    n_exc = meas_pattern.shape[0]
    idx_exc=np.ones_like(idx_meas_0, dtype=int) * np.arange(n_exc).reshape(n_exc,1)


    f_min = np.minimum(fb[meas_pattern[:, 0]], fb[meas_pattern[:, 1]]).reshape((-1, 1))
    f_max= np.maximum(fb[meas_pattern[:, 0]], fb[meas_pattern[:, 1]]).reshape((-1, 1))
    # b_matrix = []
    # for i, j in pairs:
    #     f_min, f_max = min(fb[i], fb[j]), max(fb[i], fb[j])
    #     b_matrix.append((f_min < f) & (f <= f_max))
    # return np.array(b_matrix)
    # return (f_min < f) & (f <= f_max)

    return np.array(list(map(b_matrix_init, np.arange(f.shape[0]))))


def subtract_row(v:np.ndarray, meas_pattern:np.ndarray) -> np.ndarray:
    """
    Same as subtract_row, except it takes advantage of
    Numpy's vectorization capacities.
    v_diff[k] = v[i, :] - v[j, :]

    David Metz: new implementation 33% less computation time

    Parameters
    ----------
    v: NDArray
        Nx1 boundary measurements vector or NxM matrix of shape (n_exc,n_el,1)
    meas_pattern: NDArray
        of shape (n_exc, n_meas, 2) Nx2 subtract_row pairs ??could be a list??

    Returns
    -------
    v_diff: NDArray
        difference measurements
    """

    if v.shape[:1] != meas_pattern.shape[:1]:
        raise ValueError(
            f'Measurements vector v ({v.shape=}) should have same 1stand 2nd dim as meas_pattern ({meas_pattern.shape=})')

    # creation of excitation indexe for each idx_meas
    idx_meas_0=meas_pattern[:,:,0]
    idx_meas_1=meas_pattern[:,:,1]
    n_exc = meas_pattern.shape[0]
    idx_exc=np.ones_like(idx_meas_0, dtype=int) * np.arange(n_exc).reshape(n_exc,1)

    return v[idx_exc,idx_meas_0]-v[idx_exc,idx_meas_1]


def voltage_meter(ex_mat:np.ndarray, n_el:int=16, step:int=1, parser:Union[str, list[str]]=None)->np.ndarray:
    """
    Faster implementation using numpy's native funcs.
    Made to work with a full matrix, unlike voltage_meter.

    David Metz: new implementation 35% less computation time

    extract subtract_row-voltage measurements on boundary electrodes.
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
    ex_mat: NDArray
        Nx2 array, [positive electrode, negative electrode]. of shape (n_exc, 2)
    n_el: int
        number of total electrodes.
    step: int
        measurement method (two adjacent electrodes are used for measuring).
    parser: str or list[str]
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
    v: NDArray
        (N-1)*2 arrays of subtract_row pairs
    """
    # local node
    ex_mat= ex_mat.astype(int)
    n_exc= ex_mat.shape[0]
    drv_a= np.ones((n_exc, n_exc), dtype=int) * ex_mat[:, 0].reshape(n_exc, 1)
    drv_b= np.ones((n_exc, n_exc), dtype=int) * ex_mat[:, 1].reshape(n_exc, 1)
    # print(f'{drv_a=}, {drv_a.shape=}')
    # print(f'{drv_b=}, {drv_b.shape=}')

    if not isinstance(parser, list):  # transform parser in list
        parser = [parser]

    meas_current = "meas_current" in parser 
    fmmu_rotate = any(p in ("fmmu", "rotate_meas") for p in parser)
    i0 = drv_a if fmmu_rotate else np.zeros_like(drv_a)

    # print(f'{i0=}, {i0.shape=}')
    idx_el=np.ones((n_exc, n_exc), dtype=int) * np.arange(n_el)
    # print(f'{idx_el=}, {idx_el.shape=}')
    a= i0+idx_el
    # print(f'{a=}, {a.shape=}')
    m= a % n_el
    # print(f'{m=}, {m.shape=}')
    n= (m + step) % n_el
    all_meas_pattern = np.concatenate((n[:,:,np.newaxis],m[:,:,np.newaxis]), 2)
    # print(f'1st {all_meas_pattern=}, {all_meas_pattern.shape=}')

    if meas_current:
        return all_meas_pattern

    # print(f'{m=}, {m.shape=}')
    # print(f'{n=}, {n.shape=}')
    # print(f'{drv_a=}, {drv_a.shape=}')
    diff_pairs_mask = np.logical_and.reduce((m != drv_a, m != drv_b , n != drv_a, n != drv_b))
    # print(f'{diff_pairs_mask=}, {diff_pairs_mask.shape=}')
    filtered_meas_pattern=all_meas_pattern[diff_pairs_mask]
    # to reshape after masking
    return filtered_meas_pattern.reshape(n_exc, -1, 2)


def assemble(ke, tri, perm, n_pts, ref=0):
    """
    Assemble the stiffness matrix (using sparse matrix)

    Parameters
    ----------
    ke: NDArray
        n_tri x (n_dim x n_dim) 3d matrix
    tri: NDArray
        the structure of mesh
    perm: NDArray
        n_tri x 1 conductivities on elements
    n_pts: int
        number of nodes
    ref: int
        reference electrode

    Returns
    -------
    K: NDArray
        , NxN array of complex stiffness matrix

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
    k_matrix = sparse.csr_matrix((data, (row, col)), shape=(n_pts, n_pts), dtype=perm.dtype)

    # the stiffness matrix may not be sparse
    k_matrix = k_matrix.toarray()

    # place reference electrode
    if 0 <= ref < n_pts:
        k_matrix[ref, :] = 0.0
        k_matrix[:, ref] = 0.0
        k_matrix[ref, ref] = 1.0

    return k_matrix


def calculate_ke(pts, tri):
    """
    Calculate local stiffness matrix on all elements.

    Parameters
    ----------
    pts: NDArray
        Nx2 (x,y) or Nx3 (x,y,z) coordinates of points
    tri: NDArray
        Mx3 (triangle) or Mx4 (tetrahedron) connectivity of elements

    Returns
    -------
    ke_array: NDArray
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


def _k_triangle(xy):
    """
    given a point-matrix of an element, solving for Kij analytically
    using barycentric coordinates (simplex coordinates)

    Parameters
    ----------
    xy: NDArray
        (x,y) of nodes 1,2,3 given in counterclockwise manner

    Returns
    -------
    ke_matrix: NDArray
        local stiffness matrix
    """
    # edges (vector) of triangles
    s = xy[[2, 0, 1]] - xy[[1, 2, 0]]
    # s1 = xy[2, :] - xy[1, :]
    # s2 = xy[0, :] - xy[2, :]
    # s3 = xy[1, :] - xy[0, :]

    # area of triangles. Note, abs is removed since version 2020,
    # user must make sure all triangles are CCW (conter clock wised).
    # at = 0.5 * la.det(s[[0, 1]])
    at = 0.5 * det2x2(s[0], s[1])
    # TODO maybe replace with:
    # at= 0.5 * np.linalg.det(s)

    # (e for element) local stiffness matrix
    ke_matrix = np.dot(s, s.T) / (4.0 * at)

    return ke_matrix


def det2x2(s1, s2):
    """Calculate the determinant of a 2x2 matrix"""
    return s1[0] * s2[1] - s1[1] * s2[0]


def _k_tetrahedron(xy):
    """
    given a point-matrix of an element, solving for Kij analytically
    using barycentric coordinates (simplex coordinates)

    Parameters
    ----------
    xy: NDArray
        (x,y) of nodes 1, 2, 3, 4 given in counterclockwise manner,
        see notes.

    Returns
    -------
    ke_matrix: NDArray
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
    a = [sign * np.cross(s[i], s[j]) for (i, j), sign in zip(ij_pairs, signs)]
    a = np.array(a)

    # local (e for element) stiffness matrix
    ke_matrix = np.dot(a, a.transpose()) / (36.0 * vt)

    return ke_matrix


if __name__ == "__main__":

    from glob_utils.debug.debugging_help import print_np
    print_np(np.arange(3))
    print_np(np.arange(3))


    import numpy as np

    import pyeit.mesh as mesh
    from pyeit.mesh import quality
    from pyeit.eit.fem import Forward
    from pyeit.eit.utils import eit_scan_lines

    """ 0. build mesh """
    # Mesh shape is specified with fd parameter in the instantiation, e.g : fd=thorax , Default :fd=circle
    mesh_obj, el_pos = mesh.create(16, h0=0.1)

    # extract node, element, alpha
    pts = mesh_obj["node"]
    tri = mesh_obj["element"]
    x, y = pts[:, 0], pts[:, 1]
    quality.stats(pts, tri)

    # change permittivity
    anomaly = [{"x": 0.40, "y": 0.50, "d": 0.20, "perm": 100.0}]
    mesh_new = mesh.set_perm(mesh_obj, anomaly=anomaly, background=1.0)
    perm = mesh_new["perm"]

    """ 1. FEM forward simulations """
    # setup EIT scan conditions
    ex_dist, step = 7, 1
    ex_mat = eit_scan_lines(16, ex_dist)
    # Define electrode current sink and current source
    ex_line = ex_mat[0].ravel()

    # calculate simulated data using FEM
    fwd = Forward(mesh_obj, el_pos)
    p= fwd.solve_eit(ex_mat, perm=perm, vector=True)
    print(f'{p.b_matrix=}, {p.b_matrix.shape=}')
    print(f'{p.v=}, {p.v.shape=}')
    # f, _ = fwd.solve(ex_line, perm=perm)
    






    # a = np.array([[1, 2], [3, 4]])

    # print(np.linalg.det(a), det2x2(a[0], a[1]))

