# coding: utf-8
# pylint: disable=invalid-name, no-member, too-many-locals
""" 2D FEM routines for EIT """
from __future__ import absolute_import

from collections import namedtuple
import numpy as np
import scipy.linalg as la
import scipy.sparse as sparse

from .utils import eit_scan_lines


class forward(object):
    """ FEM forward computing code """

    def __init__(self, mesh, elPos):
        """
        a GOOD FEM forward solver should ONLY depend on
        mesh structure and elPos

        Parameters
        ----------
        mesh : dict
            mesh structure
        elPos : NDArray
            numbering of electrodes positions
        """
        self.no2xy = mesh['node']
        self.el2no = mesh['element']
        self.tri_perm = mesh['alpha']
        self.elPos = elPos

    def solve(self, exMtx=None, step=1, perm=None, parser=None):
        """
        generate pertubation matrix and forward v, following laberge2008

        Parameters
        ----------
        mesh : dict
            mesh object
        elPos : NDArray
            numElx1 ndarray, position of electrodes
        exMtx : NDArray
            numLines x numEl ndarray, excitation matrix
        perm : NDArray
            Mx1 ndarray, initial x0
        parser : str
            if parser is 'et3', then diff_pairs are re-arranged
            if parser is 'std', standard diff from the 1st electrode

        Returns
        -------
        Jac : NDArray
            numMeasure x n_E complex ndarray, the Jacobian
        v : NDArray
            numMeasure x 1 ndarray, simulated boundary measures
        B : NDArray
            back-projection mappings (smear matrix)
        """
        # initialize permitivity on elements
        if perm is not None:
            tri_perm = perm
        else:
            tri_perm = self.tri_perm

        # extratct scan lines of EIT
        if exMtx is None:
            exMtx = eit_scan_lines(16, 8)
        numLines = np.shape(exMtx)[0]

        # calculate f and Jacobian loop over all excitation lines
        Jac, vb, B = None, None, None
        for i in range(numLines):
            # fem solver
            exLine = exMtx[i, :].ravel()
            f, J = self.solve_once(exLine, tri_perm)
            diff_array = diff_pairs(exLine, step, parser)

            # 1. concat vb. voltage at the electrodes is differenced
            v_diff = diff(f[self.elPos], diff_array)
            vb = v_diff if vb is None else np.hstack([vb, v_diff])

            # 2. concat Jac. Jac or sensitivity matrix is formed vstack
            Ji = diff(J, diff_array)
            Jac = Ji if Jac is None else np.vstack([Jac, Ji])

            # 3. build bp map B
            # 3.1 we can either smear at the center of elements, using
            #     >> fe = np.mean(f[self.el2no], axis=1)
            # 3.2 or, more simply, smear at the nodes using f.
            fe = np.mean(f[self.el2no], axis=1)
            Bi = smear(fe, f[self.elPos], diff_array)
            B = Bi if B is None else np.vstack([B, Bi])

        # update output
        r = namedtuple("forward", ['Jac', 'v', 'B'])
        return r(Jac=Jac, v=vb, B=B)

    def solve_once(self, exLine, tri_perm):
        """
        with one-{pos, neg} driven pairs, calculate and
        compute the potential distribution (complex variable)

        Parameters
        exLine : NDArray
            excitation pattern/scan line
        tri_perm : NDArray
            permitivity on elements (initial)

        Returns
        -------
        f : NDArray
            potential on nodes
        R : NDArray
            inv(K), electrodes impedance
        """
        noNum = np.size(self.no2xy, 0)
        elNum = np.size(self.el2no, 0)

        # boundary conditions (current to voltage)
        b = np.zeros((noNum, 1))
        Vpos = self.elPos[np.where(exLine == 1)]
        Vneg = self.elPos[np.where(exLine == -1)]
        b[Vpos] = 1.
        b[Vneg] = -1.

        # assemble
        A, Ke = assembpde(self.no2xy, self.el2no, perm=tri_perm)

        # place reference node
        ref_el = self.elPos[0]
        A[ref_el, :] = 0.
        A[:, ref_el] = 0.
        A[ref_el, ref_el] = 1.

        # electrodes impedance
        R = la.inv(A)
        # nodes potential
        f = np.dot(R, b).ravel()

        # build pertubation on each element, Je = R*J*Ve
        Ne = len(self.elPos)
        J = np.zeros((Ne, elNum), dtype='complex')
        R_el = R[self.elPos]
        for i in range(elNum):
            ei = self.el2no[i, :]
            J[:, i] = np.dot(np.dot(R_el[:, ei], Ke[i]), f[ei])

        return f, J


def smear(f, fb, pairs):
    """ build smear matrix B for bp

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
    Bi = []
    L = len(f)
    for i, j in pairs:
        fmin, fmax = min(fb[i], fb[j]), max(fb[i], fb[j])
        Bi.append((fmin < f) & (f <= fmax))
    return np.array(Bi) / float(L)


def diff(v, pairs):
    """
    vdiff[k] = v[i, :] - v[j, :]

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
    vdiff = []
    for i, j in pairs:
        vdiff.append(v[i] - v[j])

    return vdiff


def diff_pairs(exPat, step=1, parser=None):
    """
    extract diff-voltage measurements on boundary electrodes
    support free-mode of diff pairs

    Parameters
    ----------
    exPat : NDArray
        nEx1 ndarray, 1 for positive, -1 for negative, 0 otherwise
    step : int
        measurement method (which two electrodes are used for measuring)
    parser : str
        if parser is 'et3', data are trimmed, start index (i) is always 'A'.

    Returns
    -------
    v : NDArray
        (N-1)*2 arrarys of diff pairs, i - k, for neighbore mode
    """
    A = np.where(exPat == 1)[0][0]
    B = np.where(exPat == -1)[0][0]
    L = len(exPat)
    v = []
    if parser is 'et3':
        for i in range(L):
            j = (i + A) % L
            k = (j + step) % L
            if not(j == A or j == B or k == A or k == B):
                v.append([j, k])
    else:
        for i in range(L):
            j = (i + step) % L
            if not(i == A or i == B or j == A or j == B):
                v.append([i, j])
    return v


def assembpde(no2xy, el2no, perm=None):
    """
    assemble the stiffness matrix for PDE

    Parameters
    ----------
    no2xy : NDArray
        Nx2 ndarray, (x,y) locations for points
    el2no : NDArray
        Mx3 ndarray, elements (triangles) connectivity
    perm : NDArray
        the conductivities on elements

    Returns
    -------
    NDArray
        A, NxN ndarray of complex stiffness matrix
    """
    noNum = np.size(no2xy, 0)
    elNum = np.size(el2no, 0)

    # initialize the permitivity on element
    if perm is None:
        perm = np.ones(elNum)

    # check dimension
    ndim = no2xy.shape[1]

    # for triangle, the shape of local Ke is (3, 3)
    # for tetrahedron, the shape is (4, 4)
    if ndim == 2:
        nshape = 3
        CmpElMtx = CmpElMtx2D
    elif ndim==3:
        nshape = 4
        CmpElMtx = CmpElMtx3D

    # Assemble the matrix A
    A = np.zeros((noNum, noNum), dtype='complex')
    Ke = np.zeros((elNum, nshape, nshape), dtype='complex')

    for ei in range(elNum):
        # get the nodes and their coordinates for element ei
        no = el2no[ei, :]
        xy = no2xy[no, :]
        pe = perm[ei]

        # compute the KIJ (without permitivity)
        KIJ = CmpElMtx(xy)
        Ke[ei] = KIJ

        # 'add' the 'contribution' to the 'global' matrix.
        # warning, in python A[no, no] will return a 3x1 array,
        # use np.ix_ to construct an open mesh from multiple sequences.
        ij = np.ix_(no, no)
        A[ij] = A[ij] + (KIJ * pe)

    # return
    return A, Ke


def assembpde_sparse(no2xy, el2no, perm=None):
    """
    assemble the stiffness matrix for PDE using coo_sparse

    Notes
    -----
    A.toarray() should not be used, as the major advantage of sparse
    matrix is in solving linear equations. Should figure out how to
    set the reference node in sparse matrix.
    """
    noNum = np.size(no2xy, 0)
    elNum = np.size(el2no, 0)

    # initialize the permitivity on element
    if perm is None:
        perm = np.ones(elNum)

    # prepare IJV
    Ke = np.zeros((elNum, 3, 3), dtype='complex')
    row = np.zeros((elNum, 3, 3), dtype=np.int32)
    col = np.zeros((elNum, 3, 3), dtype=np.int32)

    for ei in range(elNum):
        # get the nodes and their coordinates for element ei
        no = el2no[ei, :]
        xy = no2xy[no, :]

        # compute the KIJ (without permitivity)
        KIJ = CmpElMtx2D(xy)
        Ke[ei] = KIJ

        # build row, col
        row[ei], col[ei] = np.meshgrid(no, no)

    # 'add' the 'contribution' to the 'global' matrix.
    K = np.array([Ke[i]*perm[i] for i in range(elNum)])
    A = sparse.coo_matrix((K.ravel(), (row.ravel(), col.ravel())),
                          shape=(noNum, noNum),
                          dtype='complex')

    # return
    return A, Ke


def CmpElMtx2D(xy):
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
        Ae, local stiffness matrix
    """
    # s1 = xy[2, :] - xy[1, :]
    # s2 = xy[0, :] - xy[2, :]
    # s3 = xy[1, :] - xy[0, :]
    s = xy[[2, 0, 1]] - xy[[1, 2, 0]]

    # Atot = 0.5 * (s2[0]*s3[1] - s3[0]*s2[1])
    Atot = 0.5 * la.det(s[[0, 1]])
    if Atot < 0:
        # idealy nodes should be given in anti-clockwise,
        # but Yang Bin's .mes file is in clockwise manner,
        # so we make this script compatible to his file format
        Atot *= -1.

    # using for-loops
    # Ae = np.zeros((3, 3))
    # for i in range(3):
    #     for j in range(3):
    #         Ae[i, j] = np.dot(grad_phi[i, :], grad_phi[j, :]) * Atot

    # vectorize
    Ae = np.dot(s, s.transpose()) / (4. * Atot)

    return Ae


def CmpElMtx3D(xy):
    """
    given a point-matrix of an element, solving for Kij analytically
    using barycentric coordinates (simplex coordinates)

    Parameters
    ----------
    xy : NDArray
        (x,y) of nodes 1,2,3,4 given in counterclockwise manner

    Returns
    -------
    NDArray
        Ae, local stiffness matrix
    """
    s = xy[[2, 3, 0, 1]] - xy[[1, 2, 3, 0]]

    # calculate Volume from vertices (make volume compatible)
    Vtot = 1/6. * la.det(s[[0, 1, 2]])
    if Vtot < 0:
        Vtot = -Vtot

    # calculate area vector
    A = [cross_product(s[ij]) for ij in [[0, 1], [1, 2], [2, 3], [3, 0]]]
    A = np.array(A)

    # vectorize
    Ae = np.dot(A, A.transpose()) / (36. * Vtot)

    return Ae


def cross_product(xyz):
    """ calculate cross product of xyz[0] and xyz[1] """
    v = [la.det(xyz[:, [1, 2]]),   #  x
         -la.det(xyz[:, [0, 2]]),  #  y
         la.det(xyz[:, [0, 1]])]   #  z
    return np.array(v)


def CmpAoE(no2xy, el2no):
    """
    loop over all elements and find the Area of Elements (aoe)
    return a vector triangle area of n_E

    Parameters
    ----------
    no2xy : NDArray
        Nx2 ndarray, (x,y) locations for points
    el2no : NDArray
        Mx3 ndarray, elements (triangles) connectivity

    Returns
    -------
    NDArray
        ae, area of each element
    """
    elNum = np.size(el2no, 0)
    ae = np.zeros(elNum)
    for ei in range(elNum):
        no = el2no[ei, :]
        xy = no2xy[no, :]
        ae[ei] = tri_area(xy)

    return ae


def tri_area(xy):
    """
    return area of a triangle, given its tri-coordinates xy

    Parameters
    ----------
    xy : NDArray
        (x,y) of nodes 1,2,3 given in counterclockwise manner

    Returns
    -------
    float
        area of this element
    """
    # s2 = xy[0, :] - xy[2, :]
    # s3 = xy[1, :] - xy[0, :]
    s = xy[[0, 1]] - xy[[2, 0]]
    # Atot = 0.5*(s2[0]*s3[1] - s3[0]*s2[1])
    Atot = 0.5*la.det(s)
    # (should be possitive if tri-points are counter-clockwise)
    # abs is for compatibility with Yang-Bin's .mes file
    # whose tri-points are clockwise
    return abs(Atot)


def tet_volume(xyz):
    """ calculate the volume of tetrahedron """
    s = xyz[[2, 3, 0]] - xyz[[1, 2, 3]]
    Vtot = 1/6. * la.det(s)
    return abs(Vtot)


def pdeintrp(no2xy, el2no, node_value):
    """
    given the values on nodes, calculate the interpolated value on elements
    this function was tested and equivalent to MATLAB 'pdeintrp'
    except for the shapes of 'no2xy' and 'el2no'

    Parameters
    ----------
    no2xy : NDArray
        Nx2 ndarray, (x,y) locations for points
    el2no : NDArray
        Mx3 ndarray, elements (triangles) connectivity
    node_value : NDArray
        Nx1 ndarray, real/complex valued

    Returns
    -------
    NDArray
        el_value, Mx1 ndarray, real/complex valued
    """
    N = np.size(no2xy, 0)
    M = np.size(el2no, 0)
    # build e->n matrix, could be accel by sparse
    e2n = np.zeros([M, N], dtype='int')
    for i in range(M):
        e2n[i, el2no[i, :]] = 1
    # in tri-mesh, we average by simply deviding 3.0
    el_value = np.dot(e2n, node_value) / 3.0
    return el_value


def pdetrg(no2xy, el2no):
    """
    analytical calculate the Area and grad(phi_i) using
    barycentric coordinates (simplex coordinates)
    this function is tested and equivalent to MATLAB pdetrg
    except for the shape of 'no2xy' and 'el2no' and the outputs

    note: each node may have multiple gradients in each neighbore
    elements' coordinates. you may averaged all the gradient to
    get one node gradient.

    Parameters
    ----------
    no2xy : NDArray
        Nx2 ndarray, (x,y) locations for points
    el2no : NDArray
        Mx3 ndarray, elements (triangles) connectivity

    Returns
    -------
    Atot : NDArray
        Mx1 ndarray, Element Area
    grad_phi_x : NDArray
        Mx3 ndarray, x-gradient on elements' local coordinate
    grad_phi_y : NDArray
        Mx3 ndarray, y-gradient on elements' local coordinate
    """
    M = np.size(el2no, 0)
    ix = el2no[:, 0]
    iy = el2no[:, 1]
    iz = el2no[:, 2]

    s1 = no2xy[iz, :] - no2xy[iy, :]
    s2 = no2xy[ix, :] - no2xy[iz, :]
    s3 = no2xy[iy, :] - no2xy[ix, :]

    Atot = 0.5*(s2[:, 0]*s3[:, 1] - s3[:, 0]*s2[:, 1])
    if any(Atot) < 0:
        exit("nodes are given in clockwise manner")

    # note in python, reshape place elements first on the right-most index
    grad_phi_x = np.reshape([-s1[:, 1] / (2. * Atot),
                             -s2[:, 1] / (2. * Atot),
                             -s3[:, 1] / (2. * Atot)], [-1, M]).T
    grad_phi_y = np.reshape([s1[:, 0] / (2. * Atot),
                             s2[:, 0] / (2. * Atot),
                             s3[:, 0] / (2. * Atot)], [-1, M]).T

    return Atot, grad_phi_x, grad_phi_y


def pdegrad(no2xy, el2no, node_value):
    """
    given the values on nodes, calculate the averaged-grad on element
    this function was tested and equivalent to MATLAB 'pdegrad'
    except for the shape of 'no2xy', 'el2no'

    Parameters
    ----------
    no2xy : NDArray
        Nx2 ndarray, (x,y) locations for points
    el2no : NDArray
        Mx3 ndarray, elements (triangles) connectivity
    node_value : NDArray
        Nx1 ndarray, real/complex valued

    Returns
    -------
    NDArray
        el_grad, Mx2 ndarray, real/complex valued
    """
    M = np.size(el2no, 0)
    _, grad_phi_x, grad_phi_y = pdetrg(no2xy, el2no)
    trinode_values = np.reshape(node_value[el2no.ravel()], [M, -1])
    grad_el_x = np.sum(grad_phi_x * trinode_values, axis=1)
    grad_el_y = np.sum(grad_phi_y * trinode_values, axis=1)
    return grad_el_x, grad_el_y


def pdeprtni(no2xy, el2no, el_value):
    """
    given the value of element, interpolate the nodes
    prtni is the reverse-interp :)
    this code was tested and equivalent to MATLAB pdeprtni
    except for the shape of 'no2xy' and 'el2no'

    Parameters
    ----------
    no2xy : NDArray
        Nx2 ndarray, (x,y) locations for points
    el2no : NDArray
        Mx3 ndarray, elements (triangles) connectivity
    el_value : NDArray
        Mx1 value, real/complex valued on elements

    Returns
    -------
    NDArray
        no_value, piecewise reverse-interpolate of el_value on nodes
    """
    N = np.size(no2xy, 0)
    M = np.size(el2no, 0)
    # build n->e matrix, this could be accelerated using sparse matrix
    n2e = np.zeros([N, M], dtype='int')
    for i in range(M):
        n2e[el2no[i, :], i] = 1
    # equivalent to,
    # pick a node, find all the triangles sharing this node,
    # and average all the values on these triangles
    node_value = np.dot(n2e, el_value) / np.sum(n2e, axis=1)
    return node_value
