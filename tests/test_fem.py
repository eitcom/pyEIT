# test for fem.py
import unittest

import numpy as np
import pyeit.eit.fem
from pyeit.eit.protocol import PyEITProtocol, build_meas_pattern_std
from pyeit.mesh import PyEITMesh


def _assemble(ke, tri, perm, n):
    # assemble global stiffness matrix
    k = np.zeros((n, n), dtype=perm.dtype)
    for ei in range(ke.shape[0]):
        k_local = ke[ei]
        pe = perm[ei]
        no = tri[ei, :]
        ij = np.ix_(no, no)
        k[ij] += k_local * pe

    k[0, :] = 0.0
    k[:, 0] = 0.0
    k[0, 0] = 1.0

    return k


def _meas_pattern(ex_line, n_el, dist, parser):
    """a simple voltage meter (meas_pattern)"""
    meas_current = parser == "meas_current"
    rel_electrode = parser in ["fmmu", "rotate_meas"]
    i0 = ex_line[0] if rel_electrode else 0
    m = np.arange(i0, i0 + n_el) % n_el
    n = np.arange(i0 + dist, i0 + dist + n_el) % n_el
    v = np.array([[ni, mi] for ni, mi in zip(n, m)])
    keep = [~np.any(np.isin(vi, ex_line), axis=0) for vi in v]
    return v if meas_current else v[keep]


def _mesh_obj():
    """build a simple, determinant mesh model/dataset"""
    node = np.array([[0.13, 0.15], [0.2, 0.2], [0.1, 0.1], [0.18, 0.12]])
    element = np.array([[0, 2, 3], [0, 3, 1]])
    # assemble uses perm.dtype, perm MUST not be np.int (result rounding error in K)
    perm = np.array([3.0, 1.0])
    el_pos = np.array([1, 2])
    # new mesh structure or dataset
    return PyEITMesh(node=node, element=element, perm=perm, el_pos=el_pos, ref_node=3)


def _mesh_obj_large():
    """build a large, random mesh model/dataset"""
    n_tri, n_pts = 400, 1000
    node = np.random.randn(n_pts, 2)
    element = np.array([np.random.permutation(n_pts)[:3] for _ in range(n_tri)])
    perm = np.random.randn(n_tri)
    np.random.seed(0)
    el_pos = np.random.permutation(n_pts)[:16]
    return PyEITMesh(node=node, element=element, perm=perm, el_pos=el_pos, ref_node=0)


def _protocol_obj(ex_mat, n_el, step_meas, parser_meas):
    meas_mat, keep_ba = build_meas_pattern_std(ex_mat, n_el, step_meas, parser_meas)
    return PyEITProtocol(ex_mat, meas_mat, keep_ba)


class TestFem(unittest.TestCase):
    def test_ke_triangle(self):
        """test ke calculation using triangle (2D)"""
        pts = np.array([[0, 1], [0, 0], [1, 0]])
        tri = np.array([[0, 1, 2]])
        k_truth = np.array([[1, -1, 0], [-1, 2, -1], [0, -1, 1]])
        area = 0.5
        ke = pyeit.eit.fem.calculate_ke(pts, tri)

        self.assertTrue(ke.shape == (1, 3, 3))
        self.assertTrue(np.allclose(ke[0], k_truth * area))

    def test_ke_tetrahedron(self):
        """test ke calculation using tetrahedron (3D)"""
        pts = np.array([[0, 0, 0], [0, 0, 1], [0, 1, 0], [1, 0, 0]])
        tri = np.array([[0, 1, 2, 3]])
        k_truth = np.array(
            [[3, -1, -1, -1], [-1, 1, 0, 0], [-1, 0, 1, 0], [-1, 0, 0, 1]]
        )
        volumn = 1.0 / 6.0
        ke = pyeit.eit.fem.calculate_ke(pts, tri)

        self.assertTrue(ke.shape == (1, 4, 4))
        self.assertTrue(np.allclose(ke[0], k_truth * volumn))

    def test_assemble(self):
        """test assembling coefficients matrix, {se, perm} -> K"""
        np.random.seed(0)
        n, ne = 10, 42
        pts = np.arange(n)
        tri = np.array([pts[np.random.permutation(n)[:3]] for _ in range(ne)])
        perm = np.random.randn(ne)
        se = np.random.randn(ne, 3, 3)
        k_truth = _assemble(se, tri, perm, n)
        k = pyeit.eit.fem.assemble(se, tri, perm, n).toarray()

        self.assertTrue(np.allclose(k, k_truth))

    def test_meas_pattern(self):
        """test measurement pattern/voltage meter"""
        # @libuenyan shoul be in test_eit.py or test_protocol.py
        n_el = 16
        np.random.seed(42)
        mesh = _mesh_obj_large()
        for parser in ["meas_current", "fmmu", "rotate_meas"]:
            ex_lines = [np.random.permutation(n_el)[:2] for _ in range(10)]
            for ex_line in ex_lines:
                ex_mat = np.array([ex_line])
                # build protocol dict/dataset
                protocol = _protocol_obj(ex_mat, n_el, 1, parser)
                fwd = pyeit.eit.fem.EITForward(mesh, protocol)
                diff_truth = _meas_pattern(ex_line, n_el, 1, parser)
                diff = fwd.protocol.meas_mat[:, :-1]

                assert np.allclose(diff, diff_truth)

    def test_subtract_row(self):
        """calculate f[diff_op[0]] - f[diff_op[1]]"""
        n_exe = 10
        n_el = 16
        v = np.random.randn(n_el)
        diff_pairs = np.array([np.random.permutation(n_el)[:2] for _ in range(n_exe)])
        vd_truth = np.array([v[d[0]] - v[d[1]] for d in diff_pairs])
        vd = pyeit.eit.fem.subtract_row(v, diff_pairs)

        self.assertTrue(vd_truth.size == vd.size)
        self.assertTrue(np.allclose(vd.ravel(), vd_truth.ravel()))

    def test_subtract_row_vectorized(self):
        """calculate f[exc_id, diff_op[0]] - f[exc_id, diff_op[1]]"""
        n_el = 16
        n_meas = 16
        n_exc = 3
        n_meas_tot = n_exc * n_meas
        v = np.full((n_exc, n_el), np.random.randn(n_el))
        # build measurement pattern, [m, n, exc_id] per row
        diff_pairs = []
        for i in range(n_exc):
            for _ in range(n_meas):
                diff_pairs.append(np.hstack([np.random.permutation(n_el)[:2], i]))
        meas_pattern = np.vstack(diff_pairs)
        print(meas_pattern)
        # calculate ground truth
        vd_truth = np.zeros((n_meas_tot,))
        for i in range(n_meas_tot):
            v_exc = v[meas_pattern[i, 2]]
            vd_truth[i] = v_exc[meas_pattern[i, 0]] - v_exc[meas_pattern[i, 1]]
        vd = pyeit.eit.fem.subtract_row_vectorized(v, meas_pattern)

        self.assertTrue(vd_truth.size == vd.size)
        self.assertTrue(np.allclose(vd.ravel(), vd_truth.ravel()))

    def test_k(self):
        """test Forward.kg using a simple, determinant mesh structure"""
        k_truth = np.array(
            [
                [3.7391, -0.1521, -1.5, 0.0],
                [-0.1521, 0.3695, 0.0, 0.0],
                [-1.5, 0.0, 1.5, 0.0],
                [0.0, 0.0, 0.0, 1.0],
            ]
        )
        mesh = _mesh_obj()
        fwd = pyeit.eit.fem.Forward(mesh)
        k = fwd.kg.toarray()  # sparse COO to dense

        self.assertTrue(np.allclose(k, k_truth, rtol=0.01))

    def test_solve(self):
        """test solve using a simple mesh structure"""
        mesh = _mesh_obj()
        f_truth = np.array([-0.27027027, 2.59459459, -0.93693694, 0.0])
        fwd = pyeit.eit.fem.Forward(mesh)
        ex_line = np.array([0, 1])
        f = fwd.solve(ex_line)

        self.assertTrue(np.allclose(f, f_truth))
        # test without passing any argument
        f = fwd.solve()
        self.assertTrue(isinstance(f, np.ndarray))

    def test_solve_eit(self):
        """test solve_eit using a simple mesh structure"""
        mesh = _mesh_obj()
        el_pos = mesh.el_pos
        ex_mat = np.array([[0, 1], [1, 0]])
        protocol = _protocol_obj(ex_mat, mesh.n_el, 1, "meas_current")
        fwd = pyeit.eit.fem.EITForward(mesh, protocol)

        # include voltage differences on driving electrodes
        v = fwd.solve_eit()
        f_truth = np.array([-0.27027027, 2.59459459, -0.93693694, 0.0])
        vdiff_truth = f_truth[el_pos[1]] - f_truth[el_pos[0]]
        v_truth = vdiff_truth * np.array([1, -1, -1, 1])
        self.assertTrue(np.allclose(v, v_truth))

    def test_compute_jac(self):
        """test solve using a simple mesh structure"""
        mesh = _mesh_obj()
        ex_mat = np.array([[0, 1]])
        protocol = _protocol_obj(ex_mat, mesh.n_el, 1, "meas_current")
        fwd = pyeit.eit.fem.EITForward(mesh, protocol)

        # testing solve
        jac_truth = np.array([[-0.25874523, -2.75529584], [0.25874523, 2.75529584]])
        jac, _ = fwd.compute_jac()
        self.assertTrue(np.allclose(jac, jac_truth))

    def test_compute_b_matrix(self):
        """test compute_jac using a simple mesh structure"""
        mesh = _mesh_obj()
        ex_mat = np.array([[0, 1]])
        protocol = _protocol_obj(ex_mat, mesh.n_el, 1, "meas_current")

        # smear: (f_min < f) & (f <= f_max)
        b_truth = np.array([[1, 1, 0, 1], [1, 1, 0, 1]])
        # fix ref to be exactly the one in mesh
        fwd = pyeit.eit.fem.EITForward(mesh, protocol)
        b = fwd.compute_b_matrix()
        self.assertTrue(np.allclose(b, b_truth))


if __name__ == "__main__":
    unittest.main()
