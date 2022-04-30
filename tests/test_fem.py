# test for fem.py
import unittest
import numpy as np
import pyeit.eit.fem


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

def _voltage_meter(ex_line, n_el, dist, parser):
    """a simple voltage meter"""
    meas_current = parser == "meas_current"
    rel_electrode = parser in ["fmmu", "rotate_meas"]
    i0 = ex_line[0] if rel_electrode else 0
    m = np.arange(i0, i0 + n_el) % n_el
    n = np.arange(i0 + dist, i0 + dist + n_el) % n_el
    v = np.array([[ni, mi] for ni, mi in zip(n, m)])
    keep = [~np.any(np.isin(vi, ex_line), axis=0) for vi in v]
    return v if meas_current else v[keep]

def _mesh_obj():
    """build a simple mesh, which is used in FMMU.CEM"""
    node = np.array([[0.13, 0.15], [0.2, 0.2], [0.1, 0.1], [0.18, 0.12]])
    element = np.array([[0, 2, 3], [0, 3, 1]])
    # assemble uses perm.dtype, perm MUST not be np.int
    perm = np.array([3.0, 1.0])
    mesh = {"node": node, "element": element, "perm": perm, "ref": 3}
    el_pos = np.array([1, 2])

    return mesh, el_pos

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
        k_truth = np.array([[3, -1, -1, -1], [-1, 1, 0, 0], [-1, 0, 1, 0], [-1, 0, 0, 1]])
        volumn = 1 / 6.0
        ke = pyeit.eit.fem.calculate_ke(pts, tri)

        self.assertTrue(ke.shape == (1, 4, 4))
        self.assertTrue(np.allclose(ke[0], k_truth * volumn))


    def test_assemble(self):
        """test assembling coefficients matrix, K"""
        np.random.seed(0)
        n, ne = 10, 42
        nodes = np.arange(n)
        ke = np.random.randn(ne, 3, 3)
        tri = np.array([nodes[np.random.permutation(n)[:3]] for _ in range(ne)])
        perm = np.random.randn(ne)
        k_truth = _assemble(ke, tri, perm, n)
        k = pyeit.eit.fem.assemble(ke, tri, perm, n)

        self.assertTrue(np.allclose(k, k_truth))

    def test_voltage_meter(self):
        """test voltage meter"""
        n_el = 16
        np.random.seed(42)
        for parser in ["meas_current", "fmmu", "rotate_meas"]:
            ex_lines = [np.random.permutation(n_el)[:2] for _ in range(10)]
            for ex_line in ex_lines:
                ex_mat = np.array([ex_line])  # two dim
                diff_truth = _voltage_meter(ex_line, n_el, 1, parser)
                diff = pyeit.eit.fem.voltage_meter(ex_mat, n_el, 1, parser)
                
                self.assertTrue(np.allclose(diff, diff_truth))

    def test_subtract_row(self):
        """
        subtract the last dimension
        v is [n_exe, n_el, 1] where n_exe is the number of execution (stimulations)
        and n_el is the number of voltage sensing electrode,
        meas_pattern is [n_exe, n_meas, 2], where n_meas is the effective number
        of voltage differences.

        for simplification, we let n_meas=1
        """
        n_exe = 10
        n_el = 16
        v = np.random.randn(n_exe, n_el, 1)
        diff_pairs = np.array([np.random.permutation(n_el)[:2] for _ in range(n_exe)])
        vd_truth = np.array([v[i, d[0]] - v[i, d[1]] for i, d in enumerate(diff_pairs)])
        meas_pattern = diff_pairs.reshape(n_exe, 1, 2)
        vd = pyeit.eit.fem.subtract_row(v, meas_pattern)

        self.assertTrue(vd_truth.size == vd.size)
        self.assertTrue(np.allclose(vd.ravel(), vd_truth.ravel()))

    def test_k(self):
        """test K using a simple mesh structure"""
        k_truth = np.array(
            [
                [3.7391, -0.1521, -1.5, 0.0],
                [-0.1521, 0.3695, 0.0, 0.0],
                [-1.5, 0.0, 1.5, 0.0],
                [0.0, 0.0, 0.0, 1.0],
            ]
        )
        mesh, _ = _mesh_obj()
        n_pts = mesh["node"].shape[0]
        ke = pyeit.eit.fem.calculate_ke(mesh["node"], mesh["element"])
        # fix ref to be exactly the one in mesh
        k = pyeit.eit.fem.assemble(
            ke, mesh["element"], mesh["perm"], n_pts, ref=mesh["ref"]
        )

        self.assertTrue(np.allclose(k, k_truth, rtol=0.01))

    def test_solve(self):
        """test solve using a simple mesh structure"""
        mesh, el_pos = _mesh_obj()
        f_truth = np.array([-0.27027027, 2.59459459, -0.93693694, 0.0])
        fwd = pyeit.eit.fem.Forward(mesh, el_pos)
        # fix ref to be exactly the one in mesh
        ex_mat = np.array([[0, 1]])
        fwd.set_ref_el(mesh["ref"])
        f= fwd.solve(ex_mat, perm=mesh["perm"])

        self.assertTrue(np.allclose(f, f_truth))

    def test_solve_eit(self):
        """test solve using a simple mesh structure"""
        mesh, el_pos = _mesh_obj()
        f_truth = np.array([-0.27027027, 2.59459459, -0.93693694, 0.0])

        fwd = pyeit.eit.fem.Forward(mesh, el_pos)
        # fix ref to be exactly the one in mesh
        fwd.set_ref_el(mesh["ref"])
        ex_mat = np.array([[0, 1], [1, 0]])
        # include voltage differences on driving electrodes
        fwd = fwd.solve_eit(ex_mat, parser="meas_current")
        vdiff_truth = f_truth[el_pos[1]] - f_truth[el_pos[0]]
        v_truth = vdiff_truth * np.array([1, -1, -1, 1])

        self.assertTrue(np.allclose(fwd.v, v_truth))

    def test_compute_jac(self):
        """test solve using a simple mesh structure"""
        #TODO @ liubenyuan please checkt this test
        # compute_jac return jac with the "subtract_row part" here you wanted to test only the jac_i get from old solve method
        # the jac_i_truth correspond to the jac_truth!
        mesh, el_pos = _mesh_obj()
        jac_i_truth = np.array([[-0.02556611, 2.67129291], [-0.28431134, -0.08400292]])
        jac_truth = np.array([[-0.25874523, -2.75529584], [ 0.25874523,  2.75529584]])

        # testing solve
        ex_mat = np.array([[0, 1]])
        fwd = pyeit.eit.fem.Forward(mesh, el_pos)
        # fix ref to be exactly the one in mesh
        fwd.set_ref_el(mesh["ref"])
        jac = fwd.compute_jac(ex_mat, perm=mesh["perm"], parser="meas_current" )
        
        self.assertTrue(np.allclose(jac, jac_truth))


if __name__ == "__main__":
    unittest.main()