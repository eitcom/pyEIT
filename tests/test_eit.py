# test for eit
import unittest
import numpy as np
import pyeit.eit
from pyeit.eit.protocol import PyEITProtocol, build_meas_pattern_std
from pyeit.mesh import PyEITMesh


def _mesh_obj():
    """build a simple mesh, which is used in FMMU.CEM"""
    node = np.array([[0, 0], [1, 1], [1, 2], [0, 1]])
    element = np.array([[0, 1, 3], [1, 2, 3]])
    perm = np.array([3.0, 1.0])  # assemble should not use perm.dtype
    el_pos = np.array([1, 2])

    return PyEITMesh(node=node, element=element, perm=perm, el_pos=el_pos, ref_node=3)


def _protocol_obj(ex_mat, n_el, step_meas, parser_meas):
    meas_mat = build_meas_pattern_std(ex_mat, n_el, step_meas, parser_meas)
    return PyEITProtocol(ex_mat, meas_mat)


class TestFem(unittest.TestCase):
    def test_bp(self):
        """test back projection"""
        mesh = _mesh_obj()
        ex_mat = np.array([[0, 1], [1, 0]])
        protocol = _protocol_obj(ex_mat, mesh.n_el, 1, "meas_current")
        solver = pyeit.eit.BP(mesh=mesh, protocol=protocol)
        solver.setup()

        assert solver.B.shape[0] > 0


if __name__ == "__main__":
    unittest.main()
