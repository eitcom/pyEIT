# test for eit
import unittest
import numpy as np
import pyeit.eit
from pyeit.mesh.wrapper import PyEITMesh


def _mesh_obj():
    """build a simple mesh, which is used in FMMU.CEM"""
    node = np.array([[0, 0], [1, 1], [1, 2], [0, 1]])
    element = np.array([[0, 1, 3], [1, 2, 3]])
    perm = np.array([3.0, 1.0])  # assemble should not use perm.dtype
    el_pos = np.array([1, 2])

    return PyEITMesh(node= node, element= element, perm= perm, el_pos= el_pos, ref_el= 3)


class TestFem(unittest.TestCase):
    def test_bp(self):
        """test back projection"""
        mesh = _mesh_obj()
        ex_mat = np.array([[0, 1], [1, 0]])
        protocol = {"ex_mat": ex_mat, "step": 1, "parser": "meas_current"}
        solver = pyeit.eit.BP(mesh=mesh, protocol=protocol)
        solver.setup()

        assert solver.B.shape[0] > 0


if __name__ == "__main__":
    unittest.main()
