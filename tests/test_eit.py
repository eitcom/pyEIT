# test for eit
import numpy as np
import pyeit.eit


def _mesh_obj():
    """build a simple mesh, which is used in FMMU.CEM"""
    node = np.array([[0, 0], [1, 1], [1, 2], [0, 1]])
    element = np.array([[0, 1, 3], [1, 2, 3]])
    perm = np.array([3.0, 1.0])  # assemble should not use perm.dtype
    mesh = {"node": node, "element": element, "perm": perm, "ref": 3}
    el_pos = np.array([1, 2])

    return mesh, el_pos


def test_bp():
    """test back projection"""
    mesh, el_pos = _mesh_obj()
    ex_mat = np.array([[0, 1], [1, 0]])
    solver = pyeit.eit.BP(mesh, el_pos, ex_mat, parser="meas_current")
    solver.setup()

    assert solver.B.shape[0] > 0


if __name__ == "__main__":
    pass
