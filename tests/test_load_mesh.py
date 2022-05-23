from pyeit.mesh.external import load_mesh
import numpy as np
from pathlib import Path

parent_dir = str(Path(__file__).parent)


def test_load_triangle():
    correct_mesh = {
        "element": [[0, 1, 2]],
        "node": [[17.32050896, 30.0, 0.0], [34.64101791, 0.0, 0.0], [0.0, 0.0, 0.0]],
        "perm": [1.0],
    }

    triangle_mesh = load_mesh(parent_dir + "/test_data/triangle.STL")

    np.testing.assert_almost_equal(correct_mesh["element"], triangle_mesh.element)
    np.testing.assert_almost_equal(correct_mesh["node"], triangle_mesh.node)
    np.testing.assert_almost_equal(correct_mesh["perm"], triangle_mesh.perm)
