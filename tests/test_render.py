from pyeit.eit.render import (
    pt_in_triang,
    map_image,
    model_inverse_uv,
    get_bounds,
    scale_uv_list,
)
from pyeit.mesh.external import load_mesh
import numpy as np
from pathlib import Path
from numpy.testing import assert_almost_equal
import matplotlib.pyplot as plt
from pyeit.visual.plot import create_mesh_plot

parent_dir = str(Path(__file__).parent)


def test_pt_in_triang():
    p0, p1, p2 = ([0, 0], [1, 0], [1, 1])
    p_out = [0.25, 0.75]
    p_in = [0.75, 0.25]

    should_be_false = pt_in_triang(p_out, p0, p1, p2)
    should_be_true = pt_in_triang(p_in, p0, p1, p2)

    assert not should_be_false
    assert should_be_true


def test_pt_in_triang_neg():
    p0, p1, p2 = ([-1, -1], [1, -1], [0, 1])
    p_out_1 = [-0.5, -1.5]
    p_out_2 = [0.5, -1.5]
    p_in = [-0.5, -0.75]

    out_1 = pt_in_triang(p_out_1, p0, p1, p2)
    out_2 = pt_in_triang(p_out_2, p0, p1, p2)
    in_1 = pt_in_triang(p_in, p0, p1, p2)

    assert not out_1
    assert not out_2
    assert in_1


def test_get_bounds():
    array = [[1, 1], [2, 3], [4, 5], [5, 7], [3, 2]]
    bounds = (1, 5, 1, 7)

    result = get_bounds(array)

    assert result == bounds


def test_get_bounds_neg():
    array = [[-5, -4], [5, -5], [3, 3], [-3, 4]]
    bounds = (-5, 5, -5, 4)

    result = get_bounds(array)

    assert result == bounds


def test_model_inverse_uv():
    mesh = load_mesh(parent_dir + "/data/circle.STL")
    image = model_inverse_uv(
        {"node": mesh.node[:, :2], "element": mesh.element},
        (100, 100),
        preserve_aspect_ratio=False,
    )
    image = image.T[
        :, ::-1
    ]  # Flip back because this test was created before we corrected the orientation

    circle_image = np.load(parent_dir + "/data/circle_image.npy")

    assert np.all(image == circle_image)


def test_model_inverse_uv_neg():
    mesh = load_mesh(parent_dir + "/data/circle.STL")
    mesh.node -= 5
    image = model_inverse_uv(
        {"node": mesh.node[:, :2], "element": mesh.element},
        (100, 100),
        preserve_aspect_ratio=False,
    )
    image = image.T[
        :, ::-1
    ]  # Flip back because this test was created before we corrected the orientation

    circle_image = np.load(parent_dir + "/data/circle_image.npy")

    assert np.all(image == circle_image)


def test_render():
    mesh = load_mesh(parent_dir + "/data/L_shape.STL")
    image = model_inverse_uv(
        {"node": mesh.node[:, :2], "element": mesh.element}, (100, 100)
    )
    mapped = map_image(image, mesh.perm)

    correct_image = np.load(parent_dir + "/data/L_image.npy")
    correct_mapped = np.load(parent_dir + "/data/L_mapped.npy")

    # fig, axs = plt.subplots(1, 3)
    # create_mesh_plot(axs[0], mesh)
    # axs[1].imshow(image)
    # axs[2].imshow(mapped)
    # fig.tight_layout()
    # plt.show()

    np.testing.assert_array_equal(image, correct_image)
    np.testing.assert_array_equal(mapped, correct_mapped)


def test_map_image():
    circle_image = np.load(parent_dir + "/data/circle_image.npy")

    values = np.array(
        [
            5.0,
            5.0,
            10.0,
            5.0,
            5.0,
            5.0,
            5.0,
            5.0,
            5.0,
            5.0,
            5.0,
            5.0,
            5.0,
            5.0,
            5.0,
            5.0,
            5.0,
            5.0,
            10,
            5.0,
            5.0,
            10.0,
            10.0,
            10.0,
            10.0,
            10.0,
            10.0,
            10.0,
            10.0,
            10.0,
            10.0,
            10.0,
            10.0,
            10.0,
            10.0,
            10.0,
            10.0,
            10.0,
            10.0,
        ]
    )

    image = map_image(circle_image, values)

    mapped_image = np.load(parent_dir + "/data/circle_image_mapped.npy")

    equal = True
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            if image[i][j] == mapped_image[i][j] or (
                np.isnan(image[i][j]) and np.isnan((mapped_image[i][j]))
            ):
                pass
            else:
                equal = False
                break

    assert equal


def test_scale_uv_list():
    uv_list = np.array([[1, 1], [2, 1], [2, 4], [1, 4]])

    scaled_0 = scale_uv_list(
        uv_list,
        resolution=[10, 10],
        preserve_aspect_ratio=True,
        bounds=np.array([[0, 0], [2, 4]]),
    )
    scaled_1 = scale_uv_list(
        uv_list,
        resolution=[10, 10],
        preserve_aspect_ratio=False,
        bounds=np.array([[0, 0], [2, 4]]),
    )
    scaled_2 = scale_uv_list(
        uv_list, resolution=[10, 10], preserve_aspect_ratio=True, bounds=None
    )

    correct_scaled_0 = np.array([[2.5, 2.5], [5.0, 2.5], [5.0, 10.0], [2.5, 10.0]])
    correct_scaled_1 = np.array([[5.0, 2.5], [10.0, 2.5], [10.0, 10.0], [5.0, 10.0]])
    correct_scaled_2 = np.array(
        [[0.0, 0.0], [3.33333333, 0.0], [3.33333333, 10.0], [0.0, 10.0]]
    )

    assert np.array_equal(scaled_0, correct_scaled_0)
    assert np.array_equal(scaled_1, correct_scaled_1)
    assert_almost_equal(scaled_2, correct_scaled_2)


def test_scale_uv_list_neg():
    uv_list = np.array([[-1, -1], [0, -1], [0, 2], [-1, 2]])

    scaled_0 = scale_uv_list(
        uv_list,
        resolution=[10, 10],
        preserve_aspect_ratio=True,
        bounds=np.array([[-2, -2], [0, 2]]),
    )
    scaled_1 = scale_uv_list(
        uv_list,
        resolution=[10, 10],
        preserve_aspect_ratio=False,
        bounds=np.array([[-2, -2], [0, 2]]),
    )
    scaled_2 = scale_uv_list(
        uv_list, resolution=[10, 10], preserve_aspect_ratio=True, bounds=None
    )

    correct_scaled_0 = np.array([[2.5, 2.5], [5.0, 2.5], [5.0, 10.0], [2.5, 10.0]])
    correct_scaled_1 = np.array([[5.0, 2.5], [10.0, 2.5], [10.0, 10.0], [5.0, 10.0]])
    correct_scaled_2 = np.array(
        [[0.0, 0.0], [3.33333333, 0.0], [3.33333333, 10.0], [0.0, 10.0]]
    )

    assert np.array_equal(scaled_0, correct_scaled_0)
    assert np.array_equal(scaled_1, correct_scaled_1)
    assert_almost_equal(scaled_2, correct_scaled_2)
