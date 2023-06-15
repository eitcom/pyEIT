import numpy as np
from pyeit.quality.merit import (
    calc_circle,
    calc_amplitude,
    calc_position_error,
    calc_fractional_amplitude_set,
    calc_resolution,
    calc_ringing,
    calc_shape_deformation,
    classify_target_and_background,
    get_image_bounds,
    lambda_max,
)
from imageio.v2 import imread
import scipy.ndimage as ndi
from pathlib import Path
from matplotlib import (
    pyplot as plt,
    patches as mpatches,
    axes as mpl_axes,
)
from numpy import NaN

parent_dir = str(Path(__file__).parent)

test_image = np.array(
    [
        [NaN, NaN, NaN, NaN, NaN, NaN, NaN],
        [NaN, 0, 0, 0, 0, 0, NaN],
        [NaN, 0, 1, 1, 1, 0, NaN],
        [NaN, 0, 1, 1, 1, 0, NaN],
        [NaN, 0, 1, 1, 1, 0, NaN],
        [NaN, 0, 1, 1, 1, 0, NaN],
        [NaN, 0, 2, 2, 2, 0, NaN],
        [NaN, 0, 0, 0, 0, 0, NaN],
        [NaN, NaN, NaN, NaN, NaN, NaN, NaN],
    ]
)

test_image_2 = np.array(
    [
        [NaN, NaN, NaN, NaN, NaN, NaN, NaN],
        [NaN, 0, 0, 0, 0, 0, NaN],
        [NaN, 0, 1, 1, 1, 0, NaN],
        [NaN, 0, 1, 1, 1, 0, NaN],
        [NaN, 0, 1, 1, 1, 0, NaN],
        [NaN, 0, 1, 1, 1, 0, NaN],
        [NaN, 0, 0, 0, 0, 0, NaN],
        [NaN, 0, 0, 0, 0, 0, NaN],
        [NaN, NaN, NaN, NaN, NaN, NaN, NaN],
    ]
)

test_image_3 = np.array(
    [
        [NaN, NaN, NaN, NaN, NaN, NaN, NaN],
        [NaN, 0, 0, 0, 0, 0, NaN],
        [NaN, 0, -1, -1, -1, 0, NaN],
        [NaN, 0, 0.1, 0.1, 0.1, 0, NaN],
        [NaN, 0, 0.1, 0.1, 0.1, 0, NaN],
        [NaN, 0, 0.75, 0.75, 0.75, 0, NaN],
        [NaN, 0, 2, 2, 2, 0, NaN],
        [NaN, 0, 0, 0, 0, 0, NaN],
        [NaN, NaN, NaN, NaN, NaN, NaN, NaN],
    ]
)


def test_calc_circle():
    square = imread(parent_dir + "/data/square_image.bmp", pilmode="RGB")

    fractional_image = np.full(np.shape(square)[0:2], NaN)
    fractional_image[
        np.where(
            (square[:, :, 0] == 255)
            & (square[:, :, 1] == 255)
            & (square[:, :, 2] == 255)
        )[0:2]
    ] = 0

    fractional_image[
        np.where(
            (square[:, :, 0] == 0) & (square[:, :, 1] == 0) & (square[:, :, 2] == 0)
        )[0:2]
    ] = 1

    circle = calc_circle(fractional_image)

    circle_center = ndi.center_of_mass(circle)
    fractional_image_center = ndi.center_of_mass(np.nan_to_num(fractional_image, nan=0))

    circle_area = np.sum(circle == 1)
    fractional_image_area = np.sum(fractional_image == 1)

    # # Display test images
    # fig, axs = plt.subplots(2, 2)
    # axs[0, 0].imshow(square)
    # axs[0, 0].set_title("Test Image")
    #
    # axs[0, 1].remove()
    #
    # img = axs[1, 0].imshow(fractional_image)
    # axs[1, 0].set_title("Fractional Image")
    # colors = [img.cmap(img.norm(value)) for value in [NaN, 0, 1]]
    # patches = [
    #     mpatches.Patch(color=colors[0], label="NAN"),
    #     mpatches.Patch(color=colors[1], label="0"),
    #     mpatches.Patch(color=colors[2], label="1")
    # ]
    # axs[1, 0].legend(handles=patches, bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.0)
    #
    # axs[1, 1].imshow(circle)
    # axs[1, 1].set_title("Equivalent Area Circle")
    # axs[1, 1].legend(handles=patches, bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.0)
    #
    # fig.tight_layout()
    # plt.show()

    assert circle_center == fractional_image_center
    assert np.isclose(circle_area, fractional_image_area, rtol=0.01)


def test_calc_amplitude():
    correct_amplitude = 18 / 35

    # fig, ax = plt.subplots()
    # ax.imshow(test_image)
    # plt.show()

    amplitude = calc_amplitude(test_image)
    assert amplitude == correct_amplitude


def test_calc_position_error():
    test_image_p1 = np.array(
        [
            [NaN, NaN, NaN, NaN, NaN, NaN, NaN],
            [NaN, 0, 0, 0, 0, 0, NaN],
            [NaN, 0, 1, 1, 1, 0, NaN],
            [NaN, 0, 0, 0, 0, 0, NaN],
            [NaN, 0, 0, 0, 0, 0, NaN],
            [NaN, 0, 0, 0, 0, 0, NaN],
            [NaN, 0, 0, 0, 0, 0, NaN],
            [NaN, 0, 0, 0, 0, 0, NaN],
            [NaN, NaN, NaN, NaN, NaN, NaN, NaN],
        ]
    )

    test_image_p2 = np.array(
        [
            [NaN, NaN, NaN, NaN, NaN, NaN, NaN],
            [NaN, 0, 0, 0, 0, 0, NaN],
            [NaN, 0, 0, 0, 0, 0, NaN],
            [NaN, 0, 1, 1, 1, 0, NaN],
            [NaN, 0, 0, 0, 0, 0, NaN],
            [NaN, 0, 0, 0, 0, 0, NaN],
            [NaN, 0, 0, 0, 0, 0, NaN],
            [NaN, 0, 0, 0, 0, 0, NaN],
            [NaN, NaN, NaN, NaN, NaN, NaN, NaN],
        ]
    )

    test_image_p2_flipped = np.array(
        [
            [NaN, NaN, NaN, NaN, NaN, NaN, NaN],
            [NaN, 0, 0, 0, 0, 0, NaN],
            [NaN, 0, 0, 0, 0, 0, NaN],
            [NaN, 0, 0, 0, 0, 0, NaN],
            [NaN, 0, 0, 0, 0, 0, NaN],
            [NaN, 0, 1, 1, 1, 0, NaN],
            [NaN, 0, 0, 0, 0, 0, NaN],
            [NaN, 0, 0, 0, 0, 0, NaN],
            [NaN, NaN, NaN, NaN, NaN, NaN, NaN],
        ]
    )

    position_error = calc_position_error(test_image_p1, test_image_p2, method="GREIT")
    position_error_reversed = calc_position_error(
        test_image_p2, test_image_p1, method="GREIT"
    )

    position_error_greit_p2_flipped = calc_position_error(
        test_image_p2, test_image_p2_flipped, method="GREIT"
    )
    position_error_euclidean_p2_flipped = calc_position_error(
        test_image_p2, test_image_p2_flipped, method="Euclidean"
    )

    correct_position_error = 1 / 7
    correct_position_error_reversed = -1 / 7
    correct_position_error_greit_p2_flipped = 0 / 7
    correct_position_error_euclidean_p2_flipped = 2 / 7

    # fig, axs = plt.subplots(1, 2)
    # axs[0].imshow(test_image_p1)
    # axs[0].set_title("p1")
    # axs[1].imshow(test_image_p2)
    # axs[1].set_title("p2")
    # plt.show()

    assert position_error == correct_position_error
    assert position_error_reversed == correct_position_error_reversed
    assert position_error_greit_p2_flipped == correct_position_error_greit_p2_flipped
    assert (
        position_error_euclidean_p2_flipped
        == correct_position_error_euclidean_p2_flipped
    )


def test_calc_fractional_amplitude_set():
    fractional_amplitude_set = calc_fractional_amplitude_set(
        test_image_3, fraction=0.25, conductive_target=True, method="GREIT"
    )
    fractional_amplitude_set_non_conductive = calc_fractional_amplitude_set(
        test_image_3, fraction=0.25, conductive_target=False, method="GREIT"
    )
    fractional_amplitude_set_range = calc_fractional_amplitude_set(
        test_image_3, fraction=0.75, conductive_target=True, method="Range"
    )
    fractional_amplitude_set_range_non_conductive = calc_fractional_amplitude_set(
        test_image_3, fraction=0.75, conductive_target=False, method="Range"
    )

    correct_fractional_amplitude_set = np.array(
        [
            [NaN, NaN, NaN, NaN, NaN, NaN, NaN],
            [NaN, 0, 0, 0, 0, 0, NaN],
            [NaN, 0, 0, 0, 0, 0, NaN],
            [NaN, 0, 0, 0, 0, 0, NaN],
            [NaN, 0, 0, 0, 0, 0, NaN],
            [NaN, 0, 1, 1, 1, 0, NaN],
            [NaN, 0, 1, 1, 1, 0, NaN],
            [NaN, 0, 0, 0, 0, 0, NaN],
            [NaN, NaN, NaN, NaN, NaN, NaN, NaN],
        ]
    )

    correct_fractional_amplitude_set_range = np.array(
        [
            [NaN, NaN, NaN, NaN, NaN, NaN, NaN],
            [NaN, 0, 0, 0, 0, 0, NaN],
            [NaN, 0, 0, 0, 0, 0, NaN],
            [NaN, 0, 0, 0, 0, 0, NaN],
            [NaN, 0, 0, 0, 0, 0, NaN],
            [NaN, 0, 0, 0, 0, 0, NaN],
            [NaN, 0, 1, 1, 1, 0, NaN],
            [NaN, 0, 0, 0, 0, 0, NaN],
            [NaN, NaN, NaN, NaN, NaN, NaN, NaN],
        ]
    )

    correct_fractional_amplitude_set_negative_target = np.array(
        [
            [NaN, NaN, NaN, NaN, NaN, NaN, NaN],
            [NaN, 0, 0, 0, 0, 0, NaN],
            [NaN, 0, 1, 1, 1, 0, NaN],
            [NaN, 0, 0, 0, 0, 0, NaN],
            [NaN, 0, 0, 0, 0, 0, NaN],
            [NaN, 0, 0, 0, 0, 0, NaN],
            [NaN, 0, 0, 0, 0, 0, NaN],
            [NaN, 0, 0, 0, 0, 0, NaN],
            [NaN, NaN, NaN, NaN, NaN, NaN, NaN],
        ]
    )

    np.testing.assert_array_equal(
        fractional_amplitude_set, correct_fractional_amplitude_set
    )
    np.testing.assert_array_equal(
        fractional_amplitude_set_non_conductive,
        correct_fractional_amplitude_set_negative_target,
    )
    np.testing.assert_array_equal(
        fractional_amplitude_set_range, correct_fractional_amplitude_set_range
    )
    np.testing.assert_array_equal(
        fractional_amplitude_set_range_non_conductive,
        correct_fractional_amplitude_set_negative_target,
    )


def test_calc_resolution():
    resolution = calc_resolution(test_image)
    correct_resolution = 0.6546536707079771

    np.testing.assert_almost_equal(resolution, correct_resolution)


def test_calc_shape_deformation():
    shape_deformation, extras = calc_shape_deformation(
        reconstruction_image=test_image,
        target_image=test_image_2,
        circular=False,
        conductive_target=True,
        return_extras=True,
    )
    shape_deformation_circular, extras2 = calc_shape_deformation(
        reconstruction_image=test_image,
        circular=True,
        conductive_target=True,
        return_extras=True,
    )

    correct_shape_deformation = 0.2
    correct_shape_deformation_circular = 0.26666666666666666

    assert shape_deformation == correct_shape_deformation
    np.testing.assert_almost_equal(
        shape_deformation_circular, correct_shape_deformation_circular
    )


def test_classify_target_and_background():
    target, _ = classify_target_and_background(test_image_2, conductive_target=True)
    target_negative, _ = classify_target_and_background(
        test_image_3, conductive_target=False
    )
    correct_target = np.array(
        [
            [False, False, False, False, False, False, False],
            [False, False, False, False, False, False, False],
            [False, False, True, True, True, False, False],
            [False, False, True, True, True, False, False],
            [False, False, True, True, True, False, False],
            [False, False, True, True, True, False, False],
            [False, False, False, False, False, False, False],
            [False, False, False, False, False, False, False],
            [False, False, False, False, False, False, False],
        ]
    )

    correct_target_non_conductive = np.array(
        [
            [False, False, False, False, False, False, False],
            [False, False, False, False, False, False, False],
            [False, False, True, True, True, False, False],
            [False, False, False, False, False, False, False],
            [False, False, False, False, False, False, False],
            [False, False, False, False, False, False, False],
            [False, False, False, False, False, False, False],
            [False, False, False, False, False, False, False],
            [False, False, False, False, False, False, False],
        ]
    )

    np.testing.assert_array_equal(target, correct_target)
    np.testing.assert_array_equal(target_negative, correct_target_non_conductive)


def test_calc_ringing():
    test_image_ringing = np.array(
        [
            [NaN, NaN, NaN, NaN, NaN, NaN, NaN],
            [NaN, 0, 0, 0, 0, 0, NaN],
            [NaN, 0, 1, 1, 1, 0, NaN],
            [NaN, 0, 1, 1, 1, 0, NaN],
            [NaN, 0, 1, 1, 1, 0, NaN],
            [NaN, 0, 1, 1, 1, 0, NaN],
            [NaN, 0, -1, -1, -1, 0, NaN],
            [NaN, 0, 0, 0, 0, 0, NaN],
            [NaN, NaN, NaN, NaN, NaN, NaN, NaN],
        ]
    )

    test_image_target_non_conductive = np.array(
        [
            [NaN, NaN, NaN, NaN, NaN, NaN, NaN],
            [NaN, 3, 3, 3, 3, 3, NaN],
            [NaN, 3, 3, 3, 3, 3, NaN],
            [NaN, 3, 3, 3, 3, 3, NaN],
            [NaN, 3, 3, 3, 3, 3, NaN],
            [NaN, 3, 3, 3, 3, 3, NaN],
            [NaN, 3, 2, 2, 2, 3, NaN],
            [NaN, 3, 3, 3, 3, 3, NaN],
            [NaN, NaN, NaN, NaN, NaN, NaN, NaN],
        ]
    )

    test_image_recon_non_conductive = np.array(
        [
            [NaN, NaN, NaN, NaN, NaN, NaN, NaN],
            [NaN, -0.4, -0.4, -0.4, -0.4, -0.4, NaN],
            [NaN, -0.4, -0.4, -0.4, -0.4, -0.4, NaN],
            [NaN, -0.4, -0.4, -0.4, -0.4, -0.4, NaN],
            [NaN, -0.4, -0.4, -0.4, -0.4, -0.4, NaN],
            [NaN, -0.4, 1, 1, 1, -0.4, NaN],
            [NaN, -0.4, -2, -2, -2, -0.4, NaN],
            [NaN, -0.4, -0.4, -0.4, -0.4, -0.4, NaN],
            [NaN, NaN, NaN, NaN, NaN, NaN, NaN],
        ]
    )

    ringing1 = calc_ringing(test_image, test_image_2, circular=False)
    ringing2 = calc_ringing(test_image_ringing, test_image_2, circular=False)

    ringing_non_conductive = calc_ringing(
        test_image_recon_non_conductive,
        test_image_target_non_conductive,
        conductive_target=False,
    )

    correct_ringing1 = 0
    correct_ringing2 = 0.25
    correct_ringing_non_conductive = 0.5

    assert ringing1 == correct_ringing1
    assert ringing2 == correct_ringing2
    assert ringing_non_conductive == correct_ringing_non_conductive


def test_get_image_bounds():
    rowmin, rowmax, colmin, colmax = get_image_bounds(test_image)

    image = test_image[rowmin:rowmax, colmin:colmax]

    assert ~np.any(np.isnan(image))
    assert (rowmin, rowmax, colmin, colmax) == (1, 8, 1, 6)


def test_lambda_max():
    arr = np.array([1, 2, 9, -11, 3])

    lm = lambda_max(arr, key=abs)

    correct_lm = -11

    assert lm == correct_lm
