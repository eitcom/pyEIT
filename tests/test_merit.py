import numpy as np
from pyeit.quality.merit import calc_circle
from imageio.v2 import imread
import scipy.ndimage as ndi
from pathlib import Path

parent_dir = str(Path(__file__).parent)


def test_calc_circle():
    square = imread(parent_dir + "/data/square_image.bmp", pilmode="RGB")

    fractional_image = np.full(np.shape(square)[0:2], np.nan)
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

    assert circle_center == fractional_image_center
    assert np.isclose(circle_area, fractional_image_area, rtol=0.01)
