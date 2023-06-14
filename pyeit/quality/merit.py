import numpy as np
import math
import scipy.ndimage as ndi
from typing import Tuple, Callable, Union
from numpy.typing import ArrayLike

"""
merit.py contains figures of merit as defined in "GREIT: a unified approach to 2D linear EIT
reconstruction of lung images" by Andy Adler et al 2009 Physiol. Meas. 30 S35
doi:10.1088/0967-3334/30/6/S03
Also cf implementation in EIDORS
"""


def calc_greit_figures_of_merit(
    target_image, reconstruction_image, conductive_target=True, return_extras=False
):
    """
    Calculate 5 GRIET figures of merit. Units are pixels of the input images. Target image and reconstruction image are
    rendered rectangular pixel arrays, and are assumed to have the same pixel resolution

    With circular=True (default), calculations are as defined in "GREIT: a unified approach to 2D linear EIT
    reconstruction of lung images" by Andy Adler et al.

    With circular=False, Shape deformation and Ringing are adapted to work with non circular targets. These adaptions are
    as defined in "Tracking boundary movement and exterior shape modelling in lung EIT imaging" by A Biguri et al.

    Parameters
    ----------
    target_image: np.Array(width, height)
        Render of target mesh with conductivities as pixel values. If target_value is not supplied, the target is
        classified as the pixels of value encountered least in the image (i.e., the region of lowest area).
    reconstruction_image: np.Array(width, height)
        Render of reconstructed mesh with conductivities as pixel values
    circular: Bool
        Assume circular targets
    conductive_target:
        If true, value of pixels in target image is higher than that of the surrounding pixels
    return_extras
        return extra images and calculations

    Returns
    -------
    Amplitude:
        Sum of image amplitudes. Units: pixels*value
    Position Error:
        Distance between center of gravity of reconstructed image and center of gravity of target image. Units: pixels length
    Resolution:
        Size of reconstructed targets as a fraction of the medium. Non dimensional
    Shape Deformation
        Sum of pixels in reconstructed target that are outside the reference target. Unit: pixels
    Ringing
        Sum of pixels of opposite value to reconstructed target in the reconstruction image

    """
    out = {"shape_deformation":{}, "ringing":{}}


    # Amplitude
    amplitude = calc_amplitude(reconstruction_image)

    # Position error
    position_error = calc_position_error(target_image, reconstruction_image, method="GREIT", conductive_target=conductive_target)

    # Resolution
    resolution = calc_resolution(reconstruction_image, conductive_target=conductive_target)

    # Shape Deformation
    shape_deformation, shape_out = calc_shape_deformation(
        reconstruction_image,
        target_image=None,
        circular=True,
        conductive_target=conductive_target,
        return_extras=True
    )
    out["shape_deformation"] = shape_out

    # Ringing
    ringing, ringing_out = calc_ringing(
        reconstruction_image,
        target_image=target_image,
        circular=False,
        conductive_target=conductive_target,
        return_extras=True
    )
    out["ringing"] = ringing_out

    if return_extras:
        return (amplitude, position_error, resolution, shape_deformation, ringing), out

    return amplitude, position_error, resolution, shape_deformation, ringing


def calc_fractional_amplitude_set(image, fraction=0.25, conductive_target: bool = True, method: str = "GREIT"):
    """
    The fractional amplitude set is equal to one where the image amplitude is greater than or equal to the given fraction
    times the maximum amplitude, and zero otherwise.

    If conductive_target is set to true, the maximum amplitude is the largest positive value in the image. Otherwise
    it is the largest negative value in the image.

    If method is set to "GREIT", the fraction is calculated based on distance between the maximum amplitude and zero.
    If it is set to "Range", the fraction is calculated based on the proportion of the entire range of values in the
    image.

    Parameters
    ----------
    image: np.Array(width,height)
    fraction: float
    conductive_target
    method:
        Options: GREIT, Range

    Returns
    ---------
    image_set: np.Array(width,height)
    """

    if not conductive_target:
        image = image*-1

    max_amplitude = np.nanmax(image)

    if method == "GREIT":
        min_amplitude = 0

    elif method == "Range":
        # Minimum amplitude is the value furthest from the max
        offset_image = image - max_amplitude
        abs_max = lambda_max(np.nan_to_num(offset_image, nan=0), axis=None, key=np.abs)
        min_amplitude = abs_max + max_amplitude
    else:
        raise ValueError("Invalid method specified for fractional amplitude set")

    dist_from_min = image - min_amplitude
    range = max_amplitude - min_amplitude
    threshold = np.abs(range * fraction)

    image_set = np.full(np.shape(image), np.nan)
    with np.errstate(invalid="ignore"):
        image_set[dist_from_min < threshold] = 0
        image_set[dist_from_min >= threshold] = 1

    return image_set


def calc_amplitude(recon_image):
    """
    Image amplitude is the sum of the values of the reconstructed image.

    In Adler's GREIT: a unified approach to 2D linear EIT reconstruction of lung images, this is further divided by a
    quantity involving the target, but in Adler's implementation in EIDORS, just the reconstruciton image amplitude is
    calculated
    Parameters
    ----------
    recon_image: np.Array(width, height)
        list of reconstruction image amplitudes

    Returns
    -------
    amplitude: float
        Image Amplitude

    """

    recon_image = np.nan_to_num(recon_image, nan=0)

    amplitude = np.sum(recon_image)

    return amplitude


def calc_position_error(target_image, reconstruction_image, method="GREIT", conductive_target=True):
    """
    Calculate position error using one of two methods:

    GREIT:
        Difference between the two distances from the center of mass to the center of the image.
        One of the target, and one of the reconstruction. (Target - Recon) So positive values mean the reconstruction
        center is closer to the image center than the target is.

    Euclidean:
        Calculate the Euclidean distance between the center of gravity of the target image and the center of gravity of the
        reconstruction image

    Parameters
    ----------
    target_image: np.Array(width,height)
        reference target image
    reconstruction_image: np.Array(width,height)
        reconstructed image
    method
        Options: GREIT, Euclidean
    conductive_target

    Returns
    -------
    position_error: float

    """

    fractional_image = calc_fractional_amplitude_set(reconstruction_image, conductive_target=conductive_target)

    fractional_image_nonan = np.nan_to_num(fractional_image, nan=0)
    recon_center = ndi.center_of_mass(fractional_image_nonan)

    target_classified, _ = classify_target_and_background(target_image, conductive_target)
    target_center = ndi.center_of_mass(target_classified)

    if method == "GREIT":
        # This definition allows + and - PE, but can also give zero in unexpected places
        homogeneous_image = calc_fractional_amplitude_set(target_image, 0)
        homogeneous_image_nonan = np.nan_to_num(homogeneous_image, nan=0)
        medium_center = ndi.center_of_mass(homogeneous_image_nonan)

        r_target = math.sqrt(
            (target_center[0] - medium_center[0]) ** 2
            + (target_center[1] - medium_center[1]) ** 2
        )

        r_recon = math.sqrt(
            (recon_center[0] - medium_center[0]) ** 2
            + (recon_center[1] - medium_center[1]) ** 2
        )

        position_error = r_target - r_recon

    elif method == "Euclidean":
        # This definition gives the absolute PE, but can't be negative
        position_error = math.sqrt((target_center[0]-recon_center[0])**2 + (target_center[1]-recon_center[1])**2)
    else:
        raise ValueError("Invalid method specified for position error")

    return position_error


def calc_resolution(reconstruction_image, conductive_target:bool=True):
    """
    Resolution measures the size of reconstructed targets as a fraction of the medium. Per Adler: the square root is used
    so that RES measures radius ratios rather than area ratios.

    *Note* this is intended for use with point targets

    Parameters
    ----------
    reconstruction_image: np.array(width, height)

    Returns
    -------
    resolution: float

    """
    fractional_image = calc_fractional_amplitude_set(reconstruction_image, conductive_target=conductive_target)

    target_area = np.count_nonzero(fractional_image == 1)

    medium_area = np.count_nonzero(~np.isnan(fractional_image))

    resolution = math.sqrt(target_area / medium_area)
    return resolution


def calc_circle(fractional_image):
    """
    Calculate a circle with equal area to the fractional image

    Parameters
    ----------
    fractional_image: np.Array(width, height)

    Returns
    -------
    circle: np.Array(width, height)

    """
    target_area = np.count_nonzero(fractional_image == 1)
    cx, cy = ndi.center_of_mass(np.nan_to_num(fractional_image, nan=0))

    radius = math.sqrt(target_area / math.pi)

    circle = np.zeros(fractional_image.shape)
    for i in range(circle.shape[0]):
        for j in range(circle.shape[1]):
            distance = math.sqrt((cx - i) ** 2 + (cy - j) ** 2)
            if distance <= radius:
                circle[i, j] = 1

    return circle


def calc_shape_deformation(
    reconstruction_image, target_image=None, circular=True, conductive_target=True, return_extras=False
):
    """
    Calculate shape deformation: Proportion of pixels in thresholded reconstruction that are outside the target to total
    pixels in the thresholded reconstruction

    With circular=True (default), the calculation is as defined in "GREIT: a unified approach to 2D linear EIT
    reconstruction of lung images" by Andy Adler et al.

    With circular=False, the calculation is an adaption of the shape deformation measure to account for non circular targets.
    This adaption of the GREIT shape deformation measure was made by A Biguri in "Tracking boundary movement and exterior
    shape modelling in lung EIT imaging". A drawback of this adaption is that it will give low values if the reconstructed
    image is smaller than the target even if it's not the same shape. This is exacerbated by the fractional thresholding,
    which uses and arbitrary threshold value of .25

    Parameters
    ----------
    target_image: np.Array(width,height)
        reference target image
    reconstruction_image: np.Array(width,height)
        reconstructed image
    circular: bool
    target_value:
        Value of targets in target image. If set to none, targets are classified automatically
    return_extras
        if true, a dict with extra calculated images is returned

    Returns
    -------

    """
    if not circular and target_image is None:
        raise ValueError("target_image must not be None if circular is False")

    fractional_image = calc_fractional_amplitude_set(reconstruction_image, conductive_target=conductive_target)
    reconstructed_area = np.count_nonzero(fractional_image == 1)

    if circular:
        circle = calc_circle(fractional_image)
        shape_deformation_target = circle

    else:
        target, _ = classify_target_and_background(target_image, conductive_target)
        shape_deformation_target = target

    outside = np.logical_and(fractional_image == 1, shape_deformation_target == 0)
    area_outside = np.sum(outside)

    shape_deformation = area_outside / reconstructed_area

    if return_extras:
        out = {"shape_deformation_target": shape_deformation_target,
               "outside_positions": outside}

        return shape_deformation, out

    return shape_deformation


def calc_ringing(
    reconstruction_image, target_image=None, circular=False, conductive_target=True, return_extras=False
):
    """
    Calculate ringing: Sum of pixels of opposite value to reconstructed target in the reconstruction image

    With circular=True, the calculation is as defined in "GREIT: a unified approach to 2D linear EIT
    reconstruction of lung images" by Andy Adler et al.

    With circular=False (default), the calculation is an adaption of the ringing measure to account for non circular targets.
    This adaption of the GREIT shape deformation measure was made by A Biguri in "Tracking boundary movement and exterior
    shape modelling in lung EIT imaging".

    Eidors 3.11 does not use the equivalent circle for the ringing calculation, it simply uses the thresholded image,
    so circular is set to False by default in this function.

    Parameters
    ----------
    target_image: np.Array(width,height)
        reference target image
    reconstruction_image: np.Array(width,height)
        reconstructed image
    circular: bool
    conductive_target:
        conductive target. Must be positive! and greater than background
    return_extras
        return extras
    """

    if not circular and target_image is None:
        raise ValueError("target_image must not be None if circular is False")

    if circular:
        fractional_image = calc_fractional_amplitude_set(reconstruction_image, conductive_target=conductive_target)
        circle = calc_circle(fractional_image)
        ringing_target = circle
    else:
        target, _ = classify_target_and_background(target_image, conductive_target)
        ringing_target = target

    if not conductive_target:
        reconstruction_image = reconstruction_image * -1

    sum_inside = np.sum(reconstruction_image[np.where(ringing_target == 1)])

    with np.errstate(invalid="ignore"):
        opposite_outside_positions = np.logical_and(
            reconstruction_image < 0, np.logical_not(ringing_target)
        )
    ringing = -1*np.sum(reconstruction_image[opposite_outside_positions]) / sum_inside

    if return_extras:
        out = {"ringing_target": ringing_target, "opposite_outside_positions": opposite_outside_positions}
        return ringing, out

    return ringing


def classify_target_and_background(target_image, conductive_target:bool=True):
    """
    We have to decide what is target and what is background in the target image.
    There should only be two numbers. So the one with the least area should be the target. Otherwise the user can input
    a target value

    Parameters
    ----------
    target_image: np.Array(width,height)
        reference target image
    conductive_target:

    Returns
    -------
    target: np.array(width,height)
        array with true at pixels representing target
    background: np.array(width,height)
        array with true at pixels representing background

    """
    target_rtol = 0.001

    if conductive_target:
        target_value = np.nanmax(target_image)
    else:
        target_value = np.nanmin(target_image)

    background = np.logical_and(
        np.logical_not(np.isnan(target_image)),
        np.logical_not(np.isclose(target_image, target_value, rtol=target_rtol)),
    )

    target = np.logical_and(
        np.logical_not(np.isnan(target_image)),
        np.isclose(target_image, target_value, rtol=target_rtol),
    )

    return target, background


def lambda_max(arr: ArrayLike, axis: int = None, key: Callable = None, keepdims: bool = False) -> Union[
    float, ArrayLike]:
    """

    Applies the callable "key" to the input array "arr", then finds the index of the maximum value of this transformed
    array. Finally, returns the value in the *original* input array at that index. This is equivalent to the "key"
    parameter for the built-in max function, but can be applied to multi-dimensional arrays.

    A useful application of this is to select elements of an array where the absolute value is maximum. (In which case,
    key should be np.abs)

    See the following stackoverflow thread for discussion:
        https://stackoverflow.com/questions/61703879/in-numpy-how-to-select-elements-based-on-the-maximum-of-their-absolute-values

    Parameters
    ----------
    arr: Input array
    axis: Axis to take along
    key: Callable to transform arr before finding the max index
    keepdims: Keep the dimensions of the input array

    Returns
    ---------
    max: Number or ArrayLike

    """
    arr = np.asarray(arr)

    if not callable(key):
        return np.amax(arr, axis)

    idxs = np.argmax(key(arr), axis)
    if axis is not None:
        idxs = np.expand_dims(idxs, axis)
        result = np.take_along_axis(arr, idxs, axis)
        if not keepdims:
            result = np.squeeze(result, axis=axis)
        return result
    else:
        return arr.flatten()[idxs]