import numpy as np
import math
import scipy.ndimage as ndi

"""
merit.py contains figures of merit as defined in "GREIT: a unified approach to 2D linear EIT
reconstruction of lung images" by Andy Adler et al 2009 Physiol. Meas. 30 S35
doi:10.1088/0967-3334/30/6/S03
Also cf implementation in EIDORS
"""


def calc_greit_figures_of_merit(
    target_image, reconstruction_image, circular=True, target_value=None, out=None
):
    """
    Calculate 5 GRIET figures of merit. Units are pixels of the input images. Target image and reconstruction image are
    assumed to have the same pixel resolution

    With circular=True (default), calculations are as defined in "GREIT: a unified approach to 2D linear EIT
    reconstruction of lung images" by Andy Adler et al.

    With circular=False, Shape deformation and Ringing are adapted to work with non circular targets. These adaptions are
    as defined in "Tracking boundary movement and exterior shape modelling in lung EIT imaging" by A Biguri et al.

    Parameters
    ----------
    target_image: np.Array(width, height)
        Render of target mesh with conductivities as pixel values
    reconstruction_image: np.Array(width, height)
        Render of reconstructed mesh with conductivities as pixel values
    circular: Bool
        Assume circular targets
    target_value:
        Value of targets in target image. If set to none, targets are classified automatically
    out
        output dict where intermediate results are placed

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
    if out is not None:
        out["shape_deformation"] = {}
        out["ringing"] = {}

    # Amplitude
    amplitude = calc_amplitude(reconstruction_image)

    # Position error
    position_error = calc_position_error(target_image, reconstruction_image)

    # Resolution
    resolution = calc_resolution(reconstruction_image)

    # Shape Deformation
    shape_deformation = calc_shape_deformation(
        reconstruction_image,
        target_image=target_image if not circular else None,
        circular=circular,
        target_value=target_value,
        out=out["shape_deformation"] if out is not None else None,
    )

    # Ringing
    ringing = calc_ringing(
        reconstruction_image,
        target_image=target_image if not circular else None,
        circular=circular,
        target_value=target_value,
        out=out["ringing"] if out is not None else None,
    )

    return amplitude, position_error, resolution, shape_deformation, ringing


def calc_fractional_amplitude_set(image, fraction=0.25):
    """
    The fractional amplitude set is equal to one where the image amplitude is greater than or equal to the given fraction
    times the maximum amplitude, and zero otherwise.

    Parameters
    ----------
    image: np.Array(width,height)
    fraction: float

    Returns
    ---------
    image_set: np.Array(width,height)
    """

    abs_max = np.max(np.abs(np.nan_to_num(image, nan=0)))
    max_ind = np.unravel_index(np.argmax(np.abs(image) == abs_max), np.shape(image))

    image_set = np.full(np.shape(image), np.nan)

    if image[max_ind] < 0:
        negative_image = image * -1

        with np.errstate(invalid="ignore"):
            image_set[negative_image < abs_max * fraction] = 0
            image_set[negative_image >= abs_max * fraction] = 1

    else:
        with np.errstate(invalid="ignore"):
            image_set[image < abs_max * fraction] = 0
            image_set[image >= abs_max * fraction] = 1

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


def calc_position_error(target_image, reconstruction_image):
    """
    Calculate the Euclidean distance between the center of gravity of the target image and the center of gravity of the
    reconstruction image

    Parameters
    ----------
    target_image: np.Array(width,height)
        reference target image
    reconstruction_image: np.Array(width,height)
        reconstructed image

    Returns
    -------
    position_error: float

    """

    fractional_image = calc_fractional_amplitude_set(reconstruction_image)

    fractional_image_nonan = np.nan_to_num(fractional_image, nan=0)
    recon_center = ndi.center_of_mass(fractional_image_nonan)

    target_nonan = np.nan_to_num(target_image, nan=0)
    target_center = ndi.center_of_mass(target_nonan)

    # This definition allows + and - PE, but can also give zero in unexpected places
    position_error = math.sqrt(
        target_center[0] ** 2 + target_center[1] ** 2
    ) - math.sqrt(recon_center[0] ** 2 + recon_center[1] ** 2)

    # # This definition gives the absolute PE, but can't be negative
    # position_error = math.sqrt((target_center[0]-recon_center[0])**2 + (target_center[1]-recon_center[1])**2)

    return position_error


def calc_resolution(reconstruction_image):
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
    fractional_image = calc_fractional_amplitude_set(reconstruction_image)

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
    reconstruction_image, target_image=None, circular=True, target_value=None, out=None
):
    """
    Calculate shape deformation: Sum of pixels in reconstructed target that are outside the reference target. Unit: pixels

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
    out
        output dict where intermediate results are placed

    Returns
    -------

    """
    if not circular and target_image is None:
        raise ValueError("target_image must not be None if circular is False")

    fractional_image = calc_fractional_amplitude_set(reconstruction_image)
    reconstructed_area = np.count_nonzero(fractional_image == 1)

    if circular:
        circle = calc_circle(fractional_image)
        shape_deformation_target = circle

    else:
        target, _ = classify_target_and_background(target_image, target_value)
        shape_deformation_target = target

    outside = np.logical_and(fractional_image == 1, shape_deformation_target == 0)
    area_outside = np.sum(outside)
    if out is not None:
        out["shape_deformation_target"] = shape_deformation_target
        out["outside_positions"] = outside

    shape_deformation = area_outside / reconstructed_area

    return shape_deformation


def calc_ringing(
    reconstruction_image, target_image=None, circular=True, target_value=None, out=None
):
    """
    Calculate ringing: Sum of pixels of opposite value to reconstructed target in the reconstruction image

    With circular=True (default), the calculation is as defined in "GREIT: a unified approach to 2D linear EIT
    reconstruction of lung images" by Andy Adler et al.

    With circular=False, the calculation is an adaption of the ringing measure to account for non circular targets.
    This adaption of the GREIT shape deformation measure was made by A Biguri in "Tracking boundary movement and exterior
    shape modelling in lung EIT imaging".


    Parameters
    ----------
    target_image: np.Array(width,height)
        reference target image
    reconstruction_image: np.Array(width,height)
        reconstructed image
    circular: bool
    target_value:
        Value of targets in target image. If set to none, targets are classified automatically
    out
        output dict where intermediate results are placed
    """

    if not circular and target_image is None:
        raise ValueError("target_image must not be None if circular is False")

    if circular:
        fractional_image = calc_fractional_amplitude_set(reconstruction_image)
        circle = calc_circle(fractional_image)
        ringing_target = circle
    else:
        target, _ = classify_target_and_background(target_image, target_value)
        ringing_target = target

    abs_max = np.max(np.abs(np.nan_to_num(reconstruction_image, nan=0)))
    max_ind = np.unravel_index(
        np.argmax(np.abs(reconstruction_image) == abs_max),
        np.shape(reconstruction_image),
    )

    sum_inside = np.sum(reconstruction_image[np.where(ringing_target == 1)])

    if reconstruction_image[max_ind] >= 0:
        with np.errstate(invalid="ignore"):
            opposite_outside_positions = np.logical_and(
                reconstruction_image < 0, np.logical_not(ringing_target)
            )
        ringing = np.sum(reconstruction_image[opposite_outside_positions]) / sum_inside
    else:
        with np.errstate(invalid="ignore"):
            opposite_outside_positions = np.logical_and(
                reconstruction_image >= 0, np.logical_not(ringing_target)
            )
        ringing = np.sum(reconstruction_image[opposite_outside_positions]) / sum_inside

    if out is not None:
        out["ringing_target"] = ringing_target
        out["opposite_outside_positions"] = opposite_outside_positions

    return ringing


def classify_target_and_background(target_image, target_value=None):
    """
    We have to decide what is target and what is background in the target image.
    There should only be two numbers. So the one with the least area should be the target. Otherwise the user can input
    a target value

    Parameters
    ----------
    target_image: np.Array(width,height)
        reference target image
    target_value:
        Value of targets in target image. If set to none, targets are classified automatically

    Returns
    -------
    target: np.array(width,height)
        array with true at pixels representing target
    background: np.array(width,height)
        array with true at pixels representing background

    """
    target_rtol = 0.001

    unique, counts = np.unique(
        target_image[np.logical_not(np.isnan(target_image))], return_counts=True
    )

    if target_value is None:
        target_value = unique[np.argmin(counts)]

    background = np.logical_and(
        np.logical_not(np.isnan(target_image)),
        np.logical_not(np.isclose(target_image, target_value, rtol=target_rtol)),
    )

    target = np.logical_and(
        np.logical_not(np.isnan(target_image)),
        np.isclose(target_image, target_value, rtol=target_rtol),
    )

    return target, background
