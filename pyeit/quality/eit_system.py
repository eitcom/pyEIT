import numpy as np
import math
import scipy.ndimage as ndi
from typing import Tuple, Callable, Union, Dict
from numpy.typing import ArrayLike, NDArray

"""
eit_system.py contains calculation for the performance of EIT hardware systems based on measured data. These performance
measures are defined in "Evaluation of EIT system performance" by Andy Adler et. al. doi:10.1088/0967-3334/32/7/S09
"""


def calc_signal_to_noise_ratio(measurements: NDArray) -> NDArray:
    """
    Signal to noise ratio calculates the mean measurement divided by the standard deviation of measurements for each
    channel. (For this calculation, a channel is defined as a unique combination of stimulation and measurement
    electrodes)

    Parameters
    ----------
    measurements
        NDArray containing a number of repeated EIT measurements

    Returns
    -------
    snr
        NDArray of signal to noise ratio for each individual channel in the EIT measurements

    """
    # Flatten measurements in case they are in a 2d array
    measurements = measurements.reshape((2, -1))

    stdev = np.std(measurements, axis=0)
    average = np.average(measurements, axis=0)

    snr = average / stdev

    return snr


def calc_accuracy(measurements: NDArray, reference_measurements: NDArray, method="Ratio") -> NDArray:
    """
    Accuracy measures the closeness of measured quantities to a "true" reference value. In this case simulated EIT
    measurements are used as the reference

    Parameters
    ----------
    measurements
    reference_measurements
    method
        Options: EIDORS, Ratio

    Returns
    -------
    accuracy
    """
    # Flatten measurements in case they are in a 2d array
    measurements = measurements.reshape((2, -1))
    reference_measurements = reference_measurements.reshape(-1)

    average = np.average(measurements, axis=0)

    if method == "EIDORS":
        # Normalize measurement sets individually (This is like calibrating by scaling and offsetting, so we only see
        # the difference between channels. But it doesn't necessarily give you true information about any particular
        # channel. So maybe it would be better as a range?)
        average = (average - np.min(average)) / (np.max(average) - np.min(average))
        reference_measurements = (reference_measurements - np.min(reference_measurements)) / (
                    np.max(reference_measurements) - np.min(reference_measurements))
        accuracy = 1 - np.abs(average - reference_measurements)

    elif method == "Ratio":
        # This is as described in Gagnon 2010 - "A Resistive Mesh Phantom for Assissing the Performance of EIT Systems"
        accuracy = 1 - np.abs((average - reference_measurements) / reference_measurements)
    else:
        raise ValueError("Invalid method selected for accuracy")

    return accuracy


def calc_drift():
    """
    Drift is a measure of the change in average value of measurements over time. This function uses the Allan
    variance to calculate drift

    Returns
    -------

    """
    pass


def calc_reciprocity_accuracy():
    pass


def calc_detectability():
    pass


def calc_distinguishability():
    pass
