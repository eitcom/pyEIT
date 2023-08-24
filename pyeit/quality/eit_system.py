import numpy as np
from numpy.typing import NDArray
import allantools
from pyeit.eit.protocol import PyEITProtocol
from pyeit.quality.merit import calc_fractional_amplitude_set

"""
eit_system.py contains calculation for the performance of EIT hardware systems based on measured data. These performance
measures are defined in "Evaluation of EIT system performance" by Mamatjan Yasin et. al. (With Andy Adler)
doi:10.1088/0967-3334/32/7/S09
"""


def calc_signal_to_noise_ratio(measurements: NDArray, method="ratio") -> NDArray:
    """
    Signal to noise ratio calculates the mean measurement divided by the standard deviation of measurements for each
    channel. (For this calculation, a channel is defined as a unique combination of stimulation and measurement
    electrodes)

    The measurements array must contain at least two sets of measurements

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
    measurements = measurements.reshape((measurements.shape[0], -1))

    stdev = np.std(measurements, axis=0)
    average = np.average(measurements, axis=0)

    snr = average / stdev

    if method == "ratio":
        return snr
    elif method == "db":
        return (
            np.log10(np.abs(snr)) * 20
        )  # Convert to decibels as a root-power quantity
    else:
        raise ValueError("Invalid method specified (must be ratio or db)")


def calc_accuracy(
    measurements: NDArray, reference_measurements: NDArray, method="Ratio"
) -> NDArray:
    """
    Accuracy measures the closeness of measured quantities to a "true" reference value. In this case simulated EIT
    measurements are used as the reference

    The measurements array must contain at least two sets of measurements

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
    measurements = measurements.reshape((measurements.shape[0], -1))
    reference_measurements = reference_measurements.reshape(-1)

    average = np.average(measurements, axis=0)

    if method == "EIDORS":
        # Normalize measurement sets individually (This is like calibrating by scaling and offsetting, so we only see
        # the difference between channels. But it doesn't necessarily give you true information about any particular
        # channel. So maybe it would be better as a range?)
        average = (average - np.min(average)) / (np.max(average) - np.min(average))
        reference_measurements = (
            reference_measurements - np.min(reference_measurements)
        ) / (np.max(reference_measurements) - np.min(reference_measurements))
        accuracy = 1 - np.abs(average - reference_measurements)

    elif method == "Ratio":
        # This is as described in Gagnon 2010 - "A Resistive Mesh Phantom for Assissing the Performance of EIT Systems"
        accuracy = 1 - np.abs(
            (average - reference_measurements) / reference_measurements
        )
    else:
        raise ValueError("Invalid method selected for accuracy")

    return accuracy


def calc_drift(
    measurements: NDArray, sampling_rate: float = 1, sample_period=None, method="Allan"
):
    """
    Drift is a measure of the change in average value of measurements over time. There are two methods for calculating
    this. The EIDORS method uses the Allan variance, and the Delta method calculates the difference between two
    samples taken from the start and end of the total list of measurements.

    Returns
    -------
    method: "Allan"
        t2: the set of sampling periods used
        adevs: the list of allan deviations calculated for each channel

    method: "Delta"
        drifts: drifts calculated for each channel

    """
    # Flatten measurements in case they are in a 2d array
    measurements = measurements.reshape((measurements.shape[0], -1))

    if method == "Allan":
        # Iterate through each channel
        adevs = []
        for channel_measurements in measurements.T:
            (t2, ad, ade, adn) = allantools.oadev(
                channel_measurements, rate=sampling_rate, data_type="freq", taus="all"
            )
            adevs.append(ad)

        adevs = np.array(adevs)
        return t2, adevs

    elif method == "Delta":
        drifts = []
        for channel_measurements in measurements.T:
            start = np.average(channel_measurements[0 : sampling_rate * sample_period])
            end = np.average(
                np.flip(channel_measurements)[0 : sampling_rate * sample_period]
            )
            drifts.append(end - start)
        drifts = np.array(drifts)
        return drifts


def calc_reciprocity_accuracy(
    measurements: NDArray, protocol: PyEITProtocol
) -> NDArray:
    """
    Tests the closeness of reciprocal measurements to each other. This is in accordance with the principle in
    "Reciprocity Applied to Volume Conductors and the ECG" (Plonsey 1963). The interpretation of this in
    "Evaluation of EIT system performance" is as follows: "EIT measurements from a stimulation–measurement pair should
    not change if the current stimulation and voltage measurement electrodes are swapped."

    The measurements array must contain at least two sets of measurements

    Parameters
    ----------
    measurements
        Array of EIT measurements
    protocol
        PyEITProtocol object listing the excitation and measurement electrodes for each row in the EIT measurement
        array

    Returns
    -------
    reciprocal_accuracies
        Array of accuracy calculations for reciprocal pairs

    """
    combined_mat = np.hstack(
        (protocol.meas_mat[:, :2], protocol.ex_mat[protocol.meas_mat[:, 2]])
    )
    reciprocals = find_reciprocals(combined_mat)

    # Flatten measurements in case they are in a 2d array
    measurements = measurements.reshape((measurements.shape[0], -1))
    average = np.average(measurements, axis=0)

    reciprocal_accuracies = []
    for reciprocal in reciprocals:
        v = average[reciprocal[0]]
        vr = average[reciprocal[1]]

        reciprocal_accuracy = 1 - np.abs(v - vr) / np.abs(v)
        reciprocal_accuracies.append(reciprocal_accuracy)

    return np.array(reciprocal_accuracies)


def find_reciprocals(combined_mat: NDArray) -> NDArray:
    """
    Find reciprocal rows in an Nx4 array. Reciprocals are defined as a pair of rows where the two pairs of elements
    are swapped, i.e., element 0 and 1 in row a are the same as element 2 and 3 in row b, and element 2 and 3 in row a
    are the same as element 0 and 1 in row b. Order does not matter.

    If any row has no reciprocal, a ValueError will be raised.

    Parameters
    ----------
    combined_mat
        Nx4 array where all rows have a reciprocal

    Returns
    -------
    reciprocals
        Nx2 array with indices of reciprocal rows. (Duplicates are removed)

    """
    reciprocals = set()
    for i, row in enumerate(combined_mat):
        reciprocal = find_reciprocal(row, combined_mat)
        reciprocals.add(
            frozenset((i, reciprocal))
        )  # Append as inner set to outer set to filter out duplicates

    reciprocals = np.array(
        [list(r) for r in list(reciprocals)]
    )  # Convert back to array

    return reciprocals


def find_reciprocal(row: NDArray, combined_mat: NDArray) -> int:
    """
    Auxilliary function to find_reciprocals. Finds the reciprocal of a single row. Raises ValueError if one is not found.

    Parameters
    ----------
    row
        Row to find reciprocal of
    combined_mat
        Array to search

    Returns
    -------
    i
        Index of reciprocal row

    """
    # Reciprocal pairs can be in any order (because the signal is AC). So we compare rows as sets
    reciprocal_row = np.array(({*row[2:]}, {*row[0:2]}))

    for i, compare_row in enumerate(combined_mat):
        if np.array_equal(
            reciprocal_row, np.array(({*compare_row[0:2]}, {*compare_row[2:]}))
        ):
            return i

    raise ValueError("No reciprocal found")


def calc_detectability(
    image,
    conductive_target: bool = True,
    fraction=0.25,
    fraction_method="GREIT",
    method="ratio",
):
    """
    See Adler et. al. 2010 "Distinguishability in EIT using a hypothesis-testing model". This creates a z statistic
    so how do we calculate probability of null hypothesis?

    Parameters
    ----------
    image
    conductive_target
    fraction
    fraction_method

    Returns
    -------

    """

    fractional_image = calc_fractional_amplitude_set(
        image,
        conductive_target=conductive_target,
        fraction=fraction,
        method=fraction_method,
    )

    mean = np.abs(np.mean(image[fractional_image == 1]))
    std = np.std(image[fractional_image == 1])

    detectability = mean / std

    if method == "ratio":
        return detectability
    elif method == "db":
        return (
            np.log10(np.abs(detectability)) * 20
        )  # Convert to decibels as a root-power quantity
    else:
        raise ValueError("Invalid method specified (must be ratio or db)")
