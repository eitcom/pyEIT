import numpy as np
from pyeit.quality.eit_system import calc_signal_to_noise_ratio, calc_accuracy
from numpy.testing import assert_array_equal


def test_snr():
    measurements = np.array([[[1, 2, 3], [4, 5, 6]], [[2, 3, 4], [5, 6, 7]]])
    snr = calc_signal_to_noise_ratio(measurements)
    snr_with_flat = calc_signal_to_noise_ratio(measurements.reshape((2, -1)))

    correct_snr = np.array([3,  5,  7,  9, 11, 13])

    assert_array_equal(snr, correct_snr)
    assert_array_equal(snr_with_flat, correct_snr)


def test_accuracy():
    measurements = np.array([[[1, 2, 3], [4, 5, 6]], [[1, 2, 3], [4, 5, 6]]])
    reference_measurements = np.array([[1.25, 2.5, 3.75], [5, 6.25, 7.5]])

    accuracy_ratio = calc_accuracy(measurements, reference_measurements, method="Ratio")
    correct_accuracy_ratio = np.array([0.8, 0.8, 0.8, 0.8, 0.8, 0.8])

    accuracy_eidors = calc_accuracy(measurements, reference_measurements, method="EIDORS")
    correct_accuracy_eidors = np.array([1., 1., 1., 1., 1., 1.])

    assert_array_equal(accuracy_ratio, correct_accuracy_ratio)
    assert_array_equal(accuracy_eidors, correct_accuracy_eidors)
