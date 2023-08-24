import numpy as np
from pyeit.quality.eit_system import (
    calc_signal_to_noise_ratio,
    calc_accuracy,
    calc_drift,
    find_reciprocal,
    find_reciprocals,
    calc_reciprocity_accuracy,
)
from numpy.testing import assert_array_equal
from numpy.random import default_rng
import matplotlib.pyplot as plt
import pyeit.eit.protocol as protocol


def test_snr():
    measurements = np.array([[[1, 2, 3], [4, 5, 6]], [[2, 3, 4], [5, 6, 7]]])
    snr = calc_signal_to_noise_ratio(measurements)
    snr_with_flat = calc_signal_to_noise_ratio(measurements.reshape((2, -1)))

    correct_snr = np.array([3, 5, 7, 9, 11, 13])

    assert_array_equal(snr, correct_snr)
    assert_array_equal(snr_with_flat, correct_snr)


def test_accuracy():
    measurements = np.array([[[1, 2, 3], [4, 5, 6]], [[1, 2, 3], [4, 5, 6]]])
    reference_measurements = np.array([[1.25, 2.5, 3.75], [5, 6.25, 7.5]])

    accuracy_ratio = calc_accuracy(measurements, reference_measurements, method="Ratio")
    correct_accuracy_ratio = np.array([0.8, 0.8, 0.8, 0.8, 0.8, 0.8])

    accuracy_eidors = calc_accuracy(
        measurements, reference_measurements, method="EIDORS"
    )
    correct_accuracy_eidors = np.array([1.0, 1.0, 1.0, 1.0, 1.0, 1.0])

    assert_array_equal(accuracy_ratio, correct_accuracy_ratio)
    assert_array_equal(accuracy_eidors, correct_accuracy_eidors)


def test_drift_allan():
    measurements = np.arange(1, 11)
    rng = default_rng(0)
    noise = rng.normal(0, 0.5, (100, 10))
    measurements = measurements + noise

    correct_t2 = np.arange(1, 50)
    correct_adevs_0 = np.array(
        [
            0.50144798,
            0.34189729,
            0.26004536,
            0.21503377,
            0.19580965,
            0.17977422,
            0.16560979,
            0.15510351,
            0.13790106,
            0.12184227,
            0.11378077,
            0.1028532,
            0.09888532,
            0.09849245,
            0.09000506,
            0.0825436,
            0.0737621,
            0.06756625,
            0.06567533,
            0.0643697,
            0.05971835,
            0.05578692,
            0.05155231,
            0.04734716,
            0.04230214,
            0.03907241,
            0.03439275,
            0.03255361,
            0.03208408,
            0.0327237,
            0.03250187,
            0.03394281,
            0.03573658,
            0.03737299,
            0.03849052,
            0.03712749,
            0.03684584,
            0.03460951,
            0.03283292,
            0.02997242,
            0.02945563,
            0.02765693,
            0.02602859,
            0.02549113,
            0.0244943,
            0.02455454,
            0.01994969,
            0.0135816,
            0.00945514,
        ]
    )

    t2, adevs = calc_drift(measurements, sampling_rate=1, method="Allan")

    # fig, ax = plt.subplots()
    # for adev in adevs:
    #     ax.plot(t2, adev, ".")
    # ax.set_title("Allan Deviation, 10 Channels")
    # plt.show()

    np.testing.assert_array_equal(t2, correct_t2)
    np.testing.assert_array_almost_equal(adevs[0], correct_adevs_0)


def test_drift_allan_with_drift():
    measurements = np.arange(1, 11)
    rng = default_rng(0)
    noise = rng.normal(0, 0.5, (100, 10))
    drift = (np.arange(1, 101) / 100).reshape((1, -1))
    measurements = measurements + noise + drift.T

    t2, adevs = calc_drift(measurements, sampling_rate=1, method="Allan")

    correct_t2 = np.arange(1, 50)
    correct_adevs_0 = np.array(
        [
            0.50151151,
            0.34240022,
            0.26094362,
            0.21688356,
            0.19928049,
            0.18489464,
            0.17249367,
            0.16404771,
            0.15007923,
            0.13812585,
            0.13544855,
            0.13172051,
            0.13419322,
            0.13919843,
            0.13771382,
            0.13777519,
            0.13701817,
            0.13801215,
            0.14205622,
            0.14714887,
            0.14925671,
            0.15213755,
            0.1549156,
            0.15761515,
            0.16228652,
            0.1684516,
            0.17457164,
            0.17906281,
            0.18619088,
            0.19295449,
            0.19888729,
            0.20556761,
            0.21200783,
            0.21823937,
            0.22585266,
            0.23350849,
            0.24107197,
            0.249568,
            0.2585531,
            0.26629471,
            0.27500569,
            0.28331606,
            0.29105864,
            0.29524983,
            0.30406007,
            0.31541238,
            0.32955649,
            0.34293692,
            0.34950794,
        ]
    )

    # fig, ax = plt.subplots()
    # for adev in adevs:
    #     ax.plot(t2, adev, ".")
    # ax.set_title("Allan Deviation, 10 Channels")
    # plt.show()

    np.testing.assert_array_equal(t2, correct_t2)
    np.testing.assert_array_almost_equal(adevs[0], correct_adevs_0)


def test_drift_delta():
    measurements = np.arange(1, 11)
    rng = default_rng(0)
    noise = rng.normal(0, 0.5, (100, 10))
    measurements = measurements + noise

    correct_drifts = np.array(
        [
            0.04459828,
            -0.52595789,
            -0.33285465,
            0.0782566,
            -0.02511503,
            -0.09873106,
            -0.27501217,
            -0.28731926,
            -0.27797152,
            -0.02059609,
        ]
    )
    drifts = calc_drift(measurements, sampling_rate=1, sample_period=10, method="Delta")

    # print("\n")
    # for i, drift in enumerate(drifts):
    #     print(f"Channel {i + 1} drift: {drift:.4f}")

    np.testing.assert_array_almost_equal(drifts, correct_drifts)


def test_drift_delta_with_drift():
    measurements = np.arange(1, 11)
    rng = default_rng(0)
    noise = rng.normal(0, 0.5, (100, 10))
    drift = (np.arange(1, 101) / 100).reshape((1, -1))  # 1/100 drift per second
    measurements = measurements + noise + drift.T

    correct_drifts = np.array(
        [
            0.94459828,
            0.37404211,
            0.56714535,
            0.9782566,
            0.87488497,
            0.80126894,
            0.62498783,
            0.61268074,
            0.62202848,
            0.87940391,
        ]
    )
    drifts = calc_drift(measurements, sampling_rate=1, sample_period=10, method="Delta")

    # total_period = 1*100  # Sampling rate times n_samples
    # sampling_time = 1*10  # Sampling rate times sample period
    # period_between_samples = total_period - 2*sampling_time
    # drifts_per_second = drifts/period_between_samples
    # print("\n")
    # for i, (drift, drift_per_second) in enumerate(zip(drifts, drifts_per_second)):
    #     print(f"Channel {i+1} drift: {drift:.4f}, drift per second: {drift_per_second:.4f}")
    # print(f"Average drift per second: {np.average(drifts_per_second):.4f}")

    np.testing.assert_array_almost_equal(drifts, correct_drifts)


def test_find_reciprocal():
    arr = np.array([[1, 2, 3, 4], [3, 4, 2, 1], [4, 5, 6, 7]])
    correct_reciprocal = 1
    reciprocal = find_reciprocal(arr[0], arr)

    assert reciprocal == correct_reciprocal


def test_find_reciprocals():
    arr = np.array([[1, 2, 3, 4], [3, 4, 2, 1], [4, 5, 6, 7], [6, 7, 4, 5]])
    correct_reciprocals = np.array([[0, 1], [2, 3]])

    reciprocals = find_reciprocals(arr)

    np.testing.assert_array_equal(reciprocals, correct_reciprocals)


def test_calc_reciprocity_accuracy():
    protocol_obj = protocol.create(4, dist_exc=1, step_meas=1, parser_meas="std")
    data = np.array([[1, 1, 0.9, 0.8], [1, 1, 0.9, 0.8]])

    correct_reciprocity_accuracy = np.array([0.9, 0.8])

    reciprocity_accuracy = calc_reciprocity_accuracy(data, protocol_obj)

    np.testing.assert_array_equal(correct_reciprocity_accuracy, reciprocity_accuracy)
