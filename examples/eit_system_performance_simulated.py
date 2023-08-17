# coding: utf-8
from __future__ import absolute_import, division, print_function

import matplotlib.pyplot as plt
import numpy as np
import pyeit.eit.jac as jac
import pyeit.mesh as mesh
from pyeit.eit.fem import EITForward
import pyeit.eit.protocol as protocol
from pyeit.mesh.wrapper import PyEITAnomaly_Circle
from pyeit.mesh.external import place_electrodes_equal_spacing
from numpy.random import default_rng
from pyeit.quality.eit_system import (
    calc_signal_to_noise_ratio,
    calc_accuracy,
    calc_drift,
    calc_detectability,
)
from pyeit.eit.render import render_2d_mesh
from pyeit.visual.plot import colorbar


def main():
    # Configuration
    # ------------------------------------------------------------------------------------------------------------------
    n_el = 16
    render_resolution = (64, 64)
    background_value = 1
    anomaly_value = 2
    noise_magnitude = 2e-4
    drift_rate = 2e-7  # Per frame

    n_background_measurements = 10
    n_drift_measurements = 1800
    measurement_frequency = 1

    detectability_r = 0.5
    distinguishability_r = 0.35

    # Initialization
    # ------------------------------------------------------------------------------------------------------------------
    drift_period_hours = n_drift_measurements / (60 * 60 * measurement_frequency)
    conductive_target = True if anomaly_value - background_value > 0 else False
    det_center_range = np.arange(0, 0.667, 0.067)
    dist_distance_range = np.arange(0.35, 1.2, 0.09)
    rng = default_rng(0)

    # Problem setup
    # ------------------------------------------------------------------------------------------------------------------
    sim_mesh = mesh.create(n_el, h0=0.05)
    electrode_nodes = place_electrodes_equal_spacing(sim_mesh, n_electrodes=n_el)
    sim_mesh.el_pos = np.array(electrode_nodes)
    protocol_obj = protocol.create(n_el, dist_exc=1, step_meas=1, parser_meas="std")
    fwd = EITForward(sim_mesh, protocol_obj)

    recon_mesh = mesh.create(n_el, h0=0.1)
    electrode_nodes = place_electrodes_equal_spacing(recon_mesh, n_electrodes=n_el)
    recon_mesh.el_pos = np.array(electrode_nodes)
    eit = jac.JAC(recon_mesh, protocol_obj)
    eit.setup(
        p=0.5, lamb=0.03, method="kotre", perm=background_value, jac_normalized=True
    )

    # Simulate background
    # ------------------------------------------------------------------------------------------------------------------
    v0 = fwd.solve_eit(perm=background_value)
    d1 = np.array(range(n_drift_measurements)) * drift_rate  # Create drift
    n1 = noise_magnitude * rng.standard_normal(
        (n_drift_measurements, len(v0))
    )  # Create noise

    v0_dn = np.tile(v0, (n_drift_measurements, 1)) + np.tile(d1, (len(v0), 1)).T + n1

    # Calculate background performance measures
    # ------------------------------------------------------------------------------------------------------------------
    snr = calc_signal_to_noise_ratio(v0_dn[:n_background_measurements], method="db")
    accuracy = calc_accuracy(v0_dn[:n_background_measurements], v0, method="EIDORS")
    t2, adevs = calc_drift(v0_dn, method="Allan")

    drifts_delta = calc_drift(v0_dn, sample_period=10, method="Delta")
    start = np.average(v0_dn[0:10], axis=0)
    drifts_percent = 100 * drifts_delta / start

    # Simulate detectability test
    # ------------------------------------------------------------------------------------------------------------------
    detectabilities = []
    detectability_renders = []
    for c in det_center_range:
        anomaly = PyEITAnomaly_Circle(
            center=[c, 0], r=detectability_r, perm=anomaly_value
        )
        sim_mesh_new = mesh.set_perm(
            sim_mesh, anomaly=anomaly, background=background_value
        )
        v1 = fwd.solve_eit(perm=sim_mesh_new.perm)
        n = noise_magnitude * rng.standard_normal(len(v1))
        v1_n = v1 + n
        ds = eit.solve(v1_n, v0_dn[0], normalize=True)
        solution = np.real(ds)
        image = render_2d_mesh(recon_mesh, solution, resolution=render_resolution)
        detectability = calc_detectability(
            image, conductive_target=conductive_target, method="db"
        )
        detectabilities.append(detectability)
        detectability_renders.append(image)

    # Simulate distinguishability test
    # ------------------------------------------------------------------------------------------------------------------
    anomaly = PyEITAnomaly_Circle(center=[0, 0], r=detectability_r, perm=anomaly_value)
    sim_mesh_new = mesh.set_perm(sim_mesh, anomaly=anomaly, background=background_value)
    dist_v0 = fwd.solve_eit(perm=sim_mesh_new.perm)
    dist_v0n = dist_v0 + noise_magnitude * rng.standard_normal(len(dist_v0))

    distinguishabilities = []
    distinguishabilitiy_renders = []
    for d in dist_distance_range:
        a1 = PyEITAnomaly_Circle(
            center=[d / 2, 0], r=distinguishability_r, perm=anomaly_value
        )
        a2 = PyEITAnomaly_Circle(
            center=[-d / 2, 0], r=distinguishability_r, perm=anomaly_value
        )
        sim_mesh_new = mesh.set_perm(
            sim_mesh, anomaly=[a1, a2], background=background_value
        )
        v1 = fwd.solve_eit(perm=sim_mesh_new.perm)
        v1_n = v1 + noise_magnitude * rng.standard_normal(len(v1))
        ds = eit.solve(v1_n, dist_v0n, normalize=True)
        solution = np.real(ds)
        image = render_2d_mesh(recon_mesh, solution, resolution=render_resolution)
        # Distinguishability is detectability but with a target as the background.
        distinguishability = calc_detectability(
            image, conductive_target=conductive_target, method="db"
        )
        distinguishabilities.append(distinguishability)
        distinguishabilitiy_renders.append(image)

    # Plot results
    # ------------------------------------------------------------------------------------------------------------------
    fig, axs = plt.subplots(2, 2)

    axs[0, 0].plot(snr)
    axs[0, 0].set_xlabel("Channel Number")
    axs[0, 0].set_ylabel("Signal to Noise Ratio\n(dB)")
    axs[0, 0].title.set_text(f"Signal to Noise Ratio for {len(snr)} channels")

    axs[0, 1].plot(accuracy)
    axs[0, 1].set_xlabel("Channel Number")
    axs[0, 1].set_ylabel("Accuracy")
    axs[0, 1].title.set_text(f"Accuracy for {len(snr)} channels")

    axs[1, 0].set_xlabel("Averaging Window (s)")
    axs[1, 0].set_ylabel("Allan Deviation")
    axs[1, 0].title.set_text(f"Allan Deviation for {len(snr)} channels")
    for adev in adevs:
        axs[1, 0].plot(t2, adev)

    axs[1, 1].plot(drifts_percent)
    axs[1, 1].title.set_text(
        f"Drift percentage on all channels.\nDrift period (hours): {drift_period_hours}"
    )
    axs[1, 1].set_xlabel("Channel number")
    axs[1, 1].set_ylabel("Drift (% of starting value)")

    fig.tight_layout()
    fig.set_size_inches((10, 6))

    fig, axs = plt.subplots(1, 2)
    axs[0].plot(det_center_range, detectabilities, ".-", label="-x axis")
    axs[0].legend()
    axs[0].set_xlabel("Target position (radius fraction)")
    axs[0].set_ylabel("Detectability (dB)")
    axs[0].title.set_text("Detectability vs radial position")

    axs[1].plot(dist_distance_range, distinguishabilities)
    axs[1].set_xlabel("Separation distance (radius fraction)")
    axs[1].set_ylabel("Distinguishability (dB)")
    axs[1].title.set_text("Distinguishability vs separation distance")

    fig.set_size_inches((10, 4))
    fig.tight_layout()

    fig, axs = plt.subplots(1, len(det_center_range))
    for i, c in enumerate(det_center_range):
        axs[i].imshow(detectability_renders[i])
        axs[i].xaxis.set_ticks([])
        axs[i].yaxis.set_ticks([])

    fig.set_size_inches((14, 2))
    fig.suptitle("Detectability Renders")
    fig.tight_layout()

    fig, axs = plt.subplots(1, len(dist_distance_range))
    for i, d in enumerate(dist_distance_range):
        img = axs[i].imshow(distinguishabilitiy_renders[i])
        axs[i].xaxis.set_ticks([])
        axs[i].yaxis.set_ticks([])
        colorbar(img)

    fig.set_size_inches((18, 2))
    fig.suptitle("Distinguishability Renders")
    fig.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
