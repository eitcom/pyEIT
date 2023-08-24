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
from pyeit.eit.render import render_2d_mesh
from pyeit.visual.plot import (
    create_image_plot,
    create_layered_image_plot,
    create_mesh_plot,
)
from pyeit.quality.merit import calc_greit_figures_of_merit

"""
Example demonstrating the calculation of the GREIT figures of merit for a single reconstruction
"""


def main():
    n_el = 16
    render_resolution = (64, 64)
    background = 1
    anomaly = 2
    conductive_target = True if anomaly - background > 0 else False

    # Problem setup
    sim_mesh = mesh.create(n_el, h0=0.05)
    electrode_nodes = place_electrodes_equal_spacing(sim_mesh, n_electrodes=16)
    sim_mesh.el_pos = np.array(electrode_nodes)
    anomaly = PyEITAnomaly_Circle(center=[0.5, 0], r=0.05, perm=anomaly)
    sim_mesh_new = mesh.set_perm(sim_mesh, anomaly=anomaly, background=background)

    # Simulation
    protocol_obj = protocol.create(n_el, dist_exc=1, step_meas=1, parser_meas="std")
    fwd = EITForward(sim_mesh, protocol_obj)
    v0 = fwd.solve_eit(perm=background)
    v1 = fwd.solve_eit(perm=sim_mesh_new.perm)

    # Reconstruction
    recon_mesh = mesh.create(n_el, h0=0.1)
    electrode_nodes = place_electrodes_equal_spacing(recon_mesh, n_electrodes=16)
    recon_mesh.el_pos = np.array(electrode_nodes)

    eit = jac.JAC(recon_mesh, protocol_obj)
    eit.setup(p=0.5, lamb=0.03, method="kotre", perm=1, jac_normalized=True)
    ds = eit.solve(v1, v0, normalize=True)
    solution = np.real(ds)

    # Render results
    sim_render = render_2d_mesh(
        sim_mesh, sim_mesh_new.perm, resolution=render_resolution
    )
    recon_render = render_2d_mesh(recon_mesh, solution, resolution=render_resolution)

    figs, out = calc_greit_figures_of_merit(
        sim_render,
        recon_render,
        conductive_target=conductive_target,
        return_extras=True,
    )

    # Output
    # Print figures of merit
    print("")
    print(f"Amplitude: Average pixel value in reconstruction image is {figs[0]:.4f}")
    print(f"Position Error: {100 * figs[1]:.2f}% of widest axis")
    print(
        f"Resolution: Reconstructed point radius {100 * figs[2]:.2f}% of image equivalent radius"
    )
    print(
        f"Shape Deformation: {100 * figs[3]:.2f}% of pixels in the thresholded image are outside the equivalent circle"
    )
    print(
        f"Ringing: Ringing pixel amplitude is  {100 * figs[4]:.2f}% of image amplitude in thresholded region"
    )

    # Create mesh plots
    fig, axs = plt.subplots(1, 2)
    create_mesh_plot(axs[0], sim_mesh_new, ax_kwargs={"title": "Sim mesh"})
    create_mesh_plot(axs[1], recon_mesh, ax_kwargs={"title": "Recon mesh"})
    fig.set_size_inches(10, 4)

    fig, axs = plt.subplots(1, 2)
    create_image_plot(axs[0], sim_render, title="Target image")
    create_image_plot(axs[1], recon_render, title="Reconstruction image")
    fig.set_size_inches(10, 4)

    fig, axs = plt.subplots(1, 2, constrained_layout=True)
    create_layered_image_plot(
        axs[0],
        (
            recon_render,
            out["shape_deformation"]["shape_deformation_target"],
            out["shape_deformation"]["outside_positions"],
        ),
        labels=["Background", "Shape deformation\ntarget", "Area outside\ntarget"],
        title="Shape deformation",
        margin=10,
    )

    create_layered_image_plot(
        axs[1],
        (
            recon_render,
            out["ringing"]["ringing_target"],
            out["ringing"]["opposite_outside_positions"],
        ),
        labels=["Background", "Ringing\ntarget", "Opposite values\n outside target"],
        title="Ringing",
        margin=10,
    )
    fig.set_size_inches(10, 4)

    plt.show()


if __name__ == "__main__":
    main()
