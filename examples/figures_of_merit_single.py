# coding: utf-8
from __future__ import absolute_import, division, print_function

import matplotlib.pyplot as plt
import numpy as np
import pyeit.eit.jac as jac
import pyeit.mesh as mesh
from pyeit.eit.fem import EITForward
import pyeit.eit.protocol as protocol
from pyeit.mesh.wrapper import PyEITAnomaly_Circle
from pyeit.mesh.external import load_mesh, place_electrodes_equal_spacing
from pyeit.eit.render import render_2d_mesh
from pyeit.visual.plot import (
    create_image_plot,
    create_layered_image_plot,
    create_mesh_plot
)
from pyeit.quality.merit import calc_greit_figures_of_merit

"""
Example demonstrating the calculation of the GREIT figures of merit for a single reconstruction
"""


def main():
    simulation_mesh_filename = r".\example_data\imdm.stl"
    reconstruction_mesh_filename = r".\example_data\imdl.stl"
    n_el = 16
    render_resolution = (64, 64)

    # Problem setup
    sim_mesh = load_mesh(simulation_mesh_filename)
    electrode_nodes = place_electrodes_equal_spacing(sim_mesh, n_electrodes=16)
    sim_mesh.el_pos = np.array(electrode_nodes)
    anomaly = PyEITAnomaly_Circle(center=[0.5, 0], r=0.05, perm=2)
    sim_mesh_new = mesh.set_perm(sim_mesh, anomaly=anomaly)

    # Simulation
    protocol_obj = protocol.create(n_el, dist_exc=1, step_meas=1, parser_meas="std")
    fwd = EITForward(sim_mesh, protocol_obj)
    v0 = fwd.solve_eit()
    v1 = fwd.solve_eit(perm=sim_mesh_new.perm)

    # Reconstruction
    recon_mesh = load_mesh(reconstruction_mesh_filename)
    electrode_nodes = place_electrodes_equal_spacing(recon_mesh, n_electrodes=16)
    recon_mesh.el_pos = np.array(electrode_nodes)

    eit = jac.JAC(recon_mesh, protocol_obj)
    eit.setup(p=0.5, lamb=0.03, method="kotre", perm=1, jac_normalized=True)
    ds = eit.solve(v1, v0, normalize=True)
    solution = np.real(ds)

    # Render results
    sim_render = render_2d_mesh(sim_mesh, sim_mesh_new.perm, resolution=render_resolution)
    recon_render = render_2d_mesh(recon_mesh, solution, resolution=render_resolution)

    out = {}
    figs = calc_greit_figures_of_merit(sim_render, recon_render, out=out)

    #
    # Output
    #

    # Print figures of merit
    print(figs)

    # Create mesh plots
    fig, axs = plt.subplots(1, 2)
    create_mesh_plot(axs[0], sim_mesh, ax_kwargs={"title":"Sim mesh"})
    create_mesh_plot(axs[1], recon_mesh, ax_kwargs={"title":"Recon mesh"})
    fig.set_size_inches(10,4)

    fig, axs = plt.subplots(1, 2)
    im_simulation = create_image_plot(axs[0], sim_render.T, title="Target image")
    im_recon = create_image_plot(axs[1], recon_render.T, title="Reconstruction image")
    fig.set_size_inches(10,4)

    fig, axs = plt.subplots(1,2, constrained_layout=True)
    create_layered_image_plot(
        axs[0],
        (
            recon_render.T,
            out["shape_deformation"]["shape_deformation_target"].T,
            out["shape_deformation"]["outside_positions"].T,
        ),
        labels=["Background", "Shape deformation\ntarget", "Area outside\ntarget"],
        title="Shape deformation",
        margin=10,
    )

    create_layered_image_plot(
        axs[1],
        (
            recon_render.T,
            out["ringing"]["ringing_target"].T,
            out["ringing"]["opposite_outside_positions"].T,
        ),
        labels=["Background", "Ringing\ntarget", "Opposite values\n outside target"],
        title="Ringing",
        margin=10,
    )
    fig.set_size_inches(10, 4)

    plt.show()


if __name__ == "__main__":
    main()
