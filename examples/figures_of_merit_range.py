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
    anomaly_value = 2
    conductive_target = True if anomaly_value - background > 0 else False
    c_range = np.arange(0, 1, 0.05)

    # Problem setup
    sim_mesh = mesh.create(n_el, h0=0.05)
    electrode_nodes = place_electrodes_equal_spacing(sim_mesh, n_electrodes=16)
    sim_mesh.el_pos = np.array(electrode_nodes)

    # Simulation
    protocol_obj = protocol.create(n_el, dist_exc=1, step_meas=1, parser_meas="std")
    fwd = EITForward(sim_mesh, protocol_obj)
    v0 = fwd.solve_eit(perm=background)

    # Reconstruction
    recon_mesh = mesh.create(n_el, h0=0.1)
    electrode_nodes = place_electrodes_equal_spacing(recon_mesh, n_electrodes=16)
    recon_mesh.el_pos = np.array(electrode_nodes)

    eit = jac.JAC(recon_mesh, protocol_obj)
    eit.setup(p=0.5, lamb=0.03, method="kotre", perm=1, jac_normalized=True)

    figs_list = []
    for c in c_range:
        anomaly = PyEITAnomaly_Circle(center=[c, 0], r=0.05, perm=anomaly_value)
        sim_mesh_new = mesh.set_perm(sim_mesh, anomaly=anomaly, background=background)

        v1 = fwd.solve_eit(perm=sim_mesh_new.perm)

        ds = eit.solve(v1, v0, normalize=True)
        solution = np.real(ds)

        # Render results
        sim_render = render_2d_mesh(
            sim_mesh, sim_mesh_new.perm, resolution=render_resolution
        )
        recon_render = render_2d_mesh(
            recon_mesh, solution, resolution=render_resolution
        )

        figs, out = calc_greit_figures_of_merit(
            sim_render,
            recon_render,
            conductive_target=conductive_target,
            return_extras=True,
        )
        figs_list.append(figs)

    figs_list = np.array(figs_list)
    fig, axs = plt.subplots(5, 1)
    titles = [
        "Amplitude",
        "Position Error",
        "Resolution",
        "Shape Deformation",
        "Ringing",
    ]
    for i in range(5):
        axs[i].plot(c_range, figs_list[:, i])
        axs[i].set_title(titles[i])

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
