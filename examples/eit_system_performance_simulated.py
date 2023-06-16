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
from pyeit.visual.plot import (
    create_image_plot,
    create_layered_image_plot,
    create_mesh_plot,
    create_plot
)
from pyeit.quality.merit import calc_greit_figures_of_merit
from numpy.random import default_rng


def main():
    n_el = 16
    render_resolution = (64, 64)
    background = 1
    anomaly = 2
    conductive_target = True if anomaly - background > 0 else False
    noise_magnitude = 2e-5
    # noise_magnitude = 0

    # Problem setup
    sim_mesh = mesh.create(n_el, h0=0.05)
    electrode_nodes = place_electrodes_equal_spacing(sim_mesh, n_electrodes=16)
    sim_mesh.el_pos = np.array(electrode_nodes)
    anomaly = PyEITAnomaly_Circle(center=[0.5, 0], r=0.05, perm=anomaly)
    sim_mesh = mesh.set_perm(sim_mesh, anomaly=anomaly, background=background)

    # Simulation
    protocol_obj = protocol.create(n_el, dist_exc=1, step_meas=1, parser_meas="std")
    fwd = EITForward(sim_mesh, protocol_obj)
    v0 = fwd.solve_eit(perm=background)
    v1 = fwd.solve_eit(perm=sim_mesh.perm)

    # Add noise
    rng = default_rng(0)
    n1 = noise_magnitude * rng.standard_normal(len(v0))
    n2 = noise_magnitude * rng.standard_normal(len(v1))

    v0_n = v0 + n1
    v1_n = v1 + n2

    # Reconstruction
    recon_mesh = mesh.create(n_el, h0=0.1)
    electrode_nodes = place_electrodes_equal_spacing(recon_mesh, n_electrodes=16)
    recon_mesh.el_pos = np.array(electrode_nodes)

    eit = jac.JAC(recon_mesh, protocol_obj)
    eit.setup(p=0.5, lamb=0.03, method="kotre", perm=1, jac_normalized=True)
    ds = eit.solve(v1_n, v0_n, normalize=True)
    solution = np.real(ds)

    # Create plots
    fig, axs = plt.subplots(1, 2)
    create_mesh_plot(axs[0], sim_mesh, ax_kwargs={"title": "Mesh Plot"})
    create_plot(axs[1], solution, recon_mesh)

    fig.set_size_inches(10, 5)
    fig.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
