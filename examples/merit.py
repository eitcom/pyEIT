from pyeit.mesh.external import load_mesh, place_electrodes_equal_spacing
import matplotlib.pyplot as plt
from pyeit.visual.plot import (
    create_mesh_plot_2,
    create_plot,
    create_image_plot,
    create_layered_image_plot,
)
import pyeit.eit.protocol as protocol
from pyeit.eit.jac import JAC
from pyeit.eit.fem import EITForward
import numpy as np
from pyeit.eit.render import model_inverse_uv, map_image
import pyeit.quality.merit as merit


def main():
    # The example file is oriented in the following manner:
    #   Left towards the X axis
    #   Anterior direction towards the Y axis
    # This allows the 2D mesh to be displayed in the radiological view with no transformations
    simulation_mesh_filename = (
        "example_data/mesha06_bumpychestslice_radiological_view_both_lungs_1_0-3.ply"
    )
    n_electrodes = 16

    sim_mesh = load_mesh(simulation_mesh_filename)
    electrode_nodes = place_electrodes_equal_spacing(sim_mesh, n_electrodes=16)
    sim_mesh.el_pos = np.array(electrode_nodes)

    image = model_inverse_uv(
        {"node": sim_mesh.node[:, :2], "element": sim_mesh.element},
        resolution=(1000, 1000),
    )
    render = map_image(image, np.array(sim_mesh.perm))

    fig, ax = plt.subplots()
    create_image_plot(ax, render.T, title="Target image")

    protocol_obj = protocol.create(
        n_electrodes, dist_exc=int(n_electrodes / 2), step_meas=1, parser_meas="std"
    )
    fwd = EITForward(sim_mesh, protocol_obj)
    vh = fwd.solve_eit(perm=1)
    vi = fwd.solve_eit(perm=sim_mesh.perm)

    # Recon
    # Set up eit object
    pyeit_obj = JAC(sim_mesh, protocol_obj)
    pyeit_obj.setup(p=0.5, lamb=0.001, method="kotre", perm=1)

    # # Dynamic solve simulated data
    ds_sim = pyeit_obj.solve(vi, vh, normalize=False)
    solution = np.real(ds_sim)

    recon_render = map_image(image, np.array(solution))

    fig, ax = plt.subplots()
    im = create_image_plot(ax, recon_render.T, title="Reconstruction image")

    out_circular = {}
    figs_circular = merit.calc_greit_figures_of_merit(
        render, recon_render, out=out_circular
    )

    out_non_circular = {}
    figs_non_circular = merit.calc_greit_figures_of_merit(
        render, recon_render, circular=False, out=out_non_circular
    )

    print(
        "GREIT figures of merit (circular)\nAmplitude:\t\t\t%f\nPosition Error:\t\t%f\nResolution:\t\t\t%f\nShape Deformation:\t%f\nRinging:\t\t\t%f\n"
        % figs_circular
    )

    print(
        "GREIT figures of merit (non circular)\nAmplitude:\t\t\t%f\nPosition Error:\t\t%f\nResolution:\t\t\t%f\nShape Deformation:\t%f\nRinging:\t\t\t%f\n"
        % figs_non_circular
    )

    fig, ax = plt.subplots(constrained_layout=True)
    create_layered_image_plot(
        ax,
        (
            render.T,
            out_circular["shape_deformation"]["shape_deformation_target"].T,
            out_circular["shape_deformation"]["outside_positions"].T,
        ),
        labels=["Background", "Shape deformation\ntarget", "Area outside\ntarget"],
        title="Shape deformation circular",
        margin=10,
    )

    fig, ax = plt.subplots(constrained_layout=True)
    create_layered_image_plot(
        ax,
        (
            render.T,
            out_circular["ringing"]["ringing_target"].T,
            out_circular["ringing"]["opposite_outside_positions"].T,
        ),
        labels=["Background", "Ringing\ntarget", "Opposite values\n outside target"],
        title="Ringing circular",
        margin=10,
    )

    fig, ax = plt.subplots(constrained_layout=True)
    create_layered_image_plot(
        ax,
        (
            render.T,
            out_non_circular["shape_deformation"]["shape_deformation_target"].T,
            out_non_circular["shape_deformation"]["outside_positions"].T,
        ),
        labels=["Background", "Shape deformation\ntarget", "Area outside\ntarget"],
        title="Shape deformation non circular",
        margin=10,
    )

    fig, ax = plt.subplots(constrained_layout=True)
    create_layered_image_plot(
        ax,
        (
            render.T,
            out_non_circular["ringing"]["ringing_target"].T,
            out_non_circular["ringing"]["opposite_outside_positions"].T,
        ),
        labels=["Background", "Ringing\ntarget", "Opposite values\n outside target"],
        title="Ringing non circular",
        margin=10,
    )

    plt.show()


if __name__ == "__main__":
    main()
