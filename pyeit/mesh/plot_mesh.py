import matplotlib.pyplot as plt
import numpy as np

from . import PyEITMesh


def plot_mesh(mesh_obj: PyEITMesh) -> None:
    """
    Plot a PyEITMesh

    Parameters
    ----------
    mesh_obj : PyEITMesh

    """
    plt.style.use("default")
    pts = mesh_obj.node
    tri = mesh_obj.element
    x, y = pts[:, 0], pts[:, 1]
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.tripcolor(
        x,
        y,
        tri,
        np.real(mesh_obj.perm_array),
        edgecolors="k",
        shading="flat",
        alpha=0.5,
        cmap=plt.cm.viridis,
    )
    # draw electrodes
    ax.plot(x[mesh_obj.el_pos], y[mesh_obj.el_pos], "ro")
    for i, e in enumerate(mesh_obj.el_pos):
        ax.text(x[e], y[e], str(i + 1), size=12)
    ax.set_title(r"mesh")
    ax.set_aspect("equal")
    ax.set_ylim([-1.2, 1.2])
    ax.set_xlim([-1.2, 1.2])
    fig.set_size_inches(6, 6)
