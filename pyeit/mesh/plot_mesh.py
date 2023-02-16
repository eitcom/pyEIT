import matplotlib.pyplot as plt
import numpy as np

from . import PyEITMesh


def plot_mesh(mesh_obj: PyEITMesh, figsize: tuple = (6, 4), title: str = "mesh"):
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
    fig = plt.figure(figsize=figsize)
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
    ax.set_title(title)
    ax.set_aspect("equal")
    ax.set_ylim([-1.2, 1.2])
    ax.set_xlim([-1.2, 1.2])
    fig.set_size_inches(6, 6)
    plt.show()

    return fig


def plot_distmesh(p, t, el_pos=[]):
    """helper function to plot distmesh (in examples)"""
    fig, ax = plt.subplots()
    ax.triplot(p[:, 0], p[:, 1], t)
    mesh_center = np.array([np.median(p[:, 0]), np.median(p[:, 1])])
    if len(el_pos) > 0:
        ax.plot(p[el_pos, 0], p[el_pos, 1], "ro")  # ro : red circle
    for i, el in enumerate(el_pos):
        xy = np.array([p[el, 0], p[el, 1]])
        text_offset = (xy - mesh_center) * [1, -1] * 0.05
        ax.annotate(
            str(i + 1),
            xy=xy,
            xytext=text_offset,
            textcoords="offset points",
            color="k",
            fontsize=15,
            ha="center",
            va="center",
        )
    xmax, ymax = np.max(p, axis=0)
    xmin, ymin = np.min(p, axis=0)
    ax.set_xlim([1.2 * xmin, 1.2 * xmax])
    ax.set_ylim([1.2 * ymin, 1.2 * ymax])
    ax.set_aspect("equal")
    plt.show()

    return fig
