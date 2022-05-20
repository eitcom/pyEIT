# pylint: disable=no-member, invalid-name
# pylint: disable=too-many-arguments, too-many-locals
"""plot EIT data"""
# Copyright (c) Benyuan Liu. All Rights Reserved.
# Distributed under the (new) BSD License. See LICENSE.txt for more info.
from __future__ import absolute_import, division, print_function

import os.path

import numpy as np
from matplotlib import dates
from pyeit.mesh import PyEITMesh
import matplotlib.collections
from numpy.typing import ArrayLike
from matplotlib import pyplot as plt, patches as mpatches, axes as mpl_axes, artist as mpl_artist


def mesh_plot(
    mesh: PyEITMesh,
    el_pos,
    mstr="",
    figsize=(9, 6),
    alpha=0.5,
    offset_ratio=0.075,
    show_image=False,
    show_mesh=False,
    show_electrode=True,
    show_number=False,
    show_text=True,
):
    """plot mesh structure (base layout)"""
    # load mesh structure
    pts = mesh.node
    tri = mesh.element
    fig, ax = plt.subplots(figsize=figsize)
    ax.set_facecolor("black")
    ax.set_aspect("equal")

    # load background
    if show_image and os.path.exists(mstr):
        image_name = mstr.replace("mes", "bmp")
        im = plt.imread(image_name)
        ax.imshow(im)
    else:
        # without imshow, the yaxis should be inverted
        ax.invert_yaxis()

    # show mesh structure
    if show_mesh:
        ax.triplot(pts[:, 0], pts[:, 1], tri, alpha=alpha)

    # show electrodes markers
    if show_electrode:
        ax.plot(pts[el_pos, 0], pts[el_pos, 1], "yo")

    # annotate electrodes numbering
    if show_number:
        for i, e in enumerate(el_pos):
            ax.text(pts[e, 0], pts[e, 1], np.str(i + 1), color="r", size=12)

    # annotate (L) at offset_ratio*d beside node 0
    if show_text:
        xb, xa = pts[el_pos[8], 0], pts[el_pos[0], 0]
        d = np.abs(xa - xb)
        offset = d * offset_ratio
        x, y = xa + offset, pts[el_pos[0], 1]
        ax.text(x, y, "L", size=20, color="w")
        # enlarge the right of axes if using annotation
        ax.set_xlim([xb - offset, xa + 2 * offset])

    # clean up axis
    ax.grid("off")
    plt.setp(ax.get_xticklines(), visible=False)
    plt.setp(ax.get_xticklabels(), visible=False)
    plt.setp(ax.get_yticklines(), visible=False)
    plt.setp(ax.get_yticklabels(), visible=False)

    return fig, ax


def ts_plot(ts, figsize=(6, 4), ylabel="ATI (Ohm)", ylim=None, xdate_format=True):
    """plot time series data"""
    fig, ax = plt.subplots(figsize=figsize)
    ax.plot(ts)
    ax.grid("on")
    ax.set_ylabel(ylabel)
    if ylim is not None:
        assert len(ylim) == 2
        ax.set_ylim(ylim)
    # for better xticklabels
    if xdate_format:
        axis_format = dates.DateFormatter("%m/%d %H:%M")
        ax.xaxis.set_major_formatter(axis_format)
        fig.autofmt_xdate()

    return fig, ax


def create_mesh_plot(ax: mpl_axes.Axes, mesh:PyEITMesh = None, ax_kwargs: dict = None, electrodes: ArrayLike = None,
                     coordinate_labels: str = None, marker_kwargs: dict = None, marker_text_kwargs: dict = None,
                     coord_label_text_kwargs: dict = None, flat_plane: str = "z"):
    """
    Creates a plot to display a 2d mesh. Optionally plots electrode positions and adds coordinate labels.

    Parameters
    ----------
    ax:
        Matplotlib Axes
    mesh:
        Mesh to plot.
    ax_kwargs:
        kwargs for ax.set()
    electrodes:
        array of electrode node indices
    coordinate_labels: str
        Coordinate labels to place on plot. Options:
            radiological: Labels as if looking up at patient from feet.
    marker_kwargs
        kwargs for electrode markers
    marker_text_kwargs
        kwargs for electrode marker text
    coord_label_text_kwargs
        kwargs for coordinate label text
    flat_plane

    Returns
    -------
    pc, elec_markers, coord_labels:
        matplotlib artists

    """
    if ax_kwargs is None:
        ax_kwargs = {"title": 'Mesh plot'}

    ax.set_aspect('equal')  # Default aspect is auto. Set to equal so plot is not distorted

    if flat_plane not in ["x", "y", "z"]:
        raise ValueError("Please select a flat plane from x, y, or z")
    flat_ind = {"x": 0, "y": 1, "z": 2}[flat_plane]

    nodes = np.delete(mesh.node, flat_ind, axis=1)
    elements = mesh.element
    values = mesh.perm

    # Create PolyCollection representing mesh
    verts = nodes[elements]
    pc = matplotlib.collections.PolyCollection(verts, edgecolor="black")
    pc.set_array(values)

    # Add mesh to ax
    ax.add_collection(pc)
    ax.figure.colorbar(pc, ax=ax, label="Element Value")
    ax.set_xticks([], labels=None)
    ax.set_yticks([], labels=None)
    ax.set(**ax_kwargs)

    elec_markers = None
    if electrodes is not None:
        elec_markers = add_electrode_markers(ax, nodes[electrodes], marker_kwargs, marker_text_kwargs)

    coord_labels = None
    if coordinate_labels is not None:
        coord_labels = add_coordinate_labels(ax, coordinate_labels, coord_label_text_kwargs)

    return pc, elec_markers, coord_labels


def add_electrode_markers(ax: mpl_axes.Axes, electrode_points: list,
                          marker_kwargs: dict = None, text_kwargs: dict = None):
    """
    Add markers to a plot to indicate the position of electrodes

    Parameters
    ----------
    ax:
        Matplotlib axes
    electrode_points:
        List of coordinates
    marker_kwargs:
        kwargs for marker plotting
    text_kwargs:
        kwargs for marker text plotting

    Returns
    -------
    elec_markers
        list of tuple of marker and text

    """
    if marker_kwargs is None:
        marker_kwargs = {"marker": "o", "color": "black"}

    if text_kwargs is None:
        text_kwargs = {"size": 12}

    elec_markers = []
    for i, point in enumerate(electrode_points):
        marker = ax.plot(*point, **marker_kwargs)
        alignment = alignment_opposing_center(ax, *point)
        text = ax.text(*point, s=str(i + 1), **text_kwargs, **alignment)
        elec_markers.append((text, marker))
    ax.legend(marker, ["Electrodes"], loc="lower left", bbox_to_anchor=(0.65, -0.15))

    return elec_markers


def add_coordinate_labels(ax: mpl_axes.Axes, coordinate_labels: str = None, text_kwargs: dict = None):
    """
    Add labels to a plot to clarify the relationship between the plot coordinate system and the coordinate system of the
    subject (e.g. torso slice).

    *Note* This adds labels only. The user is responsible for ensuring that the plot is oriented correctly.

    Parameters
    ----------
    ax:
        Matplotlib Axes
    coordinate_labels:
        String indicating which type of coordinate labels to add. Options:
            - radiological: standard radiological view. A view as if looking up at a patient from their feet.
    text_kwargs:
        kwargs for the text labels

    Returns
    -------
    coord_labels
        tuple of text labels

    """
    if text_kwargs is None:
        text_kwargs = {}

    coord_labels = []
    if coordinate_labels == "radiological":
        l1 = ax.text(0.01, 0.5, "Right", transform=ax.transAxes, rotation="vertical", va="center", ha="left",
                     **text_kwargs)
        l2 = ax.text(0.99, 0.5, "Left", transform=ax.transAxes, rotation="vertical", va="center", ha="right",
                     **text_kwargs)
        l3 = ax.text(0.5, 0.01, "Posterior", transform=ax.transAxes, ha="center", va="bottom", **text_kwargs)
        l4 = ax.text(0.5, 0.99, "Anterior", transform=ax.transAxes, ha="center", va="top", **text_kwargs)
        ax.margins(y=.1, x=.1)  # axis autoscaling doesn't work with text, so we increase the margins to make room.
        coord_labels = (l1, l2, l3, l4)

    return coord_labels


def alignment_opposing_center(ax: mpl_axes.Axes, x: float, y: float) -> dict:
    """
    Finds an alignment for a label at a given point so that it is placed away from the center of the axes. This is used
    so that labels around the perimeter of a convex object do not overlap with the object.

    Parameters
    ----------
    ax:
        Matplotlib Axes
    x:
        x coordinate
    y:
        y coordinate

    Returns
    -------
    alignment:
        A dict that can be used as kwargs for a matplotlib label placement
    """
    alignment = {}
    ax.get_xlim()  # This is sometimes needed to ensure that transLimits.transform works. Not sure why.
    axes_pos = ax.transLimits.transform((x, y))
    if axes_pos[0] < 0.5:
        alignment["ha"] = "right"
    else:
        alignment["ha"] = "left"

    if axes_pos[1] < 0.5:
        alignment["va"] = "top"
    else:
        alignment["va"] = "bottom"

    return alignment


def create_plot(ax: mpl_axes.Axes, eit_image: ArrayLike, mesh: PyEITMesh, vmin=None, vmax=None, ax_kwargs=None,
                electrodes: ArrayLike = None, coordinate_labels: str = None, marker_kwargs: dict = None,
                marker_text_kwargs: dict = None, coord_label_text_kwargs: dict = None, flat_plane: str = "z"):
    """

    Parameters
    ----------
    ax
    eit_image
    mesh
    vmin
    vmax
    ax_kwargs
    electrodes
    coordinate_labels
    marker_kwargs
    marker_text_kwargs
    coord_label_text_kwargs
    flat_plane

    Returns
    -------

    """
    if ax_kwargs is None:
        ax_kwargs = {"title": "EIT Plot"}

    if flat_plane not in ["x", "y", "z"]:
        raise ValueError("Please select a flat plane from x, y, or z")
    flat_ind = {"x": 0, "y": 1, "z": 2}[flat_plane]

    nodes = np.delete(mesh.node, flat_ind, axis=1)
    elements = mesh.element
    x = nodes[:, 0]
    y = nodes[:, 1]

    ax.set_aspect('equal')

    tripcolor_keys_map = {"vmin": vmin, "vmax": vmax}
    tripcolor_kwargs = {k: v for k, v in tripcolor_keys_map.items() if v is not None}
    plot_image = ax.tripcolor(x, y, elements, eit_image, **tripcolor_kwargs)
    ax.figure.colorbar(plot_image)
    ax.set_xticks([], labels=None)
    ax.set_yticks([], labels=None)
    ax.set(**ax_kwargs)

    elec_markers = None
    if electrodes is not None:
        elec_markers = add_electrode_markers(ax, nodes[electrodes], marker_kwargs, marker_text_kwargs)

    coord_labels = None
    if coordinate_labels is not None:
        coord_labels = add_coordinate_labels(ax, coordinate_labels, coord_label_text_kwargs)

    return plot_image, elec_markers, coord_labels
