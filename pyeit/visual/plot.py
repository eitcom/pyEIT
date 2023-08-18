# pylint: disable=no-member, invalid-name
# pylint: disable=too-many-arguments, too-many-locals
"""plot EIT data"""
# Copyright (c) Benyuan Liu. All Rights Reserved.
# Distributed under the (new) BSD License. See LICENSE.txt for more info.
from __future__ import absolute_import, division, print_function

import numpy as np
from matplotlib import dates
from pyeit.mesh import PyEITMesh
import matplotlib.collections
from typing import Optional
from matplotlib import (
    pyplot as plt,
    patches as mpatches,
    axes as mpl_axes,
)
from mpl_toolkits.axes_grid1 import make_axes_locatable
import matplotlib.colorbar


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


def create_mesh_plot(
    ax: mpl_axes.Axes,
    mesh: PyEITMesh,
    ax_kwargs: Optional[dict] = {},
    electrodes: Optional[np.ndarray] = None,
    coordinate_labels: Optional[str] = None,
    marker_kwargs: Optional[dict] = {},
    marker_text_kwargs: Optional[dict] = {},
    coord_label_text_kwargs: Optional[dict] = {},
    flat_plane: Optional[str] = "z",
):
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
        column in PyEITMesh to consider flat

    Returns
    -------
    pc, elec_markers, coord_labels:
        matplotlib artists

    """
    ax.set_aspect(
        "equal"
    )  # Default aspect is auto. Set to equal so plot is not distorted

    if flat_plane not in ["x", "y", "z"]:
        raise ValueError("Please select a flat plane from x, y, or z")
    flat_ind = {"x": 0, "y": 1, "z": 2}[flat_plane]

    nodes = np.delete(mesh.node, flat_ind, axis=1)
    elements = mesh.element
    values = (
        mesh.perm
        if not isinstance(mesh.perm, float)
        else np.ones(len(elements)) * mesh.perm
    )

    # Create PolyCollection representing mesh
    verts = nodes[elements]
    pc = matplotlib.collections.PolyCollection(verts, edgecolor="black")
    pc.set_array(values)

    # Add mesh to ax
    ax.add_collection(pc)
    cb = colorbar(pc)
    cb.set_label("Element Value")
    ax.autoscale()
    ax.set_xticks([], labels=None)
    ax.set_yticks([], labels=None)
    ax.set(**ax_kwargs)

    elec_markers = None
    if electrodes is not None:
        elec_markers = add_electrode_markers(
            ax, nodes[electrodes], marker_kwargs, marker_text_kwargs
        )

    coord_labels = None
    if coordinate_labels is not None:
        coord_labels = add_coordinate_labels(
            ax, coordinate_labels, coord_label_text_kwargs
        )

    return pc, elec_markers, coord_labels


def add_electrode_markers(
    ax: mpl_axes.Axes,
    electrode_points: list,
    marker_kwargs: Optional[dict] = None,
    text_kwargs: Optional[dict] = None,
):
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


def add_coordinate_labels(
    ax: mpl_axes.Axes,
    coordinate_labels: Optional[str] = None,
    text_kwargs: Optional[dict] = None,
):
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

    if coordinate_labels == "radiological":
        l1 = ax.text(
            0.01,
            0.5,
            "Right",
            transform=ax.transAxes,
            rotation="vertical",
            va="center",
            ha="left",
            **text_kwargs
        )
        l2 = ax.text(
            0.99,
            0.5,
            "Left",
            transform=ax.transAxes,
            rotation="vertical",
            va="center",
            ha="right",
            **text_kwargs
        )
        l3 = ax.text(
            0.5,
            0.01,
            "Posterior",
            transform=ax.transAxes,
            ha="center",
            va="bottom",
            **text_kwargs
        )
        l4 = ax.text(
            0.5,
            0.99,
            "Anterior",
            transform=ax.transAxes,
            ha="center",
            va="top",
            **text_kwargs
        )
        ax.margins(
            y=0.1, x=0.1
        )  # axis autoscaling doesn't work with text, so we increase the margins to make room.
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


def create_plot(
    ax: mpl_axes.Axes,
    eit_image: np.ndarray,
    mesh: PyEITMesh,
    vmin: Optional[float] = None,
    vmax: Optional[float] = None,
    ax_kwargs: Optional[dict] = None,
    electrodes: Optional[np.ndarray] = None,
    coordinate_labels: Optional[str] = None,
    marker_kwargs: Optional[dict] = None,
    marker_text_kwargs: Optional[dict] = None,
    coord_label_text_kwargs: Optional[dict] = None,
    flat_plane: Optional[str] = "z",
):
    """
    Creates a plot of a reconstructed EIT image. Optionally plots electrode positions and adds coordinate labels.

    Parameters
    ----------
    ax
        Matplotlib axes on which to create the plot
    eit_image
        Real valued output of pyeit solve methods
    mesh
        PyEIT mesh
    vmin
        Minimum value to plot with ax.tripcolor
    vmax
        Maximum value to plot with ax.tripcolor
    ax_kwargs
        Additional kwargs to use in ax.set
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
        column in PyEITMesh to consider flat

    Returns
    -------
    plot_image, elec_markers, coord_labels
        matplotlib artists

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

    ax.set_aspect("equal")

    tripcolor_keys_map = {"vmin": vmin, "vmax": vmax}
    tripcolor_kwargs = {k: v for k, v in tripcolor_keys_map.items() if v is not None}
    plot_image = ax.tripcolor(x, y, elements, eit_image, **tripcolor_kwargs)
    colorbar(plot_image)
    ax.set_xticks([], labels=None)
    ax.set_yticks([], labels=None)
    ax.set(**ax_kwargs)

    elec_markers = None
    if electrodes is not None:
        elec_markers = add_electrode_markers(
            ax, nodes[electrodes], marker_kwargs, marker_text_kwargs
        )

    coord_labels = None
    if coordinate_labels is not None:
        coord_labels = add_coordinate_labels(
            ax, coordinate_labels, coord_label_text_kwargs
        )

    return plot_image, elec_markers, coord_labels


def create_image_plot(
    ax, image, title, vmin=None, vmax=None, background=np.nan, margin=10, origin="lower"
):
    """
    Create a plot using imshow and set the axis bounds to frame the image

    Parameters
    ----------
    ax
    image
        Image array
    title
        Plot title
    background
        Value of the background in the image
    margin
        Margin to place at the sides of the image
    origin
        Origin parameter for imshow

    Returns
    -------
    im

    """
    im = ax.imshow(image, origin=origin, vmin=vmin, vmax=vmax)
    img_bounds = get_img_bounds(image, background=background)
    ax.set_ybound(img_bounds[0] - margin, img_bounds[1] + margin)
    ax.set_xbound(img_bounds[2] - margin, img_bounds[3] + margin)
    ax.set_title(title)

    colorbar(im)
    return im


def create_layered_image_plot(
    ax, layers, labels=None, title=None, origin="lower", margin=None
):
    """
    Create a plot using imshow built from discrete layers, and label those layers in the legend.

    Parameters
    ----------
    ax
    layers: list(np.Array(width,height))
        list of with x height arrays with value of 1 in cells where layer should be present
    labels
        layer labels
    title
        plot title
    origin
        origin parameter for imshow
    margin
        margin to place at the sides of the image

    Returns
    -------
    img

    """
    values = list(range(1, len(labels) + 1))
    img_array = np.full(np.shape(layers[0]), np.nan)
    for (
        i,
        layer,
    ) in enumerate(layers):
        img_array[np.where(np.logical_and(~np.isnan(layer), layer))] = values[i]

    img = ax.imshow(img_array, origin=origin)

    if margin is not None:
        img_bounds = get_img_bounds(img_array)
        ax.set_ybound(img_bounds[0] - margin, img_bounds[1] + margin)
        ax.set_xbound(img_bounds[2] - margin, img_bounds[3] + margin)
        ax.set_title(title)

    if labels is not None:
        # get the colors of the values, according to the
        # colormap used by imshow
        colors = [img.cmap(img.norm(value)) for value in values]
        # create a patch (proxy artist) for every color
        patches = [
            mpatches.Patch(color=colors[i], label=labels[i]) for i in range(len(values))
        ]
        # put those patched as legend-handles into the legend
        ax.legend(handles=patches, bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.0)
        # fig.set_tight_layout(True)

    return img


def get_img_bounds(img, background=np.nan):
    """
    Get the bounds of an image represented by values on a background of an array of width x height

    Parameters
    ----------
    img: np.Array(width, height)
    background:
        value of the image background

    Returns
    -------
    xmin: first row containing image
    xmax: last row containing image
    ymin: first column containing image
    ymax: last column containing image

    """
    xmin = None
    ymin = None
    xmax = np.shape(img)[0]
    ymax = np.shape(img)[1]

    if not np.isnan(background):
        img[np.where(img == background)] = np.nan

    for i in range(img.shape[0]):
        if xmin is None and not np.all(np.isnan(img[i, :])):
            xmin = i
        if xmin is not None and np.all(np.isnan(img[i, :])):
            xmax = i - 1

    for j in range(img.shape[1]):
        if ymin is None and not np.all(np.isnan(img[:, j])):
            ymin = j
        if ymin is not None and np.all(np.isnan(img[:, j])):
            ymax = j - 1

    return xmin, xmax, ymin, ymax


def colorbar(mappable: matplotlib.cm.ScalarMappable) -> matplotlib.colorbar.Colorbar:
    """
    Add a colorbar that matches the height of its corresponding image

    Parameters
    ----------
    mappable

    Returns
    -------
    cbar

    """
    last_axes = plt.gca()
    ax = mappable.axes
    fig = ax.figure
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.1)
    cbar = fig.colorbar(mappable, cax=cax)
    plt.sca(last_axes)
    return cbar
