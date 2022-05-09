# pylint: disable=no-member, invalid-name
# pylint: disable=too-many-arguments, too-many-locals
"""plot EIT data"""
# Copyright (c) Benyuan Liu. All Rights Reserved.
# Distributed under the (new) BSD License. See LICENSE.txt for more info.
from __future__ import absolute_import, division, print_function

import os.path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import dates
from pyeit.mesh.wrapper import PyEITMesh


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
