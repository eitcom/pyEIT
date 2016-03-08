# coding: utf-8
# pylint: disable=invalid-name
""" util functions for 2D EIT """
from __future__ import absolute_import

import numpy as np


def eit_scan_lines(numEl, dist=1):
    """
    generate scan matrix,

    Note
    ----
    in the scan (or excitation matrix),
        1 for postive currrent injection,
        -1 for negative current sink
    in 'adjacent' mode, excitation dist=1
    in 'aposition' mode, excitation dist=numEl/2.
    i.e.,

    if excitation_mode=='neighbore':
        exMtx = eit_scan_lines(numEl)
    elif excitation_mode=='apposition':
        exMtx = eit_scan_lines(numEl, numEl/2)

    Parameters
    ----------
    numEl : int
        number of electrodes
    dist  : int
        distance between vpos and vneg (default=1)

    Returns
    -------
    exMtx : NDArray
        excitation matrix
    """
    exMtx = np.zeros((numEl, numEl))
    for i in range(numEl):
        exMtx[i, i % numEl] = 1
        exMtx[i, (i+dist) % numEl] = -1
    return exMtx
