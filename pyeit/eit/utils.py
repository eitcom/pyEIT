# coding: utf-8
# pylint: disable=invalid-name
"""
util functions for 2D EIT
1. generate stimulation lines/patterns
"""
# Copyright (c) Benyuan Liu. All Rights Reserved.
# Distributed under the (new) BSD License. See LICENSE.txt for more info.
from __future__ import division, absolute_import, print_function

import numpy as np


def eit_scan_lines(n_el=16, el_dist=1):
    """
    generate scan matrix

    Parameters
    ----------
    n_el : int
        number of electrodes
    el_dist  : int
        distance between A and B (default=1)

    Returns
    -------
    ex_mat : NDArray
        stimulation matrix

    Notes
    -----
    in the scan of EIT (or stimulation matrix), we use 4-electrodes
    mode, where A, B are used as positive and negative stimulation
    electrodes and M, N are used as voltage measurements

         1 (A) for positive current injection,
        -1 (B) for negative current sink

    el_dist is the distance (number of electrodes) of A to B
    in 'adjacent' mode, el_dist=1, in 'apposition' mode, el_dist=n_el/2

    WARNING
    -------
    ex_mat is local index, where it is ranged from 0...15.
    In FEM applications, you should convert ex_mat to
    global index using el_pos information.

    Examples
    --------
    # default n_el=16
    if mode=='neighbor':
        ex_mat = eit_scan_lines()
    elif mode=='apposition':
        ex_mat = eit_scan_lines(el_dist=8)
    """
    ex = np.array([[i, np.mod(i+el_dist, n_el)] for i in range(n_el)])

    return ex


if __name__ == "__main__":
    m = eit_scan_lines(el_dist=8)
    print(m)
